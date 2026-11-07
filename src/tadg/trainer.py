"""Reuse the reproduced simulator/world-model loop; replace its actor update with Q + BPTT."""

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from supermimic.config import TrainConfig
from supermimic.imagination.compiled import ImagStep
from supermimic.training.loop import Trainer

from tadg.experience import ModelInputs, sample_real_batch
from tadg.actor_safety import ActorStepGuard
from tadg.gradients import dual_actor_step
from tadg.imagination import imagine_soft
from tadg.policy import StochasticPolicy
from tadg.sac import SACBranch, TwinQ, frozen
from tadg.value_teacher import value_from_features


@dataclass
class DualGradientCfg:
    dynamics_weight: float = 0.5
    alpha: float = 1e-3  # fixed temperature, identical in SAC and imagined soft-return
    q_lr: float = 3e-4
    q_hidden: tuple[int, ...] = (512, 256)
    q_updates: int = 1
    tau: float = 0.005
    log_std_init: float = -2.3
    deterministic_collection: bool = False
    q_warmup_updates: int = 0  # Q optimizer steps before the first Q-only Actor update
    fixed_std: bool = False
    actor_lr_scale: float = 1.0
    q_teacher_v: bool = False
    require_q_calibration: bool = False
    terminal_value_mode: str = "q"  # "v": fixed pretrained V, alpha=0, BPTT-only
    q_keep_teacher: bool = False  # Separate opening the Actor from switching Q to Bellman targets.
    actor_max_change_rms: float = 0.0  # Both bounds >0 opt into local step backtracking.
    actor_max_teacher_deviation: float = 0.0
    actor_max_backtracks: int = 8


@dataclass
class TADGConfig(TrainConfig):
    run_name: str = "tadg"
    tadg: DualGradientCfg = field(default_factory=DualGradientCfg)
    warm_start: Path | None = None  # existing SuperMimic checkpoint; initializes mean and model


class DualPolicyTrainer:
    def __init__(self, wm, policy, library, cfg, dual, *, value_critic=None, teacher_policy=None):
        if dual.q_updates < 1 or not 0 <= dual.dynamics_weight <= 1:
            raise ValueError("q_updates must be positive and dynamics_weight must be in [0,1]")
        if dual.q_warmup_updates < 0 or not 0 < dual.actor_lr_scale <= 1:
            raise ValueError("q_warmup_updates must be nonnegative and actor_lr_scale in (0,1]")
        if dual.q_warmup_updates and dual.dynamics_weight >= 1:
            raise ValueError("Critic warmup requires an active Q-gradient branch")
        if dual.terminal_value_mode not in ("q", "v"):
            raise ValueError("terminal_value_mode must be q or v")
        if (dual.q_teacher_v or dual.terminal_value_mode == "v") and dual.alpha != 0:
            raise ValueError("Pretrained ordinary V guidance requires alpha=0")
        if dual.terminal_value_mode == "v" and dual.dynamics_weight <= 0:
            raise ValueError("Pretrained V endpoint requires an active BPTT branch")
        if (dual.q_teacher_v or dual.terminal_value_mode == "v") and value_critic is None:
            raise ValueError("Value guidance requires an explicitly loaded pretrained V")
        if dual.require_q_calibration and (not dual.q_teacher_v or dual.dynamics_weight >= 1):
            raise ValueError("Q calibration requires an active Q-gradient branch and pretrained V")
        if getattr(dual, 'q_keep_teacher', False) and not dual.q_teacher_v:
            raise ValueError("q_keep_teacher requires q_teacher_v")
        change = getattr(dual, 'actor_max_change_rms', 0.)
        deviation = getattr(dual, 'actor_max_teacher_deviation', 0.)
        if not (0 <= change < float('inf') and 0 <= deviation < float('inf')) or bool(change) != bool(deviation):
            raise ValueError("Both Actor step bounds must be positive, or both zero")
        if change and teacher_policy is None:
            raise ValueError("Actor step guard requires a frozen teacher policy")
        if getattr(dual, 'actor_max_backtracks', 8) < 0:
            raise ValueError("actor_max_backtracks must be nonnegative")
        self.wm, self.policy, self.cfg, self.dual = wm, policy, cfg, dual
        self.value_critic, self.teacher_policy = value_critic, teacher_policy
        if value_critic is not None:
            value_critic.requires_grad_(False).eval()
        self.inputs = ModelInputs(wm, policy, library, cfg)
        feature_dim = (wm.spec.dims.state + policy.cfg.goal_dim + wm.spec.dims.xi + 4
                       + policy.mean_policy.decoder.scale.numel())
        q = TwinQ(feature_dim, policy.cfg.n_joints, dual.q_hidden).to(next(policy.parameters()).device)
        self.sac = SACBranch(q, gamma=cfg.gamma, alpha=dual.alpha, lr=dual.q_lr, tau=dual.tau)
        self.opt = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad),
                                    lr=cfg.lr * dual.actor_lr_scale, betas=cfg.betas)
        self.step_fn = ImagStep(wm, library.default_joint_pos, compile_mode=cfg.compile)
        self.q_updates_completed = 0
        self.actor_updates_completed = 0
        self.q_calibration_passed = False
        self.q_bootstrap_initialized = False
        self.last_q_calibration_iteration = None

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr * self.dual.actor_lr_scale
        for g in self.sac.opt.param_groups:
            g["lr"] = self.dual.q_lr * lr / self.cfg.lr

    def state_dict(self):
        return {"q_updates_completed": self.q_updates_completed,
                "actor_updates_completed": self.actor_updates_completed,
                "q_calibration_passed": self.q_calibration_passed,
                "q_bootstrap_initialized": self.q_bootstrap_initialized,
                "last_q_calibration_iteration": self.last_q_calibration_iteration}

    def load_state_dict(self, state):
        q_updates = int(state["q_updates_completed"])
        actor_updates = int(state["actor_updates_completed"])
        if min(q_updates, actor_updates) < 0:
            raise ValueError("Negative policy update counters in checkpoint")
        self.q_updates_completed = q_updates
        self.actor_updates_completed = actor_updates
        self.q_calibration_passed = bool(state.get("q_calibration_passed", False))
        self.q_bootstrap_initialized = bool(state.get("q_bootstrap_initialized", False))
        self.last_q_calibration_iteration = state.get("last_q_calibration_iteration")

    def actor_step_guard(self, batch):
        change = getattr(self.dual, 'actor_max_change_rms', 0.)
        if not change:
            return None
        with torch.no_grad():
            reference = self.teacher_policy(batch.obs, batch.goal)
        return ActorStepGuard(lambda: self.policy(batch.obs, batch.goal), reference,
                              max_change_rms=change,
                              max_reference_deviation=self.dual.actor_max_teacher_deviation,
                              max_backtracks=self.dual.actor_max_backtracks)

    def teacher_value(self, features, goal):
        if self.value_critic is None:
            raise ValueError("No pretrained V teacher has been loaded")
        return value_from_features(self.value_critic, features, goal,
                                   self.wm.spec.dims.state, self.wm.spec.dims.xi)

    def update(self, buffer):
        metrics = {}
        for _ in range(self.dual.q_updates):
            batch, start = sample_real_batch(buffer, self.inputs, self.cfg.batch_size, self.cfg.history)
            teacher_stage = self.dual.q_teacher_v and (
                getattr(self.dual, 'q_keep_teacher', False)
                or (self.dual.terminal_value_mode == "v" and self.dual.dynamics_weight == 1)
                or self.q_updates_completed < self.dual.q_warmup_updates
                or (self.dual.require_q_calibration and not self.q_calibration_passed))
            if teacher_stage:
                with torch.no_grad():
                    next_value = self.teacher_value(batch.next_features, batch.next_goal)
                metrics = self.sac.update(self.policy, batch, next_value=next_value)
            else:
                if self.dual.q_teacher_v and not self.q_bootstrap_initialized:
                    # The student has passed the teacher phase; do not switch
                    # back to a Polyak target still lagging near its random init.
                    self.sac.target_q.load_state_dict(self.sac.q.state_dict())
                    self.q_bootstrap_initialized = True
                metrics = self.sac.update(self.policy, batch)
            self.q_updates_completed += 1
        waiting = (self.q_updates_completed <= self.dual.q_warmup_updates
                   or (self.dual.require_q_calibration and not self.q_calibration_passed))
        metrics.update(q_optimizer_steps=self.dual.q_updates,
                       q_updates_completed=self.q_updates_completed,
                       actor_step_performed=0,
                       critic_warmup=int(waiting),
                       calibration_pending=int(self.dual.require_q_calibration and not self.q_calibration_passed))
        if waiting:
            # SACBranch.update uses no_grad for policy sampling: no Actor
            # gradient/optimizer state is changed during critic-only warmup.
            with torch.no_grad():
                sample = self.policy.sample(batch.obs, batch.goal)
            metrics.update(policy_std_mean=float(sample.std.mean()),
                           action_mean_abs_max=float(sample.mean.abs().max()),
                           action_abs_max=float(sample.action.abs().max()))
            return metrics
        with frozen(self.wm, self.sac.q):
            step_guard = self.actor_step_guard(batch)
            if self.dual.dynamics_weight == 0:
                sample = self.policy.sample(batch.obs, batch.goal)
                q_objective = self.sac.actor_objective(batch.features, sample)
                metrics.update(dual_actor_step(self.policy, self.opt, q_objective, q_objective * 0,
                                               0.0, self.cfg.grad_clip_norm, step_guard=step_guard))
                metrics["q_actor_objective"] = float(q_objective.detach())
                metrics["policy_std_mean"] = float(sample.std.detach().mean())
                metrics["policy_entropy_sample"] = float(-sample.log_prob.detach().mean())
                metrics["action_abs_mean"] = float(sample.action.detach().abs().mean())
                metrics["action_abs_max"] = float(sample.action.detach().abs().max())
                metrics["action_mean_abs_max"] = float(sample.mean.detach().abs().max())
                self.actor_updates_completed += int(metrics["actor_step_performed"])
                return metrics
            value_endpoint = self.dual.terminal_value_mode == "v"
            rollout = imagine_soft(self.wm, self.policy, self.sac, self.inputs,
                                   start, self.cfg.horizon, self.step_fn,
                                   terminal_value_fn=self.teacher_value if value_endpoint else None,
                                   compute_q_objective=not value_endpoint or self.dual.dynamics_weight < 1)
            metrics.update(dual_actor_step(self.policy, self.opt, rollout.q_objective, rollout.objective,
                                           self.dual.dynamics_weight, self.cfg.grad_clip_norm,
                                           dynamics_scale=1 / self.cfg.horizon if value_endpoint else 1.,
                                           grad_clip_value=self.cfg.grad_clip_value if value_endpoint else None,
                                           step_guard=step_guard))
        self.actor_updates_completed += int(metrics["actor_step_performed"])
        metrics.update(imag_objective=float(rollout.objective.detach()),
                       imagination_performed=1,
                       imag_return_undisc=float(rollout.return_undisc.detach()),
                       q_actor_objective=float(rollout.q_objective.detach()),
                       imag_reward_mean=float(rollout.reward_mean.detach()),
                       imag_survival_H=float(rollout.survival_end.detach()),
                       action_abs_mean=float(rollout.action_abs_mean.detach()),
                       action_abs_max=float(rollout.action_abs_max.detach()),
                       policy_std_mean=float(rollout.std_mean.detach()),
                       policy_entropy_sample=float(rollout.entropy_sample_mean.detach()),
                       terminal_value_source_v=int(value_endpoint),
                       reward_objective=float(rollout.reward_objective.detach()),
                       entropy_objective=float(rollout.entropy_objective.detach()),
                       terminal_objective=float(rollout.terminal_objective.detach()))
        return metrics


class TADGTrainer(Trainer):
    def __init__(self, cfg):
        if cfg.resume is not None and cfg.warm_start is not None:
            raise ValueError("Choose resume OR warm_start")
        # A stochastic actor supplies collection noise. Do not add a second, unlogged policy distribution.
        cfg.env.action_noise_std = 0.0
        resume = cfg.resume
        cfg.resume = None
        try:
            super().__init__(cfg)
        finally:
            cfg.resume = resume
        if self.world != 1:
            self.env.close()
            raise ValueError("The initial TADG implementation supports a single training process")
        self.policy = StochasticPolicy(self.policy, cfg.tadg.log_std_init, learn_std=not cfg.tadg.fixed_std)
        needs_value = cfg.tadg.q_teacher_v or cfg.tadg.terminal_value_mode == "v"
        if cfg.warm_start is not None:
            ck = torch.load(cfg.warm_start, map_location=self.device, weights_only=False)
            if ck["config"].get("robot", "g1") != cfg.robot:
                raise ValueError("Warm-start robot differs from requested robot")
            self.policy.mean_policy.load_state_dict(ck["policy"])
            self.wm.load_state_dict(ck["wm"])
            if needs_value:
                if "critic" not in ck:
                    raise ValueError("Warm-start checkpoint has no pretrained V critic")
                self.critic.load_state_dict(ck["critic"])
            print(f"[INFO] initialized mean actor and world model from {cfg.warm_start}; Q starts fresh")
        if needs_value and cfg.warm_start is None and resume is None:
            raise ValueError("Value guidance requires warm_start or a value-guided resume checkpoint")
        teacher_policy = copy.deepcopy(self.policy.mean_policy).requires_grad_(False).eval() if needs_value else None
        self.policy_trainer = DualPolicyTrainer(
            self.wm, self.policy, self.lib, cfg.policy_trainer, cfg.tadg,
            value_critic=self.critic if needs_value else None, teacher_policy=teacher_policy)
        if resume is not None:
            self.load(resume)
        (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    @torch.no_grad()
    def act(self, obs):
        if self.cfg.tadg.deterministic_collection:
            return self.policy(obs["policy"], obs["goal"])
        return self.policy.sample(obs["policy"], obs["goal"]).action

    def save(self, name="ckpt_latest.pt"):
        sac = self.policy_trainer.sac
        torch.save({"iteration": self.iteration, "num_updates": self.num_updates,
                    "wm": self.wm.state_dict(),
                    # Preserve original evaluation script compatibility for deterministic mean evaluation.
                    "policy": self.policy.mean_policy.state_dict(),
                    "tadg_policy": self.policy.state_dict(), "q": sac.q.state_dict(),
                    "target_q": sac.target_q.state_dict(), "q_opt": sac.opt.state_dict(),
                    "wm_opt": self.wm_trainer.opt.state_dict(),
                    "policy_opt": self.policy_trainer.opt.state_dict(),
                    "policy_update_state": self.policy_trainer.state_dict(),
                    "teacher_value": (self.policy_trainer.value_critic.state_dict()
                                      if self.policy_trainer.value_critic is not None else None),
                    "teacher_policy": (self.policy_trainer.teacher_policy.state_dict()
                                       if self.policy_trainer.teacher_policy is not None else None),
                    "bin_return": self.command.bin_return, "bin_memory": self.command.bin_memory,
                    "config": asdict(self.cfg)}, self.log_dir / name)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck["config"].get("robot", "g1") != self.cfg.robot:
            raise ValueError("Resume robot differs from requested robot")
        sac = self.policy_trainer.sac
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["tadg_policy"])
        sac.q.load_state_dict(ck["q"])
        sac.target_q.load_state_dict(ck["target_q"])
        sac.opt.load_state_dict(ck["q_opt"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.load_state_dict(ck.get("policy_update_state", {
            "q_updates_completed": ck["num_updates"] * self.cfg.tadg.q_updates,
            "actor_updates_completed": ck["num_updates"],
        }))
        if self.policy_trainer.value_critic is not None:
            if ck.get("teacher_value") is None or ck.get("teacher_policy") is None:
                raise ValueError("Resume checkpoint lacks frozen V/Actor teachers; use pretrained warm_start")
            self.policy_trainer.value_critic.load_state_dict(ck["teacher_value"])
            self.policy_trainer.teacher_policy.load_state_dict(ck["teacher_policy"])
        self.command.bin_return.copy_(ck["bin_return"])
        self.command.bin_memory.copy_(ck["bin_memory"])
        self.iteration = self.start_iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        print(f"[INFO] resumed TADG at iteration {self.iteration}; replay refills during warmup")
