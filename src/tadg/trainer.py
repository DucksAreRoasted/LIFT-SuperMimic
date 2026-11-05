"""Reuse the reproduced simulator/world-model loop; replace its actor update with Q + BPTT."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from supermimic.config import TrainConfig
from supermimic.imagination.compiled import ImagStep
from supermimic.training.loop import Trainer

from tadg.experience import ModelInputs, sample_real_batch
from tadg.gradients import dual_actor_step
from tadg.imagination import imagine_soft
from tadg.policy import StochasticPolicy
from tadg.sac import SACBranch, TwinQ, frozen


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


@dataclass
class TADGConfig(TrainConfig):
    run_name: str = "tadg"
    tadg: DualGradientCfg = field(default_factory=DualGradientCfg)
    warm_start: Path | None = None  # existing SuperMimic checkpoint; initializes mean and model


class DualPolicyTrainer:
    def __init__(self, wm, policy, library, cfg, dual):
        if dual.q_updates < 1 or not 0 <= dual.dynamics_weight <= 1:
            raise ValueError("q_updates must be positive and dynamics_weight must be in [0,1]")
        self.wm, self.policy, self.cfg, self.dual = wm, policy, cfg, dual
        self.inputs = ModelInputs(wm, policy, library, cfg)
        feature_dim = (wm.spec.dims.state + policy.cfg.goal_dim + wm.spec.dims.xi + 4
                       + policy.mean_policy.decoder.scale.numel())
        q = TwinQ(feature_dim, policy.cfg.n_joints, dual.q_hidden).to(next(policy.parameters()).device)
        self.sac = SACBranch(q, gamma=cfg.gamma, alpha=dual.alpha, lr=dual.q_lr, tau=dual.tau)
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.step_fn = ImagStep(wm, library.default_joint_pos, compile_mode=cfg.compile)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.sac.opt.param_groups:
            g["lr"] = self.dual.q_lr * lr / self.cfg.lr

    def update(self, buffer):
        metrics = {}
        for _ in range(self.dual.q_updates):
            batch, start = sample_real_batch(buffer, self.inputs, self.cfg.batch_size, self.cfg.history)
            metrics = self.sac.update(self.policy, batch)
        with frozen(self.wm, self.sac.q):
            if self.dual.dynamics_weight == 0:
                sample = self.policy.sample(batch.obs, batch.goal)
                q_objective = self.sac.actor_objective(batch.features, sample)
                metrics.update(dual_actor_step(self.policy, self.opt, q_objective, q_objective * 0,
                                               0.0, self.cfg.grad_clip_norm))
                metrics["q_actor_objective"] = float(q_objective.detach())
                metrics["policy_std_mean"] = float(sample.std.detach().mean())
                metrics["policy_entropy_sample"] = float(-sample.log_prob.detach().mean())
                metrics["action_abs_mean"] = float(sample.action.detach().abs().mean())
                metrics["action_abs_max"] = float(sample.action.detach().abs().max())
                return metrics
            rollout = imagine_soft(self.wm, self.policy, self.sac, self.inputs,
                                   start, self.cfg.horizon, self.step_fn)
            metrics.update(dual_actor_step(self.policy, self.opt, rollout.q_objective, rollout.objective,
                                           self.dual.dynamics_weight, self.cfg.grad_clip_norm))
        metrics.update(imag_objective=float(rollout.objective.detach()),
                       imag_return_undisc=float(rollout.return_undisc.detach()),
                       q_actor_objective=float(rollout.q_objective.detach()),
                       imag_reward_mean=float(rollout.reward_mean.detach()),
                       imag_survival_H=float(rollout.survival_end.detach()),
                       action_abs_mean=float(rollout.action_abs_mean.detach()),
                       action_abs_max=float(rollout.action_abs_max.detach()),
                       policy_std_mean=float(rollout.std_mean.detach()),
                       policy_entropy_sample=float(rollout.entropy_sample_mean.detach()))
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
        self.policy = StochasticPolicy(self.policy, cfg.tadg.log_std_init)
        self.policy_trainer = DualPolicyTrainer(self.wm, self.policy, self.lib,
                                                cfg.policy_trainer, cfg.tadg)
        if cfg.warm_start is not None:
            ck = torch.load(cfg.warm_start, map_location=self.device, weights_only=False)
            if ck["config"].get("robot", "g1") != cfg.robot:
                raise ValueError("Warm-start robot differs from requested robot")
            self.policy.mean_policy.load_state_dict(ck["policy"])
            self.wm.load_state_dict(ck["wm"])
            print(f"[INFO] initialized mean actor and world model from {cfg.warm_start}; Q starts fresh")
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
        self.command.bin_return.copy_(ck["bin_return"])
        self.command.bin_memory.copy_(ck["bin_memory"])
        self.iteration = self.start_iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        print(f"[INFO] resumed TADG at iteration {self.iteration}; replay refills during warmup")
