"""Identical logging and evaluation around the original and dual-gradient training loops."""

import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

import torch
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer

from tadg.evaluation import evaluate
from tadg.recording import file_sha256, write_json
from tadg.trainer import TADGConfig, TADGTrainer


def apply_config(instance, values):
    known = {f.name for f in fields(instance)}
    for key, value in values.items():
        if key not in known:
            # A checkpoint can carry experiment-specific extras; do not pass them to the baseline.
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            apply_config(current, value)
        elif isinstance(current, Path) and value is not None:
            setattr(instance, key, Path(value))
        else:
            setattr(instance, key, value)
    return instance


def make_config(checkpoint, protocol, method, seed, output):
    saved = checkpoint["config"]
    if protocol.get("robot", saved.get("robot", "g1")) != saved.get("robot", "g1"):
        raise ValueError("The protocol robot differs from the pretrained checkpoint robot")
    cfg = TrainConfig() if method == "supermimic" else TADGConfig()
    apply_config(cfg, saved)
    cfg.robot = saved.get("robot", "g1")
    cfg.seed = seed
    cfg.device = protocol.get("device", "cuda:0")
    cfg.run_name = Path(output).name
    cfg.log_root = Path(output).parent
    cfg.resume = None
    cfg.data.npz_dir = Path(protocol["motion_dir"])
    cfg.data.clips = protocol["train_clips"]
    cfg.data.library = cfg.data.manifest = cfg.data.limit = None
    cfg.env.compact_library = False
    cfg.env.num_envs = int(protocol["num_envs"])
    cfg.env.replay_capacity = int(protocol["replay_capacity"])
    cfg.wm_trainer.batch_size = int(protocol["wm_batch_size"])
    cfg.wm_trainer.batches_per_update = int(protocol["wm_batches_per_update"])
    cfg.policy_trainer.batch_size = int(protocol["actor_batch_size"])
    cfg.policy_trainer.horizon = int(protocol["horizon"])
    cfg.policy_trainer.compile = protocol.get("compile", "default")
    cfg.wm_trainer.compile = protocol.get("compile", "default")
    cfg.schedule.total_iterations = int(protocol["train_iterations"])
    cfg.schedule.warmup_iterations = int(protocol["warmup_iterations"])
    cfg.schedule.updates_per_env_step = float(protocol["updates_per_env_step"])
    cfg.schedule.policy_updates_per_wm_update = 1
    cfg.schedule.log_interval = int(protocol["log_interval"])
    cfg.schedule.checkpoint_interval = int(protocol["eval_interval"])
    cfg.schedule.max_hours = protocol.get("max_hours_per_run")
    if method == "supermimic":
        scale = float(protocol.get("supermimic", {}).get("actor_lr_scale", 1.0))
        if not 0 < scale <= 1:
            raise ValueError("Baseline actor_lr_scale must be in (0,1]")
        cfg.policy_trainer.lr *= scale
        cfg.policy_trainer.critic_lr *= scale
    if method != "supermimic":
        cfg.warm_start = Path(protocol["checkpoint"])
        cfg.tadg.dynamics_weight = {"q_only": 0.0, "bptt_soft": 1.0,
                                    "dual_fixed": float(protocol["dynamics_weight"])}[method]
        cfg.tadg.alpha = float(protocol["alpha"])
        cfg.tadg.q_updates = int(protocol["q_updates"])
        cfg.tadg.deterministic_collection = bool(protocol.get("deterministic_collection", False))
        options = protocol.get(method, {})
        cfg.tadg.q_warmup_updates = int(options.get("q_warmup_updates", 0))
        cfg.tadg.fixed_std = bool(options.get("fixed_std", False))
        cfg.tadg.actor_lr_scale = float(options.get("actor_lr_scale", 1.0))
        cfg.tadg.q_teacher_v = bool(options.get("q_teacher_v", False))
        cfg.tadg.require_q_calibration = bool(options.get("require_q_calibration", False))
        cfg.tadg.terminal_value_mode = str(options.get("terminal_value_mode", "q"))
        cfg.tadg.q_keep_teacher = bool(options.get("q_keep_teacher", False))
        cfg.tadg.actor_max_change_rms = float(options.get("actor_max_change_rms", 0.))
        cfg.tadg.actor_max_teacher_deviation = float(options.get("actor_max_teacher_deviation", 0.))
        cfg.tadg.actor_max_backtracks = int(options.get("actor_max_backtracks", 8))
    if cfg.schedule.warmup_iterations < max(cfg.wm_trainer.window + 2, cfg.policy.history + 2):
        raise ValueError("Warmup is too short for replay windows and policy history")
    return cfg


class RecordedMixin:
    def record_evaluation(self):
        if self._last_evaluation == self.iteration:
            return
        directory = self.log_dir / "evaluations" / f"step_{self.iteration:08d}"
        summary = evaluate(self, self.protocol, directory)
        self.records.times["evaluation_s"] += summary["wall_s"]
        self.records.counters["eval_env_transitions"] += summary["env_transitions"]
        self.records.event("evaluation", iteration=self.iteration,
                           env_transitions=self.iteration * self.env.num_envs, summary=summary)
        self._last_evaluation = self.iteration
        self.record_q_calibration()

    def record_q_calibration(self):
        dual = getattr(self.cfg, "tadg", None)
        if dual is None or not dual.require_q_calibration:
            return
        updater = self.policy_trainer
        interval = int(self.protocol.get('q_calibration', {}).get('recheck_interval', 0))
        if interval < 0:
            raise ValueError("q_calibration.recheck_interval must be nonnegative")
        last = updater.last_q_calibration_iteration
        if updater.q_updates_completed < dual.q_warmup_updates:
            return
        if updater.q_calibration_passed and (
                not interval or (last is not None and self.iteration - last < interval)):
            return
        from tadg.calibration import calibrate_q

        report = calibrate_q(self, self.protocol, self.log_dir / "calibration" / f"step_{self.iteration:08d}")
        updater.q_calibration_passed = report["passed"]
        updater.last_q_calibration_iteration = self.iteration
        if not report['passed']:
            updater.q_bootstrap_initialized = False
        self.records.times["calibration_s"] += report["wall_s"]
        self.records.counters["calibration_env_transitions"] += report["env_transitions"]
        self.records.event("q_calibration", iteration=self.iteration, report=report)

    def save(self, name="ckpt_latest.pt"):
        started = time.perf_counter()
        super().save(name)
        path = self.log_dir / name
        elapsed = time.perf_counter() - started
        self.records.times["checkpoint_s"] += elapsed
        self.records.event("checkpoint", iteration=self.iteration, file=str(path),
                           bytes=path.stat().st_size, duration_s=elapsed)
        if name == "ckpt_latest.pt":
            self.record_evaluation()

    def attach_records(self, records, protocol):
        self.records, self.protocol = records, protocol
        self._last_evaluation = None
        records.attach(self)
        write_json(self.log_dir / "resolved_config.json", asdict(self.cfg))

    def export_recent_replay(self):
        fields_to_save = ("s", "a", "a_exec", "xi", "r", "done", "frame", "clip", "ep_id",
                          "terminal", "ep_start", "valid", "ptr", "ep_counter")
        path = self.log_dir / "recent_replay.pt"
        torch.save({"tensors": {name: getattr(self.buffer, name).detach().cpu() for name in fields_to_save},
                    "iteration": self.iteration, "capacity_per_env": self.buffer.T,
                    "done_semantics": "state_terminal_v2",
                    "note": "ring buffer snapshot, NOT a complete stream of all training transitions"}, path)
        self.records.event("recent_replay", path=str(path), bytes=path.stat().st_size,
                           sha256=file_sha256(path))


class RecordedTADGTrainer(RecordedMixin, TADGTrainer):
    pass


class RecordedBaselineTrainer(RecordedMixin, Trainer):
    def warm_start(self, checkpoint):
        self.policy.load_state_dict(checkpoint["policy"])
        self.wm.load_state_dict(checkpoint["wm"])
        self.critic.load_state_dict(checkpoint["critic"])
        # Fresh optimizers and reset sampler for the selected clip subset, matching a new continuation run.
