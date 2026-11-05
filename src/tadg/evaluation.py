"""Identical deterministic evaluation for baseline and TADG, including raw tracking errors."""

import time
from pathlib import Path

import numpy as np
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.state_io import StateIO
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.rewards import reanchor_reference
from supermimic.kinematics.rotations import heading_yaw, wrap_to_pi

from tadg.recording import evaluation_rng, write_json


def quaternion_angle(q1, q2):
    dot = (q1 * q2).sum(-1) / (q1.norm(dim=-1) * q2.norm(dim=-1)).clamp_min(1e-12)
    return 2 * dot.abs().clamp(0, 1).acos()


@torch.no_grad()
def evaluate(trainer, protocol, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = discover_npz(trainer.cfg.data.npz_dir, names=protocol["eval_clips"])
    if len(files) != len(protocol["eval_clips"]):
        raise ValueError("Some frozen evaluation clips were not found")
    n = len(files)
    max_steps = int(protocol.get("eval_max_steps", 0))
    stride = int(protocol.get("trace_stride", 10))
    seed = int(protocol.get("eval_seed", 12345))
    if stride < 1 or max_steps < 0:
        raise ValueError("trace_stride must be positive and eval_max_steps nonnegative")
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], robot_name=trainer.cfg.robot,
        history_length=trainer.cfg.policy.history, future_frames=trainer.cfg.env.future_frames,
        num_envs=n, episode_length_s=1e5, action_noise_std=0., obs_noise=False,
        reset_noise=False, domain_randomization=False, sampling_mode="start",
        record_transitions=True, replay_capacity=32, seed=seed, anchor_obs=trainer.cfg.env.anchor_obs)
    was_training = trainer.policy.training
    trainer.policy.eval()
    started = time.perf_counter()
    env = None
    try:
        with evaluation_rng(seed):
            env = make_env(args, device=trainer.device)
            command = env.command_manager.get_term("motion")
            library = command.library
            command.forced_frame[:] = library.clip_start
            obs, _ = env.reset()
            command.forced_frame[:] = library.clip_start
            recorder = env.recorder_manager.get_term("replay")
            buffer = recorder.buffer
            state_io = StateIO(env)
            heads = trainer.wm.heads
            alive = torch.ones(n, dtype=torch.bool, device=env.device)
            steps = torch.zeros(n, dtype=torch.long, device=env.device)
            returns = torch.zeros(n, device=env.device)
            failures = torch.zeros(n, dtype=torch.bool, device=env.device)
            task_ended = torch.zeros_like(failures)
            previous_action = torch.zeros(n, trainer.cfg.policy.n_joints, device=env.device)
            totals = {}
            traces = []
            reward_names = list(env.reward_manager.active_terms)
            reward_sums = torch.zeros(n, len(reward_names), device=env.device)
            tick = 0
            while alive.any():
                frame = command.frame.clone()
                ref = library.bodies(frame)
                reference = library.ref(frame)
                action = trainer.policy(obs["policy"], obs["goal"])
                active = alive.clone()
                obs, reward, terminated, timeout, _ = env.step(action)
                reached = state_io.read()
                # Automatic reset has already happened; recover the real terminal state from replay.
                ended = (terminated | timeout) & active
                ids = ended.nonzero().flatten()
                if ids.numel():
                    last_row = (buffer.ptr[ids] - 1) % buffer.T
                    if not buffer.terminal[ids, last_row].all():
                        raise RuntimeError("Evaluation recorder did not retain pre-reset terminal state")
                    reached[ids] = buffer.s[ids, last_row]
                bodies = heads.robot_bodies(reached)
                ai = heads.anchor_index
                relative_pos, relative_quat = reanchor_reference(
                    ref.pos_w, ref.quat_w, ref.pos_w[:, ai], ref.quat_w[:, ai],
                    bodies["pos"][:, ai], bodies["quat"][:, ai])
                qj = reached[:, 7:heads.fk.nq]
                qd = reached[:, heads.fk.nq + 6:]
                lo, hi = heads.soft_limits[..., 0], heads.soft_limits[..., 1]
                excess = (lo - qj).clamp_min(0) + (qj - hi).clamp_min(0)
                metrics = {
                    "root_pos_error_m": (reached[:, :3] - reference.root_pos_w).norm(dim=-1),
                    "root_ori_error_rad": quaternion_angle(reached[:, 3:7], reference.root_quat_w),
                    "heading_error_rad": wrap_to_pi(
                        heading_yaw(reached[:, 3:7]) - heading_yaw(reference.root_quat_w)).abs(),
                    "body_pos_global_m": (bodies["pos"] - ref.pos_w).norm(dim=-1).mean(-1),
                    "body_pos_reanchored_m": (bodies["pos"] - relative_pos).norm(dim=-1).mean(-1),
                    "body_ori_reanchored_rad": quaternion_angle(bodies["quat"], relative_quat).mean(-1),
                    "body_lin_vel_error_m_s": (bodies["lin_vel"] - ref.lin_vel_w).norm(dim=-1).mean(-1),
                    "body_ang_vel_error_rad_s": (bodies["ang_vel"] - ref.ang_vel_w).norm(dim=-1).mean(-1),
                    "joint_pos_rmse_rad": (qj - reference.joint_pos).square().mean(-1).sqrt(),
                    "joint_vel_rmse_rad_s": (qd - reference.joint_vel).square().mean(-1).sqrt(),
                    "action_delta_l2": (action - previous_action).norm(dim=-1),
                    "action_abs_mean": action.abs().mean(-1),
                    "soft_limit_fraction": (excess > 0).float().mean(-1),
                    "soft_limit_excess_rad": excess.mean(-1),
                }
                for key, value in metrics.items():
                    totals[key] = totals.get(key, torch.zeros_like(value)) + torch.where(active, value, 0.)
                returns += torch.where(active, reward, 0.)
                reward_sums += env.reward_manager.term_rewards * active[:, None]
                steps += active.long()
                failures |= terminated & active
                task_ended |= timeout & active & (frame >= library.last_frame_of(frame))
                if tick % stride == 0:
                    traces.append({"tick": tick, "frame": frame.cpu().numpy(),
                                   "active": active.cpu().numpy(), "state": reached.cpu().numpy(),
                                   "action": action.cpu().numpy(), "reward": reward.cpu().numpy(),
                                   **{k: v.cpu().numpy() for k, v in metrics.items()}})
                alive &= ~(terminated | timeout)
                if max_steps:
                    alive &= steps < max_steps
                previous_action = action
                tick += 1
            max_reward = heads.reward_cfg.max_step_reward
            results = []
            for i, clip in enumerate(library.clips):
                count = int(steps[i])
                results.append({"clip": clip.name, "steps": count, "clip_frames": clip.length,
                                "failed": bool(failures[i]), "reference_ended": bool(task_ended[i]),
                                "evaluation_capped": bool(max_steps and count >= max_steps
                                                          and not failures[i] and not task_ended[i]),
                                "completion_fraction": min(count / max(clip.length, 1), 1.0),
                                "return": float(returns[i]),
                                "norm_return": float(returns[i]) / (max_reward * max(count, 1)),
                                "window_norm_return": float(returns[i]) / (max_reward * max(
                                    min(clip.length, max_steps) if max_steps else clip.length, 1)),
                                "metrics": {k: float(v[i]) / max(count, 1) for k, v in totals.items()},
                                "reward_terms_per_step": {k: float(reward_sums[i, j]) / max(count, 1)
                                                          for j, k in enumerate(reward_names)}})
            summary = {"clips": len(results), "failed": sum(r["failed"] for r in results),
                       "capped": sum(r["evaluation_capped"] for r in results),
                       "mean_norm_return": sum(r["norm_return"] for r in results) / len(results),
                       "mean_window_norm_return": sum(r["window_norm_return"] for r in results) / len(results),
                       "mean_completion_fraction": sum(r["completion_fraction"] for r in results) / len(results),
                       "metrics": {k: sum(r["metrics"][k] for r in results) / len(results)
                                   for k in totals},
                       "env_transitions": int(steps.sum()), "wall_s": time.perf_counter() - started}
            write_json(directory / "evaluation.json", {
                "protocol": protocol, "iteration": trainer.iteration, "summary": summary,
                "clips": results, "alignment": "post-action pre-reset state against that transition's target frame",
                "unrecorded": ["mechanical energy", "contact impulses", "real-robot safety"]})
            if traces:
                np.savez_compressed(directory / "traces.npz",
                                    **{key: np.asarray([t[key] for t in traces]) for key in traces[0]})
            return summary
    finally:
        if env is not None:
            env.close()
        trainer.policy.train(was_training)
