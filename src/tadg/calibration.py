"""Held-out real-simulator local action checks before Q takes over the Actor."""

import math
import time

import torch


def calibration_metrics(predicted, targets, *, min_return_gap=.01, min_pairs=6,
                        min_pairwise_accuracy=.65, max_normalized_rmse=.25):
    if predicted.shape != targets.shape or predicted.ndim != 2 or predicted.shape[1] < 2:
        raise ValueError("Calibration needs [states, candidate actions] arrays with matching shapes")
    if (not math.isfinite(min_return_gap) or min_return_gap < 0 or min_pairs < 1
            or not 0 <= min_pairwise_accuracy <= 1 or not 0 < max_normalized_rmse):
        raise ValueError("Invalid Q calibration thresholds")
    finite = bool(torch.isfinite(predicted).all() and torch.isfinite(targets).all())
    if not finite:
        return {"passed": False, "finite": False, "normalized_rmse": None,
                "pairwise_accuracy": None, "n_comparable_pairs": 0}
    error = (predicted - targets).square().mean().sqrt()
    scale = targets.abs().mean().clamp_min(1e-3)
    target_diff = targets[..., :, None] - targets[..., None, :]
    pred_diff = predicted[..., :, None] - predicted[..., None, :]
    upper = torch.ones_like(target_diff, dtype=torch.bool).triu(1)
    mask = upper & (target_diff.abs() > min_return_gap)
    pairs = int(mask.sum())
    accuracy = None
    if pairs:
        correct = (pred_diff[mask].sign() == target_diff[mask].sign()).float()
        ties = (pred_diff[mask] == 0).float() * .5
        accuracy = float((correct + ties).mean())
    nrmse = float(error / scale)
    return {"passed": bool(pairs >= min_pairs and accuracy >= min_pairwise_accuracy
                           and nrmse <= max_normalized_rmse),
            "finite": True, "rmse": float(error), "normalized_rmse": nrmse,
            "pairwise_accuracy": accuracy, "n_comparable_pairs": pairs}


@torch.no_grad()
def calibrate_q(trainer, protocol, directory):
    # Lazy imports keep numerical metric tests independent of simulator setup.
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.state_io import StateIO

    from tadg.experience import ModelInputs
    from tadg.recording import evaluation_rng, write_json

    started = time.perf_counter()
    cfg = protocol.get("q_calibration", {})
    states, actions = int(cfg.get("states", 4)), int(cfg.get("actions", 5))
    horizon = int(cfg.get("rollout_steps", 50))
    noise = float(cfg.get("action_noise_std", .1))
    if states < 1 or actions < 2 or horizon < 1 or not math.isfinite(noise) or noise <= 0:
        raise ValueError("Invalid Q calibration sampling configuration")
    clips = cfg.get("clips", protocol["eval_clips"][:4])
    files = discover_npz(trainer.cfg.data.npz_dir, names=clips)
    if not files or len(files) != len(clips):
        raise ValueError("Q calibration motion clips are missing")
    updater = trainer.policy_trainer
    if updater.teacher_policy is None or updater.value_critic is None:
        raise ValueError("Q calibration requires a loaded frozen Actor and V")
    env = None
    with evaluation_rng(int(cfg.get("seed", 24680))):
        try:
            env = make_env(SuperMimicEnvCfgArgs(
                npz_files=[str(p) for p in files], robot_name=trainer.cfg.robot,
                history_length=trainer.cfg.policy.history, num_envs=states * actions,
                future_frames=trainer.cfg.env.future_frames, episode_length_s=1e5,
                action_noise_std=0., obs_noise=False, reset_noise=False,
                domain_randomization=False, sampling_mode="start", record_transitions=True,
                replay_capacity=32, anchor_obs=trainer.cfg.env.anchor_obs,
                seed=int(cfg.get("seed", 24680))), device=trainer.device)
            command = env.command_manager.get_term("motion")
            library = command.library
            selected = []
            for i in range(states):
                clip = library.clips[i % len(library.clips)]
                groups = math.ceil(states / len(library.clips))
                fraction = (i // len(library.clips) + 1) / (groups + 1)
                selected.append(clip.start + int((clip.length - 1) * fraction))
            frames = torch.tensor(selected, device=env.device).repeat_interleave(actions)
            command.forced_frame[:] = frames
            obs, _ = env.reset()
            command.forced_frame[:] = frames
            state_io = StateIO(env)
            recorder = env.recorder_manager.get_term("replay")
            inputs = ModelInputs(trainer.wm, trainer.policy, library, updater.cfg)
            xi = recorder.ctx.read()
            features = inputs.features(state_io.read(), xi, command.frame,
                                       obs["policy"], obs["goal"])
            mean = updater.teacher_policy(obs["policy"], obs["goal"])
            perturbation = torch.randn(states, actions, mean.shape[-1], device=env.device) * noise
            perturbation[:, 0] = 0.  # Always retain the original mean action as a candidate.
            first_action = mean + perturbation.flatten(0, 1)
            predicted = updater.sac.q.minimum(features, first_action)
            alive = torch.ones(states * actions, device=env.device, dtype=torch.bool)
            returns = torch.zeros_like(predicted)
            steps = torch.zeros_like(predicted, dtype=torch.long)
            for h in range(horizon):
                action = first_action if h == 0 else updater.teacher_policy(obs["policy"], obs["goal"])
                obs, reward, terminated, timeout, _ = env.step(action)
                returns += updater.sac.gamma**h * torch.where(alive, reward, 0.)
                steps += alive.long()
                alive &= ~(terminated | timeout)
                if not alive.any():
                    break
            # Surviving trajectories use the ORIGINAL fixed V, never the student Q.
            features = inputs.features(state_io.read(), xi, command.frame,
                                       obs["policy"], obs["goal"])
            tail = updater.teacher_value(features, obs["goal"])
            targets = returns + updater.sac.gamma**steps * torch.where(alive, tail, 0.)
            predicted, targets = predicted.reshape(states, actions), targets.reshape(states, actions)
            thresholds = {k: cfg[k] for k in ("min_return_gap", "min_pairs", "min_pairwise_accuracy",
                                             "max_normalized_rmse") if k in cfg}
            report = calibration_metrics(predicted, targets, **thresholds)
            report.update(iteration=trainer.iteration, q_updates=updater.q_updates_completed,
                          target_kind="real_n_step_return_plus_frozen_v", seed=int(cfg.get("seed", 24680)),
                          states=states, actions=actions, rollout_steps=horizon,
                          selected_frames=selected, clips=list(clips), thresholds=thresholds,
                          predicted=predicted.cpu().tolist(), targets=targets.cpu().tolist(),
                          actual_steps=steps.reshape(states, actions).cpu().tolist(),
                          surviving=alive.reshape(states, actions).cpu().tolist(),
                          env_transitions=states * actions * (h + 1),
                          wall_s=time.perf_counter() - started,
                          limitation="Local action check with pretrained V bootstrap, not full Monte Carlo certification")
            write_json(directory / "calibration.json", report)
            return report
        finally:
            if env is not None:
                env.close()
