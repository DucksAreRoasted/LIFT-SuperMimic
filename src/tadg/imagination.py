"""Stochastic imagined soft-return using SuperMimic's existing differentiable model and FK."""

from dataclasses import dataclass

import torch

from tadg.gradients import soft_return


@dataclass
class SoftRollout:
    objective: torch.Tensor
    q_objective: torch.Tensor
    reward_mean: torch.Tensor
    survival_end: torch.Tensor
    action_abs_mean: torch.Tensor
    return_undisc: torch.Tensor
    entropy_sample_mean: torch.Tensor
    std_mean: torch.Tensor
    action_abs_max: torch.Tensor
    reward_objective: torch.Tensor | None = None
    entropy_objective: torch.Tensor | None = None
    terminal_objective: torch.Tensor | None = None


def imagine_soft(wm, policy, sac, inputs, start, horizon, step_fn, *,
                 terminal_value_fn=None, compute_q_objective=True):
    if horizon < 1:
        raise ValueError("A fixed imagination horizon must be positive")
    state, frame, previous_action = start.state, start.frame, start.prev_action
    hist = inputs.history(start)
    alive = torch.ones_like(frame, dtype=state.dtype)
    survival = [alive]
    rewards, log_probs, magnitudes = [], [], []
    standard_deviations, maxima = [], []
    q_objective = None
    last = inputs.lib.last_frame_of(frame)
    for _ in range(horizon):
        obs, goal, features, ref = inputs.at(state, start.xi, frame, hist)
        sample = policy.sample(obs, goal)
        if q_objective is None and compute_q_objective:
            # Same actual starting state, observation history AND initial action sample for both paths.
            q_objective = sac.actor_objective(features, sample)
        next_state, reward, _, logits, obs_clean, _ = step_fn(
            state, sample.action, previous_action, start.xi, ref)
        termination = torch.where(frame >= last, torch.ones_like(logits), logits.sigmoid())
        alive = alive * (1 - termination)
        survival.append(alive)
        rewards.append(reward)
        log_probs.append(sample.log_prob)
        magnitudes.append(sample.action.abs().mean())
        standard_deviations.append(sample.std.mean() if getattr(sample, "std", None) is not None
                                   else sample.action.new_tensor(float("nan")))
        maxima.append(sample.action.abs().max())
        hist = hist[1:] + [inputs.add_noise(obs_clean)]
        state, frame, previous_action = next_state, torch.minimum(frame + 1, last), sample.action

    obs, goal, features, _ = inputs.at(state, start.xi, frame, hist)
    if terminal_value_fn is None:
        terminal_sample = policy.sample(obs, goal)
        terminal_value = sac.q.minimum(features, terminal_sample.action) - sac.alpha * terminal_sample.log_prob
    else:
        terminal_value = terminal_value_fn(features, goal)
    objective = soft_return(torch.stack(rewards), torch.stack(log_probs), torch.stack(survival),
                            terminal_value, gamma=sac.gamma, alpha=sac.alpha)
    if q_objective is None:
        q_objective = objective * 0  # V-guided ablation must not depend on cold Q gradients.
    discount = sac.gamma ** torch.arange(horizon, device=state.device, dtype=state.dtype)
    weights = discount[:, None] * torch.stack(survival[:-1])
    reward_objective = (weights * torch.stack(rewards)).sum(0).mean()
    entropy_objective = (-sac.alpha * weights * torch.stack(log_probs)).sum(0).mean()
    terminal_objective = (sac.gamma**horizon * survival[-1] * terminal_value).mean()
    return SoftRollout(objective, q_objective, torch.stack(rewards).mean(), survival[-1].mean(),
                       torch.stack(magnitudes).mean(),
                       (torch.stack(survival[:-1]) * torch.stack(rewards)).sum(0).mean(),
                       -torch.stack(log_probs).mean(), torch.stack(standard_deviations).mean(),
                       torch.stack(maxima).max(), reward_objective, entropy_objective, terminal_objective)
