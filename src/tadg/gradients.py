"""Combine gradients computed at one actor snapshot, then take exactly one optimizer step."""

import math

import torch


def dual_actor_step(policy, optimizer, q_objective, dynamics_objective, dynamics_weight=0.5,
                    grad_clip=1.0, *, objective_scale=1., dynamics_scale=1., grad_clip_value=None,
                    step_guard=None):
    weight = float(torch.as_tensor(dynamics_weight).detach())
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("dynamics_weight must be a finite scalar between 0 and 1")
    params = tuple(p for p in policy.parameters() if p.requires_grad)
    q_objective = q_objective * objective_scale
    dynamics_objective = dynamics_objective * objective_scale * dynamics_scale
    # Pure branches must not traverse the inactive placeholder graph. In the
    # V-only case that placeholder shares the compiled BPTT graph: retaining
    # it is incompatible with AOTAutograd donated buffers.
    q_grad = (torch.autograd.grad(q_objective, params, retain_graph=weight > 0, allow_unused=True)
              if weight < 1 else (None,) * len(params))
    d_grad = (torch.autograd.grad(dynamics_objective, params, allow_unused=True)
              if weight > 0 else (None,) * len(params))
    q_grad = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, q_grad))
    d_grad = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, d_grad))
    if not torch.stack([torch.isfinite(g).all() for g in (*q_grad, *d_grad)]).all():
        raise FloatingPointError("Non-finite actor gradient; optimizer was not stepped")
    q_norm = torch.sqrt(sum(g.square().sum() for g in q_grad))
    d_norm = torch.sqrt(sum(g.square().sum() for g in d_grad))
    dot = sum((q * d).sum() for q, d in zip(q_grad, d_grad))
    cosine = dot / (q_norm * d_norm).clamp_min(1e-12)
    optimizer.zero_grad(set_to_none=True)
    for p, q, d in zip(params, q_grad, d_grad):
        # Both inputs are ascent objectives; optimizers perform descent.
        p.grad = -((1 - weight) * q + weight * d).detach()
    mixed_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip, error_if_nonfinite=True)
    if grad_clip_value is not None:
        torch.nn.utils.clip_grad_value_(params, grad_clip_value)
    if step_guard is None:
        optimizer.step()
        step_metrics = {'actor_step_performed': 1}
    else:
        step_metrics = step_guard.step(optimizer)
    return {"dynamics_weight": weight, "q_actor_grad_norm": float(q_norm),
            "dynamics_grad_norm": float(d_norm), "gradient_cosine": float(cosine),
            "policy_grad_norm": float(mixed_norm), **step_metrics}


def soft_return(rewards, log_probs, survival, terminal_value, *, gamma, alpha):
    """rewards/log_probs [H,N], survival [H+1,N]; preserve terminal state gradients."""
    horizon = rewards.shape[0]
    discount = gamma ** torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    stage = rewards - alpha * log_probs
    return ((discount[:, None] * survival[:-1] * stage).sum(0)
            + gamma**horizon * survival[-1] * terminal_value).mean()
