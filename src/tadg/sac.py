"""SAC's real-transition critic update and Q-based actor objective."""

import copy
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@contextmanager
def frozen(*modules):
    parameters = [p for m in modules for p in m.parameters()]
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


class TwinQ(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden=(512, 256)):
        super().__init__()

        def network():
            layers = []
            d = feature_dim + action_dim
            for width in hidden:
                layers.extend((nn.Linear(d, width), nn.SiLU()))
                d = width
            layers.append(nn.Linear(d, 1))
            return nn.Sequential(*layers)

        self.q1, self.q2 = network(), network()

    def forward(self, features, action):
        x = torch.cat((features, action), dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def minimum(self, features, action):
        q1, q2 = self(features, action)
        return torch.minimum(q1, q2)


@dataclass
class SACBatch:
    features: torch.Tensor
    obs: torch.Tensor
    goal: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    bootstrap: torch.Tensor
    next_features: torch.Tensor
    next_obs: torch.Tensor
    next_goal: torch.Tensor


def bellman_target(reward, bootstrap, next_q, next_log_prob, gamma, alpha):
    return reward + gamma * bootstrap * (next_q - alpha * next_log_prob)


class SACBranch:
    def __init__(self, q, *, gamma=0.99, alpha=1e-3, lr=3e-4, tau=0.005):
        if not 0 < gamma <= 1 or alpha < 0 or not 0 < tau <= 1:
            raise ValueError("Invalid SAC discount, temperature or target update rate")
        self.q = q
        self.target_q = copy.deepcopy(q).requires_grad_(False)
        self.opt = torch.optim.Adam(q.parameters(), lr=lr)
        self.gamma, self.alpha, self.tau = gamma, alpha, tau

    def update(self, policy, batch, *, next_value=None):
        with torch.no_grad():
            if next_value is None:
                sample = policy.sample(batch.next_obs, batch.next_goal)
                next_q = self.target_q.minimum(batch.next_features, sample.action)
                target = bellman_target(batch.reward, batch.bootstrap, next_q, sample.log_prob,
                                        self.gamma, self.alpha)
            else:
                if next_value.shape != batch.reward.shape:
                    raise ValueError("Teacher V must return one value per transition")
                target = batch.reward + self.gamma * batch.bootstrap * next_value
        q1, q2 = self.q(batch.features, batch.action)
        loss = ((q1 - target).square() + (q2 - target).square()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Q loss; optimizer was not stepped")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0, error_if_nonfinite=True)
        self.opt.step()
        with torch.no_grad():
            for p_target, p in zip(self.target_q.parameters(), self.q.parameters()):
                p_target.lerp_(p, self.tau)
        return {"q_loss": float(loss.detach()), "q_grad_norm": float(norm),
                "q_target_source_v": int(next_value is not None),
                "q_target_mean": float(target.mean()), "q1_mean": float(q1.detach().mean()),
                "q2_mean": float(q2.detach().mean()),
                "q_disagreement_abs": float((q1 - q2).detach().abs().mean()),
                "q_td_abs": float((q1.detach() - target).abs().mean()),
                "q_td_max": float((q1.detach() - target).abs().max()),
                "bootstrap_fraction": float(batch.bootstrap.mean())}

    def actor_objective(self, features, sample):
        """Freeze Q parameters at the caller; retain gradients through its action input."""
        return (self.q.minimum(features, sample.action) - self.alpha * sample.log_prob).mean()
