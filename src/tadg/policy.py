"""Add a Gaussian head to the existing SuperMimic policy without changing its mean action."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class ActionSample:
    action: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor | None = None


class StochasticPolicy(nn.Module):
    def __init__(self, mean_policy: nn.Module, log_std_init: float = -2.3):
        super().__init__()
        if not -5.0 <= log_std_init <= 1.0:
            raise ValueError("log_std_init must be between -5 and 1")
        self.mean_policy = mean_policy
        self.cfg = mean_policy.cfg
        obs_dim = mean_policy.decoder.scale.numel()
        self.log_std_head = nn.Linear(obs_dim + self.cfg.latent_dim, self.cfg.n_joints)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, log_std_init)
        self.log_std_head.to(device=next(mean_policy.parameters()).device,
                             dtype=next(mean_policy.parameters()).dtype)

    def forward(self, obs_hist, goal, interface="motion"):
        """Deterministic mean, also compatible with the original evaluation interface."""
        return self.mean_policy(obs_hist, goal, interface)

    def distribution(self, obs_hist, goal, interface="motion"):
        z = self.mean_policy.encoders[interface](goal)
        x = torch.cat((z, obs_hist * self.mean_policy.decoder.scale), dim=-1)
        mean = self.mean_policy.decoder.mlp(x)
        log_std = self.log_std_head(x).clamp(-5.0, 1.0)
        # SuperMimic uses unbounded normalized joint setpoints. Adding tanh would change that interface.
        return Normal(mean, log_std.exp())

    def sample(self, obs_hist, goal, noise=None):
        dist = self.distribution(obs_hist, goal)
        action = dist.rsample() if noise is None else dist.mean + dist.stddev * noise
        return ActionSample(action, dist.log_prob(action).sum(-1), dist.mean, dist.stddev)
