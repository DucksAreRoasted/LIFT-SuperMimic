"""Numerical checks for the actual changed update rules; no simulator or training jobs."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tadg.gradients import dual_actor_step, soft_return
from tadg.policy import StochasticPolicy
from tadg.sac import SACBatch, SACBranch, TwinQ, bellman_target, frozen


@pytest.mark.parametrize('weight', [0., 1.])
def test_single_branch_does_not_differentiate_inactive_objective(weight):
    actor = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.fill_(1.)
    optimizer = torch.optim.SGD(actor.parameters(), lr=.1)
    active = actor.weight.square().sum()
    inactive = torch.tensor(0.)
    q, dynamics = (active, inactive) if weight == 0 else (inactive, active)
    metrics = dual_actor_step(actor, optimizer, q, dynamics, weight, grad_clip=100)
    torch.testing.assert_close(actor.weight, torch.tensor([[1.2]]))
    assert metrics['q_actor_grad_norm' if weight == 1 else 'dynamics_grad_norm'] == 0


def test_compiled_bptt_placeholder_is_not_backpropagated_twice():
    actor = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.fill_(1.)
    compiled = torch.compile(lambda x: x.square().sum(), backend='aot_eager', fullgraph=True)
    objective = compiled(actor.weight)
    optimizer = torch.optim.SGD(actor.parameters(), lr=.1)
    dual_actor_step(actor, optimizer, objective * 0, objective, 1., grad_clip=100)
    torch.testing.assert_close(actor.weight, torch.tensor([[1.2]]))


def test_existing_mean_policy_is_preserved():
    from supermimic.policy.nets import Policy, PolicyCfg

    cfg = PolicyCfg(n_joints=2, history=2, goal_dim=28, latent_dim=4,
                    encoder_hidden=(8,), decoder_hidden=(8,))
    base = Policy(cfg).double()
    actor = StochasticPolicy(base).double()
    obs, goal = torch.randn(5, 24, dtype=torch.float64), torch.randn(5, 28, dtype=torch.float64)
    expected = base(obs, goal)
    torch.testing.assert_close(actor(obs, goal), expected, rtol=0, atol=0)
    sample = actor.sample(obs, goal, noise=torch.zeros(5, 2, dtype=torch.float64))
    torch.testing.assert_close(sample.action, expected, rtol=0, atol=0)
    # At its mean each independent Gaussian contributes -log(std) - log(2*pi)/2.
    expected_log_prob = torch.full((5,), 2 * (2.3 - 0.5 * torch.log(torch.tensor(2 * torch.pi))),
                                  dtype=torch.float64)
    torch.testing.assert_close(sample.log_prob, expected_log_prob)
    sample.action.sum().backward()
    assert actor.mean_policy.decoder.mlp.net[-1].weight.grad is not None


def test_target_and_soft_return_match_independent_numbers():
    target = bellman_target(torch.tensor([1., 1.]), torch.tensor([1., 0.]),
                           torch.tensor([3., 3.]), torch.tensor([-2., -2.]), 0.9, 0.1)
    torch.testing.assert_close(target, torch.tensor([3.88, 1.]))
    terminal = torch.tensor([3.], requires_grad=True)
    j = soft_return(torch.tensor([[1.], [2.]]), torch.tensor([[-1.], [-2.]]),
                    torch.tensor([[1.], [.5], [.25]]), terminal, gamma=.9, alpha=.1)
    torch.testing.assert_close(j, torch.tensor(2.6975))
    j.backward()
    torch.testing.assert_close(terminal.grad, torch.tensor([.2025]))


def test_dual_step_uses_one_snapshot_and_one_optimizer_step():
    actor = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        actor.weight.fill_(1.)
    opt = torch.optim.SGD(actor.parameters(), lr=.1)
    theta = actor.weight.squeeze()
    metrics = dual_actor_step(actor, opt, 2 * theta, -theta.square(), .25, grad_clip=100)
    torch.testing.assert_close(actor.weight, torch.tensor([[1.1]], dtype=torch.float64))
    assert abs(metrics["gradient_cosine"] + 1) < 1e-10


def test_sac_update_does_not_update_actor_or_target_by_backprop():
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = nn.Parameter(torch.tensor(.3))

        def sample(self, obs, goal):
            action = self.mean.expand(obs.shape[0], 1)
            return SimpleNamespace(action=action, log_prob=action[:, 0] * 0)

    actor = Actor()
    q = TwinQ(2, 1, hidden=(4,))
    branch = SACBranch(q)
    b = SACBatch(torch.randn(3, 2), torch.zeros(3, 1), torch.zeros(3, 1),
                 torch.randn(3, 1), torch.ones(3), torch.tensor([1., 0., 1.]),
                 torch.randn(3, 2), torch.zeros(3, 1), torch.zeros(3, 1))
    before = [p.clone() for p in q.parameters()]
    branch.update(actor, b)
    assert actor.mean.grad is None
    assert any(not torch.equal(a, p) for a, p in zip(before, q.parameters()))
    assert all(p.grad is None for p in branch.target_q.parameters())
    with frozen(q):
        sample = actor.sample(b.obs, b.goal)
        grad = torch.autograd.grad(branch.actor_objective(b.features, sample), actor.mean)[0]
        assert torch.isfinite(grad)
    assert all(p.requires_grad for p in q.parameters())


def test_soft_rollout_reaches_actor_through_model_and_terminal_q():
    from tadg.imagination import imagine_soft

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.tensor(.1, dtype=torch.float64))

        def sample(self, obs, goal):
            a = self.theta * obs
            return SimpleNamespace(action=a, log_prob=a[:, 0] * 0)

    class Inputs:
        lib = SimpleNamespace(last_frame_of=lambda frame: torch.full_like(frame, 100))

        def history(self, start):
            return [start.state]

        def at(self, state, xi, frame, hist):
            return hist[-1], torch.zeros_like(state), state, None

        def add_noise(self, obs):
            return obs

    class QuadraticQ(nn.Module):
        def minimum(self, state, action):
            return -state[:, 0].square()

    actor = Actor()
    sac = SimpleNamespace(q=QuadraticQ(), alpha=0., gamma=.9)
    sac.actor_objective = lambda features, sample: sac.q.minimum(features, sample.action).mean()
    start = SimpleNamespace(state=torch.ones(1, 1, dtype=torch.float64),
                            frame=torch.zeros(1, dtype=torch.long), prev_action=torch.zeros(1, 1), xi=None)

    def step(state, action, previous, xi, ref):
        following = state + action
        # -inf logits give exactly zero termination probability.
        return following, -following[:, 0].square(), None, state[:, 0] * 0 - torch.inf, following, None

    ro = imagine_soft(None, actor, sac, Inputs(), start, 2, step)
    actual = torch.autograd.grad(ro.objective, actor.theta)[0]
    t = actor.theta.detach()
    # Two rewards and a terminal value; all states depend on the same actor parameter.
    expected = -2 * (1 + t) - 4 * (.9 + .9**2) * (1 + t)**3
    torch.testing.assert_close(actual, expected)
