"""Real value-teacher, calibration and Actor-gating regressions; no simulator needed."""
from types import SimpleNamespace

import pytest
import torch
from test_q_only_repair import make_updater
from torch import nn

from tadg.calibration import calibration_metrics
from tadg.sac import SACBatch, SACBranch, TwinQ
from tadg.value_teacher import value_from_features


def test_value_adapter_uses_raw_goal_and_keeps_input_gradient():
    class Value(nn.Module):
        def forward(self, state, goal, xi, root):
            torch.testing.assert_close(goal, torch.tensor([[5., 6.]]))
            torch.testing.assert_close(xi, torch.tensor([[7.]]))
            torch.testing.assert_close(root, torch.tensor([[8., 9., 10., 11.]]))
            return state.sum(-1)

    features = torch.tensor([[1., 2., 50., 60., 7., 8., 9., 10., 11., 100.]], requires_grad=True)
    value = value_from_features(Value(), features, torch.tensor([[5., 6.]]), 2, 1)
    value.sum().backward()
    torch.testing.assert_close(features.grad, torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]]))


def test_q_teacher_target_does_not_sample_policy_or_backprop_into_teacher():
    class NoSampling:
        def sample(self, *args):
            raise AssertionError('V teacher must replace the random target Q')

    branch = SACBranch(TwinQ(1, 1, (4,)), gamma=.9, alpha=0.)
    b = SACBatch(torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1), torch.zeros(2, 1),
                 torch.tensor([1., 2.]), torch.tensor([1., 0.]), torch.ones(2, 1),
                 torch.ones(2, 1), torch.ones(2, 1))
    teacher = torch.tensor([10., 20.], requires_grad=True)
    result = branch.update(NoSampling(), b, next_value=teacher)
    assert result['q_target_mean'] == pytest.approx(6.)  # Targets [10, 2].
    assert result['q_target_source_v'] == 1
    assert teacher.grad is None


def test_bad_action_ranking_cannot_pass_a_small_value_error():
    target = torch.tensor([[10., 10.1, 10.2], [20., 20.1, 20.2]])
    bad = target.flip(-1)
    result = calibration_metrics(bad, target, min_pairs=2)
    assert result['normalized_rmse'] < .25
    assert result['pairwise_accuracy'] == 0.
    assert not result['passed']
    assert calibration_metrics(target, target, min_pairs=2)['passed']


def test_indistinguishable_action_returns_do_not_certify_q():
    result = calibration_metrics(torch.ones(2, 3), torch.ones(2, 3), min_pairs=2)
    assert result['n_comparable_pairs'] == 0
    assert result['pairwise_accuracy'] is None
    assert not result['passed']


def test_calibration_gate_keeps_actor_fixed_past_warmup_budget():
    updater = make_updater(warmup=1)
    updater.dual.require_q_calibration = True
    updater.q_calibration_passed = False
    before = [p.detach().clone() for p in updater.policy.parameters()]
    for _ in range(3):
        assert updater.update(None)['actor_step_performed'] == 0
    assert all(torch.equal(a, p) for a, p in zip(before, updater.policy.parameters()))
    updater.q_calibration_passed = True
    assert updater.update(None)['actor_step_performed'] == 1


def test_gate_state_restores_without_implicitly_approving_q():
    original = make_updater(warmup=1)
    original.dual.require_q_calibration = True
    original.q_calibration_passed = False
    original.update(None)
    restored = make_updater(warmup=1)
    restored.dual.require_q_calibration = True
    restored.load_state_dict(original.state_dict())
    assert not restored.q_calibration_passed
    assert restored.update(None)['actor_step_performed'] == 0


def test_v_endpoint_bptt_keeps_state_gradient_without_querying_q():
    from tadg.imagination import imagine_soft

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.tensor(.1, dtype=torch.float64))

        def sample(self, obs, goal):
            action = self.theta * obs
            return SimpleNamespace(action=action, log_prob=action[:, 0] * 0,
                                   std=torch.ones_like(action) * .1)

    class Inputs:
        lib = SimpleNamespace(last_frame_of=lambda frame: torch.full_like(frame, 100))

        def history(self, start):
            return [start.state]

        def at(self, state, xi, frame, hist):
            return hist[-1], torch.zeros_like(state), state, None

        def add_noise(self, obs):
            return obs

    def forbidden(*args):
        raise AssertionError('V-guided BPTT must not query a cold Q')

    actor = Actor()
    sac = SimpleNamespace(alpha=0., gamma=.9, actor_objective=forbidden,
                          q=SimpleNamespace(minimum=forbidden))
    start = SimpleNamespace(state=torch.ones(1, 1, dtype=torch.float64),
                            frame=torch.zeros(1, dtype=torch.long), prev_action=torch.zeros(1, 1), xi=None)
    teacher = nn.Parameter(torch.tensor(-1., dtype=torch.float64), requires_grad=False)

    def step(state, action, previous, xi, ref):
        nxt = state + action
        return nxt, -nxt[:, 0].square(), None, state[:, 0] * 0 - torch.inf, nxt, None

    rollout = imagine_soft(None, actor, sac, Inputs(), start, 2, step,
                           terminal_value_fn=lambda x, goal: teacher * x[:, 0].square(),
                           compute_q_objective=False)
    actual = torch.autograd.grad(rollout.objective, actor.theta)[0]
    t = actor.theta.detach()
    expected = -2 * (1 + t) - 4 * (.9 + .9**2) * (1 + t)**3
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(rollout.objective,
                               rollout.reward_objective + rollout.entropy_objective + rollout.terminal_objective)
    assert rollout.q_objective.item() == 0 and teacher.grad is None


def test_teacher_to_bellman_transition_initializes_target_q_only_once():
    updater = make_updater(warmup=1)
    updater.dual.q_teacher_v = True
    updater.teacher_value = lambda features, goal: torch.ones(features.shape[0]) * 5
    seen = []
    original = updater.sac.update

    def observe(policy, batch, **kwargs):
        if kwargs.get('next_value') is None:
            seen.append(all(torch.equal(a, b) for a, b in zip(
                updater.sac.q.parameters(), updater.sac.target_q.parameters())))
        return original(policy, batch, **kwargs)

    updater.sac.update = observe
    assert updater.update(None)['actor_step_performed'] == 0
    assert updater.update(None)['actor_step_performed'] == 1
    updater.update(None)
    assert seen == [True, False]
    assert updater.state_dict()['q_bootstrap_initialized']


def test_fixed_fusion_uses_matching_q_and_horizon_normalized_bptt_units():
    from tadg.gradients import dual_actor_step

    actor = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        actor.weight.fill_(1.)
    optimizer = torch.optim.SGD(actor.parameters(), lr=.1)
    theta = actor.weight.squeeze()
    metrics = dual_actor_step(actor, optimizer, 2 * theta, 8 * theta, .5,
                               grad_clip=100., dynamics_scale=.25)
    torch.testing.assert_close(actor.weight, torch.tensor([[1.2]], dtype=torch.float64))
    assert metrics['q_actor_grad_norm'] == pytest.approx(2.)
    assert metrics['dynamics_grad_norm'] == pytest.approx(2.)


def test_fixed_fusion_keeps_actor_frozen_until_q_is_calibrated():
    updater = make_updater(warmup=1, dynamics_weight=.5, terminal_value_mode='v',
                           q_teacher_v=True, require_q_calibration=True)
    before = [p.detach().clone() for p in updater.policy.parameters()]
    for _ in range(3):
        metrics = updater.update(None)
        assert metrics['q_target_source_v'] == 1
        assert metrics['actor_step_performed'] == 0
    assert all(torch.equal(p, old) for p, old in zip(updater.policy.parameters(), before))
    assert all(p.grad is None for p in updater.value_critic.parameters())


def test_fixed_fusion_backpropagates_both_real_objectives_in_one_step():
    from tadg.gradients import dual_actor_step
    from tadg.imagination import imagine_soft

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.tensor(.1, dtype=torch.float64))

        def sample(self, obs, goal):
            a = self.theta * obs
            return SimpleNamespace(action=a, log_prob=a[:, 0] * 0, std=torch.ones_like(a) * .1)

    class Inputs:
        lib = SimpleNamespace(last_frame_of=lambda frame: torch.full_like(frame, 100))

        def history(self, start):
            return [start.state]

        def at(self, state, xi, frame, hist):
            return hist[-1], state, state, None

        def add_noise(self, obs):
            return obs

    actor = Actor()
    sac = SimpleNamespace(alpha=0., gamma=.9,
                          actor_objective=lambda features, sample: -sample.action[:, 0].square().mean())
    start = SimpleNamespace(state=torch.ones(1, 1, dtype=torch.float64), frame=torch.zeros(1, dtype=torch.long),
                            prev_action=torch.zeros(1, 1), xi=None)

    def step(state, action, previous, xi, ref):
        nxt = state + action
        return nxt, -nxt[:, 0].square(), None, state[:, 0] * 0 - torch.inf, nxt, None

    rollout = imagine_soft(None, actor, sac, Inputs(), start, 2, step,
                           terminal_value_fn=lambda x, goal: -x[:, 0].square())
    t = actor.theta.detach().clone()
    gq = -2 * t
    gb = (-2 * (1 + t) - 4 * (.9 + .9**2) * (1 + t)**3) / 2
    optimizer = torch.optim.SGD(actor.parameters(), lr=.001)
    steps = []
    optimizer.register_step_post_hook(lambda *args: steps.append(1))
    dual_actor_step(actor, optimizer, rollout.q_objective, rollout.objective, .5,
                     grad_clip=100., dynamics_scale=.5)
    torch.testing.assert_close(actor.theta, t + .001 * (.5 * gq + .5 * gb))
    assert steps == [1]


def test_baseline_protocol_applies_the_matched_actor_learning_rate(tmp_path):
    import json
    from pathlib import Path

    from test_q_only_repair import ROOT, source_classes

    def config():
        return SimpleNamespace(data=SimpleNamespace(), env=SimpleNamespace(),
                               wm_trainer=SimpleNamespace(window=8),
                               policy_trainer=SimpleNamespace(lr=.01, critic_lr=.01),
                               schedule=SimpleNamespace(), policy=SimpleNamespace(history=10))

    module = source_classes('src/tadg/experiment.py', ['make_config'], Path=Path,
                            TrainConfig=config, TADGConfig=config, apply_config=lambda cfg, _: cfg)
    protocol = json.loads((ROOT / 'configs/experiments/supermimic_matched_5000.json').read_text())
    protocol.update(checkpoint='initial.pt', motion_dir='motions')
    cfg = module.make_config({'config': {'robot': 'r1'}}, protocol, 'supermimic', 0, tmp_path / 'supermimic')
    assert cfg.policy_trainer.lr == pytest.approx(.001)
    assert cfg.policy_trainer.critic_lr == pytest.approx(.001)
