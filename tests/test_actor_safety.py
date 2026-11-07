"""Regression for the post-calibration mean drift observed in the November 6 runs."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tadg.actor_safety import ActorStepGuard
from tadg.gradients import dual_actor_step
from test_q_only_repair import make_updater, source_classes


def test_clipped_q_updates_stay_in_the_checked_teacher_neighborhood():
    actor = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        actor.weight.zero_()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.02)
    obs = torch.ones(4, 1, dtype=torch.float64)
    steps, rejected = 0, 0
    for _ in range(100):
        guard = ActorStepGuard(lambda: actor(obs), torch.zeros_like(obs),
                               max_change_rms=.01, max_reference_deviation=.1)
        objective = actor(obs).mean() * 5
        metrics = dual_actor_step(actor, optimizer, objective, objective * 0, 0., step_guard=guard)
        steps += metrics['actor_step_performed']
        rejected += metrics['actor_step_rejected']
        assert actor(obs).abs().max() <= .1
    assert steps > 0 and rejected > 0


def test_backtracking_commits_one_adam_step_and_preserves_scheduled_lr():
    actor = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        actor.weight.zero_()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.2)
    obs = torch.ones(1, 1, dtype=torch.float64)
    guard = ActorStepGuard(lambda: actor(obs), torch.zeros_like(obs),
                           max_change_rms=.01, max_reference_deviation=.1)
    objective = actor(obs).mean()
    result = dual_actor_step(actor, optimizer, objective, objective * 0, 0., step_guard=guard)
    assert result['actor_step_performed'] == 1 and result['actor_step_scale'] < 1
    assert 0 < actor.weight.item() <= .01
    assert optimizer.param_groups[0]['lr'] == .2
    assert optimizer.state[actor.weight]['step'].item() == 1


def test_rejected_step_restores_parameters_and_existing_adam_moments():
    actor = nn.Linear(1, 1, bias=False).double()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.1)
    obs = torch.ones(1, 1, dtype=torch.float64)
    initial_objective = actor(obs).mean()
    dual_actor_step(actor, optimizer, initial_objective, initial_objective * 0, 0.)
    before = actor.weight.detach().clone()
    state = copy.deepcopy(optimizer.state_dict())
    guard = ActorStepGuard(lambda: actor(obs), actor(obs).detach(), max_change_rms=1e-12,
                           max_reference_deviation=1e-12, max_backtracks=2)
    objective = actor(obs).mean()
    result = dual_actor_step(actor, optimizer, objective, objective * 0, 0., step_guard=guard)
    assert result['actor_step_performed'] == 0 and result['actor_step_rejected'] == 1
    assert torch.equal(actor.weight, before)
    restored = optimizer.state_dict()
    assert restored['param_groups'] == state['param_groups']
    for index, values in state['state'].items():
        for key, value in values.items():
            torch.testing.assert_close(restored['state'][index][key], value, rtol=0, atol=0)


def test_nonfinite_candidate_is_rejected_without_creating_adam_state():
    actor = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.zero_()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.1)
    obs = torch.ones(1, 1)
    def predict():
        return actor(obs) if actor.weight.item() == 0 else torch.full_like(obs, torch.inf)
    guard = ActorStepGuard(predict, torch.zeros_like(obs), max_change_rms=.01,
                           max_reference_deviation=.1, max_backtracks=0)
    objective = actor(obs).mean()
    result = dual_actor_step(actor, optimizer, objective, objective * 0, 0., step_guard=guard)
    assert result['actor_step_performed'] == 0
    assert actor.weight.item() == 0 and not optimizer.state


def test_opening_actor_can_keep_v_supervision_without_bellman_switch():
    updater = make_updater(warmup=1, q_teacher_v=True, q_keep_teacher=True)
    first = updater.update(None)
    second = updater.update(None)
    third = updater.update(None)
    assert first['actor_step_performed'] == 0
    assert second['actor_step_performed'] == third['actor_step_performed'] == 1
    assert first['q_target_source_v'] == second['q_target_source_v'] == third['q_target_source_v'] == 1
    assert not updater.q_bootstrap_initialized


def test_rejected_real_updater_step_is_not_counted_as_actor_update():
    updater = make_updater(warmup=0, q_teacher_v=True, q_keep_teacher=True,
                           actor_max_change_rms=1e-12, actor_max_teacher_deviation=1e-12,
                           actor_max_backtracks=0)
    before = [p.detach().clone() for p in updater.policy.parameters()]
    result = updater.update(None)
    assert result['actor_step_performed'] == 0 and result['actor_step_rejected'] == 1
    assert updater.q_updates_completed == 1 and updater.actor_updates_completed == 0
    assert all(torch.equal(p, saved) for p, saved in zip(updater.policy.parameters(), before))


def test_periodic_calibration_can_revoke_a_previous_pass(monkeypatch):
    from tadg import calibration
    calls = []
    def reject(*args):
        calls.append(1)
        return {'passed': False, 'wall_s': .1, 'env_transitions': 10}
    monkeypatch.setattr(calibration, 'calibrate_q', reject)
    module = source_classes('src/tadg/experiment.py', ['RecordedMixin'])
    result = module.RecordedMixin()
    updater = SimpleNamespace(q_calibration_passed=True, q_updates_completed=500,
                              last_q_calibration_iteration=100, q_bootstrap_initialized=True)
    result.cfg = SimpleNamespace(tadg=SimpleNamespace(require_q_calibration=True, q_warmup_updates=400))
    result.protocol = {'q_calibration': {'recheck_interval': 50}}
    result.policy_trainer = updater
    result.log_dir = __import__('pathlib').Path('/tmp')
    result.records = SimpleNamespace(times={'calibration_s': 0}, counters={'calibration_env_transitions': 0},
                                     event=lambda *args, **kwargs: None)
    result.iteration = 149
    result.record_q_calibration()
    assert not calls
    result.iteration = 150
    result.record_q_calibration()
    assert calls == [1] and not updater.q_calibration_passed
    assert not updater.q_bootstrap_initialized and updater.last_q_calibration_iteration == 150


@pytest.mark.parametrize('change,deviation', [(-1., .1), (.1, 0.), (.1, -.1),
                                            (float('nan'), .1), (.1, float('inf'))])
def test_guard_configuration_requires_two_positive_bounds(change, deviation):
    with pytest.raises(ValueError):
        make_updater(q_teacher_v=True, actor_max_change_rms=change,
                     actor_max_teacher_deviation=deviation)


def test_revoked_gate_refreezes_actor_and_persists_last_check():
    updater = make_updater(warmup=0, q_teacher_v=True, q_keep_teacher=True,
                           require_q_calibration=True)
    updater.q_calibration_passed = True
    assert updater.update(None)['actor_step_performed'] == 1
    updater.q_calibration_passed = False
    updater.last_q_calibration_iteration = 500
    before = [p.detach().clone() for p in updater.policy.parameters()]
    result = updater.update(None)
    assert result['actor_step_performed'] == 0 and result['q_target_source_v'] == 1
    assert all(torch.equal(p, saved) for p, saved in zip(updater.policy.parameters(), before))
    restored = make_updater(warmup=0, q_teacher_v=True, q_keep_teacher=True,
                            require_q_calibration=True)
    restored.load_state_dict(updater.state_dict())
    assert restored.last_q_calibration_iteration == 500 and not restored.q_calibration_passed


def test_rejected_bptt_counts_computation_without_counting_actor_step(tmp_path, monkeypatch):
    from tadg import recording
    monkeypatch.setattr(recording, 'gpu_status', dict)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    updater = SimpleNamespace(update=lambda _: {'actor_step_performed': 0,
                                               'actor_step_rejected': 1,
                                               'imagination_performed': 1})
    dual = SimpleNamespace(dynamics_weight=1., q_updates=1)
    trainer = SimpleNamespace(policy_trainer=updater, wm_trainer=SimpleNamespace(update=lambda _: {}),
                               cfg=SimpleNamespace(tadg=dual, policy_trainer=SimpleNamespace(batch_size=4,horizon=8)),
                               iteration=1, env=SimpleNamespace(num_envs=1), writer=None)
    recorder = recording.RunRecorder(tmp_path, {}, resource_interval=3600)
    try:
        recorder.attach(trainer)
        updater.update(None)
        assert recorder.counters['actor_updates'] == 0
        assert recorder.counters['actor_rejected_updates'] == 1
        assert recorder.counters['imagination_steps'] == 32
    finally:
        recorder.finish('completed')
