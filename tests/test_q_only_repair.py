"""CPU regression checks for the real recorder and update code, without importing mjlab.

Only simulator construction and batch generation are substituted. The actual class
methods are loaded unchanged from source because the training entry imports GPU-only
simulator modules eagerly.
"""
import ast
import copy
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from supermimic.data.replay_buffer import ReplayBuffer, StartBatch
from torch import nn

from tadg.gradients import dual_actor_step
from tadg.actor_safety import ActorStepGuard
from tadg.policy import StochasticPolicy
from tadg.sac import SACBatch, SACBranch, TwinQ, frozen
from tadg.value_teacher import value_from_features

ROOT = Path(__file__).resolve().parents[1]


def source_classes(path, names, **dependencies):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes += [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    name = '_repair_' + '_'.join(names)
    module = ModuleType(name)
    module.__dict__.update(dependencies)
    sys.modules[name] = module
    # Only trusted repository class definitions are loaded, never external input.
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), module.__dict__)  # noqa: S102
    return module


@pytest.mark.parametrize('fall,reference_end', [(False, False), (True, False), (False, True)])
def test_real_recorder_labels_only_reached_terminal_state(fall, reference_end):
    module = source_classes('baselines/supermimic/src/supermimic/data/recorder.py',
                            ['ReplayRecorder'], RecorderTerm=object, torch=torch)
    recorder = module.ReplayRecorder.__new__(module.ReplayRecorder)
    recorder.buffer = ReplayBuffer(1, 8, 1, 1, 1, 'cpu')
    recorder.prev_s = torch.tensor([[1.]])
    recorder.prev_xi = torch.zeros(1, 1)
    recorder.prev_frame = torch.tensor([1])
    recorder.prev_clip = torch.tensor([0])
    recorder.ep_start = torch.tensor([False])
    recorder.enabled = True
    recorder.n_joints = 1
    action = SimpleNamespace(executed_raw=torch.zeros(1, 1))
    command = SimpleNamespace(frame=torch.tensor([2]), library=SimpleNamespace(frame_clip=torch.zeros(3, dtype=torch.long)))
    recorder._env = SimpleNamespace(
        action_manager=SimpleNamespace(action=torch.zeros(1, 1), get_term=lambda _: action),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        reward_buf=torch.tensor([.1]), reset_terminated=torch.tensor([fall]), device='cpu')
    recorder.action_name = 'joint_pos'
    recorder.command_name = 'motion'
    recorder.state_io = SimpleNamespace(read=lambda: torch.tensor([[2.]]))
    # A real nonterminal transition, followed by the final transition and its terminal state.
    recorder.prev_s = torch.tensor([[0.]])
    recorder.prev_frame = torch.tensor([0])
    recorder._write_rows(torch.tensor([0]), torch.zeros(1))
    recorder.prev_s = torch.tensor([[1.]])
    recorder.prev_frame = torch.tensor([1])
    recorder.record_pre_reset(torch.tensor([0]))
    assert recorder.buffer.done[0, 1].item() == 0, 'pre-fall state was incorrectly labeled terminal'
    assert recorder.buffer.done[0, 2].item() == int(fall)
    assert recorder.buffer.terminal[0, 2]
    sampler = source_classes('src/tadg/experience.py', ['_start_at', 'sample_real_batch'],
                             torch=torch, StartBatch=StartBatch, SACBatch=SACBatch)

    class Inputs:
        lib = SimpleNamespace(last_frame_of=lambda frame: torch.full_like(frame, 1 if reference_end else 100))

        def history(self, start):
            return [start.state]

        def at(self, state, xi, frame, history):
            return state, state, state, None

    recorder.buffer._sample_from_mask = lambda mask, n: (torch.tensor([0]), torch.tensor([0]))
    assert recorder.buffer.sample_windows(1, 1).done.item() == 0
    batch, _ = sampler.sample_real_batch(recorder.buffer, Inputs(), 1, 1)
    assert batch.bootstrap.item() == 1  # No premature cutoff on s0 -> s1.
    recorder.buffer._sample_from_mask = lambda mask, n: (torch.tensor([0]), torch.tensor([1]))
    assert recorder.buffer.sample_windows(1, 1).done.item() == int(fall)
    batch, _ = sampler.sample_real_batch(recorder.buffer, Inputs(), 1, 1)
    assert batch.bootstrap.item() == int(not fall and not reference_end)


class MeanPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(n_joints=1, latent_dim=1)
        self.encoders = nn.ModuleDict({'motion': nn.Linear(1, 1)})
        self.decoder = nn.Module()
        self.decoder.register_buffer('scale', torch.ones(1))
        self.decoder.mlp = nn.Linear(2, 1)

    def forward(self, obs, goal, interface='motion'):
        return self.decoder.mlp(torch.cat((self.encoders[interface](goal), obs), -1))


def test_fixed_std_preserves_reparameterized_mean_gradient():
    policy = StochasticPolicy(MeanPolicy(), learn_std=False)
    with torch.no_grad():
        policy.log_std_head.weight.fill_(20.)
        policy.log_std_head.bias.fill_(1.)
    sample = policy.sample(torch.ones(4, 1), torch.ones(4, 1), noise=torch.ones(4, 1))
    torch.testing.assert_close(sample.std, torch.full((4, 1), math.exp(-2.3)))
    sample.action.sum().backward()
    assert policy.mean_policy.decoder.mlp.weight.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in policy.log_std_head.parameters())


def make_updater(warmup=2, **dual_overrides):
    policy = StochasticPolicy(MeanPolicy())
    policy.cfg.goal_dim = 1
    batch = SACBatch(torch.ones(4, 7), torch.ones(4, 1), torch.ones(4, 1),
                     torch.zeros(4, 1), torch.ones(4), torch.ones(4),
                     torch.ones(4, 7), torch.ones(4, 1), torch.ones(4, 1))
    module = source_classes('src/tadg/trainer.py', ['DualPolicyTrainer'],
                            torch=torch, frozen=frozen, dual_actor_step=dual_actor_step,
                            ActorStepGuard=ActorStepGuard,
                            TwinQ=TwinQ, SACBranch=SACBranch, value_from_features=value_from_features,
                            ModelInputs=lambda *args: None, ImagStep=lambda *args, **kwargs: None,
                            sample_real_batch=lambda *args: (batch, None))
    wm = nn.Linear(1, 1)
    wm.spec = SimpleNamespace(dims=SimpleNamespace(state=1, xi=0))
    cfg = SimpleNamespace(batch_size=4, history=1, grad_clip_norm=1., lr=.01,
                          betas=(.9, .95), gamma=.99, compile=None, grad_clip_value=.1)
    dual = SimpleNamespace(q_updates=1, dynamics_weight=0., q_warmup_updates=warmup,
                           actor_lr_scale=.1, q_lr=.003, q_hidden=(4,), alpha=0., tau=.005,
                           q_teacher_v=False, require_q_calibration=False, terminal_value_mode="q")
    for key, value in dual_overrides.items():
        setattr(dual, key, value)

    class Value(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.))

        def forward(self, state, goal, xi, root):
            return state.sum(-1) * self.weight

    needs_value = dual.q_teacher_v or dual.terminal_value_mode == 'v'
    return module.DualPolicyTrainer(
        wm, policy, SimpleNamespace(default_joint_pos=torch.zeros(1)), cfg, dual,
        value_critic=Value() if needs_value else None,
        teacher_policy=copy.deepcopy(policy.mean_policy).requires_grad_(False) if needs_value else None)


def test_critic_warmup_changes_q_but_not_actor_then_releases_actor():
    torch.manual_seed(7)
    updater = make_updater()
    before_actor = [p.detach().clone() for p in updater.policy.parameters()]
    before_q = [p.detach().clone() for p in updater.sac.q.parameters()]
    for _ in range(2):
        metrics = updater.update(None)
        assert metrics['actor_step_performed'] == 0
        assert all(torch.equal(a, p) for a, p in zip(before_actor, updater.policy.parameters()))
        assert all(p.grad is None for p in updater.policy.parameters())
        assert not updater.opt.state
    assert any(not torch.equal(a, p) for a, p in zip(before_q, updater.sac.q.parameters()))
    metrics = updater.update(None)
    assert metrics['actor_step_performed'] == 1
    assert any(not torch.equal(a, p) for a, p in zip(before_actor, updater.policy.parameters()))
    assert updater.q_updates_completed == 3 and updater.actor_updates_completed == 1


def test_warmup_progress_survives_checkpoint_restore_and_q_lr_is_independent():
    updater = make_updater()
    updater.update(None)
    restored = make_updater()
    restored.load_state_dict(updater.state_dict())
    assert restored.q_updates_completed == 1
    assert restored.update(None)['actor_step_performed'] == 0
    assert restored.update(None)['actor_step_performed'] == 1
    restored.set_lr(.005)
    assert restored.opt.param_groups[0]['lr'] == pytest.approx(.0005)
    assert restored.sac.opt.param_groups[0]['lr'] == pytest.approx(.0015)


def test_real_checkpoint_preserves_critic_only_phase(tmp_path):
    module = source_classes('src/tadg/trainer.py', ['TADGTrainer'],
                            Trainer=object, torch=torch, asdict=lambda cfg: {'robot': cfg.robot})

    def trainer():
        result = module.TADGTrainer.__new__(module.TADGTrainer)
        result.policy_trainer = make_updater()
        result.policy = result.policy_trainer.policy
        result.wm = result.policy_trainer.wm
        result.wm_trainer = SimpleNamespace(opt=torch.optim.Adam(result.wm.parameters()))
        result.cfg = SimpleNamespace(robot='r1', tadg=result.policy_trainer.dual)
        result.command = SimpleNamespace(bin_return=torch.zeros(1), bin_memory=torch.zeros(1))
        result.iteration = 50
        result.num_updates = 1
        result.device = 'cpu'
        result.log_dir = tmp_path
        return result

    original = trainer()
    original.policy_trainer.update(None)
    original.save()
    restored = trainer()
    restored.load(tmp_path / 'ckpt_latest.pt')
    assert restored.policy_trainer.q_updates_completed == 1
    assert restored.policy_trainer.update(None)['actor_step_performed'] == 0
    assert restored.policy_trainer.update(None)['actor_step_performed'] == 1


def test_recorder_counts_critic_only_updates_separately(tmp_path, monkeypatch):
    import json

    from tadg import recording

    monkeypatch.setattr(recording, 'gpu_status', dict)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    updater = make_updater()
    trainer = SimpleNamespace(policy_trainer=updater, wm_trainer=SimpleNamespace(update=lambda _: {}),
                              cfg=SimpleNamespace(tadg=updater.dual), iteration=1,
                              env=SimpleNamespace(num_envs=1), writer=None)
    recorder = recording.RunRecorder(tmp_path, {}, resource_interval=3600)
    try:
        recorder.attach(trainer)
        for _ in range(3):
            updater.update(None)
        assert recorder.counters['actor_updates'] == 1
        assert recorder.counters['critic_only_updates'] == 2
        assert recorder.counters['q_optimizer_steps'] == 3
        assert recorder.counters['imagination_steps'] == 0
    finally:
        recorder.finish('completed')
    kinds = [json.loads(line)['event'] for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert kinds == ['critic_warmup', 'critic_warmup', 'actor_update', 'run_end']


def test_nonfinite_q_target_fails_before_any_optimizer_update():
    updater = make_updater()
    before = [p.detach().clone() for p in updater.sac.q.parameters()]
    batch = SACBatch(torch.ones(1, 7), torch.ones(1, 1), torch.ones(1, 1),
                     torch.ones(1, 1), torch.tensor([float('inf')]), torch.ones(1),
                     torch.ones(1, 7), torch.ones(1, 1), torch.ones(1, 1))
    with pytest.raises(FloatingPointError, match='Non-finite Q loss'):
        updater.sac.update(updater.policy, batch)
    assert all(torch.equal(p, old) for p, old in zip(updater.sac.q.parameters(), before))


def test_repair_protocol_only_changes_q_only_schedule(tmp_path):
    import json

    def config():
        return SimpleNamespace(data=SimpleNamespace(), env=SimpleNamespace(),
                               wm_trainer=SimpleNamespace(window=8), policy_trainer=SimpleNamespace(),
                               schedule=SimpleNamespace(), policy=SimpleNamespace(history=10),
                               tadg=SimpleNamespace(q_warmup_updates=0, fixed_std=False, actor_lr_scale=1.))

    module = source_classes('src/tadg/experiment.py', ['make_config'], Path=Path,
                            TrainConfig=config, TADGConfig=config, apply_config=lambda cfg, _: cfg)
    protocol = json.loads((ROOT / 'configs/experiments/q_only_repair_5000.json').read_text())
    protocol.update(checkpoint='initial.pt', motion_dir='motions')
    repaired = module.make_config({'config': {'robot': 'r1'}}, protocol, 'q_only', 0, tmp_path / 'q_only')
    assert repaired.tadg.q_warmup_updates == 400
    assert repaired.tadg.fixed_std is True
    assert repaired.tadg.actor_lr_scale == .1
    other = module.make_config({'config': {'robot': 'r1'}}, protocol, 'bptt_soft', 0, tmp_path / 'bptt_soft')
    assert other.tadg.q_warmup_updates == 0
    assert other.tadg.fixed_std is False
    assert other.tadg.actor_lr_scale == 1.
