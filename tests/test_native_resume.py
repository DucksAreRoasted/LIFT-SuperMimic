"""Native continuation must retain its saved config and LR clock under a short probe."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_q_only_repair import source_classes


def test_native_plan_preserves_training_settings_and_optimizer_restore():
    module = source_classes('scripts/validate_native_resume.py', ['native_resume_plan'], copy=copy)
    cfg = {'schedule': {'total_iterations': 320000, 'warmup_iterations': 40},
           'env': {'num_envs': 4096, 'replay_capacity': 1000},
           'policy_trainer': {'horizon': 32, 'batch_size': 2048, 'lr': .0002},
           'wm_trainer': {'batches_per_update': 10, 'batch_size': 2048},
           'data': {'npz_dir': 'old', 'clips': None}}
    ck = {k: {} for k in ['policy', 'wm', 'critic', 'policy_opt', 'wm_opt', 'critic_opt',
                          'bin_return', 'bin_memory']}
    ck.update(iteration=51350, num_updates=20000, config=cfg)
    before = copy.deepcopy(cfg)
    plan = module.native_resume_plan(ck, Path('initial.pt'), Path('motions'), Path('probe'), 200, 'cuda:0')
    assert plan['start_iteration'] == 51350 and plan['stop_iteration'] == 51550
    for k in ['schedule', 'env', 'policy_trainer', 'wm_trainer']:
        assert plan['config'][k] == before[k]
    assert cfg == before
    assert plan['config']['resume'] == 'initial.pt' and plan['restore_optimizer_state']


def test_native_run_budget_does_not_modify_lr_schedule(monkeypatch):
    import time
    import torch
    calls, saved = [], []
    class Env:
        num_envs = 1
        def reset(self):
            return {}, {}
        def step(self, action):
            return {}, torch.zeros(1), torch.tensor([False]), torch.tensor([False]), {}
    module = source_classes('baselines/supermimic/src/supermimic/training/loop.py', ['Trainer'],
                            torch=torch, time=time, cosine_lr=lambda step, total, *args: calls.append(total) or .1,
                            finalize=lambda: None)
    result = module.Trainer.__new__(module.Trainer)
    sch = SimpleNamespace(total_iterations=320000, warmup_iterations=0, updates_per_env_step=1.,
                          lr_final_ratio=.03, lr_decay_fraction=.78, policy_updates_per_wm_update=1,
                          log_interval=10000, checkpoint_interval=10000, max_hours=None)
    result.cfg = SimpleNamespace(schedule=sch, wm_trainer=SimpleNamespace(lr=.1),
                                 policy_trainer=SimpleNamespace(lr=.1))
    result.env = Env()
    result.iteration = result.start_iteration = 51350
    result.ep_reward_sum = torch.zeros(1)
    result.ep_len = torch.zeros(1)
    result.max_step_reward = 1.
    result._utd_acc = 0.
    result.num_updates = 0
    result.act = lambda obs: torch.zeros(1, 1)
    result.wm_trainer = result.policy_trainer = SimpleNamespace(set_lr=lambda lr: None, update=lambda _: {})
    result.buffer = None
    result.save = lambda name: saved.append(name)
    result.writer = None
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    result.run(stop_iteration=51352)
    assert result.iteration == 51352 and calls == [320000, 320000]
    assert sch.total_iterations == 320000 and saved == ['ckpt_latest.pt']


def test_native_plan_rejects_weight_only_checkpoint():
    module = source_classes('scripts/validate_native_resume.py', ['native_resume_plan'], copy=copy)
    with pytest.raises(ValueError, match='missing'):
        module.native_resume_plan({'policy': {}}, Path('x'), Path('m'), Path('o'), 100, 'cuda:0')
