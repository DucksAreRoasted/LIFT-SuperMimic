"""Assert real GPU V-teacher, calibration-gate and V-endpoint integration behavior."""
import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path

import torch

from tadg.experiment import RecordedTADGTrainer, make_config
from tadg.recording import RunRecorder, write_json


def clone(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def unchanged(module, snapshot):
    return all(torch.equal(p, snapshot[n]) for n, p in module.named_parameters())


def run_case(checkpoint, motions, output, method, compile_mode=None):
    protocol_path = Path('configs/experiments') / (
        {'q_only': 'q_only_v_teacher_smoke.json', 'bptt_soft': 'bptt_v_teacher_smoke.json',
         'dual_fixed': 'dual_fixed_v_teacher_smoke.json'}[method])
    protocol = json.loads(protocol_path.read_text())
    if compile_mode is not None:
        protocol['compile'] = compile_mode
    protocol.update(checkpoint=str(checkpoint.resolve()), motion_dir=str(motions.resolve()))
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = make_config(source, protocol, method, 0, output)
    records = RunRecorder(output, {'method': method, 'seed': 0, 'protocol': protocol, 'diagnostic_only': True})
    trainer = None
    try:
        trainer = RecordedTADGTrainer(cfg)
        trainer.attach_records(records, protocol)
        updater = trainer.policy_trainer
        for n, value in source['critic'].items():
            torch.testing.assert_close(updater.value_critic.state_dict()[n].cpu(), value, rtol=0, atol=0)
        for n, value in source['policy'].items():
            torch.testing.assert_close(updater.teacher_policy.state_dict()[n].cpu(), value, rtol=0, atol=0)
        del source
        value_before, actor_before = clone(updater.value_critic), clone(trainer.policy)
        teacher_actor_before, q_before = clone(updater.teacher_policy), clone(updater.sac.q)
        checks = {'teacher_loaded_exactly': True, 'actor_frozen_checks': 0,
                  'actor_steps': 0, 'v_target_steps': 0, 'v_endpoint_steps': 0}
        original_update = updater.update
        if method == 'bptt_soft':
            def forbidden(*args):
                raise AssertionError('Cold Q was queried on the V-guided Actor path')
            updater.sac.actor_objective = forbidden
            updater.sac.q.minimum = forbidden

        def checked_update(buffer):
            before = clone(trainer.policy) if method != 'bptt_soft' else None
            metrics = original_update(buffer)
            assert unchanged(updater.value_critic, value_before)
            assert unchanged(updater.teacher_policy, teacher_actor_before)
            checks['v_target_steps'] += int(metrics['q_target_source_v'])
            if not metrics['actor_step_performed']:
                assert unchanged(trainer.policy, before)
                checks['actor_frozen_checks'] += 1
                if updater.q_updates_completed == 2:
                    trainer.save('ckpt_teacher_probe.pt')
            else:
                checks['actor_steps'] += 1
                if method == 'q_only':
                    assert updater.q_calibration_passed
                else:
                    assert metrics['terminal_value_source_v'] == 1
                    assert metrics['entropy_objective'] == 0.
                    decomposition = metrics['reward_objective'] + metrics['entropy_objective'] + metrics['terminal_objective']
                    assert abs(decomposition - metrics['imag_objective']) < 1e-5
                    checks['v_endpoint_steps'] += 1
                    if method == 'dual_fixed':
                        assert metrics['dynamics_weight'] == .5
                        assert metrics['q_actor_grad_norm'] > 0
                        assert metrics['dynamics_grad_norm'] > 0
            return metrics

        updater.update = checked_update
        trainer.record_evaluation()
        trainer.run()
        trainer.export_recent_replay()
        assert not unchanged(updater.sac.q, q_before)
        checks.update(q_updated=True, teacher_v_unchanged=unchanged(updater.value_critic, value_before),
                      teacher_actor_unchanged=unchanged(updater.teacher_policy, teacher_actor_before),
                      q_calibration_required=cfg.tadg.require_q_calibration,
                      q_calibration_passed=updater.q_calibration_passed,
                      actor_changed=not unchanged(trainer.policy, actor_before),
                      update_state=updater.state_dict(), counters=copy.deepcopy(records.counters))
        if not cfg.tadg.require_q_calibration:
            assert checks['v_endpoint_steps'] > 0 and checks['actor_changed']
        elif not updater.q_calibration_passed:
            assert checks['actor_steps'] == 0 and not checks['actor_changed']
        write_json(output/'integration_checks.json', checks)
        records.finish('smoke_passed', **checks)
        records = None
    except BaseException:
        if records is not None:
            records.finish('failed', failure=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            trainer.env.close()
            trainer.writer.close()
            del trainer
            torch.cuda.empty_cache()
    if method == 'q_only':
        probe = output/'ckpt_teacher_probe.pt'
        assert probe.is_file()
        resumed_cfg = copy.deepcopy(cfg)
        resumed_cfg.run_name += '_resume_probe'
        resumed_cfg.warm_start = None
        resumed_cfg.resume = probe
        resumed = RecordedTADGTrainer(resumed_cfg)
        try:
            assert resumed.policy_trainer.q_updates_completed == 2
            assert not resumed.policy_trainer.q_calibration_passed
            assert unchanged(resumed.policy_trainer.value_critic, value_before)
            assert unchanged(resumed.policy_trainer.teacher_policy, teacher_actor_before)
            checks['teacher_checkpoint_restore_passed'] = True
            write_json(output/'integration_checks.json', checks)
        finally:
            resumed.env.close()
            resumed.writer.close()
            del resumed
            torch.cuda.empty_cache()
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile', choices=('default',), default=None,
                        help='Exercise the compiled forward/backward path used by long runs.')
    parser.add_argument('--methods', nargs='+', choices=('q_only', 'bptt_soft', 'dual_fixed'),
                        default=['q_only', 'bptt_soft'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = list(Path('src/tadg').glob('*.py')) + [Path(__file__)]
    write_json(args.output/'source_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    results = {}
    try:
        for method in args.methods:
            results[method] = run_case(args.checkpoint, args.motions, args.output/method, method, args.compile)
        write_json(args.output/'checks.json', {'passed': True, 'cases': results})
    except BaseException:
        write_json(args.output/'checks.json', {'passed': False, 'cases': results, 'failure': traceback.format_exc()})
        raise
    print('VALUE_GUIDANCE_SMOKE_PASSED', json.dumps(results))


if __name__ == '__main__':
    main()
