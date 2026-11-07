"""Validate a bounded native SuperMimic resume, preserving optimizers and the LR clock."""

import argparse
import copy
import json
from pathlib import Path


def native_resume_plan(checkpoint, source, motions, output, steps, device):
    if steps <= 0:
        raise ValueError("steps must be positive")
    required = ('policy', 'wm', 'critic', 'policy_opt', 'wm_opt', 'critic_opt',
                'iteration', 'num_updates', 'bin_return', 'bin_memory', 'config')
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"Native resume checkpoint is missing {missing}")
    cfg = copy.deepcopy(checkpoint['config'])
    start = int(checkpoint['iteration'])
    stop = start + steps
    if stop > cfg['schedule']['total_iterations']:
        raise ValueError("Probe would exceed the original training schedule")
    cfg.update(resume=str(source), device=device, run_name=output.name, log_root=str(output.parent))
    cfg['data']['npz_dir'] = str(motions)
    return {'config': cfg, 'start_iteration': start, 'stop_iteration': stop,
            'additional_iterations': steps, 'restore_optimizer_state': True,
            'restore_sampler_state': True, 'preserve_original_lr_schedule': True,
            'note': 'Fresh simulator/replay and existing resume warmup; original native updates retained. '
                    'Motion files must match the pretrained data. Shared state-terminal label fix is active.'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--eval-max-steps', type=int, default=200,
                        help='0 for full clips; the default is only a short integration check')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    import torch

    if args.output.exists():
        parser.error('Choose a new output directory')
    if not args.motions.is_dir() or args.eval_max_steps < 0:
        parser.error('Motions directory must exist and eval-max-steps must be nonnegative')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    plan = native_resume_plan(checkpoint, args.checkpoint.resolve(), args.motions.resolve(),
                              args.output.resolve(), args.steps, args.device)
    args.output.mkdir(parents=True)
    (args.output / 'native_resume_plan.json').write_text(json.dumps(plan, indent=2, default=str))
    if args.dry_run:
        print(f"Prepared native resume {plan['start_iteration']} -> {plan['stop_iteration']}; no training")
        return
    # GPU simulator imports are deliberately deferred until after dry-run.
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    from tadg.evaluation import evaluate
    from tadg.experiment import apply_config

    cfg = apply_config(TrainConfig(), plan['config'])
    trainer = None
    try:
        trainer = Trainer(cfg)  # cfg.resume calls native load(), including Adam and sampler state.
        clips = [clip.name for clip in trainer.lib.clips]
        protocol = {'eval_clips': clips, 'eval_seed': 12345,
                    'eval_max_steps': args.eval_max_steps, 'trace_stride': 10}
        evaluate(trainer, protocol, args.output / 'evaluation_before')
        trainer.run(stop_iteration=plan['stop_iteration'])
        evaluate(trainer, protocol, args.output / 'evaluation_after')
    finally:
        if trainer is not None:
            trainer.env.close()
            if trainer.writer is not None:
                trainer.writer.close()


if __name__ == '__main__':
    main()
