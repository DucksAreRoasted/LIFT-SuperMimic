"""Backtrack an Actor step using observable action changes, restoring Adam on rejection."""

import copy
import math

import torch


class ActorStepGuard:
    """Local constraints on one fixed replay batch; not a global safety certificate."""

    def __init__(self, predict_mean, reference_mean, *, max_change_rms,
                 max_reference_deviation, max_backtracks=8):
        if (not math.isfinite(max_change_rms) or max_change_rms <= 0
                or not math.isfinite(max_reference_deviation) or max_reference_deviation <= 0
                or not isinstance(max_backtracks, int) or max_backtracks < 0):
            raise ValueError("Invalid Actor step guard bounds")
        self.predict_mean = predict_mean
        self.reference = reference_mean.detach().clone()
        self.max_change_rms = max_change_rms
        self.max_reference_deviation = max_reference_deviation
        self.max_backtracks = max_backtracks
        with torch.no_grad():
            self.before = predict_mean().detach().clone()
        if self.before.shape != self.reference.shape:
            raise ValueError("Actor and reference action shapes differ")
        if not torch.isfinite(self.before).all() or not torch.isfinite(self.reference).all():
            raise FloatingPointError("Non-finite Actor or teacher mean before guarded update")

    @torch.no_grad()
    def step(self, optimizer):
        params = [p for group in optimizer.param_groups for p in group['params']]
        before_params = [p.detach().clone() for p in params]
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        learning_rates = [g['lr'] for g in optimizer.param_groups]

        def restore():
            for p, saved in zip(params, before_params):
                p.copy_(saved)
            optimizer.load_state_dict(copy.deepcopy(before_optimizer))

        try:
            for attempt in range(self.max_backtracks + 1):
                if attempt:
                    restore()
                scale = .5 ** attempt
                for group, lr in zip(optimizer.param_groups, learning_rates):
                    group['lr'] = lr * scale
                optimizer.step()
                after = self.predict_mean()
                finite = bool(torch.isfinite(after).all())
                change = float((after - self.before).square().mean().sqrt()) if finite else math.inf
                deviation = float((after - self.reference).abs().max()) if finite else math.inf
                if (finite and change <= self.max_change_rms
                        and deviation <= self.max_reference_deviation):
                    # Backtracking changes this step's size, not the external LR schedule.
                    for group, lr in zip(optimizer.param_groups, learning_rates):
                        group['lr'] = lr
                    return {'actor_step_performed': 1, 'actor_step_rejected': 0,
                            'actor_step_scale': scale, 'actor_backtracks': attempt,
                            'actor_mean_change_rms': change,
                            'actor_teacher_deviation_max': deviation}
            restore()
        except BaseException:
            restore()
            raise
        return {'actor_step_performed': 0, 'actor_step_rejected': 1,
                'actor_step_scale': 0., 'actor_backtracks': self.max_backtracks,
                'actor_mean_change_rms': 0.,
                'actor_teacher_deviation_max': float((self.before - self.reference).abs().max())}
