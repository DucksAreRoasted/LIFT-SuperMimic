"""Server worker: one frozen method/seed, with identical records and evaluation for every arm."""

import argparse
import importlib.metadata
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--method", choices=("supermimic", "q_only", "bptt_soft", "dual_fixed"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch

    from tadg.experiment import RecordedBaselineTrainer, RecordedTADGTrainer, make_config
    from tadg.recording import RunRecorder

    protocol = json.loads(args.protocol.read_text())
    random.seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = torch.load(protocol["checkpoint"], map_location="cpu", weights_only=False)
    cfg = make_config(checkpoint, protocol, args.method, args.seed, args.output.resolve())
    records = RunRecorder(args.output, {
        "method": args.method, "seed": args.seed, "protocol": protocol,
        "python": sys.executable, "command": sys.argv,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "torch_settings": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "pretrained_iteration": checkpoint.get("iteration"),
        "pretraining_cost": "shared source checkpoint; original pretraining cost is not included in continuation",
        "packages": sorted((d.metadata.get("Name", "unknown"), d.version)
                           for d in importlib.metadata.distributions()),
        "interpretation": "warm-start continuation pilot, not a from-scratch or full-paper reproduction",
    })
    trainer = None
    status, failure = "failed", None
    try:
        started = time.perf_counter()
        if args.method == "supermimic":
            trainer = RecordedBaselineTrainer(cfg)
            trainer.warm_start(checkpoint)
        else:
            trainer = RecordedTADGTrainer(cfg)
        del checkpoint
        trainer.attach_records(records, protocol)
        records.event("initialized", duration_s=time.perf_counter() - started)
        trainer.record_evaluation()
        records.event("training_started", iteration=trainer.iteration)
        trainer.run()
        if protocol.get("save_recent_replay", True):
            trainer.export_recent_replay()
        status = "completed" if trainer.iteration >= cfg.schedule.total_iterations else "time_budget_stopped"
        if (args.method != "supermimic" and cfg.tadg.require_q_calibration
                and not trainer.policy_trainer.q_calibration_passed):
            status = "calibration_not_passed"
            raise RuntimeError("Q calibration did not pass at the final check; Actor updates are gated. "
                               "See calibration reports and actual Actor update counters.")
    except BaseException:
        failure = traceback.format_exc()
        (args.output / "failure.txt").write_text(failure)
        records.event("exception", traceback=failure)
        raise
    finally:
        values = {"failure": failure}
        if trainer is not None:
            values.update(iteration=trainer.iteration,
                          train_env_transitions=trainer.iteration * trainer.env.num_envs,
                          planned_iterations=cfg.schedule.total_iterations)
            trainer.env.close()
            if trainer.writer is not None:
                trainer.writer.close()
        records.finish(status, **values)


if __name__ == "__main__":
    main()
