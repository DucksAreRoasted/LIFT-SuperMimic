"""Launch the R1/LAFAN experiment pipeline; default background jobs survive SSH logout.

This launcher uses only the standard library. uv installs the training environment.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["supermimic", "q_only", "bptt_soft", "dual_fixed"]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_plan(args, uv):
    run = [uv, "run", "--extra", "train", "--locked", "--no-sync", "python"]
    prepare = [*run, str(ROOT / "scripts/prepare_r1_lafan.py"), "--output", str(args.motions)]
    if args.data_source:
        prepare.extend(["--source", str(args.data_source)])
    if args.download_data:
        prepare.extend(["--download", "--workers", str(args.workers)])
    plan = []
    if not args.no_sync:
        plan.append({"name": "environment", "command": [uv, "sync", "--extra", "train", "--locked"]})
    plan.append({"name": "data", "command": prepare})
    stages = ([] if args.skip_smoke else ["smoke"]) + (
        ["pilot", "confirm"] if args.stage == "all" else [args.stage])
    for stage in stages:
        if stage == "smoke":
            cfg = json.loads((ROOT / "configs/experiments/r1_lafan1_smoke.json").read_text())
            methods, seeds = ["supermimic", "dual_fixed"], [0]
        else:
            cfg = json.loads((ROOT / "configs/experiments/r1_lafan1_40clips.json").read_text())
            cfg.update(name=f"r1_lafan1_{stage}", train_iterations=5000 if stage == "pilot" else 20000,
                       num_envs=256, eval_max_steps=0, eval_interval=1000 if stage == "pilot" else 5000,
                       max_hours_per_run=None)
            methods, seeds = METHODS, [0] if stage == "pilot" else [0, 1, 2]
        protocol = args.output / f"protocol_{stage}.json"
        directory = args.output / stage
        command = [*run, str(ROOT / "scripts/run_suite.py"), "--checkpoint", str(args.checkpoint),
                   "--motions", str(args.motions), "--protocol", str(protocol),
                   "--methods", *methods, "--seeds", *map(str, seeds), "--output", str(directory)]
        plan.append({"name": stage, "command": command, "protocol": cfg,
                     "protocol_path": str(protocol), "results": str(directory),
                     "methods": methods, "seeds": seeds,
                     "transitions_per_run": cfg["num_envs"] * cfg["train_iterations"]})
    return plan


def execute(args):
    import fcntl  # Linux server / local Linux; lock is released even if the worker crashes.

    lock = (args.output / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"status": "running", "pid": os.getpid(), "gpu": args.gpu,
             "started_unix": time.time(), "stages": []}
    state_path = args.output / "pipeline_status.json"
    write_json(state_path, state)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venv")
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    try:
        for entry in json.loads((args.output / "plan.json").read_text()):
            current = {"name": entry["name"], "status": "running", "started_unix": time.time()}
            state["stages"].append(current)
            state["current_stage"] = entry["name"]
            write_json(state_path, state)
            print(f"\n[{entry['name']}] {shlex.join(entry['command'])}", flush=True)
            subprocess.run(entry["command"], cwd=ROOT, env=env, check=True)
            if "results" in entry:
                # A worker may exit normally after a time budget. Require actual completed budgets.
                directory = Path(entry["results"])
                for seed in entry["seeds"]:
                    for method in entry["methods"]:
                        path = directory / f"seed_{seed}" / method / "status.json"
                        status = json.loads(path.read_text())
                        if status["status"] != "completed":
                            raise RuntimeError(f"Incomplete run: {path}: {status['status']}")
            current.update(status="completed", finished_unix=time.time())
            write_json(state_path, state)
        state["status"] = "completed"
    except BaseException:
        state["status"] = "failed"
        if state["stages"]:
            state["stages"][-1]["status"] = "failed"
        state["failure"] = traceback.format_exc()
        print(state["failure"], file=sys.stderr, flush=True)
        raise
    finally:
        state["finished_unix"] = time.time()
        write_json(state_path, state)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "confirm", "all"], default="pilot",
                        help="pilot=5000 steps/seed 0; confirm=20000 steps/seeds 0,1,2; all=both")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "data/checkpoints/r1_supermimic_pretrained.pt")
    parser.add_argument("--motions", type=Path, default=ROOT / "data/retarget/lafan1_modelscope_v1/r1")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-source", type=Path, help="Import existing R1 NPZ before training")
    source.add_argument("--download-data", action="store_true", help="Download and retarget if no R1 NPZ exists")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", default="0", help="One CUDA_VISIBLE_DEVICES index or UUID")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--no-sync", action="store_true", help="Use an already installed uv train environment")
    parser.add_argument("--foreground", action="store_true", help="Stay attached; default is a detached worker")
    parser.add_argument("--dry-run", action="store_true", help="Write plan only; no installs, downloads or GPU use")
    parser.add_argument("--status", action="store_true", help="Read pipeline status; requires --output")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.status:
        if args.output is None:
            parser.error("--status requires --output")
        print((args.output / "pipeline_status.json").read_text())
        return
    if "," in args.gpu or not args.gpu.strip() or args.workers < 1:
        parser.error("Choose one GPU and a positive worker count")
    if args.download_data and args.motions.name != "r1":
        parser.error("--download-data requires a motions directory ending in /r1")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output = ROOT / "outputs" / f"r1_research_{stamp}"
    for key in ["checkpoint", "motions", "output", "data_source"]:
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.worker:
        execute(args)
        return
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv was not found in PATH")
    if not args.dry_run and not args.checkpoint.is_file():
        parser.error(f"Pretrained R1 checkpoint not found: {args.checkpoint}")
    if not args.dry_run and not args.download_data:
        source_path = args.data_source or args.motions
        if not list(source_path.glob("*.npz")):
            parser.error(f"No NPZ under {source_path}; use --data-source or --download-data")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = build_plan(args, uv)
    write_json(args.output / "plan.json", plan)
    for entry in plan:
        if "protocol" in entry:
            write_json(Path(entry["protocol_path"]), entry["protocol"])
    write_json(args.output / "pipeline_status.json", {"status": "planned", "gpu": args.gpu})
    print(f"Results: {args.output}", flush=True)
    for entry in plan:
        print(f"{entry['name']}: {shlex.join(entry['command'])}", flush=True)
    if args.dry_run:
        return
    if args.foreground:
        execute(args)
        return
    # The detached process and all training children keep running after SSH logout.
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
               "--output", str(args.output), "--worker"]
    with (args.output / "pipeline.log").open("a") as log:
        worker = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(args.output / "launcher.json", {"pid": worker.pid, "command": command})
    print(f"Background PID: {worker.pid}\nLog: {args.output / 'pipeline.log'}", flush=True)


if __name__ == "__main__":
    main()
