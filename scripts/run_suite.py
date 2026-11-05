"""Launch the comparison on a server. No dependency installation or implicit remote connection."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def git_state(root):
    result = {}
    for label, arguments in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--short"]),
                             ("diff", ["diff", "HEAD"])):
        process = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True)
        result[label] = process.stdout if process.returncode == 0 else process.stderr
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="defaults to the current uv environment's Python")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="a NEW output directory")
    parser.add_argument("--protocol", type=Path, default=root / "configs/experiments/pilot.json")
    parser.add_argument("--methods", nargs="+", default=["supermimic", "q_only", "bptt_soft", "dual_fixed"],
                        choices=["supermimic", "q_only", "bptt_soft", "dual_fixed"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, help="override train_iterations for a short server run")
    parser.add_argument("--dry-run", action="store_true", help="write protocol and commands; do not import torch or train")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    baseline = root / "baselines/supermimic"
    interpreter = args.python.absolute()  # Preserve the venv path; resolving its symlink loses venv semantics.
    checkpoint, motions, output = args.checkpoint.resolve(), args.motions.resolve(), args.output.resolve()
    if not interpreter.exists() or not checkpoint.is_file() or not motions.is_dir():
        parser.error("Server Python, checkpoint, and motion directory must exist")
    if not (baseline / "src/supermimic/training/loop.py").exists():
        parser.error("The integrated baseline at baselines/supermimic is missing")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate methods or seeds are not allowed")
    if output.exists():
        parser.error("Output directory already exists; choose a new directory to preserve prior results")
    protocol = json.loads(args.protocol.read_text())
    protocol.update(checkpoint=str(checkpoint), motion_dir=str(motions))
    if args.steps is not None:
        if args.steps <= protocol["warmup_iterations"]:
            parser.error("steps must exceed the configured warmup_iterations")
        protocol["train_iterations"] = args.steps
    selected = sorted(set(protocol["train_clips"]) | set(protocol["eval_clips"]))
    data_files = [motions / f"{name}.npz" for name in selected]
    missing = [str(p) for p in data_files if not p.is_file()]
    if missing:
        parser.error(f"Frozen motion list has missing files: {missing}")
    output.mkdir(parents=True)
    save_json(output / "protocol.json", protocol)
    save_json(output / "suite_manifest.json", {
        "created_unix": time.time(), "command": sys.argv, "methods": args.methods, "seeds": args.seeds,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "motions": [{"path": str(p), "sha256": sha256(p)} for p in data_files],
        "project": git_state(root),
        "baseline_source": json.loads((baseline / "UPSTREAM.json").read_text()),
        "python": str(interpreter),
        "uv_lock_sha256": sha256(root / "uv.lock"),
        "dry_run": args.dry_run,
        "limitations": ["fixed-weight dual gradient; confidence and adaptive horizon are not implemented",
                        "q_only retains world-model training for a controlled shared-loop ablation",
                        ("evaluation is step-capped" if protocol.get("eval_max_steps", 0)
                         else "evaluation runs until task termination; no step cap"),
                        "this protocol is not the full Appendix-F benchmark",
                        "LIFT locomotion results are not directly comparable to this mimic task"],
    })
    # Archive actual source, including uncommitted new modules, rather than recording only a git hash.
    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for prefix, source, directories in (("project", root, ("src", "scripts", "configs", "third_party")),
                                             ("project/baselines/supermimic", baseline, ("src", "assets", "config"))):
            for directory in directories:
                for path in sorted((source / directory).rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                        archive.add(path, arcname=f"{prefix}/{path.relative_to(source)}")
            for name in ("pyproject.toml", "uv.lock", ".python-version", "README.md", "LICENSE", "UPSTREAM.json"):
                path = source / name
                if path.is_file():
                    archive.add(path, arcname=f"{prefix}/{name}")
    environment = os.environ.copy()
    # Both packages are editable workspace members installed by uv, not injected external checkouts.
    environment.pop("PYTHONPATH", None)
    runs, commands = [], []
    for seed in args.seeds:
        for method in args.methods:
            directory = output / f"seed_{seed}" / method
            command = [str(interpreter), "-u", str(root / "scripts/train_experiment.py"),
                       "--protocol", str(output / "protocol.json"), "--method", method,
                       "--seed", str(seed), "--output", str(directory)]
            commands.append({"seed": seed, "method": method, "command": command})
    save_json(output / "commands.json", commands)
    for entry in commands:
        print(shlex.join(entry["command"]), flush=True)
        if args.dry_run:
            continue
        directory = output / f"seed_{entry['seed']}" / entry["method"]
        directory.mkdir(parents=True)
        start = time.time()
        with (directory / "console.log").open("w") as log:
            result = subprocess.run(entry["command"], cwd=root, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        runs.append({"method": entry["method"], "seed": entry["seed"], "returncode": result.returncode,
                     "wall_s": time.time() - start, "directory": str(directory)})
        save_json(output / "suite_progress.json", runs)
        if result.returncode != 0 and not args.continue_on_error:
            print(f"Run failed; records retained in {directory}. Stopping the suite.", flush=True)
            break
    if not args.dry_run:
        subprocess.run([str(interpreter), str(root / "scripts/summarize_suite.py"), str(output)], check=True)
        if any(run["returncode"] != 0 for run in runs):
            raise SystemExit("One or more runs failed; partial summary and logs were saved.")


if __name__ == "__main__":
    main()
