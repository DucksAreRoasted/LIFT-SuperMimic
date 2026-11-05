"""Download ModelScope LAFAN G1 CSVs, then retarget them to R1 and K1.

The G1 subset is 30 Hz, 36 columns (XYZ + xyzw + 29 joint angles).
Downloads stay in a separate cache. Existing input CSVs are never overwritten.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from modelscope_hub.errors import HubError

REPO = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO.parents[1]
DEFAULT_DATASET = "AI-ModelScope/LAFAN1_Retargeting_Dataset"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_names(pattern: str) -> list[str]:
    names = (REPO / "config/data/lafan1_g1_files.txt").read_text().splitlines()
    selected = [name for name in names if name and fnmatch.fnmatchcase(name, pattern)]
    if not selected:
        raise ValueError(f"No LAFAN G1 file in the manifest matches {pattern!r}")
    return selected


def validate_csv(path: Path) -> dict:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] != 36:
        raise ValueError(f"{path}: expected at least 3 rows and exactly 36 columns, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: NaN or infinity in motion")
    norms = np.linalg.norm(data[:, 3:7], axis=1)
    if np.max(np.abs(norms - 1)) > 0.01:
        raise ValueError(f"{path}: root quaternion is not normalized; expected xyzw in columns 3:7")
    return dict(name=path.name, frames=int(len(data)), fps=30, columns=36,
                duration_s=(len(data) - 1) / 30, sha256=sha256(path))


def install_snapshot(snapshot: Path, input_dir: Path, names: list[str]) -> list[dict]:
    """Validate the whole selected set before publishing any CSV into input_dir."""
    rows = []
    for name in names:
        source = snapshot / "g1" / name
        if not source.is_file():
            raise FileNotFoundError(f"Incomplete ModelScope snapshot: missing {source}")
        row = validate_csv(source)
        target = input_dir / name
        if target.exists() and sha256(target) != row["sha256"]:
            raise FileExistsError(
                f"Existing input differs from ModelScope: {target}. "
                "Choose a new --input-dir; existing data will not be overwritten."
            )
        rows.append(row)
    input_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = input_dir / row["name"]
        if target.exists():
            continue
        # A hard-link publication fails atomically if another run created target.
        with tempfile.NamedTemporaryFile(dir=input_dir, suffix=".download.tmp", delete=False) as f:
            tmp = Path(f.name)
        try:
            shutil.copyfile(snapshot / "g1" / row["name"], tmp)
            try:
                target.hardlink_to(tmp)
            except FileExistsError:
                if sha256(target) != row["sha256"]:
                    raise
        finally:
            tmp.unlink(missing_ok=True)
    return rows


def check_assets():
    required = [
        "third_party/UPSTREAM.json",
        "third_party/robot_retargeter/scripts/robot_replay.py",
        "third_party/robot_retargeter/scripts/robot_retarget.py",
        "third_party/robot_retargeter/asset/robot/g1_description/mjcf/g1.xml",
        "third_party/booster_assets/robots/K1/K1_22dof.xml",
        "third_party/unitree_ros/robots/r1_description/R1.urdf",
    ]
    missing = [name for name in required if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Upload the project including third_party robot models and their meshes. Missing:\n  "
            + "\n  ".join(missing)
        )


def run_script(relative: str, *args: str):
    command = [sys.executable, str(REPO / relative), *map(str, args)]
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/modelscope/lafan1")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/lafan1/g1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/retarget/lafan1_modelscope_v1")
    parser.add_argument("--pattern", default="*.csv", help="Select from the fixed 40-file G1 manifest")
    parser.add_argument("--workers", type=int, default=8, help="IK worker processes")
    parser.add_argument("--robots", nargs="+", choices=["r1", "k1"], default=["r1", "k1"])
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0, help="Input frame cap for short validation runs")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Reuse the ModelScope local snapshot without network")
    parser.add_argument("--render-preview", action="store_true", help="Render G1/K1/R1 video after retargeting (EGL required)")
    args = parser.parse_args()
    if args.render_preview and set(args.robots) != {"r1", "k1"}:
        parser.error("--render-preview currently requires both --robots r1 k1")
    if args.workers < 1 or args.download_workers < 1 or args.max_frames < 0:
        parser.error("Workers must be positive and --max-frames must be nonnegative")
    names = selected_names(args.pattern)
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    if not args.download_only:
        check_assets()
        extras = {p.name for p in args.input_dir.glob("*.csv") if fnmatch.fnmatchcase(p.name, args.pattern)} - set(names)
        if extras:
            raise ValueError(f"Unexpected matching CSVs in --input-dir: {sorted(extras)}. Choose a clean input directory.")
    # Keep snapshots of different repositories and revisions separate.
    identity = hashlib.sha256(f"{args.dataset_id}@{args.revision}".encode()).hexdigest()[:16]
    local_dir = args.cache_dir / identity
    from modelscope_hub import HubApi

    print(f"[download] ModelScope {args.dataset_id}@{args.revision}; {len(names)} G1 clips", flush=True)
    snapshot = HubApi(endpoint="https://modelscope.cn").download_repo(
        args.dataset_id, "dataset", revision=args.revision,
        local_dir=local_dir, cache_dir=args.cache_dir / "hub",
        allow_patterns=[f"g1/{name}" for name in names], max_workers=args.download_workers,
        local_files_only=args.offline,
    )
    rows = install_snapshot(Path(snapshot), args.input_dir, names)
    receipt = dict(dataset_id=args.dataset_id, revision=args.revision, endpoint="https://modelscope.cn",
                   snapshot=str(snapshot), input_dir=str(args.input_dir), clips=len(rows), per_clip=rows)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.cache_dir / f"download_manifest_{identity}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[validated] {len(rows)} CSVs -> {args.input_dir}; receipt: {receipt_path}", flush=True)
    if args.download_only:
        return
    # Regenerate absolute mesh/model/config paths in the server's actual directory.
    run_script("scripts/retarget/prepare_models.py")
    run_script("scripts/retarget/retarget_lafan.py", "--input-dir", args.input_dir,
               "--output-dir", args.output_dir, "--robots", *args.robots, "--pattern", args.pattern,
               "--workers", args.workers, "--max-frames", args.max_frames)
    run_script("scripts/retarget/audit_results.py", args.output_dir)
    if args.render_preview:
        clip = "dance1_subject2" if "dance1_subject2.csv" in names else Path(names[0]).stem
        short = bool(args.max_frames and args.max_frames <= 660)
        run_script("scripts/retarget/replay_motion.py", "--source-csv", args.input_dir / f"{clip}.csv",
                   "--motion", args.output_dir / "k1" / f"{clip}.npz", args.output_dir / "r1" / f"{clip}.npz",
                   "--start-time", "0" if short else "20", "--seconds", "4", "--fps", "15",
                   "--video", args.output_dir / "preview_g1_k1_r1.mp4")
    print(f"[complete] {len(names)} source clips retargeted to {args.robots}: {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError, HubError) as exc:
        raise SystemExit(str(exc)) from exc
