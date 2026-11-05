"""Prepare shared R1 LAFAN1 NPZ data, without importing Torch or starting training."""

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/retarget/lafan1_modelscope_v1/r1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(path):
    """Check the loader's actual R1 schema; embedded old machine paths are provenance only."""
    model = ET.parse(ROOT / "baselines/supermimic/assets/retarget/r1.xml").getroot()
    expected = [j.get("name") for j in model.findall(".//worldbody//joint")]
    required_bodies = {b.get("name") for b in model.findall(".//worldbody//body")}
    with np.load(path, allow_pickle=False) as arr:
        joints = arr["joint_names"].tolist()
        bodies = arr["body_names"].tolist()
        if joints != expected or len(expected) != 26:
            raise ValueError(f"{path}: joint names/order do not match R1")
        if len(set(bodies)) != len(bodies) or not required_bodies.issubset(bodies):
            raise ValueError(f"{path}: body names do not match the R1 model")
        fps = float(np.asarray(arr["fps"]).reshape(-1)[0])
        if not np.isclose(fps, 50):
            raise ValueError(f"{path}: expected 50 Hz, got {fps}")
        frames = arr["joint_pos"].shape[0]
        if frames < 32:
            raise ValueError(f"{path}: too few frames for history and model windows")
        shapes = {"joint_pos": (frames, 26), "joint_vel": (frames, 26),
                  "body_pos_w": (frames, len(bodies), 3),
                  "body_quat_w": (frames, len(bodies), 4),
                  "body_lin_vel_w": (frames, len(bodies), 3),
                  "body_ang_vel_w": (frames, len(bodies), 3)}
        for key, shape in shapes.items():
            value = arr[key]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{path}: invalid {key}, expected finite {shape}")
        if np.max(np.abs(np.linalg.norm(arr["body_quat_w"], axis=-1) - 1)) > 1e-3:
            raise ValueError(f"{path}: body quaternions are not normalized")
    return {"name": path.stem, "frames": frames, "fps": fps,
            "duration_s": (frames - 1) / fps, "bytes": path.stat().st_size,
            "sha256": digest(path)}


def publish(source, target, checksum):
    """Idempotent copy with atomic publication; never overwrite different existing data."""
    if target.exists():
        if digest(target) != checksum:
            raise FileExistsError(f"Existing file differs: {target}; choose another --output")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary)
        if digest(temporary) != checksum:
            raise ValueError(f"Source changed while copying: {source}")
        try:
            target.hardlink_to(temporary)
        except FileExistsError:
            if digest(target) != checksum:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def download(args):
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("Install uv before using --download")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venv-retarget")
    command = [uv, "run", "--extra", "retarget", "--no-dev", "--locked", "python",
               str(ROOT / "baselines/supermimic/scripts/data/prepare_lafan_modelscope.py"),
               "--robots", "r1", "--output-dir", str(args.output.parent),
               "--pattern", args.pattern, "--workers", str(args.workers),
               "--dataset-id", args.dataset_id, "--revision", args.revision]
    if args.offline:
        command.append("--offline")
    print("[download + retarget]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source", type=Path, help="Existing R1 NPZ directory to copy")
    mode.add_argument("--download", action="store_true", help="Download source CSV and retarget R1 in .venv-retarget")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Shared NPZ directory (last component must be r1 for --download)")
    parser.add_argument("--pattern", default="*", help="Clip stem pattern, e.g. walk*; no extension")
    parser.add_argument("--checkpoint", type=Path, help="Copy your existing R1 SuperMimic checkpoint too")
    parser.add_argument("--checkpoint-output", type=Path,
                        default=ROOT / "data/checkpoints/r1_supermimic_pretrained.pt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset-id", default="AI-ModelScope/LAFAN1_Retargeting_Dataset")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--offline", action="store_true", help="Reuse cached ModelScope source with --download")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.download:
        if args.output.name != "r1":
            parser.error("--download requires an --output ending in /r1")
        # The old pipeline selects source filenames including their CSV extension.
        args.pattern += ".csv"
        download(args)
        args.pattern = args.pattern[:-4]
    source = args.source.resolve() if args.source else args.output
    files = [p for p in sorted(source.glob("*.npz")) if fnmatch.fnmatchcase(p.stem, args.pattern)]
    if not files:
        parser.error(f"No matching R1 NPZ under {source}; pass --source or --download")
    rows = [validate(p) for p in files]
    # Validate the selected set and detect conflicts before publishing anything.
    for path, row in zip(files, rows):
        target = args.output / path.name
        if target.exists() and digest(target) != row["sha256"]:
            raise FileExistsError(f"Existing file differs: {target}; choose another --output")
    for path, row in zip(files, rows):
        publish(path, args.output / path.name, row["sha256"])
    checkpoint = None
    if args.checkpoint:
        path = args.checkpoint.resolve()
        checksum = digest(path)
        publish(path, args.checkpoint_output.resolve(), checksum)
        checkpoint = {"path": str(args.checkpoint_output.resolve()), "sha256": checksum,
                      "note": "Copied as supplied; training validates checkpoint robot and network shapes"}
    manifest = {"schema_version": 1, "robot": "r1", "dataset": "LAFAN1",
                "source": str(source), "output": str(args.output), "pattern": args.pattern,
                "download_request": {"dataset_id": args.dataset_id, "revision": args.revision}
                                    if args.download else None,
                "count": len(rows), "frames": sum(r["frames"] for r in rows),
                "clips": rows, "checkpoint": checkpoint,
                "split": "No train/test split assigned; this manifest is the selected data inventory"}
    dest = args.output / "dataset_manifest.json"
    temporary = dest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(dest)
    print(f"[ready] {len(rows)} R1 clips, {manifest['frames']} frames -> {args.output}")
    print(f"[manifest] {dest}")


if __name__ == "__main__":
    main()
