"""Root training data must be ignored without excluding identically named source folders."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_ignore_rule_preserves_baseline_sources():
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("This check requires a Git checkout")
    for name in ["baselines/supermimic/src/supermimic/data/replay_buffer.py",
                 "baselines/supermimic/scripts/data/prepare_lafan_modelscope.py",
                 "baselines/supermimic/config/data/lafan1_g1_files.txt"]:
        assert (ROOT / name).is_file()
        result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", name], cwd=ROOT)
        assert result.returncode == 1, f"Deployment source ignored: {name}"
    result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", "data/example.npz"], cwd=ROOT)
    assert result.returncode == 0
