"""Data publication must preserve prior datasets; download must use the retarget environment."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_r1_lafan", Path(__file__).resolve().parents[1] / "scripts/prepare_r1_lafan.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_publish_is_idempotent_and_preserves_conflicting_data(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"existing motion")
    prepare.publish(source, target, prepare.digest(source))
    prepare.publish(source, target, prepare.digest(source))
    assert target.read_bytes() == b"existing motion"
    source.write_bytes(b"different retarget version")
    with pytest.raises(FileExistsError):
        prepare.publish(source, target, prepare.digest(source))
    assert target.read_bytes() == b"existing motion"


def test_download_only_retargets_r1_in_separate_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(prepare.shutil, "which", lambda name: "/bin/uv")
    monkeypatch.setattr(prepare.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setenv("PYTHONPATH", "/old/robot/environment")
    prepare.download(SimpleNamespace(output=tmp_path / "r1", pattern="walk*.csv", workers=2,
                                     dataset_id="dataset", revision="revision", offline=True))
    command, options = calls[0]
    assert command[command.index("--robots") + 1] == "r1"
    assert "k1" not in command and "--offline" in command
    assert command[command.index("--extra") + 1] == "retarget"
    assert Path(options["env"]["UV_PROJECT_ENVIRONMENT"]).name == ".venv-retarget"
    assert "PYTHONPATH" not in options["env"]
