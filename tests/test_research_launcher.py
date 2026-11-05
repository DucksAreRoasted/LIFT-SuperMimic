"""Exercise orchestration without installing packages or starting a simulator."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "launch_research", Path(__file__).resolve().parents[1] / "scripts/launch_research.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def arguments(tmp_path, **overrides):
    values = dict(output=tmp_path, motions=tmp_path / "r1", checkpoint=tmp_path / "pretrained.pt",
                  data_source=None, download_data=False, workers=4, no_sync=False,
                  skip_smoke=False, stage="pilot", gpu="0", background_backend="tmux")
    return SimpleNamespace(**(values | overrides))


def test_default_plan_has_matching_budgets_and_full_evaluation(tmp_path):
    plan = launcher.build_plan(arguments(tmp_path), "/bin/uv")
    assert [p["name"] for p in plan] == ["environment", "data", "smoke", "pilot"]
    pilot = plan[-1]
    assert pilot["methods"] == ["supermimic", "q_only", "bptt_soft", "dual_fixed"]
    assert pilot["seeds"] == [0] and pilot["transitions_per_run"] == 1_280_000
    assert pilot["protocol"]["eval_max_steps"] == 0
    assert pilot["protocol"]["max_hours_per_run"] is None
    assert len(pilot["protocol"]["train_clips"]) == 40
    assert len(pilot["protocol"]["eval_clips"]) == 40


def test_failed_stage_stops_later_training_and_keeps_state(tmp_path, monkeypatch):
    calls = []
    plan = [{"name": "smoke", "command": ["smoke"]},
            {"name": "pilot", "command": ["pilot"]}]
    launcher.write_json(tmp_path / "plan.json", plan)

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.execute(arguments(tmp_path))
    assert calls == [["smoke"]]
    state = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert state["status"] == "failed" and state["stages"][0]["status"] == "failed"
    assert "CalledProcessError" in state["failure"]


def test_time_budget_stopped_run_is_not_treated_as_success(tmp_path, monkeypatch):
    directory = tmp_path / "pilot"
    status = directory / "seed_0" / "dual_fixed" / "status.json"
    status.parent.mkdir(parents=True)
    launcher.write_json(status, {"status": "time_budget_stopped"})
    launcher.write_json(tmp_path / "plan.json", [{"name": "pilot", "command": ["pilot"],
                        "results": str(directory), "methods": ["dual_fixed"], "seeds": [0]}])
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Incomplete run"):
        launcher.execute(arguments(tmp_path))
    assert json.loads((tmp_path / "pipeline_status.json").read_text())["status"] == "failed"


def test_successful_pipeline_records_completion_and_isolates_train_environment(tmp_path, monkeypatch):
    launcher.write_json(tmp_path / "plan.json", [{"name": "data", "command": ["prepare"]},
                                               {"name": "pilot", "command": ["train"]}])
    calls = []
    monkeypatch.setenv("PYTHONPATH", "/old/python")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/old/retarget")
    monkeypatch.setattr(launcher.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    launcher.execute(arguments(tmp_path, gpu="2"))
    assert [cmd for cmd, _ in calls] == [["prepare"], ["train"]]
    for _, options in calls:
        assert options["env"]["UV_PROJECT_ENVIRONMENT"] == str(launcher.ROOT / ".venv")
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        assert "PYTHONPATH" not in options["env"]
    state = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert state["status"] == "completed"
    assert all(stage["status"] == "completed" for stage in state["stages"])


def test_tmux_uses_quoted_command_and_writes_session_record(tmp_path, monkeypatch):
    import shlex
    calls = []
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/bin/tmux")
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    command = ["/path with spaces/python", "worker.py", "--checkpoint", "literal$(false); name.pt"]
    launcher.start_background(arguments(tmp_path), command)
    args, options = calls[0]
    assert args[:3] == ["/bin/tmux", "new-session", "-d"]
    assert options["check"] is True
    # Shell parsing must preserve literal arguments rather than interpreting checkpoint text.
    tokens = shlex.split(args[-1])
    assert tokens[0] == "exec" and tokens[1:len(command)+1] == command
    record = json.loads((tmp_path / "launcher.json").read_text())
    assert record["backend"] == "tmux" and record["session"].startswith("r1_tadg_")


def test_missing_tmux_does_not_silently_change_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="tmux is required"):
        launcher.start_background(arguments(tmp_path), ["worker.py"])


def test_github_mirror_preserves_existing_git_settings():
    initial = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "existing.key", "GIT_CONFIG_VALUE_0": "keep"}
    env = launcher.github_environment(initial, mirror="https://mirror.example/https://github.com/")
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert env["GIT_CONFIG_KEY_1"] == "url.https://mirror.example/https://github.com/.insteadOf"
    assert env["GIT_CONFIG_VALUE_1"] == "https://github.com/"
    assert env["GIT_CONFIG_VALUE_0"] == "keep"
    assert initial["GIT_CONFIG_COUNT"] == "1"


def test_github_proxy_is_scoped_and_connectivity_precedes_install(tmp_path):
    env = launcher.github_environment({}, proxy="http://127.0.0.1:7890")
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.proxy"
    assert "HTTPS_PROXY" not in env
    plan = launcher.build_plan(arguments(tmp_path, github_proxy="http://127.0.0.1:7890"), "/bin/uv")
    assert plan[0]["name"] == "github_access" and plan[0]["timeout_s"] == 90
    assert plan[1]["name"] == "environment"


def test_real_git_recognizes_mirror_and_host_scoped_proxy():
    import os
    import shutil
    if not shutil.which("git"):
        pytest.skip("Git is required for this offline configuration test")
    env = launcher.github_environment(os.environ, mirror="https://mirror.example/https://github.com/")
    url = subprocess.check_output(["git", "ls-remote", "--get-url", "https://github.com/mujocolab/mjlab"],
                                  env=env, text=True).strip()
    assert url == "https://mirror.example/https://github.com/mujocolab/mjlab"
    env = launcher.github_environment(os.environ, proxy="http://127.0.0.1:7890")
    proxy = subprocess.check_output(["git", "config", "--get-urlmatch", "http.proxy",
                                    "https://github.com/mujocolab/mjlab"], env=env, text=True).strip()
    assert proxy == "http://127.0.0.1:7890"


def test_modelscope_download_request_preserves_dataset_and_revision(tmp_path):
    args = arguments(tmp_path, download_data=True,
                     dataset_id=launcher.MODELSCOPE_DATASET, dataset_revision="pinned-revision")
    plan = launcher.build_plan(args, "/bin/uv")
    command = next(p for p in plan if p["name"] == "data")["command"]
    assert "--download" in command
    assert command[command.index("--dataset-id") + 1] == launcher.MODELSCOPE_DATASET
    assert command[command.index("--revision") + 1] == "pinned-revision"
