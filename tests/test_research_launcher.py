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
                  skip_smoke=False, stage="pilot", gpu="0")
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
