"""Server-side experiment records shared by the original baseline and all TADG arms."""

import hashlib
import json
import math
import platform
import random
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def gpu_status():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "csv": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


@contextmanager
def evaluation_rng(seed):
    """Evaluation must not change subsequent training randomness."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class RunRecorder:
    def __init__(self, directory, metadata, resource_interval=30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._events = (self.directory / "events.jsonl").open("a", buffering=1)
        self.counters = {"wm_updates": 0, "wm_optimizer_steps": 0, "actor_updates": 0,
                         "q_optimizer_steps": 0, "imagination_steps": 0,
                         "eval_env_transitions": 0}
        self.times = {"wm_s": 0.0, "actor_and_q_s": 0.0, "evaluation_s": 0.0,
                      "checkpoint_s": 0.0}
        write_json(self.directory / "run_metadata.json", {
            **metadata, "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "gpu_initial": gpu_status(),
            "timing_note": "wall time includes instrumentation; evaluation/checkpoint time is also itemized",
        })

        def monitor():
            while not self._stop.wait(resource_interval):
                self.event("resources", gpu=gpu_status())

        self._monitor = threading.Thread(target=monitor, daemon=True)
        self._monitor.start()

    def event(self, kind, **values):
        record = {"event": kind, "unix_time": time.time(),
                  "wall_s": time.perf_counter() - self.started, **values}
        with self._lock:
            self._events.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")

    def attach(self, trainer):
        def wrap(original, kind):
            def update(*args, **kwargs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                metrics = original(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                duration = time.perf_counter() - start
                if kind == "wm_update":
                    self.counters["wm_updates"] += 1
                    self.counters["wm_optimizer_steps"] += trainer.cfg.wm_trainer.batches_per_update
                    self.times["wm_s"] += duration
                else:
                    self.counters["actor_updates"] += 1
                    is_dual = hasattr(trainer.cfg, "tadg")
                    self.counters["q_optimizer_steps"] += trainer.cfg.tadg.q_updates if is_dual else 0
                    imagines = not is_dual or trainer.cfg.tadg.dynamics_weight > 0
                    if imagines:
                        self.counters["imagination_steps"] += (
                            trainer.cfg.policy_trainer.batch_size * trainer.cfg.policy_trainer.horizon)
                    self.times["actor_and_q_s"] += duration
                self.event(kind, iteration=trainer.iteration,
                           env_transitions=trainer.iteration * trainer.env.num_envs,
                           duration_s=duration, counters=dict(self.counters), metrics=metrics,
                           allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
                return metrics
            return update

        trainer.wm_trainer.update = wrap(trainer.wm_trainer.update, "wm_update")
        trainer.policy_trainer.update = wrap(trainer.policy_trainer.update, "actor_update")
        if trainer.writer is not None:
            original_scalar = trainer.writer.add_scalar

            def add_scalar(tag, scalar_value, global_step=None, *args, **kwargs):
                self.event("scalar", tag=tag, value=scalar_value, iteration=global_step)
                return original_scalar(tag, scalar_value, global_step, *args, **kwargs)

            trainer.writer.add_scalar = add_scalar

    def finish(self, status, **values):
        self._stop.set()
        self._monitor.join(timeout=12)
        result = {"status": status, "total_wall_s": time.perf_counter() - self.started,
                  "counters": self.counters, "times": self.times, **values}
        self.event("run_end", **result)
        write_json(self.directory / "status.json", result)
        with self._lock:
            self._events.close()
