"""Persistent experiment requests and replayable events from isolated worker processes."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import shutil
from tempfile import TemporaryDirectory
import subprocess
import sys
from threading import Lock
import time
from uuid import uuid4

from .library import identifier
from .models import json_bytes


def write(path, value):
    tmp = path.with_name(path.name + "." + uuid4().hex)
    tmp.write_bytes(json_bytes(value))
    os.replace(tmp, path)


def events(folder):
    file = folder / "events.jsonl"
    if not file.exists():
        return []
    lines = file.read_bytes().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b"\n")]


class Jobs:
    def __init__(self, root, library):
        self.root = root
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.library = library
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.lock = Lock()
        self.running = {}
        for state in root.glob("*/status.json"):
            if json.loads(state.read_text())["status"] in (
                "running",
                "queued",
                "paused",
            ):
                self.cleanup_container(state.parent)
                write(state, {"status": "interrupted", "detail": "Studio restarted"})

    def folder(self, job):
        p = self.root / identifier(job)
        if not (p / "request.json").exists():
            raise KeyError("Unknown job")
        return p

    def start(self, spec):
        staging = self.root / ".staging"
        staging.mkdir(exist_ok=True)
        with self.lock, TemporaryDirectory(dir=staging) as tmp:
            if (
                sum(
                    1
                    for p in self.root.glob("*/status.json")
                    if json.loads(p.read_text())["status"]
                    in ("queued", "running", "paused")
                )
                >= 12
            ):
                raise ValueError("Experiment queue is full")
            job = uuid4().hex
            folder = Path(tmp)
            spec = {**spec, "id": job, "created": time.time()}
            revision = self.library.revision(spec["model"], spec["revision"])
            if spec["revision"] != revision:
                raise ValueError("Select the model's own revision")
            if spec.get("tape"):
                other = self.folder(spec["tape"]) / "tape.json"
                (folder / "tape.json").write_bytes(other.read_bytes())
            (folder / "model.cffgz").write_bytes(
                self.library.bundle(spec["model"], revision)
            )
            write(folder / "request.json", spec)
            write(
                folder / "control.json",
                {"mode": "pause" if spec.get("paused") else "run", "steps": 0},
            )
            write(folder / "status.json", {"status": "queued"})
            folder.rename(self.root / job)
            self.pool.submit(self._work, self.root / job, spec)
        return self.get(job)

    def shutdown(self):
        for status in self.root.glob("*/status.json"):
            if self.status(status.parent) in ("running", "queued", "paused"):
                try:
                    self.control(status.parent.name, "cancel")
                except ValueError:
                    pass
        self.pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def cleanup_container(folder):
        spec = json.loads((folder / "request.json").read_text())
        docker = shutil.which("docker")
        if spec["kind"] == "simulate" and docker:
            try:
                subprocess.run(
                    [
                        docker,
                        "--context",
                        os.getenv("CFFG_DOCKER_CONTEXT", "colima"),
                        "rm",
                        "-f",
                        "cffg-simulator-" + identifier(folder.name),
                    ],
                    capture_output=True,
                    timeout=8,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _work(self, folder, spec):
        try:
            with self.lock, (folder / "worker.log").open("wb") as log:
                if self.status(folder) == "cancelled":
                    return
                env = {
                    **os.environ,
                    "CFFG_WORKER_STARTED": str(time.time()),
                    "JAX_ENABLE_COMPILATION_CACHE": "false",
                    "PYTHONUNBUFFERED": "1",
                }
                proc = subprocess.Popen(
                    [sys.executable, "-m", "cffg.studio.worker", str(folder)],
                    stdout=log,
                    stderr=log,
                    env=env,
                    start_new_session=True,
                )
                self.running[folder.name] = proc
                write(folder / "status.json", {"status": "running", "pid": proc.pid})
            try:
                code = proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                self._terminate(proc)
                raise ValueError("Worker time limit exceeded")
            with self.lock:
                if self.status(folder) == "cancelled":
                    return
                record = events(folder)
                last = record[-1] if record else {}
                complete = code == 0 and last.get("type") == "finish"
                write(
                    folder / "status.json",
                    {
                        "status": "finished" if complete else "failed",
                        "detail": last.get(
                            "message", "" if complete else f"Worker exited {code}"
                        ),
                    },
                )
        except Exception as e:
            write(folder / "status.json", {"status": "failed", "detail": str(e)})
        finally:
            if self.status(folder) in ("cancelled", "failed"):
                self.cleanup_container(folder)
            with self.lock:
                self.running.pop(folder.name, None)

    @staticmethod
    def _terminate(proc):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    @staticmethod
    def status(folder):
        return json.loads((folder / "status.json").read_text())["status"]

    def control(self, job, action):
        folder = self.folder(job)
        with self.lock:
            if self.status(folder) not in ("queued", "running", "paused"):
                raise ValueError("This job has ended")
            if action == "cancel":
                write(folder / "status.json", {"status": "cancelled"})
                if proc := self.running.get(job):
                    self._terminate(proc)
            elif action in ("pause", "resume", "step"):
                if json.loads((folder / "request.json").read_text())["kind"] != "trace":
                    raise ValueError("Step controls require trace mode")
                current = json.loads((folder / "control.json").read_text())
                write(
                    folder / "control.json",
                    {
                        "mode": "run" if action == "resume" else "pause",
                        "steps": current["steps"] + int(action == "step"),
                    },
                )
            else:
                raise ValueError("Unknown run control")
        return self.get(job)

    def get(self, job):
        folder = self.folder(job)
        spec = json.loads((folder / "request.json").read_text())
        info = json.loads((folder / "status.json").read_text())
        record = events(folder)
        result = (
            json.loads((folder / "result.json").read_text())
            if (folder / "result.json").exists()
            else None
        )
        if (
            spec["kind"] == "trace"
            and info["status"] in ("queued", "running")
            and json.loads((folder / "control.json").read_text())["mode"] == "pause"
        ):
            info["status"] = "paused"
        return {**spec, **info, "events": record, "result": result}

    def list(self, model):
        rows = []
        for path in self.root.glob("*/request.json"):
            req = json.loads(path.read_text())
            if req["model"] == model:
                rows.append(
                    {k: req[k] for k in ("id", "model", "revision", "kind", "created")}
                    | {"status": self.status(path.parent)}
                )
        return sorted(rows, key=lambda row: row["created"], reverse=True)
