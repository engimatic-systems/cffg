"""Private model, review, discovery and experiment API for CFFG Studio."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4
from zipfile import BadZipFile

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cffg import inits
from cffg.core.ir import FACTOR_TYPES
from .access import Access
from .discovery import Discovery, file_digest
from .editing import apply_commands
from .jobs import Jobs
from .library import Library, Conflict, digest, identifier
from .models import Document, MAX_ARCHIVE_BYTES, starter


class NewModel(BaseModel):
    kind: Literal["blank", "kalman", "rslds"] = "blank"
    name: str = Field(min_length=1, max_length=160)
    steps: int = 3
    latent: int = 1
    observed: int = 1
    regimes: int = 2
    seed: int = 0


class Transaction(BaseModel):
    base: str = Field(pattern=r"^[0-9a-f]{40}$")
    request_id: str = Field(
        default_factory=lambda: uuid4().hex, pattern=r"^[a-zA-Z0-9_-]{8,96}$"
    )
    commands: list[dict[str, Any]] = Field(default_factory=list, max_length=256)
    restore: str | None = None
    summary: str = Field(default="Edit graph", min_length=1, max_length=160)


class Run(BaseModel):
    model: str
    revision: str
    kind: Literal["run", "trace", "profile", "generate", "simulate"] = "run"
    seed: int = Field(default=7, ge=0, le=2**32 - 1)
    iterations: int = Field(default=12, ge=1, le=1024)
    tolerance: float = Field(default=1.0e-5, gt=0, allow_inf_nan=False)
    repetitions: int = Field(default=5, ge=1, le=50)
    paused: bool = False
    observations: dict | None = None
    tape: str | None = None
    include_structure: bool = False
    prompt: str = Field(default="", max_length=16000)
    source: str = Field(default="", max_length=60000)
    steps: int = Field(default=3, ge=1, le=10000)
    parameters: dict = Field(default_factory=dict)


def difference(a, b):
    out = {}
    for group in ("variables", "factors", "ports", "blobs", "plates", "learnables"):
        before = a["template"].get(group, {})
        after = b["template"].get(group, {})
        changed = sorted(
            n for n in before.keys() | after.keys() if before.get(n) != after.get(n)
        )
        if changed:
            out[group] = changed
    for group in ("arrays", "learning"):
        before = a.get(group, {})
        after = b.get(group, {})
        changed = sorted(
            n for n in before.keys() | after.keys() if before.get(n) != after.get(n)
        )
        if changed:
            out[group] = changed
    if out.get("arrays"):
        changed = set(out["arrays"])
        out["parameter_consumers"] = [
            f"{name}.{role} → {blob}"
            for name, factor in b["template"]["factors"].items()
            for role, blob in factor["param_blobs"].items()
            if blob in changed
        ]
        out["array_shapes"] = {
            n: {
                "from": a["arrays"].get(n, {}).get("shape"),
                "to": b["arrays"].get(n, {}).get("shape"),
            }
            for n in sorted(changed)
        }
    if out.get("variables"):
        out["variable_domains"] = {
            n: {
                "from": a["template"]["variables"].get(n),
                "to": b["template"]["variables"].get(n),
            }
            for n in out["variables"]
        }
    for group in ("name", "constraints", "agent"):
        if a["template"].get(group) != b["template"].get(group):
            out[group] = b["template"].get(group)
    if a["metadata"].get("layout", {}) != b["metadata"].get("layout", {}):
        out["layout"] = b["metadata"].get("layout", {})
    old_tours = a["metadata"].get("tours", {})
    new_tours = b["metadata"].get("tours", {})
    if old_tours != new_tours:
        out["tours"] = sorted(
            k
            for k in old_tours.keys() | new_tours.keys()
            if old_tours.get(k) != new_tours.get(k)
        )
    for group in ("notes", "reviews"):
        if a["metadata"].get(group, []) != b["metadata"].get(group, []):
            out[group] = len(b["metadata"].get(group, []))
    return out


def create_app(home=None, repository=None, roots=None):
    home = Path(
        home or os.getenv("CFFG_STUDIO_HOME", Path.home() / ".local/share/cffg-studio")
    ).resolve()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    library = Library(
        repository
        or os.getenv("CFFG_MODEL_LIBRARY", Path.home() / "projects/cffg-models")
    )
    access = Access(home)
    jobs = Jobs(home / "runs", library)
    search = Discovery(home, roots if roots is not None else [Path.home() / "projects"])
    static = Path(__file__).with_name("static")
    asset_version = digest(
        {p.name: file_digest(p) for p in static.iterdir() if p.is_file()}
    )[:16]
    shell = (static / "index.html").read_text().replace(
        "/static/", f"/ui/{asset_version}/"
    )

    watchers = {}
    loop = None

    def announce(model):
        def wake():
            for event in watchers.get(model, ()):
                event.set()

        if loop and loop.is_running():
            loop.call_soon_threadsafe(wake)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal loop
        loop = asyncio.get_running_loop()
        from watchfiles import awatch

        async def reconcile():
            # Poll for changes as a fallback when native watcher resources are unavailable.
            while True:
                await asyncio.sleep(180)
                search.scan()

        async def watch():
            while True:
                directories = [root for root in search.roots() if root.is_dir()]
                if not directories:
                    await asyncio.sleep(1)
                    continue
                try:
                    async for changes in awatch(
                        *directories,
                        raise_interrupt=False,
                        rust_timeout=1000,
                        yield_on_timeout=True,
                    ):
                        if directories != [
                            root for root in search.roots() if root.is_dir()
                        ]:
                            break
                        if any(
                            Path(p).suffix.lower()
                            in (".cffgz", ".cffgstudio", ".cffg", ".json")
                            for _, p in changes
                        ):
                            search.scan()
                except (OSError, RuntimeError):
                    await asyncio.sleep(3)

        tasks = [asyncio.create_task(reconcile()), asyncio.create_task(watch())]
        yield
        search.stop.set()
        for task in tasks:
            task.cancel()
        jobs.shutdown()

    app = FastAPI(title="CFFG Studio", version="1.0", lifespan=lifespan)
    app.state.library = library
    app.state.access = access
    app.state.jobs = jobs

    @app.middleware("http")
    async def authenticate(request, call_next):
        host = urlsplit("//" + request.headers.get("host", "")).hostname
        public = os.getenv("CFFG_STUDIO_HOST", "")
        if host not in {"127.0.0.1", "localhost", "::1", public, "testserver"}:
            return JSONResponse({"detail": "Unknown Studio host"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and not (
            urlsplit(origin).netloc == request.headers.get("host")
            and urlsplit(origin).scheme in ("http", "https")
        ):
            return JSONResponse(
                {"detail": "Cross-origin request refused"}, status_code=403
            )
        if (
            request.method not in ("GET", "HEAD")
            and request.headers.get("sec-fetch-site") == "cross-site"
        ):
            return JSONResponse(
                {"detail": "Cross-site changes refused"}, status_code=403
            )
        principal = access.identify(request)
        if not principal:
            return JSONResponse(
                {
                    "detail": "Use the owner's Tailscale account or a scoped Studio bearer token"
                },
                status_code=401,
            )
        request.state.actor, request.state.scope = principal
        if principal[1] == "read" and request.method not in ("GET", "HEAD"):
            return JSONResponse(
                {"detail": "This agent has read-only access"}, status_code=403
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path == "/" or request.url.path.startswith(("/static/", "/ui/")):
            response.headers["Cache-Control"] = "private, no-cache"
        return response

    @app.exception_handler(Conflict)
    async def conflict(_, err):
        return JSONResponse(
            {"detail": str(err), "current": err.current}, status_code=409
        )

    @app.exception_handler(KeyError)
    async def missing(_, err):
        return JSONResponse({"detail": str(err)}, status_code=404)

    async def invalid(_, err):
        return JSONResponse({"detail": str(err)}, status_code=422)

    for exc in (ValueError, TypeError, BadZipFile, FileNotFoundError, IndexError):
        app.add_exception_handler(exc, invalid)

    def owner(request):
        from fastapi import HTTPException

        if request.state.scope != "owner":
            raise HTTPException(403, "Owner access is required")

    @lru_cache(maxsize=40)
    def snapshot(model, rev, plate=None):
        doc = library.get(model, rev)
        if plate is not None:
            parent = doc
            doc = parent.fragment(plate)
            doc.meta["model_digest"] = parent.meta.get("model_digest")
            doc.meta["resources"] = parent.meta.get("resources", {})
            doc.sync_anchors()
        return {**doc.describe(), "id": model, "revision": rev}

    @app.get("/")
    def index():
        return HTMLResponse(shell)

    # The prefix gives the entire ES module graph a new cache key on deployment.
    # Older open tabs can still request late-loaded assets after the server restarts.
    app.mount("/ui/{asset_version}", StaticFiles(directory=static), name="ui_assets")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/v1/capabilities")
    def capabilities():
        from .tours import Tour

        return {
            "tour_schema": Tour.model_json_schema(),
            "factors": {name: asdict(f) for name, f in FACTOR_TYPES.items()},
            "initializers": sorted(inits.INITIALIZERS),
            "recipes": ["blank", "kalman", "rslds"],
            "commands": [
                "edit_plate",
                "set_tour",
                "remove_tour",
                "rename_variable",
                "resize_variable",
                "initialize_factor",
                "add_channel",
                "connect_ports",
                "set_array",
                "position_factor",
                "edit_annotation",
                "remove_annotation",
                "model_name",
                "annotate",
                "critique",
                "propose",
                "reply",
                "resolve",
                "accept",
                "decline",
                "add_factor",
                "replace_factor",
                "remove_factor",
                "add_variable",
                "remove_variable",
                "declare_port",
                "remove_port",
                "add_constraint",
                "remove_constraint",
                "add_blob",
                "rebind_blob_path",
                "set_plate_dims",
                "fan_out",
                "observe_interior",
            ],
        }

    @app.get("/v1/models")
    def models(q: str = ""):
        return [m for m in library.list() if q.casefold() in m["name"].casefold()]

    @app.post("/v1/models")
    def new_model(body: NewModel, request: Request):
        return library.create(starter(**body.model_dump()), request.state.actor)

    @app.get("/v1/models/{model}")
    def get(model: str, plate: str | None = None):
        return snapshot(model, library.revision(model), plate)

    @app.get("/v1/models/{model}/revisions/{rev}")
    def revision(model: str, rev: str, plate: str | None = None):
        return snapshot(model, library.revision(model, rev), plate)

    @app.get("/v1/models/{model}/history")
    def history(model: str):
        return library.history(model)

    @app.get("/v1/models/{model}/diff")
    def compare(model: str, base: str, to: str = "HEAD"):
        return difference(
            snapshot(model, library.revision(model, base)),
            snapshot(model, library.revision(model, to)),
        )

    @app.get("/v1/models/{model}/export")
    def export(model: str, revision: str = "HEAD", project: bool = False):
        revision = library.revision(model, revision)
        data = library.bundle(model, revision)
        if project:
            data = library.get(model, revision).project(data)
        extension = "cffgstudio" if project else "cffgz"
        return Response(
            data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{identifier(model)}.{extension}"'
            },
        )

    @app.post("/v1/models/{model}/fork")
    def fork(model: str, body: dict, request: Request):
        rev = library.revision(model, body.get("revision", "HEAD"))
        doc = library.get(model, rev)
        doc.template.name = body["name"]
        doc.meta.pop("receipts", None)
        doc.meta["fork"] = {"id": model, "revision": rev}
        return library.create(doc, request.state.actor, "Fork " + doc.template.name)

    def transaction(model, body, request, preview):
        current = library.revision(model)
        actor = request.state.actor
        fingerprint = digest([model, actor, body.model_dump(exclude={"request_id"})])
        previous = library.meta(model, current)
        if not preview and body.request_id in previous.get("receipts", {}):
            if previous["receipts"][body.request_id] != fingerprint:
                raise Conflict("Request ID reused with different content", current)
            return {
                "id": model,
                "revision": library.receipt(model, body.request_id),
                "replayed": True,
            }
        if current != body.base:
            raise Conflict("Model changed since this edit started", current)
        before = snapshot(model, current)
        if body.restore:
            if body.commands:
                raise ValueError("Use restore or commands in one transaction")
            doc = library.get(model, body.restore)
        else:
            doc = library.get(model, current)
            apply_commands(doc, body.commands, actor, current)
        metadata_commands = {
            "set_tour",
            "remove_tour",
            "annotate",
            "critique",
            "propose",
            "reply",
            "resolve",
            "decline",
            "edit_annotation",
            "remove_annotation",
            "position_factor",
        }
        if (
            not body.restore
            and all(c.get("action") in metadata_commands for c in body.commands)
            and doc.meta.get("resources")
        ):
            data = library.bundle(model, current)
        else:
            data = doc.certify()
        changes = difference(before, doc.describe())
        for command in body.commands:
            if command.get("action") == "edit_plate":
                plate = command["plate"]
                changes.setdefault("fragments", {})[plate] = difference(
                    snapshot(model, current, plate), doc.fragment(plate).describe()
                )
        if preview:
            return {
                "base": current,
                "changes": changes,
                "resources": doc.meta["resources"],
            }
        rev = library.publish(
            model, current, doc, data, actor, body.summary, body.request_id, fingerprint
        )
        announce(model)
        return {"id": model, "revision": rev, "changes": changes}

    @app.post("/v1/models/{model}/transactions/preview")
    def preview(model: str, body: Transaction, request: Request):
        return transaction(model, body, request, True)

    @app.post("/v1/models/{model}/transactions")
    def commit(model: str, body: Transaction, request: Request):
        return transaction(model, body, request, False)

    @app.post("/v1/models/{model}/critiques")
    @app.post("/v1/models/{model}/proposals")
    def review(model: str, body: dict, request: Request):
        op = "propose" if request.url.path.endswith("/proposals") else "critique"
        cmd = {k: v for k, v in body.items() if k in ("ref", "text", "commands")}
        return transaction(
            model,
            Transaction(
                base=body["base"],
                commands=[{**cmd, "action": op}],
                request_id=body.get("request_id", uuid4().hex),
                summary=op.title(),
            ),
            request,
            False,
        )

    @app.get("/v1/search")
    def find(q: str = "", compatible_only: bool = False):
        return {
            "files": search.find(q, compatible_only=compatible_only),
            "scanning": search.lock.locked(),
            "error": search.error,
            "roots": [str(p) for p in search.roots()],
        }

    @app.post("/v1/search/scan")
    def rescan(request: Request, body: dict):
        owner(request)
        if "roots" in body:
            search.configure(body["roots"])
        else:
            search.scan()
        return {"scanning": True}

    @app.post("/v1/import")
    def import_path(body: dict, request: Request):
        return library.create(
            search.load(body["path"]), request.state.actor, "Import model"
        )

    @app.post("/v1/import/bundle")
    async def upload(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_ARCHIVE_BYTES:
                raise ValueError("Bundle exceeds size limit")
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(
            lambda: library.create(
                Document.exchange(body), request.state.actor, "Import bundle"
            )
        )

    @app.post("/v1/models/{model}/save-back")
    def save_back(model: str, request: Request, body: dict):
        owner(request)
        with library.lock():
            rev = library.revision(model)
            if rev != body["base"]:
                raise Conflict("Model changed", rev)
            source = library.meta(model, rev).get("source", {})
            path = search.allowed(source["path"])
            if path.suffix.lower() not in (".cffgz", ".cffgstudio"):
                raise ValueError("Export a portable bundle for this loose template")
            receipts = home / "source-saves"
            receipts.mkdir(exist_ok=True, mode=0o700)
            stamp = receipts / model
            expected = stamp.read_text().strip() if stamp.exists() else source["sha256"]
            if file_digest(path) != expected:
                raise Conflict("The original file changed", rev)
            data = library.bundle(model, rev)
            if path.suffix.lower() == ".cffgstudio":
                data = library.get(model, rev).project(data)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                os.fchmod(f.fileno(), path.stat().st_mode & 0o777)
                f.write(data)
                temp = f.name
            os.replace(temp, path)
            saved = file_digest(path)
            stamp.write_text(saved)
        return {"path": str(path), "sha256": saved, "revision": rev}

    @app.post("/v1/runs")
    def start(body: Run):
        if body.kind == "generate" and not body.prompt.strip():
            raise ValueError("Describe the simulator dynamics")
        if body.kind == "simulate" and not body.source.strip():
            raise ValueError("Supply simulator source")
        if body.kind == "trace" and body.iterations > 128:
            raise ValueError("Interactive traces support up to 128 budget samples")
        return jobs.start(body.model_dump())

    @app.get("/v1/runs")
    def runs(model: str):
        return jobs.list(identifier(model))

    @app.get("/v1/runs/{job}")
    def job(job: str):
        return jobs.get(job)

    @app.post("/v1/runs/{job}/control")
    def control(job: str, body: dict):
        return jobs.control(job, body["action"])

    @app.get("/v1/runs/{job}/tape")
    def tape(job: str):
        path = jobs.folder(job) / "tape.json"
        if not path.exists():
            raise KeyError("Tape unavailable")
        return FileResponse(path, media_type="application/json")

    @app.get("/v1/runs/{job}/events")
    @app.get("/v1/models/{model}/events")
    async def stream(
        request: Request,
        job: str | None = None,
        model: str | None = None,
        after: str = "",
    ):
        cursor = request.headers.get("last-event-id") or after

        async def generate():
            nonlocal cursor
            from starlette.concurrency import run_in_threadpool

            signal = asyncio.Event()
            if model:
                watchers.setdefault(model, set()).add(signal)
            try:
                while not await request.is_disconnected():
                    signal.clear()
                    if job:
                        row = await run_in_threadpool(jobs.get, job)
                        for i, event in enumerate(row["events"], 1):
                            if i > int(cursor or 0):
                                yield f"id: {i}\ndata: {json.dumps(event)}\n\n"
                                cursor = str(i)
                        if row["status"] not in ("queued", "running", "paused"):
                            yield "event: complete\ndata: {}\n\n"
                            break
                    else:
                        head = await run_in_threadpool(library.revision, model)
                        if head != cursor:
                            rows = await run_in_threadpool(library.history, model)
                            ids = [r["revision"] for r in rows]
                            updates = (
                                rows[: ids.index(cursor)] if cursor in ids else rows[:1]
                            )
                            for event in reversed(updates):
                                cursor = event["revision"]
                                yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                    if job:
                        await asyncio.sleep(0.12)
                    else:
                        try:
                            await asyncio.wait_for(signal.wait(), timeout=12)
                        except TimeoutError:
                            yield ": heartbeat\n\n"
            finally:
                if model:
                    watchers[model].discard(signal)
                    if not watchers[model]:
                        del watchers[model]

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app
