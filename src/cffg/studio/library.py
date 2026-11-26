"""Git-backed model library. A conditional ref update publishes each complete edit."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import shutil
from functools import lru_cache
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

from .models import Document, json_bytes


class Conflict(ValueError):
    def __init__(self, message, current=None):
        super().__init__(message)
        self.current = current


def digest(value):
    return sha256(json_bytes(value)).hexdigest()


def identifier(value):
    if not re.fullmatch("[0-9a-f]{32}", value):
        raise ValueError("Invalid model identifier")
    return value


class Library:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.tool_dir = (
            Path(
                os.getenv("CFFG_STUDIO_HOME", Path.home() / ".local/share/cffg-studio")
            )
            / "bin"
        )
        with self.lock():
            if not (self.path / "HEAD").exists():
                if set(p.name for p in self.path.iterdir()) - {"studio.lock"}:
                    raise ValueError(
                        f"Choose an empty directory for the model library: {self.path}"
                    )
                self.git("init", "--bare", "-b", "main", str(self.path))
                tree = self.git("mktree", data=b"").strip().decode()
                commit = (
                    self.git("commit-tree", tree, "-m", "Create CFFG model library")
                    .strip()
                    .decode()
                )
                self.git("update-ref", "refs/heads/main", commit)
                # Library revisions are published only after Studio verifies their exact content.
                self.git("config", "receive.denyNonFastForwards", "true")
            if (self.tool_dir / "git-lfs").is_file() or shutil.which("git-lfs"):
                self.git("lfs", "install", "--local", "--skip-repo")
                self.large_files = True
            else:
                self.large_files = False
            hook = self.path / "hooks/pre-receive"
            if not hook.exists():
                hook.write_text(
                    "#!/bin/sh\necho 'Publish edits with the CFFG Studio transaction API.' >&2\nexit 1\n"
                )
                hook.chmod(0o700)

    @contextmanager
    def lock(self):
        with (self.path / "studio.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def git(self, *args, data=None, env=None):
        config = {
            **os.environ,
            "GIT_AUTHOR_NAME": "CFFG Studio",
            "GIT_AUTHOR_EMAIL": "studio@localhost",
            "GIT_COMMITTER_NAME": "CFFG Studio",
            "GIT_COMMITTER_EMAIL": "studio@localhost",
            **(env or {}),
        }
        config["PATH"] = str(self.tool_dir) + os.pathsep + config.get("PATH", "")
        config.pop("GIT_LFS_SKIP_SMUDGE", None)
        proc = subprocess.run(
            ["git", "-C", str(self.path), *args],
            input=data,
            capture_output=True,
            env=config,
        )
        if proc.returncode:
            raise ValueError(
                proc.stderr.decode(errors="replace").strip() or "Git operation failed"
            )
        return proc.stdout

    def revision(self, model, at="HEAD"):
        identifier(model)
        if at != "HEAD" and not re.fullmatch("[0-9a-f]{40}", at):
            raise ValueError("A revision must be a full Git commit ID")
        rev = (
            self.git("log", "-1", "--format=%H", at, "--", f"models/{model}")
            .decode()
            .strip()
        )
        if not rev:
            raise KeyError("Unknown model or revision")
        return rev

    def history(self, model):
        identifier(model)
        rows = (
            self.git("log", "--format=%H%x09%aI%x09%s", "HEAD", "--", f"models/{model}")
            .decode()
            .splitlines()
        )
        return [
            dict(zip(("revision", "date", "summary"), row.split("\t", 2)))
            for row in rows
        ]

    def meta(self, model, rev="HEAD"):
        rev = self.revision(model, rev)
        return json.loads(self.git("show", f"{rev}:models/{model}/studio.json"))

    def list(self):
        return self._catalog(self.git("rev-parse", "HEAD").decode().strip())

    @lru_cache(maxsize=8)
    def _catalog(self, head):
        ids = (
            self.git("ls-tree", "--name-only", head + ":models").decode().splitlines()
            if self.git("ls-tree", head, "models")
            else []
        )
        result = []
        for model in ids:
            rev = self.revision(model, head)
            m = self.meta(model, rev)
            result.append(
                {
                    "id": model,
                    "name": m["name"],
                    "revision": rev,
                    "recipe": m.get("recipe", {}).get("kind", "import"),
                }
            )
        return sorted(result, key=lambda v: v["name"].casefold())

    def bundle(self, model, rev="HEAD"):
        rev = self.revision(model, rev)
        raw = self.git("archive", "--format=zip", f"{rev}:models/{model}")
        result = BytesIO()
        with (
            ZipFile(BytesIO(raw)) as archive,
            ZipFile(result, "w", ZIP_DEFLATED) as output,
        ):
            for name in archive.namelist():
                if not name.endswith("/") and name not in (
                    "studio.json",
                    ".gitattributes",
                ):
                    raw = archive.read(name)
                    if name.endswith(".npy") and not raw.startswith(b"\x93NUMPY"):
                        raise ValueError(
                            "A library array is unavailable. Restore the Git LFS object before export."
                        )
                    output.writestr(name, raw)
        return result.getvalue()

    def get(self, model, rev="HEAD"):
        rev = self.revision(model, rev)
        return Document.load(self.bundle(model, rev), self.meta(model, rev))

    def publish(
        self, model, base, document, bundle, actor, summary, request_id, fingerprint
    ):
        identifier(model)
        with self.lock(), TemporaryDirectory() as tmp:
            head = self.git("rev-parse", "HEAD").decode().strip()
            try:
                current = self.revision(model)
                previous = self.meta(model, current)
            except KeyError:
                current, previous = None, {}
            receipt = previous.get("receipts", {}).get(request_id)
            if receipt:
                if receipt != fingerprint:
                    raise Conflict(
                        "Request ID already used for different content", current
                    )
                return self.receipt(model, request_id)
            if current != base:
                raise Conflict(
                    "The model changed. Compare revisions and preview again.", current
                )
            document.meta.update(name=document.template.name, author=actor)
            document.meta["receipts"] = {
                **previous.get("receipts", {}),
                request_id: fingerprint,
            }
            relative = f"models/{model}"
            stage = Path(tmp, "tree")
            folder = (stage / relative).resolve()
            folder.mkdir(parents=True)
            with ZipFile(BytesIO(bundle)) as z:
                for info in z.infolist():
                    dest = (folder / info.filename).resolve()
                    if (
                        not dest.is_relative_to(folder)
                        or dest == folder / "studio.json"
                    ):
                        raise ValueError("Invalid staged bundle member")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(info))
            if any(p.stat().st_size >= 1024 * 1024 for p in folder.rglob("*.npy")):
                if not self.large_files:
                    raise ValueError("Install Git LFS for arrays above 1 MiB")
                (folder / ".gitattributes").write_text(
                    "*.npy filter=lfs diff=lfs merge=lfs -text\n"
                )
            (folder / "studio.json").write_bytes(json_bytes(document.meta))
            env = {
                "GIT_INDEX_FILE": str(Path(tmp, "index")),
                "GIT_WORK_TREE": str(stage),
            }
            self.git("read-tree", head, env=env)
            self.git("--work-tree=" + str(stage), "add", "-A", "--", relative, env=env)
            tree = self.git("write-tree", env=env).decode().strip()
            msg = f"{actor}: {summary.splitlines()[0][:160]}\n\nStudio-Request: {request_id}"
            revision = (
                self.git("commit-tree", tree, "-p", head, "-m", msg).decode().strip()
            )
            self.git("update-ref", "refs/heads/main", revision, head)
            return revision

    def receipt(self, model, request_id):
        return (
            self.git(
                "log",
                "-1",
                "--format=%H",
                "--fixed-strings",
                "--grep=Studio-Request: " + request_id,
                "HEAD",
                "--",
                f"models/{identifier(model)}",
            )
            .decode()
            .strip()
        )

    def create(self, document, actor, summary="Create model"):
        model, nonce = uuid4().hex, uuid4().hex
        revision = self.publish(
            model, None, document, document.certify(), actor, summary, nonce, nonce
        )
        return {"id": model, "revision": revision}
