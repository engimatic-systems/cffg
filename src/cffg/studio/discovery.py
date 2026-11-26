"""Index model files in selected directories and import complete model contents."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from threading import Event, Lock, Thread
from time import time
from zipfile import ZipFile

from cffg.core.plates import expand_plates
from cffg.core.serialize import SCHEMA_VERSION, from_dict
from cffg.resolution.sources import _load_blob, resolve_arrays
from .jobs import write
from .models import Document, MAX_ARCHIVE_BYTES
from cffg.bundle import BUNDLE_SCHEMA_VERSION


def file_digest(path):
    import hashlib

    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def bundle_header(file):
    with ZipFile(file) as archive:
        model = archive.getinfo("model.json")
        manifest = archive.getinfo("manifest.json")
        if max(model.file_size, manifest.file_size) > 8 * 1024 * 1024:
            raise ValueError("Model header exceeds index size limit")
        return json.loads(archive.read(model)), json.loads(archive.read(manifest)).get(
            "schema_version"
        )


class Discovery:
    def __init__(self, home, roots):
        self.config = home / "roots.json"
        if not self.config.exists():
            write(self.config, [str(Path(p).resolve()) for p in roots])
        self.db = home / "catalog.sqlite"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, stamp INTEGER, size INTEGER, name TEXT, terms TEXT, scan REAL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS formats (path TEXT PRIMARY KEY, format TEXT, diagnostic TEXT)"
            )
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(path,name,terms)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS files_name ON files(name)")
        self.lock = Lock()
        self.queue = Lock()
        self.again = False
        self.stop = Event()
        self.error = None
        self.scan()

    def connect(self):
        return sqlite3.connect(self.db, timeout=10)

    def roots(self):
        return [Path(p) for p in json.loads(self.config.read_text())]

    def configure(self, roots):
        paths = [Path(p).expanduser().resolve(strict=True) for p in roots]
        if not 1 <= len(paths) <= 64 or any(not p.is_dir() for p in paths):
            raise ValueError("Choose between 1 and 64 readable directory roots")
        write(self.config, [str(p) for p in paths])
        self.scan()

    def allowed(self, path):
        path = Path(path).expanduser().resolve(strict=True)
        if not any(path.is_relative_to(root) for root in self.roots()):
            raise ValueError("File lies outside enabled search roots")
        return path

    def scan(self):
        with self.queue:
            if self.lock.acquire(blocking=False):
                Thread(target=self._drain, daemon=True).start()
            else:
                self.again = True

    def _drain(self):
        while True:
            try:
                self._scan()
            except Exception as exc:
                self.error = str(exc)
            with self.queue:
                if not self.again or self.stop.is_set():
                    self.again = False
                    self.lock.release()
                    return
                self.again = False

    def _scan(self):
        skip = {
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            "target",
            ".cache",
            "build",
            "dist",
            ".tox",
        }
        try:
            started = time()
            with self.connect() as db:
                for root in self.roots():
                    for directory, dirs, files in os.walk(root):
                        if self.stop.is_set():
                            return
                        dirs[:] = [
                            d
                            for d in dirs
                            if d not in skip
                            and not d.startswith(".")
                            and not Path(directory, d).is_symlink()
                        ]
                        for fname in files:
                            p = Path(directory, fname)
                            if (
                                p.suffix.lower()
                                not in (".cffgz", ".cffgstudio", ".cffg", ".json")
                                or p.is_symlink()
                            ):
                                continue
                            try:
                                st = p.stat()
                                if st.st_size > MAX_ARCHIVE_BYTES:
                                    continue
                                stamp = f"2:{st.st_mtime_ns}:{st.st_ctime_ns}"
                                old = db.execute(
                                    "SELECT stamp,size FROM files WHERE path=?",
                                    (str(p),),
                                ).fetchone()
                                if (
                                    old == (stamp, st.st_size)
                                    and db.execute(
                                        "SELECT 1 FROM formats WHERE path=?", (str(p),)
                                    ).fetchone()
                                ):
                                    db.execute(
                                        "UPDATE files SET scan=? WHERE path=?",
                                        (started, str(p)),
                                    )
                                    continue
                                fmt = "Template JSON"
                                diagnostic = ""
                                if p.suffix.lower() in (".cffgstudio", ".cffgz"):
                                    if p.suffix.lower() == ".cffgstudio":
                                        fmt = "Studio project"
                                        with ZipFile(p) as project:
                                            nested = project.getinfo("model.cffgz")
                                            if nested.file_size > MAX_ARCHIVE_BYTES:
                                                continue
                                            with project.open(nested) as file:
                                                obj, version = bundle_header(file)
                                    else:
                                        obj, version = bundle_header(p)
                                        fmt = f"Bundle v{version}"
                                    if version != BUNDLE_SCHEMA_VERSION:
                                        diagnostic = f"Requires bundle schema {BUNDLE_SCHEMA_VERSION}; found {version}"
                                else:
                                    if st.st_size > 8 * 1024 * 1024:
                                        continue
                                    obj = json.loads(p.read_bytes())
                                    if (
                                        isinstance(obj, dict)
                                        and obj.get("schema_version") != SCHEMA_VERSION
                                    ):
                                        diagnostic = f"Requires template schema {SCHEMA_VERSION}; found {obj.get('schema_version')}"
                                if (
                                    not isinstance(obj, dict)
                                    or not {
                                        "name",
                                        "schema_version",
                                        "variables",
                                        "factors",
                                    }
                                    <= obj.keys()
                                ):
                                    continue
                                name = str(obj["name"])
                                terms = " ".join(
                                    sorted({f["type"] for f in obj["factors"].values()})
                                )
                                db.execute(
                                    "INSERT OR REPLACE INTO formats VALUES (?,?,?)",
                                    (str(p), fmt, diagnostic),
                                )
                                db.execute(
                                    "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?)",
                                    (
                                        str(p),
                                        stamp,
                                        st.st_size,
                                        name,
                                        terms,
                                        started,
                                    ),
                                )
                            except (
                                OSError,
                                KeyError,
                                ValueError,
                                TypeError,
                                AttributeError,
                                RuntimeError,
                            ):
                                continue
                            except Exception:
                                continue
                        db.commit()
                db.execute("DELETE FROM files WHERE scan<?", (started,))
                db.execute(
                    "DELETE FROM formats WHERE path NOT IN (SELECT path FROM files)"
                )
                db.execute("DELETE FROM search")
                db.execute("INSERT INTO search SELECT path,name,terms FROM files")
            self.error = None
        except Exception as e:
            self.error = str(e)

    def find(self, q):
        prefixes = [str(root).rstrip("/") + "/" for root in self.roots()]
        if not prefixes:
            return []
        tokens = q.strip().split()[:12]
        table = "search" if tokens else "files"
        root_filter = " OR ".join(f"instr({table}.path, ?) = 1" for _ in prefixes)
        clause = f"({root_filter})"
        params = prefixes
        order = "name"
        if tokens:
            words = " AND ".join('"' + s.replace('"', '""') + '"*' for s in tokens)
            clause = f"search MATCH ? AND {clause}"
            params = [words, *prefixes]
            order = "rank"
        with self.connect() as db:
            rows = db.execute(
                f"SELECT {table}.path,name,terms,format,diagnostic FROM {table} "
                f"LEFT JOIN formats ON formats.path={table}.path WHERE {clause} "
                f"ORDER BY {order} LIMIT 120",
                params,
            ).fetchall()
        return [
            {"path": p, "name": n, "families": f, "format": fmt, "diagnostic": diag}
            for p, n, f, fmt, diag in rows
        ]

    def load(self, path):
        path = self.allowed(path)
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("File exceeds import size limit")
        raw = path.read_bytes()
        if path.suffix.lower() in (".cffgz", ".cffgstudio"):
            d = Document.exchange(raw)
        elif path.suffix.lower() in (".json", ".cffg"):
            t = from_dict(json.loads(raw))
            expanded = expand_plates(t)
            arrays = {}
            for n, b in expanded.blobs.items():
                ref = (path.parent / b.path).resolve()
                if ref.exists():
                    self.allowed(ref)
                    if (
                        ref.suffix not in (".npy", ".safetensors")
                        or ref.stat().st_size > MAX_ARCHIVE_BYTES
                    ):
                        raise ValueError(f"Unsupported blob file for {n}")
                    arrays[n] = _load_blob(ref, n)
            resolved = resolve_arrays(expanded, arrays, None)
            arrays = {**resolved, **arrays}
            if t.learnables:
                raise ValueError(
                    "Import a complete .cffgz with priors and posteriors for this learning model"
                )
            d = Document(t, arrays)
        else:
            raise ValueError("Choose a CFFG bundle or template JSON")
        d.meta["source"] = {"path": str(path), "sha256": sha256(raw).hexdigest()}
        return d
