"""Checkpoint and bundle format v1 for the deterministic categorical hierarchy."""

from dataclasses import replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import jax.numpy as jnp
import numpy as np

from cffg.bundle import load_bundle, save_bundle
from cffg.hierarchy_model import HierarchySpec


_ARRAYS = ("prior", "frontier", "cavity", "cost", "child_statistics", "parent_statistics")


def checkpoint(runtime):
    state = runtime._state
    data = {k: np.array(getattr(state, k)) for k in _ARRAYS}
    data.update(version=np.array(1), artifact=np.array(runtime.artifact_hash),
                ticks=np.array(state.ticks), periods=np.array(state.periods))
    goals = runtime._goals
    if goals is not None:
        data.update(goal_episodes=np.array(goals.episodes), goal_pending=np.array(goals.pending is not None))
        for i, name in enumerate(sorted(goals.priors)):
            data[f"goal_prior_{i}"] = goals.priors[name].copy()
            if goals.pending is not None:
                data[f"goal_pending_{i}"] = goals.pending[name].copy()
    return data


def _integer(data, key):
    item = np.asarray(data[key])
    if item.ndim or item.dtype.kind not in "iu" or int(item) < 0:
        raise ValueError(f"hierarchy checkpoint has invalid {key}")
    return int(item)


def _counts(data, key, shape):
    value = np.asarray(data[key])
    if (value.shape != shape or value.dtype.kind not in "fiu" or
            not np.isfinite(value).all() or np.any(value <= 0)):
        raise ValueError(f"hierarchy checkpoint has invalid goal concentrations for {key}")
    return value.copy()


def restore(runtime, data):
    if "version" not in data or _integer(data, "version") != 1:
        raise ValueError("unsupported or unversioned hierarchy checkpoint")
    required = set(_ARRAYS) | {"version", "artifact", "ticks", "periods"}
    if not required.issubset(data) or np.asarray(data["artifact"]).shape != () or str(data["artifact"]) != runtime.artifact_hash:
        raise ValueError("hierarchy checkpoint is incomplete or belongs to a different artifact")
    ticks, periods = _integer(data, "ticks"), _integer(data, "periods")
    if ticks // runtime.spec.thickness != periods:
        raise ValueError("hierarchy checkpoint has inconsistent child/parent clocks")
    arrays = {}
    for key in _ARRAYS:
        actual = np.asarray(data[key])
        target = getattr(runtime._state, key)
        if actual.shape != target.shape or actual.dtype != target.dtype:
            raise ValueError(f"hierarchy checkpoint {key} shape or dtype differs")
        log_message = key in ("prior", "frontier", "cavity")
        valid = np.isfinite(actual) | (log_message & np.isneginf(actual))
        if not valid.all():
            raise ValueError(f"hierarchy checkpoint has nonfinite {key}")
        arrays[key] = actual
    # The two clocks count exactly one normalized statistic each time they advance.
    for k, n in (("child_statistics", ticks), ("parent_statistics", periods)):
        stat = arrays[k]
        eps = np.finfo(stat.dtype).eps
        if np.any(stat < 0) or abs(stat.sum(dtype=np.float64)-n) > 32*eps*max(n,1)*max(1,np.sqrt(n)):
            raise ValueError(f"hierarchy checkpoint {k} disagrees with clock")
    prior, cond, cavity = (arrays[k].astype(np.float64) for k in ("prior", "frontier", "cavity"))
    eps = np.finfo(arrays["prior"].dtype).eps
    rows = np.logaddexp.reduce(cond, axis=1)
    if (abs(np.logaddexp.reduce(prior)) > 32*eps or not np.isfinite(np.logaddexp.reduce(prior+cavity))
            or np.any(np.abs(rows[np.isfinite(cavity)]) > 32*eps)):
        raise ValueError("hierarchy checkpoint has improper boundary message")
    # Every supported endpoint stores E[log q(path|r,x) - log p(path,data|r)].
    # Its independently stored endpoint likelihood must be the same free energy.
    log_mass = cond + cavity[:, None]
    supported = np.isfinite(log_mass)
    terms = arrays["cost"].astype(np.float64)
    mismatch = np.abs(terms.sum(axis=-1)[supported] + log_mass[supported])
    scale = np.maximum(1, np.abs(terms).sum(axis=-1)[supported])
    if np.any(mismatch > 128 * eps * scale):
        raise ValueError("hierarchy checkpoint path costs disagree with the saved cavity")
    if ticks % runtime.spec.thickness == 0 and np.any(cavity != 0):
        raise ValueError("hierarchy checkpoint contains evidence from a completed period")
    goals = runtime._goals
    if goals is not None:
        required.update(("goal_episodes", "goal_pending"))
        if not required.issubset(data):
            raise ValueError("hierarchy checkpoint is missing goal rollover state")
        episodes = _integer(data, "goal_episodes")
        staged = np.asarray(data["goal_pending"])
        if staged.ndim or staged.dtype != bool:
            raise ValueError("hierarchy checkpoint has invalid goal rollover flag")
        priors, pending = {}, {}
        for i, name in enumerate(sorted(goals.priors)):
            key = f"goal_prior_{i}"
            required.add(key)
            if key not in data:
                raise ValueError("missing committed goal prior")
            priors[name] = _counts(data, key, goals.priors[name].shape)
            if staged:
                key = f"goal_pending_{i}"
                required.add(key)
                if key not in data:
                    raise ValueError("missing staged goal posterior")
                pending[name] = _counts(data, key, goals.priors[name].shape)
    if required != set(data):
        raise ValueError("hierarchy checkpoint has unknown or missing fields")
    state = replace(runtime._state, ticks=ticks, periods=periods, **{k: jnp.asarray(v) for k,v in arrays.items()})
    if goals is not None:
        goals.priors, goals.pending, goals.episodes = priors, pending if staged else None, episodes
    runtime._state = state


def save(runtime, path):
    levels = {"parent": (runtime.parent_template, runtime._arrays[0]), "child": (runtime.child_template, runtime._arrays[1])}
    if runtime._goals is not None:
        levels["goals"] = runtime._goals.template, runtime._goals.arrays
    members = {"hierarchy.json": runtime.spec.to_json().encode()}
    buffer = io.BytesIO()
    np.savez(buffer, **checkpoint(runtime))
    members["checkpoint.npz"] = buffer.getvalue()
    with tempfile.TemporaryDirectory() as tmp:
        for name, (template, arrays) in levels.items():
            file = Path(tmp) / (name+".cffgz")
            save_bundle(file, template, arrays)
            members[file.name] = file.read_bytes()
    hashes = {k: sha256(v).hexdigest() for k,v in members.items()}
    members["manifest.json"] = json.dumps(dict(version=1, sha256=hashes, artifact=runtime.artifact_hash)).encode()
    dest = Path(path)
    sibling = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=dest.name+".", delete=False) as f:
            sibling = Path(f.name)
            with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as archive:
                for k, v in members.items():
                    archive.writestr(k, v)
            f.flush()
            os.fsync(f.fileno())
        # Inspect the exact outgoing bytes. A caller may have edited an exposed template.
        load(type(runtime), sibling)
        os.replace(sibling, dest)
    finally:
        if sibling is not None:
            sibling.unlink(missing_ok=True)


def load(cls, path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {"manifest.json","hierarchy.json","parent.cffgz","child.cffgz","checkpoint.npz"}
        if names - {"goals.cffgz"} != required or len(names) != len(archive.namelist()):
            raise ValueError("invalid hierarchy bundle member inventory")
        manifest = json.loads(archive.read("manifest.json"))
        if type(manifest.get("version")) is not int or manifest["version"] != 1:
            raise ValueError("unsupported hierarchy bundle version")
        raw = {k: archive.read(k) for k in names - {"manifest.json"}}
    if manifest.get("sha256") != {k: sha256(v).hexdigest() for k,v in raw.items()}:
        raise ValueError("hierarchy bundle digest mismatch")
    def level(name):
        model = load_bundle(io.BytesIO(raw[name+".cffgz"]))
        return model.template, model.arrays
    spec = HierarchySpec.from_json(raw["hierarchy.json"])
    runtime = cls(spec, *level("parent"), *level("child"), goal_window=level("goals") if "goals.cffgz" in raw else None)
    if runtime.artifact_hash != manifest["artifact"]:
        raise ValueError("hierarchy bundle declaration and tables differ from the manifest")
    with np.load(io.BytesIO(raw["checkpoint.npz"]), allow_pickle=False) as saved:
        restore(runtime, saved)
    return runtime
