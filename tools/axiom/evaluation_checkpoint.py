"""Exact disk checkpoints for the fixed Gameworld evaluation actor.

Metadata identifies the source, stage, two seeds, action budget and the trace
prefix ending at this boundary. The caller supplies it independently on restore.
The simulator and its action log belong to the evaluation runner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import jax
import numpy as np

from cffg.component_bank_device import to_host
from cffg.frame_regression import _valid_model
from cffg.mixed_scoring import _matrix_valid
from cffg.neural_artifacts import write_artifact


FORMAT = "cffg.axiom.evaluation"
VERSION = 1
_FIELDS = {"source_sha256", "stage", "game", "model_seed", "game_seed", "goal_steps", "tick", "trace_sha256"}


def _metadata(meta):
    if not isinstance(meta, dict) or set(meta) != _FIELDS:
        raise ValueError("incomplete or unknown evaluation checkpoint metadata")
    for key in ("stage", "game"):
        if not isinstance(meta[key], str) or not meta[key]:
            raise ValueError(f"invalid evaluation {key}")
    for key in ("model_seed", "game_seed", "goal_steps", "tick"):
        if type(meta[key]) is not int or meta[key] < 0:
            raise ValueError(f"invalid evaluation {key}")
    if meta["goal_steps"] == 0 or meta["tick"] > meta["goal_steps"]:
        raise ValueError("invalid evaluation action clock or budget")
    for key in ("source_sha256", "trace_sha256"):
        if not isinstance(meta[key], str) or not re.fullmatch("[0-9a-f]{64}", meta[key]):
            raise ValueError(f"invalid evaluation {key}")
    return dict(meta)


def _check_actor(actor, meta):
    if actor.game != meta["game"] or actor.seed != meta["model_seed"]:
        raise ValueError("actor factory game or model seed differs from checkpoint metadata")


def _check_clock(arrays, meta):
    # Actor.state_dict names the leaves of (scene, proposal, planning_key).
    tick = np.asarray(arrays.get("scene[0].tick"))
    if tick.shape != () or tick.dtype != np.int32 or tick.item() != meta["tick"]:
        raise ValueError("saved scene tick differs from the evaluation trace clock")


def _check_reduction_layout(actor, arrays):
    impl = np.asarray(arrays.get("reduction/key_impl"))
    key = np.asarray(arrays.get("reduction/key"))
    template = jax.random.key_data(actor.reduction.key)
    if (impl.shape != () or impl.item() != str(jax.random.key_impl(actor.reduction.key))
            or key.dtype != template.dtype or key.shape != template.shape):
        raise ValueError("incompatible reduction random key")
    if np.asarray(arrays.get("reduction/categories")).dtype != actor.reduction.records.categories.dtype:
        raise ValueError("incompatible reduction category dtype")


def validate_restored_scene(actor, scene, proposal):
    """Check fixed model inputs and staged learner and planner state before installation."""
    def fixed(s):
        return (s.absent_distance, s.identity.prior, s.interaction.prior,
                s.spatial.appearance_prior, s.spatial.prior_counts, s.spatial.position_prior,
                s.motion.affine[:4], s.motion.occupied[:4])

    if any(not np.array_equal(old, new) for old, new in
           zip(jax.tree.leaves(fixed(actor.scene)), jax.tree.leaves(fixed(scene)), strict=True)):
        raise ValueError("checkpoint changes a fixed scene prior or seed")
    for bank in (scene.identity, scene.interaction):
        to_host(bank)  # Reuse exact lifecycle, dormant-row and conjugate parameter validation.
    if (not bool(_valid_model(scene.spatial.appearance))
            or not bool(_matrix_valid(scene.positions.cov, strict=False))
            or not bool(_matrix_valid(scene.tracks.covariance, strict=False))
            or np.any(np.asarray(scene.spatial.counts) <= 0)):
        raise ValueError("invalid restored spatial belief")
    p, best = map(np.asarray, proposal)
    # Match the sampled planner's row-sum tolerance at the stored precision.
    tolerance = (8 + actor.actions.bit_length()) * np.finfo(p.dtype).eps
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(np.abs(p.sum(axis=1) - 1) > tolerance)
            or np.any(best < 0) or np.any(best >= actor.actions)):
        raise ValueError("invalid restored action proposal")


def _digest(meta, arrays):
    digest = hashlib.sha256()
    digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for name in sorted(arrays):
        a = np.asarray(arrays[name])
        if a.dtype.kind not in "bifuU":
            raise ValueError(f"unsupported actor checkpoint dtype for {name}")
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            raise ValueError(f"nonfinite actor checkpoint field {name}")
        descriptor = json.dumps([name, a.dtype.str, a.shape], separators=(",", ":")).encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(memoryview(np.ascontiguousarray(a)).cast("B") if a.size else b"")
    return digest.hexdigest()


def save(path, actor, metadata):
    """Atomically replace path once the complete actor and trace identity are bound."""
    meta = _metadata(metadata)
    _check_actor(actor, meta)
    arrays = actor.state_dict()
    _check_clock(arrays, meta)
    header = {"format": FORMAT, "version": VERSION, "evaluation": meta}
    header["sha256"] = _digest(header, arrays)
    write_artifact(Path(path), header, arrays)


def restore(path, actor, metadata):
    """Validate the envelope, then let the actor stage and install its complete state."""
    expected = _metadata(metadata)
    _check_actor(actor, expected)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or "metadata" not in archive.files:
            raise ValueError("duplicate or missing evaluation checkpoint fields")
        raw = archive["metadata"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("invalid evaluation checkpoint header")
        header = json.loads(raw.item())
        if (not isinstance(header, dict) or set(header) != {"format", "version", "evaluation", "sha256"}
                or header["format"] != FORMAT or type(header["version"]) is not int
                or header["version"] != VERSION):
            raise ValueError("unsupported evaluation checkpoint format or version")
        actual = _metadata(header["evaluation"])
        if actual != expected:
            difference = ", ".join(sorted(key for key in expected if expected[key] != actual[key]))
            raise ValueError(f"evaluation checkpoint identity mismatch for {difference}")
        arrays = {k: archive[k] for k in archive.files if k != "metadata"}
    digest = header.pop("sha256")
    if digest != _digest(header, arrays):
        raise ValueError("evaluation checkpoint content digest mismatch")
    _check_clock(arrays, actual)
    _check_reduction_layout(actor, arrays)
    actor.load_state_dict(arrays)
