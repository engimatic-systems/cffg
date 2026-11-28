"""Exact disk checkpoints for the fixed Gameworld evaluation actor.

Metadata identifies the source, stage, two seeds, action budget and the trace
prefix ending at this boundary. The caller supplies it independently on restore.
The simulator and its action log belong to the evaluation runner.
"""

from __future__ import annotations

from dataclasses import asdict
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
from cffg.experiments.axiom_planning import SearchSettings
from cffg.experiments.axiom_reduction_schedule import PERIOD, SAMPLE_COUNT, PAIR_COUNT


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


def _model(actor):
    return {
        "scene": asdict(actor.settings), "search": asdict(SearchSettings()), "horizon": 32,
        "model_version": "cffg.axiom.rgb.v1", "jax_enable_x64": bool(jax.config.x64_enabled),
        "frame": {"shape": [210, 160, 3], "dtype": "uint8", "channels": "RGB",
                  "features": "x/(width/2)-1,y/(height/2)-1,rgb/128-1"},
        "precision": actor.scene.positions.mean.dtype.name,
        "plan_key_impl": str(jax.random.key_impl(actor.key)),
        "reduction_key_impl": str(jax.random.key_impl(actor.reduction.key)),
        "reduction": {"period": PERIOD, "samples": SAMPLE_COUNT, "pairs": PAIR_COUNT},
        "reward": "native signed integer", "policy": "released open-loop reward plus count novelty",
    }


_FIXED = ("scene[0].identity.prior.", "scene[0].interaction.prior.",
          "scene[0].spatial.appearance_prior.", "scene[0].spatial.position_prior.")
_FIXED_NAMES = {"scene[0].absent_distance", "scene[0].spatial.position_prior", "scene[0].spatial.prior_counts"}
_LOW = ("scene[0].identity.posterior.carry.", "scene[0].interaction.posterior.carry.")


def _fixed(name):
    return name in _FIXED_NAMES or name.startswith(_FIXED)


def save_agent(path, actor, metadata, *, exact=True):
    """Store mutable state and bind regenerated prior values and settings to the archive."""
    if type(exact) is not bool:
        raise ValueError("exact must be a boolean")
    meta = _metadata(metadata)
    _check_actor(actor, meta)
    arrays = actor.state_dict()
    _check_clock(arrays, meta)
    fixed = {key: arrays.pop(key) for key in list(arrays) if _fixed(key)}
    if not exact:
        # The factory's reset priors have zero remainder, so each saved primary
        # count is already the compact lifecycle codec's rounded count.
        if any(np.any(value) for key, value in fixed.items() if ".carry." in key):
            raise ValueError("compact actor snapshots require the factory's zero-remainder reset priors")
        arrays = {key: value for key, value in arrays.items() if not key.startswith(_LOW)}
    header = {"format": "cffg.axiom.agent", "version": 2, "evaluation": meta,
              "model": _model(actor), "fixed_sha256": _digest({}, fixed),
              "mode": "exact" if exact else "compact"}
    header["sha256"] = _digest(header, arrays)
    write_artifact(Path(path), header, arrays)


def _expand_agent(actor, header, arrays):
    if header["mode"] not in ("exact", "compact"):
        raise ValueError("unknown actor continuation mode")
    if json.dumps(header["model"], sort_keys=True) != json.dumps(_model(actor), sort_keys=True):
        raise ValueError("checkpoint scene, planner, precision or random key configuration differs")
    fixed = {}
    for path, leaf in actor._leaves()[0]:
        key = "scene" + jax.tree_util.keystr(path)
        if _fixed(key):
            fixed[key] = np.asarray(leaf)
        elif header["mode"] == "compact" and key.startswith(_LOW):
            if key in arrays:
                raise ValueError("compact checkpoint contains a remainder")
            arrays[key] = np.zeros(leaf.shape, leaf.dtype)
    if arrays.keys() & fixed.keys() or _digest({}, fixed) != header["fixed_sha256"]:
        raise ValueError("checkpoint fixed inputs differ from the actor factory")
    arrays.update(fixed)


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
        common = {"format", "version", "evaluation", "sha256"}
        if not isinstance(header, dict) or type(header.get("version")) is not int:
            raise ValueError("invalid evaluation checkpoint header")
        legacy = header.get("format") == FORMAT and header["version"] == VERSION
        current = header.get("format") == "cffg.axiom.agent" and header["version"] == 2
        if (not (legacy or current)
                or set(header) != (common if legacy else common | {"mode", "model", "fixed_sha256"})):
            raise ValueError("unsupported evaluation checkpoint format or version")
        actual = _metadata(header["evaluation"])
        if actual != expected:
            difference = ", ".join(sorted(key for key in expected if expected[key] != actual[key]))
            raise ValueError(f"evaluation checkpoint identity mismatch for {difference}")
        arrays = {k: archive[k] for k in archive.files if k != "metadata"}
    digest = header.pop("sha256")
    if digest != _digest(header, arrays):
        raise ValueError("evaluation checkpoint content digest mismatch")
    if current:
        _expand_agent(actor, header, arrays)
    _check_clock(arrays, actual)
    actor.load_state_dict(arrays)
