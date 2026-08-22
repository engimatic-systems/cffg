"""Executable/statistic identity for flat learnable checkpoints.

The generic ``v2:`` array keys identify names and hyperparameters, but they do
not identify the sufficient statistic that produced a posterior. D8, D9, and
D10 use the same template and parameter shapes, so learned recurrence state
carries this compact manifest in addition to its numeric arrays.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


CHECKPOINT_MANIFEST_KEY = "__cffg_learnables_recurrence_manifest__"
RECURRENCE_CHECKPOINT_VERSION = "1.0.0"
RECURRENCE_STATISTIC_IDENTITY = (
    "cffg.recurrence.d9-structured-pair/1.0.0"
)
RECURRENCE_PRIOR_IDENTITY = "cffg.recurrence.unlearned-prior/1.0.0"
D10_CHECKPOINT_VERSION = "2.0.0"
D10_EXECUTABLE_IDENTITY = "cffg.structured-hybrid.d10/1.0.0"
D10_OBJECTIVE_IDENTITY = "cffg.structured-hybrid.d10-jj-bfe/1.0.0"
D10_COORDINATE_IDENTITY = (
    "cffg.structured-hybrid.singleton-pair-gaussian-lag/1.0.0"
)
D10_INITIAL_STATISTIC_IDENTITY = (
    "cffg.structured-hybrid.d10-initial-singleton/1.0.0"
)
D10_DYNAMICS_STATISTIC_IDENTITY = (
    "cffg.structured-hybrid.d10-component-mniw/1.0.0"
)
D10_RECURRENCE_STATISTIC_IDENTITY = (
    "cffg.structured-hybrid.d10-pg-pair/1.0.0"
)
D10_EMISSION_STATISTIC_IDENTITY = (
    "cffg.structured-hybrid.d10-emission-mniw/1.0.0"
)

_D10_STATISTICS = {
    ("dirichlet", "categorical"): D10_INITIAL_STATISTIC_IDENTITY,
    (
        "component_mniw",
        "gaussian_mixture_linear",
    ): D10_DYNAMICS_STATISTIC_IDENTITY,
    ("pg_weights", "recurrence_link"): D10_RECURRENCE_STATISTIC_IDENTITY,
    ("mniw", "gaussian_linear"): D10_EMISSION_STATISTIC_IDENTITY,
}


def _recurrence_plans(plans: Mapping[str, object]) -> list[object]:
    return sorted(
        (
            plan
            for plan in plans.values()
            if plan.family == "pg_weights"
            and plan.ftype == "recurrence_link"
        ),
        key=lambda plan: plan.name,
    )


def _descriptor(
    plan,
    params: Mapping[str, np.ndarray],
    *,
    statistic_identity: str | None = None,
) -> dict:
    targets = sorted(
        {
            (
                target.blob,
                target.role,
                target.form,
                target.layout,
                target.transform,
            )
            for target in plan.targets
        }
    )
    descriptor = {
        "name": plan.name,
        "family": plan.family,
        "factor_type": plan.ftype,
        "sources": list(plan.sources),
        "parameter_blobs": [
            {"name": name, "shape": list(shape)}
            for name, shape in sorted(plan.shapes.items())
        ],
        "targets": [
            {
                "blob": blob,
                "role": role,
                "form": form,
                "layout": layout,
                "transform": transform,
            }
            for blob, role, form, layout, transform in targets
        ],
        "hyperparameters": [
            {
                "name": key,
                "dtype": np.asarray(value).dtype.str,
                "shape": list(np.shape(value)),
                "sha256": array_sha256(value),
            }
            for key, value in sorted(params.items())
        ],
    }
    if statistic_identity is not None:
        descriptor["statistic_identity"] = statistic_identity
    return descriptor


def d10_checkpoint_identity(engine) -> dict | None:
    """Describe the executable D10 component, or return ``None``."""
    plans = tuple(engine.arena.structured_hybrid_plans)
    if not plans:
        return None
    if len(plans) != 1:
        raise ValueError("D10 checkpoint identity requires one component")
    plan = plans[0]
    topology = plan.topology
    return {
        "executable_identity": D10_EXECUTABLE_IDENTITY,
        "objective_identity": D10_OBJECTIVE_IDENTITY,
        "coordinate_identity": D10_COORDINATE_IDENTITY,
        "component": {
            "name": plan.name,
            "cardinality": topology.cardinality,
            "dimension": topology.dimension,
            "initial_regime_member": topology.initial_regime_member,
            "initial_state_member": topology.initial_state_member,
            "recurrence_members": list(topology.recurrence_members),
            "dynamics_members": list(topology.dynamics_members),
            "emission_members": list(topology.emission_members),
            "regime_edges": [list(row) for row in topology.regime_edges],
            "state_edges": [list(row) for row in topology.state_edges],
            "observation_edges": list(topology.observation_edges),
            "observation_rows": np.asarray(plan.observation_rows).tolist(),
        },
    }


def manifest_document(
    plans: Mapping[str, object],
    params: Mapping[str, Mapping[str, np.ndarray]],
    identity: Mapping[str, object] | None = None,
) -> dict | None:
    """Return the canonical D9 or D10 learning manifest."""
    if identity is not None:
        selected = sorted(plans.values(), key=lambda plan: plan.name)
        return {
            "checkpoint_version": D10_CHECKPOINT_VERSION,
            **dict(identity),
            "learnables": [
                _descriptor(
                    plan,
                    params[plan.name],
                    statistic_identity=_D10_STATISTICS[(plan.family, plan.ftype)],
                )
                for plan in selected
            ],
        }
    recurrence = _recurrence_plans(plans)
    if not recurrence:
        return None
    return {
        "checkpoint_version": RECURRENCE_CHECKPOINT_VERSION,
        "statistic_identity": RECURRENCE_STATISTIC_IDENTITY,
        "learnables": [
            _descriptor(plan, params[plan.name]) for plan in recurrence
        ],
    }


def encode_manifest(
    plans: Mapping[str, object],
    params: Mapping[str, Mapping[str, np.ndarray]],
    identity: Mapping[str, object] | None = None,
) -> np.ndarray | None:
    """Encode the learning manifest as one scalar Unicode npz value."""
    document = manifest_document(plans, params, identity)
    if document is None:
        return None
    return np.asarray(
        json.dumps(document, sort_keys=True, separators=(",", ":"))
    )


def _decode_scalar(value) -> dict:
    encoded = np.asarray(value)
    if encoded.shape != () or encoded.dtype.kind not in "US":
        raise ValueError(
            "recurrence checkpoint manifest must be a scalar Unicode string"
        )
    raw = encoded.item()
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "recurrence checkpoint manifest is not valid UTF-8"
            ) from exc
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            "recurrence checkpoint manifest is not valid JSON"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError("recurrence checkpoint manifest must be a JSON object")
    return document


def array_sha256(value) -> str:
    """Digest one array's exact dtype-specific C-order bytes."""
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _ownership_and_payload(learnables) -> tuple[list[dict], dict]:
    if not isinstance(learnables, list):
        raise ValueError("recurrence checkpoint learnables must be a list")
    ownership = []
    digests = {}
    for descriptor in learnables:
        if not isinstance(descriptor, dict):
            raise ValueError(
                "recurrence checkpoint learnable descriptor must be an object"
            )
        copied = dict(descriptor)
        hyperparameters = copied.get("hyperparameters")
        if not isinstance(hyperparameters, list):
            raise ValueError(
                "recurrence checkpoint hyperparameters must be a list"
            )
        copied_hyperparameters = []
        for row in hyperparameters:
            if not isinstance(row, dict) or set(row) != {
                "name",
                "dtype",
                "shape",
                "sha256",
            }:
                raise ValueError(
                    "recurrence checkpoint hyperparameter fields do not "
                    "match the current checkpoint schema"
                )
            digest = row["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(
                    "recurrence checkpoint hyperparameter digest is invalid"
                )
            try:
                dtype = np.dtype(row["dtype"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "recurrence checkpoint hyperparameter dtype is invalid"
                ) from exc
            if dtype.kind != "f" or dtype.str != row["dtype"]:
                raise ValueError(
                    "recurrence checkpoint hyperparameter dtype must be a "
                    "canonical real floating dtype"
                )
            digests[(copied.get("name"), row["name"])] = (
                dtype,
                digest,
            )
            copied_hyperparameters.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"dtype", "sha256"}
                }
            )
        copied["hyperparameters"] = copied_hyperparameters
        ownership.append(copied)
    return ownership, digests


def validate_manifest(
    value,
    plans: Mapping[str, object],
    params: Mapping[str, Mapping[str, np.ndarray]],
    identity: Mapping[str, object] | None = None,
) -> dict[tuple[str, str], tuple[np.dtype, str]]:
    """Validate identity/ownership and return required array digests."""
    expected = manifest_document(plans, params, identity)
    if expected is None:
        raise ValueError(
            "recurrence checkpoint manifest cannot restore a graph without "
            "recurrence learnables"
        )
    document = _decode_scalar(value)
    allowed = (
        {
            "checkpoint_version",
            "executable_identity",
            "objective_identity",
            "coordinate_identity",
            "component",
            "learnables",
        }
        if identity is not None
        else {"checkpoint_version", "statistic_identity", "learnables"}
    )
    expected_version = (
        D10_CHECKPOINT_VERSION
        if identity is not None
        else RECURRENCE_CHECKPOINT_VERSION
    )
    if document.get("checkpoint_version") != expected_version:
        raise ValueError(
            "unknown learning checkpoint version "
            f"{document.get('checkpoint_version')!r}"
        )
    if set(document) != allowed:
        raise ValueError(
            "recurrence checkpoint manifest fields do not match the current "
            "checkpoint schema"
        )
    if (
        identity is None
        and document["statistic_identity"] != RECURRENCE_STATISTIC_IDENTITY
    ):
        raise ValueError(
            "recurrence checkpoint statistic identity is not the current "
            "D9 structured-pair identity"
        )
    if identity is not None:
        recorded_identity = {
            key: document[key]
            for key in (
                "executable_identity",
                "objective_identity",
                "coordinate_identity",
                "component",
            )
        }
        if recorded_identity != dict(identity):
            raise ValueError(
                "D10 checkpoint executable, objective, coordinate, or topology "
                "identity does not match this engine"
            )
    ownership, payload = _ownership_and_payload(document["learnables"])
    expected_ownership, _ = _ownership_and_payload(expected["learnables"])
    if ownership != expected_ownership:
        raise ValueError(
            "recurrence checkpoint factor/source ownership or shapes "
            "do not match this engine"
        )
    return payload


def recurrence_names(plans: Mapping[str, object]) -> set[str]:
    """Names whose posterior statistics use the D9 pair authority."""
    return {plan.name for plan in _recurrence_plans(plans)}


@dataclass(frozen=True)
class RecurrenceCheckpointContext:
    """Validated manifest state used by the generic flat-key decoder."""

    names: frozenset[str]
    manifest_present: bool
    payload: Mapping[tuple[str, str], tuple[np.dtype, str]]
    d10: bool = False

    def require_identity(self, name: str) -> None:
        if name in self.names and not self.manifest_present:
            raise ValueError(
                "legacy learned checkpoint has no statistic identity; it may "
                "contain D8/D9 state and cannot be resumed under current "
                f"{'D10' if self.d10 else 'D9'} semantics"
            )

    def restore_value(self, name: str, key: str, value) -> np.ndarray:
        if name not in self.names:
            return np.asarray(value, dtype=float).copy()
        dtype, digest = self.payload[(name, key)]
        array = np.asarray(value)
        if array.dtype != dtype:
            raise ValueError(
                f"learnable {name!r}: recurrence checkpoint dtype "
                f"{array.dtype} != {dtype} recorded in its manifest"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"learnable {name!r}: recurrence checkpoint values must be "
                "finite"
            )
        if array_sha256(array) != digest:
            raise ValueError(
                f"learnable {name!r}: recurrence checkpoint value digest "
                "does not match its manifest"
            )
        return array.copy()

    def require_complete(self, seen: set[tuple[str, str]]) -> None:
        if not self.manifest_present:
            return
        present = {(name, key) for name, key in seen if name in self.names}
        required = set(self.payload)
        if present != required:
            raise KeyError(
                "structured recurrence checkpoint is incomplete; missing "
                f"hyperparameters {sorted(required - present)}"
            )


def checkpoint_context(
    data: Mapping[str, np.ndarray],
    plans: Mapping[str, object],
    params: Mapping[str, Mapping[str, np.ndarray]],
    identity: Mapping[str, object] | None = None,
) -> RecurrenceCheckpointContext:
    """Validate optional metadata before the generic decoder touches values."""
    names = frozenset(plans if identity is not None else recurrence_names(plans))
    manifest = data.get(CHECKPOINT_MANIFEST_KEY)
    if identity is not None and manifest is None:
        raise ValueError(
            "D10 learned checkpoint has no executable/statistic identity"
        )
    payload = (
        validate_manifest(manifest, plans, params, identity)
        if manifest is not None
        else {}
    )
    return RecurrenceCheckpointContext(
        names, manifest is not None, payload, identity is not None
    )


__all__ = [
    "CHECKPOINT_MANIFEST_KEY",
    "D10_CHECKPOINT_VERSION",
    "D10_COORDINATE_IDENTITY",
    "D10_DYNAMICS_STATISTIC_IDENTITY",
    "D10_EMISSION_STATISTIC_IDENTITY",
    "D10_EXECUTABLE_IDENTITY",
    "D10_INITIAL_STATISTIC_IDENTITY",
    "D10_OBJECTIVE_IDENTITY",
    "D10_RECURRENCE_STATISTIC_IDENTITY",
    "RECURRENCE_CHECKPOINT_VERSION",
    "RecurrenceCheckpointContext",
    "RECURRENCE_PRIOR_IDENTITY",
    "RECURRENCE_STATISTIC_IDENTITY",
    "array_sha256",
    "checkpoint_context",
    "d10_checkpoint_identity",
    "encode_manifest",
    "manifest_document",
    "recurrence_names",
    "validate_manifest",
]
