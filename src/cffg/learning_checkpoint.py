"""Recurrence-specific identity for flat learnable checkpoints.

The generic ``v2:`` array keys identify names and hyperparameters, but they do
not identify the sufficient statistic that produced a posterior. D8 and D9
use the same template, blob shapes, and hyperparameter shapes, so recurrence
checkpoints carry this compact manifest in addition to their numeric arrays.
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


def _descriptor(plan, params: Mapping[str, np.ndarray]) -> dict:
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
    return {
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


def manifest_document(
    plans: Mapping[str, object],
    params: Mapping[str, Mapping[str, np.ndarray]],
) -> dict | None:
    """Return the canonical D9 manifest, or ``None`` without recurrence."""
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
) -> np.ndarray | None:
    """Encode the recurrence manifest as one scalar Unicode npz value."""
    document = manifest_document(plans, params)
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
) -> dict[tuple[str, str], tuple[np.dtype, str]]:
    """Validate identity/ownership and return required array digests."""
    expected = manifest_document(plans, params)
    if expected is None:
        raise ValueError(
            "recurrence checkpoint manifest cannot restore a graph without "
            "recurrence learnables"
        )
    document = _decode_scalar(value)
    allowed = {"checkpoint_version", "statistic_identity", "learnables"}
    if set(document) != allowed:
        raise ValueError(
            "recurrence checkpoint manifest fields do not match the current "
            "checkpoint schema"
        )
    if document["checkpoint_version"] != RECURRENCE_CHECKPOINT_VERSION:
        raise ValueError(
            "unknown recurrence checkpoint version "
            f"{document['checkpoint_version']!r}"
        )
    if document["statistic_identity"] != RECURRENCE_STATISTIC_IDENTITY:
        raise ValueError(
            "recurrence checkpoint statistic identity is not the current "
            "D9 structured-pair identity"
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

    def require_identity(self, name: str) -> None:
        if name in self.names and not self.manifest_present:
            raise ValueError(
                "legacy recurrence checkpoint has no statistic identity; it "
                "may contain D8 singleton-product state and cannot be resumed "
                "as D9"
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
) -> RecurrenceCheckpointContext:
    """Validate optional metadata before the generic decoder touches values."""
    names = frozenset(recurrence_names(plans))
    manifest = data.get(CHECKPOINT_MANIFEST_KEY)
    payload = (
        validate_manifest(manifest, plans, params)
        if manifest is not None
        else {}
    )
    return RecurrenceCheckpointContext(names, manifest is not None, payload)


__all__ = [
    "CHECKPOINT_MANIFEST_KEY",
    "RECURRENCE_CHECKPOINT_VERSION",
    "RecurrenceCheckpointContext",
    "RECURRENCE_PRIOR_IDENTITY",
    "RECURRENCE_STATISTIC_IDENTITY",
    "array_sha256",
    "checkpoint_context",
    "encode_manifest",
    "manifest_document",
    "recurrence_names",
    "validate_manifest",
]
