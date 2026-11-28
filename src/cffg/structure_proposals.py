"""Replay host-side structure proposals using explicit, content-identified inputs."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np

from cffg.array_identity import array_descriptor, json_digest
from cffg.core import Template, expand_plates, validate
from cffg.core.compose import compose
from cffg.core.edits import (
    RebindBlobPath,
    apply_edits,
    bundle_from_json,
    bundle_to_json,
)
from cffg.core.serialize import structure_hash


class ProposalError(ValueError):
    """A proposal could not be reconstructed from its declared artifacts."""


def _id(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ProposalError(f"{field} must be a nonempty string")


def edit_step(actions, *, parameters=None):
    """Apply one atomic edit bundle; map changed blob names to numeric input IDs."""
    return {
        "op": "edit",
        "actions": json.loads(bundle_to_json(actions)),
        "parameters": dict(parameters or {}),
    }


def compose_step(fragment_id, bonds, *, name, scope_a="a", scope_b="b"):
    """Attach a retained fragment using the usual compose names and port bindings."""
    return {
        "op": "compose",
        "fragment": fragment_id,
        "bonds": list(map(list, bonds.items())),
        "name": name,
        "scope_a": scope_a,
        "scope_b": scope_b,
    }


def _array(value):
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf" or not np.isfinite(arr).all():
        raise ProposalError("expected finite real numeric blob values")
    return arr


def _check_blobs(expected, arrays):
    if missing := set(expected) - set(arrays):
        raise ProposalError(f"missing numeric inputs for blobs {sorted(missing)}")
    if extra := set(arrays) - set(expected):
        raise ProposalError(f"unexpected numeric inputs for blobs {sorted(extra)}")


def model_identity(template, arrays):
    """Identify a model structure and its explicitly resolved named arrays."""
    validate(template)
    _check_blobs(expand_plates(template).blobs, arrays)
    descriptors = {k: array_descriptor(_array(v)) for k, v in arrays.items()}
    return {
        "structure_sha256": structure_hash(template),
        "arrays_sha256": json_digest(descriptors),
    }


def _lookup(mapping, key, kind):
    try:
        return mapping[key]
    except KeyError as error:
        raise ProposalError(f"missing {kind} {key!r}") from error


def _edited(template, arrays, step, parameter):
    if set(step) != {"op", "actions", "parameters"}:
        raise ProposalError("invalid edit fields")
    actions = bundle_from_json(json.dumps(step["actions"]))
    result = apply_edits(template, actions)
    declared = expand_plates(result).blobs
    updated = {k: v for k, v in arrays.items() if k in declared}
    for action in actions:
        if isinstance(action, RebindBlobPath) and action.name not in step["parameters"]:
            raise ProposalError(
                f"rebound blob {action.name!r} needs an explicit numeric input"
            )
    for blob, key in step["parameters"].items():
        updated[blob] = parameter(key)
    _check_blobs(declared, updated)
    return result, updated


def _composed(template, arrays, step, model):
    if set(step) != {"op", "fragment", "bonds", "name", "scope_a", "scope_b"}:
        raise ProposalError("invalid composition fields")
    fragment, fragment_arrays = model(step["fragment"])
    a, b = step["scope_a"], step["scope_b"]
    bonds = dict(step["bonds"])
    if len(bonds) != len(step["bonds"]):
        raise ProposalError("a composition names the same source port twice")
    result = compose(template, fragment, bonds, name=step["name"], scope_a=a, scope_b=b)
    scoped = {
        f"{scope}/{key}": value
        for scope, operand in ((a, arrays), (b, fragment_arrays))
        for key, value in operand.items()
    }
    return result, scoped


def _apply(source_id, steps, models, inputs, *, expected=None):
    identities, parameters = {}, {}
    snapshots, replacements = {}, {}

    def check(group, key, actual):
        if expected is not None and expected[group].get(key) != actual:
            raise ProposalError(f"{group} content identity mismatch for {key!r}")

    def model(key):
        _id(key, "model")
        if key not in snapshots:
            template, arrays = _lookup(models, key, "model")
            template = deepcopy(template)
            arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
            try:
                identities[key] = model_identity(template, arrays)
            except ValueError as error:
                raise ProposalError(f"model {key!r}: {error}") from error
            check("models", key, identities[key])
            snapshots[key] = template, arrays
        return snapshots[key]

    def parameter(key):
        _id(key, "numeric input")
        if key not in replacements:
            value = _array(_lookup(inputs, key, "numeric input")).copy()
            parameters[key] = array_descriptor(value)
            check("inputs", key, parameters[key])
            replacements[key] = value
        return replacements[key]

    template, arrays = model(source_id)
    for i, step in enumerate(steps):
        try:
            if step["op"] == "edit":
                template, arrays = _edited(template, arrays, step, parameter)
            elif step["op"] == "compose":
                template, arrays = _composed(template, arrays, step, model)
            else:
                raise ProposalError(f"unknown operation {step['op']!r}")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ProposalError(f"proposal step {i}: {error}") from error
    return template, arrays, identities, parameters, model_identity(template, arrays)


def record_proposal(
    *, source_id, parent_id, proposal_source, steps, models, inputs=None
):
    """Make a JSON record for any producer. Models map IDs to (Template, resolved arrays).

    The parent candidate ID is optional for a root proposal. Replacements and new
    blobs refer to entries in inputs by ID, regardless of their filesystem paths.
    """
    if parent_id is not None:
        _id(parent_id, "parent_id")
    _id(proposal_source, "proposal_source")
    steps = deepcopy(list(steps))
    _, _, identities, parameters, result = _apply(
        source_id, steps, models, inputs or {}
    )
    record = {
        "version": 1,
        "source_id": source_id,
        "parent_id": parent_id,
        "proposal_source": proposal_source,
        "steps": steps,
        "models": identities,
        "inputs": parameters,
        "result": result,
    }
    return {**record, "candidate_id": json_digest(record)}


def replay_proposal(
    record, models, *, inputs=None, expected_id=None
) -> tuple[Template, dict[str, np.ndarray]]:
    """Verify a saved record and reconstruct independent candidate inputs for fitting or scoring."""
    data = dict(record)
    candidate_id = data.pop("candidate_id", None)
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ProposalError("unsupported proposal version")
    if json_digest(data) != candidate_id or (
        expected_id is not None and expected_id != candidate_id
    ):
        raise ProposalError("candidate ID mismatch")
    if set(data) != {
        "version",
        "source_id",
        "parent_id",
        "proposal_source",
        "steps",
        "models",
        "inputs",
        "result",
    }:
        raise ProposalError("invalid proposal fields")
    template, arrays, identities, parameters, result = _apply(
        data["source_id"], data["steps"], models, inputs or {}, expected=data
    )
    for label, actual in (
        ("models", identities),
        ("inputs", parameters),
        ("result", result),
    ):
        if actual != data[label]:
            raise ProposalError(f"{label} content identity mismatch")
    return template, {k: np.array(v, copy=True) for k, v in arrays.items()}
