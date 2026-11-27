"""Compound document commands shared by the keyboard editor and agent clients."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from time import time
from uuid import uuid4

import numpy as np

from cffg import Engine, compile_template, inits
from cffg.core.edits import action_from_dict
from cffg.core.ir import (
    BlobRef,
    EDGE_CONSTRAINTS,
    FactorSpec,
    PortRole,
    PortSpec,
    VariableSpec,
)
from cffg.learn import Learnables
from cffg.resolution.sources import _blob_shape

from .models import starter
from .dimensions import dependencies


def new_name(value, existing):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 160
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError("Choose a nonempty name of up to 160 characters")
    if value in existing:
        raise ValueError(f"{value!r} already exists")
    return value


def rename_refs(t, old, name):
    for f in t.factors.values():
        for roles in (f.edges, f.param_edges):
            for role, edge in roles.items():
                if edge == old:
                    roles[role] = name
    t.ports = {
        key: replace(p, variable=name) if p.variable == old else p
        for key, p in t.ports.items()
    }
    t.constraints = [
        replace(c, target=name) if c.kind in EDGE_CONSTRAINTS and c.target == old else c
        for c in t.constraints
    ]
    if t.agent:
        bindings = tuple(
            replace(b, posterior=name) if b.posterior == old else b
            for b in t.agent.beliefs
        )
        t.agent = replace(
            t.agent,
            action=name if t.agent.action == old else t.agent.action,
            beliefs=bindings,
        )


def add_channel(d, cmd):
    t = d.template
    name = new_name(cmd["name"], t.variables)
    port = t.ports.get(cmd.get("source", ""))
    if cmd.get("source") and not port:
        raise ValueError("Select an existing boundary port")
    f_name = new_name(cmd.get("factor") or name + "_factor", t.factors)
    port_name = new_name(cmd.get("port") or name, t.ports)
    size = int(cmd.get("size", 2))
    if not 1 <= size <= 4096:
        raise ValueError("Dimension must be between 1 and 4096")
    source = t.variables[port.variable] if port else None
    gaussian = (
        source.family if source else cmd.get("family", "gaussian")
    ) == "gaussian"
    if source and source.ef:
        raise ValueError(
            "Use a typed factor command to extend an exponential-family boundary"
        )
    t.variables[name] = (
        VariableSpec(name, dim=size)
        if gaussian
        else VariableSpec(name, cardinality=size)
    )
    edges = {"out": name}
    if source:
        edges["in"] = source.name
        del t.ports[port.name]
    if gaussian:
        kind = "gaussian_linear" if source else "gaussian"
        values = {"cov": np.eye(size), ("offset" if source else "mean"): np.zeros(size)}
        if source:
            values["matrix"] = np.eye(size, source.dim)
    else:
        kind = "matrix_cat" if source else "categorical"
        values = (
            {"matrix": np.full((size, source.cardinality), 1 / size)}
            if source
            else {"statistic": np.full(size, 1 / size)}
        )
    blobs = {}
    for role, array in values.items():
        blob = new_name(f"{f_name}.{role}", t.blobs)
        t.blobs[blob] = BlobRef(blob, f"blobs/{blob}.npy")
        d.arrays[blob] = array
        blobs[role] = blob
    t.factors[f_name] = FactorSpec(f_name, kind, edges=edges, param_blobs=blobs)
    t.ports[port_name] = PortSpec(
        port_name, name, PortRole.DATA if source else PortRole.GLUE
    )


def factor_init(d, c):
    factor = d.template.factors[c["factor"]]
    kind = c["preset"]
    seed = int(c.get("seed", 0))
    noise = float(c.get("noise", 0.1))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Noise variance must be positive and finite")
    rng = np.random.default_rng(seed)
    if kind in ("linear_gaussian", "noisy_gaussian"):
        if factor.type not in (
            "gaussian",
            "gaussian_linear",
            "gaussian_mixture_linear",
        ):
            raise ValueError("Select a Gaussian prior or linear Gaussian factor")
        for role, blob in factor.param_blobs.items():
            prev = d.arrays[blob]
            if role == "matrix":
                value = np.broadcast_to(np.eye(*prev.shape[-2:]), prev.shape).copy()
                if kind == "noisy_gaussian":
                    value += rng.normal(0, np.sqrt(noise), prev.shape)
            elif role == "cov":
                value = np.broadcast_to(np.eye(prev.shape[-1]) * noise, prev.shape)
            else:
                value = (
                    rng.normal(0, np.sqrt(noise), prev.shape)
                    if kind == "noisy_gaussian"
                    else np.zeros(prev.shape)
                )
            d.arrays[blob] = value.astype(prev.dtype)
        return
    blob = factor.param_blobs[c["role"]]
    prev = d.arrays[blob]
    params = c.get("parameters", {})
    if kind in ("random_spd", "random_stochastic"):
        params = {**params, "seed": seed}
    d.arrays[blob] = inits.make(kind, prev.shape, **params).astype(prev.dtype)
    d.template.blobs[blob] = replace(
        d.template.blobs[blob], init=inits.spec(kind, **params)
    )


def resize(d, c):
    t = d.template
    v = t.variables[c["name"]]
    size = int(c["size"])
    if not 1 <= size <= 4096 or v.ef:
        raise ValueError(
            "Resize requires a categorical or Gaussian variable of dimension 1–4096"
        )
    policy = c.get("policy", "preserve")
    if policy not in ("preserve", "map", "reinitialize"):
        raise ValueError("Choose preserve, map or reinitialize")
    old_size = v.dim or v.cardinality
    if old_size == size and policy != "map":
        return
    mapping = c.get("mapping")
    if policy == "map":
        if (
            not isinstance(mapping, list)
            or len(mapping) != size
            or any(
                i is not None and (type(i) is not int or not 0 <= i < old_size)
                for i in mapping
            )
        ):
            raise ValueError("Provide one old index or null for each new coordinate")
        indexes = [i for i in mapping if i is not None]
        if len(set(indexes)) != len(indexes):
            raise ValueError("Mapped source indices must be unique")
    recipe = d.meta.get("recipe", {})
    axes = d.meta.get("axes", {})
    axis = axes.get(v.name)
    names, bindings = dependencies(t, v.name, axes)
    for n in names:
        t.variables[n] = replace(
            t.variables[n], **{"dim" if v.dim else "cardinality": size}
        )
    baseline = None
    if axis and recipe.get("kind") in ("rslds", "kalman"):
        recipe[axis] = size
        baseline = starter(**recipe, name=t.name)
    changed = set()
    for blob, prev in list(d.arrays.items()):
        reference = baseline.arrays.get(blob) if baseline else None
        binding = bindings.get(blob, [])
        bound = any(n in names for n, _ in binding)
        if binding:
            shape = tuple(
                (t.variables[n].dim or t.variables[n].cardinality) - sub
                for n, sub in binding
            )
        else:
            shape = reference.shape if reference is not None else _blob_shape(t, blob)
        if shape is None or (
            tuple(shape) == prev.shape and not (policy == "map" and bound)
        ):
            continue
        changed.add(blob)
        roles = {
            role
            for f in t.factors.values()
            for role, b in f.param_blobs.items()
            if b == blob
        }
        if reference is not None:
            out = reference.astype(prev.dtype, copy=True)
        elif "cov" in roles:
            out = np.broadcast_to(np.eye(shape[-1]), shape).astype(
                prev.dtype, copy=True
            )
        elif any(
            f.type in ("categorical", "matrix_cat") and blob in f.param_blobs.values()
            for f in t.factors.values()
        ):
            out = np.full(shape, 1 / shape[0], dtype=prev.dtype)
        else:
            out = np.zeros(shape, dtype=prev.dtype)
        if policy in ("map", "preserve"):
            if policy == "map" and bound:
                if any(sub and n in names for n, sub in binding):
                    if not c.get("reset_learning"):
                        raise ValueError(
                            "Changing stick order requires explicit recurrence reinitialization"
                        )
                else:
                    src = []
                    dest = []
                    for i, (n, _) in enumerate(binding):
                        pairs = (
                            [(j, k) for j, k in enumerate(mapping) if k is not None]
                            if n in names
                            else [(k, k) for k in range(shape[i])]
                        )
                        dest.append([j for j, _ in pairs])
                        src.append([k for _, k in pairs])
                    out[np.ix_(*dest)] = prev[np.ix_(*src)]
            else:
                region = tuple(
                    slice(0, min(a, b)) for a, b in zip(prev.shape, shape, strict=True)
                )
                out[region] = prev[region]
            if any(
                f.type in ("categorical", "matrix_cat")
                and blob in f.param_blobs.values()
                for f in t.factors.values()
            ):
                totals = out.sum(axis=0, keepdims=True)
                if np.any(totals <= 0):
                    raise ValueError(
                        "Resize removes all probability mass. Choose reinitialize."
                    )
                out = out / totals
        d.arrays[blob] = out
    owners = {name for name, l in t.learnables.items() if changed.intersection(l.blobs)}
    if owners:
        if not (
            (policy == "reinitialize" or c.get("reset_learning"))
            and baseline
            and baseline.learning
        ) and not c.get("learning"):
            raise ValueError(
                "Resize affects learned priors and posteriors: "
                + ", ".join(sorted(owners))
                + ". Choose explicit reinitialization for a supported recipe or supply a complete replacement bundle."
            )
        priors = dict(d.learning.initial_params)
        posts = dict(d.learning.params)
        for name in owners:
            replacement = c.get("learning", {}).get(name)
            if replacement:
                priors[name] = {
                    k: np.asarray(x) for k, x in replacement["prior"].items()
                }
                posts[name] = {
                    k: np.asarray(x) for k, x in replacement["posterior"].items()
                }
            else:
                priors[name] = baseline.learning.initial_params[name]
                posts[name] = baseline.learning.params[name]
        d.learning = Learnables.from_persistent_state(
            Engine(compile_template(t, arrays=d.arrays)), priors, posts
        )


def apply_fragment(d, command, actor, base):
    """Edit one compact specification while preserving each copy's distinct point data."""
    from cffg.core.plates import _prefix

    name = command["plate"]
    plate = d.template.plates[name]
    commands = command["commands"]
    if any(c.get("action") in ("set_tour", "remove_tour") for c in commands):
        raise ValueError(
            "Author tours on the whole model; use a step’s plate field to target its fragment"
        )
    if any(c.get("action") == "edit_plate" for c in commands):
        raise ValueError("Nested plates are unsupported by the compiler")
    values_change = any(
        c.get("action")
        in (
            "add_channel",
            "initialize_factor",
            "resize_variable",
            "set_array",
            "accept",
        )
        for c in commands
    )
    indices = (
        np.ndindex(*plate.dims)
        if values_change and plate.per_copy_blobs
        else [(0,) * len(plate.dims)]
    )
    copies = {}
    updated = None
    for idx in indices:
        fragment = d.fragment(name, idx)
        fragment.template = deepcopy(fragment.template)
        fragment.meta = deepcopy(fragment.meta)
        fragment.meta["model_digest"] = d.meta.get("model_digest")
        if fragment.template.learnables and any(
            c.get("action") == "resize_variable" for c in commands
        ):
            raise ValueError(
                "Resizing a plate with learning owners requires a complete replacement bundle"
            )
        apply_commands(fragment, commands, actor, base)
        fragment.sync_anchors()
        for blob, array in fragment.arrays.items():
            key = (
                _prefix(name, idx) + "/" + blob
                if blob in plate.per_copy_blobs
                else blob
            )
            copies[key] = array
        if updated is None:
            updated = fragment
    d.template.plates[name] = replace(plate, fragment=updated.template)
    d.arrays.update(copies)
    d.meta.setdefault("scopes", {})[name] = updated.meta


def apply_commands(d, commands, actor, base):
    if not 1 <= len(commands) <= 256:
        raise ValueError("A transaction needs 1–256 commands")
    t = d.template
    for index, c in enumerate(commands):
        try:
            op = c["action"]
            if op == "edit_plate":
                apply_fragment(d, c, actor, base)
            elif op in ("set_tour", "remove_tour"):
                from .tours import edit_tour

                edit_tour(d, c)
            elif op == "rename_variable":
                old = c["name"]
                name = new_name(c["to"], t.variables)
                t.variables[name] = replace(t.variables.pop(old), name=name)
                rename_refs(t, old, name)
                anchors = d.meta.get("anchors", {})
                if "var:" + old in anchors:
                    anchors["var:" + name] = anchors.pop("var:" + old)
                axes = d.meta.get("axes", {})
                if old in axes:
                    axes[name] = axes.pop(old)
            elif op == "resize_variable":
                resize(d, c)
            elif op == "initialize_factor":
                factor_init(d, c)
            elif op == "add_channel":
                add_channel(d, c)
            elif op == "connect_ports":
                left, right = (t.ports[n] for n in (c["source"], c["target"]))
                if (
                    left == right
                    or t.variables[left.variable].domain_key
                    != t.variables[right.variable].domain_key
                ):
                    raise ValueError(
                        "Choose two distinct boundary ports with matching domains"
                    )
                del t.ports[left.name], t.ports[right.name]
                rename_refs(t, right.variable, left.variable)
                del t.variables[right.variable]
            elif op == "set_array":
                blob = c["name"]
                if blob not in t.blobs:
                    raise ValueError("Declare the blob before supplying its values")
                value = np.asarray(c["values"], dtype=c.get("dtype", "float64"))
                if value.dtype.kind not in "fibu" or not np.isfinite(value).all():
                    raise ValueError("Supply finite numeric parameter values")
                d.arrays[blob] = value
            elif op == "position_factor":
                name = c["name"]
                if (
                    name not in t.factors
                    or len(c["offset"]) != 2
                    or any(type(x) is not int or abs(x) > 1000 for x in c["offset"])
                ):
                    raise ValueError(
                        "Choose a factor and two integer grid offsets within 1000 cells"
                    )
                if "layout_version" not in d.meta:
                    d.meta["layout_version"] = (
                        d.layout_scene()[1] if d.meta.get("layout") else 2
                    )
                d.meta.setdefault("layout", {})[name] = c["offset"]
            elif op in ("edit_annotation", "remove_annotation"):
                note = next(
                    (n for n in d.meta.get("notes", []) if n["id"] == c["id"]), None
                )
                if note is None:
                    raise ValueError("Unknown annotation")
                if op == "remove_annotation":
                    d.meta["notes"].remove(note)
                else:
                    text = c["text"].strip()
                    if not text or len(text) > 16000:
                        raise ValueError("Enter a note of 1–16000 characters")
                    note.update(text=text, edited_by=actor, edited=time())
            elif op == "model_name":
                t.name = new_name(c["name"], {})
            elif op in ("annotate", "critique", "propose"):
                ref = c.get("ref", "")
                if ref and ref not in d.meta.get("anchors", {}):
                    raise ValueError(f"Unknown anchor {ref!r}")
                body = c.get("text", "").strip()
                if not body or len(body) > 16000:
                    raise ValueError("Enter a note of 1–16000 characters")
                entry = {
                    "id": uuid4().hex,
                    "anchor": d.meta.get("anchors", {}).get(ref),
                    "text": body,
                    "author": actor,
                    "base": base,
                    "created": time(),
                    "status": "open",
                    "type": op,
                    "replies": [],
                }
                if op == "propose":
                    entry["commands"] = c["commands"]
                    entry["model_digest"] = d.meta.get("model_digest")
                d.meta.setdefault(
                    "notes" if op == "annotate" else "reviews", []
                ).append(entry)
            elif op in ("reply", "resolve", "decline", "accept"):
                matches = [r for r in d.meta.get("reviews", []) if r["id"] == c["id"]]
                if not matches:
                    raise ValueError("Unknown critique or proposal")
                r = matches[0]
                if op == "reply":
                    r["replies"].append(
                        {"text": c["text"], "author": actor, "created": time()}
                    )
                else:
                    if r["status"] != "open":
                        raise ValueError("Review is already closed")
                    if op == "accept":
                        if r["type"] != "propose":
                            raise ValueError("Select a proposed edit")
                        if r.get("model_digest") != d.meta.get(
                            "model_digest"
                        ) and c.get("reviewed_digest") != d.meta.get("model_digest"):
                            raise ValueError(
                                "The model changed since this proposal. Preview the diff and submit the current reviewed_digest."
                            )
                        if any(
                            a.get("action") in ("accept", "propose")
                            for a in r["commands"]
                        ):
                            raise ValueError("Nested proposals cannot be applied")
                        apply_commands(d, r["commands"], actor, base)
                    r["status"] = {
                        "accept": "accepted",
                        "resolve": "resolved",
                        "decline": "declined",
                    }[op]
            else:
                action_from_dict(c).apply(t)
        except (TypeError, KeyError, ValueError, IndexError, OverflowError) as e:
            raise ValueError(
                f"Command {index + 1} ({c.get('action', '?')}): {e}"
            ) from e
    return d
