"""Portable, typed structure-search fragments and three declared example models."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from cffg.array_identity import json_digest
from cffg.core import (
    BlobRef,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
    expand_plates,
)
from cffg.core.serialize import from_dict, to_dict
from cffg.neural_artifacts import write_artifact
from cffg.structure_proposals import compose_step, model_identity


@dataclass(frozen=True)
class Fragment:
    template: Template
    arrays: dict[str, np.ndarray]
    source: str
    description: str
    outcomes: dict[str, tuple[str, ...]]

    @property
    def ports(self):
        """The native role, domain and ordered category labels of each external port."""
        if set(self.outcomes) != set(self.template.ports):
            raise ValueError("fragment outcomes must describe every port")
        result = {}
        for name, port in self.template.ports.items():
            var = self.template.variables[port.variable]
            labels = self.outcomes[name]
            if (
                not isinstance(labels, (tuple, list))
                or len(labels) != var.cardinality
                or any(not isinstance(v, str) or not v.strip() for v in labels)
                or len(set(labels)) != len(labels)
            ):
                raise ValueError(f"invalid ordered outcome labels for port {name!r}")
            result[name] = {
                "variable": var.name,
                "role": port.role.value,
                "domain": list(var.domain_key),
                "outcomes": list(labels),
            }
        return result

    def _manifest(self):
        if not all(
            isinstance(s, str) and s.strip() for s in (self.source, self.description)
        ):
            raise ValueError("fragment requires a source and description")
        return {
            "format": "cffg.fragment",
            "version": 1,
            "source": self.source,
            "description": self.description,
            "template": to_dict(self.template),
            "ports": self.ports,
            "model": model_identity(self.template, self.arrays),
        }

    @property
    def fragment_id(self):
        """Content ID, including provenance, graph, port meanings and numeric inputs."""
        return json_digest(self._manifest())

    def compose_step(self, host, bonds, *, name, scope_a="a", scope_b="b"):
        """Connect GLUE ports while preserving DATA bindings on both operands."""
        host_template = host.template if isinstance(host, Fragment) else host
        for a, b in bonds.items():
            for model, key in ((host_template, a), (self.template, b)):
                if key not in model.ports or model.ports[key].role is not PortRole.GLUE:
                    raise ValueError(f"expected a GLUE port at {model.name}.{key}")
            left = host_template.variables[host_template.ports[a].variable]
            right = self.template.variables[self.template.ports[b].variable]
            if left.domain_key != right.domain_key:
                raise ValueError(
                    f"port domain mismatch: {left.domain_key} and {right.domain_key}"
                )
            if isinstance(host, Fragment) and list(host.outcomes[a]) != list(
                self.outcomes[b]
            ):
                raise ValueError(f"ordered port outcomes disagree for {a!r} and {b!r}")
        return compose_step(
            self.fragment_id, bonds, name=name, scope_a=scope_a, scope_b=scope_b
        )


def save_fragment(path, fragment: Fragment):
    """Atomically write one portable archive. Physical member names never use model paths."""
    fragment = deepcopy(fragment)
    manifest = fragment._manifest()
    fragment_id = json_digest(manifest)
    arrays = {
        f"blob_{i}": fragment.arrays[name]
        for i, name in enumerate(sorted(fragment.arrays))
    }
    write_artifact(path, {**manifest, "fragment_id": fragment_id}, arrays)
    return fragment_id


def load_fragment(path, *, expected_id=None):
    """Reconstruct the stored template and verify every port and blob before use."""
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive or len(archive.files) != len(set(archive.files)):
            raise ValueError("invalid fragment archive inventory")
        raw = archive["metadata"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("invalid fragment metadata")
        data = json.loads(raw.item())
        if (
            not isinstance(data, dict)
            or data.get("format") != "cffg.fragment"
            or type(data.get("version")) is not int
            or data["version"] != 1
        ):
            raise ValueError("unsupported fragment format or version")
        if set(data) != {
            "format",
            "version",
            "source",
            "description",
            "template",
            "ports",
            "model",
            "fragment_id",
        }:
            raise ValueError("invalid fragment manifest fields")
        fragment_id = data.pop("fragment_id")
        if fragment_id != json_digest(data) or (
            expected_id is not None and fragment_id != expected_id
        ):
            raise ValueError("fragment ID mismatch")

        template = from_dict(data["template"])
        names = sorted(expand_plates(template).blobs)
        keys = [f"blob_{i}" for i in range(len(names))]
        if set(archive.files) != {"metadata", *keys}:
            raise ValueError("fragment blob inventory mismatch")
        arrays = {name: archive[key] for name, key in zip(names, keys, strict=True)}
    outcomes = {name: tuple(port["outcomes"]) for name, port in data["ports"].items()}
    fragment = Fragment(template, arrays, data["source"], data["description"], outcomes)
    if fragment.fragment_id != fragment_id:
        raise ValueError("fragment model, port or array content mismatch")
    return fragment


def _binary_transition(positive):
    """P(next | previous), or a selector-indexed stack of those matrices."""
    p = np.asarray(positive, dtype=np.float64)
    return np.stack((1 - p, p), axis=-2)


def _slice(
    name,
    state,
    sensor,
    reading_labels,
    transition,
    detection,
    description,
    *,
    selector=None,
):
    # Expose current state through equality so it can feed both a reading and the next slice.
    edges = {"in": "previous", "out": "predicted"}
    outcomes = {"previous": state, "current": state, sensor: reading_labels}
    ports = {v: PortSpec(v, v, PortRole.GLUE) for v in ("previous", "current")}
    ports[sensor] = PortSpec(sensor, sensor, PortRole.DATA)
    if selector:
        channel, labels = selector
        edges["selector"] = channel
        ports[channel] = PortSpec(channel, channel, PortRole.DATA)
        outcomes[channel] = labels
    factor_type, role = (
        ("transition_mixture", "matrices") if selector else ("matrix_cat", "matrix")
    )
    arrays = {
        "transition": _binary_transition(transition),
        "sensor": _binary_transition(detection),
    }
    template = Template(
        name,
        variables={v: VariableSpec(v, 2) for v in (*ports, "predicted", "emitted")},
        factors={
            "transition": FactorSpec(
                "transition", factor_type, edges, param_blobs={role: "transition"}
            ),
            "branch": FactorSpec(
                "branch", "equality", {"a": "predicted", "b": "emitted", "c": "current"}
            ),
            "emission": FactorSpec(
                "emission",
                "matrix_cat",
                {"in": "emitted", "out": sensor},
                param_blobs={"matrix": "sensor"},
            ),
        },
        ports=ports,
        blobs={key: BlobRef(key, f"{key}.npy") for key in arrays},
    )
    return Fragment(template, arrays, f"cffg.examples/{name}/1", description, outcomes)


def object_persistence():
    """A binary object can persist between readings and be missed by a detector."""
    return _slice(
        "object-persistence",
        ("absent", "present"),
        "detection",
        ("miss", "hit"),
        (0.02, 0.90),
        (0.03, 0.95),
        "Previous presence predicts current presence. A missing detection reading supplies no sensor evidence; an observed miss is informative.",
    )


def contact_causality():
    """A recorded contact selects the transition law for latent motion."""
    return _slice(
        "contact-causality",
        ("still", "moving"),
        "motion",
        ("still", "moving"),
        ((0.05, 0.70), (0.80, 0.90)),
        (0.05, 0.95),
        "The observed contact indicator selects a motion transition. The motion reading is noisy. The table encodes a causal hypothesis.",
        selector=("contact", ("absent", "present")),
    )


def agency():
    """A recorded applied command selects a stochastic position transition."""
    return _slice(
        "agency",
        ("left", "right"),
        "position_reading",
        ("left", "right"),
        ((0.03, 0.10), (0.90, 0.97)),
        (0.05, 0.95),
        "The applied left or right command selects the position transition. A noisy reading measures the resulting position.",
        selector=("applied_action", ("left", "right")),
    )
