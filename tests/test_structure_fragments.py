"""Typed fragment archives preserve observed channels and the native conditional model."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from cffg import Engine, compile_template
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.serialize import to_json
from cffg.structure_fragments import (
    agency,
    contact_causality,
    load_fragment,
    object_persistence,
    save_fragment,
)
from cffg.structure_proposals import record_proposal, replay_proposal


def initial(card=2):
    """Supply the first latent and an unrelated, observed marker on the host."""
    variables = {v: VariableSpec(v, k) for v, k in (("latent", card), ("marker", 2))}
    return Template(
        "initial",
        variables=variables,
        ports={
            v: PortSpec(v, v, role)
            for v, role in (("latent", PortRole.GLUE), ("marker", PortRole.DATA))
        },
        factors={
            v: FactorSpec(v, "categorical", {"out": v}, param_blobs={"statistic": v})
            for v in variables
        },
        blobs={v: BlobRef(v, v + ".npy") for v in variables},
    ), {"latent": np.eye(card)[0], "marker": np.array([1.0, 0.0])}


def attached(f, host, key="latent"):
    models = {"host": host, f.fragment_id: (f.template, f.arrays)}
    step = f.compose_step(host[0], {key: "previous"}, name="attached")
    record = record_proposal(
        source_id="host",
        parent_id=None,
        proposal_source="test",
        models=models,
        steps=[step],
    )
    return replay_proposal(record, models)


def test_all_archives_compose_in_a_fresh_process(tmp_path):
    row = []
    for build in (object_persistence, contact_causality, agency):
        f = build()
        path = tmp_path / f.template.name
        row.append([str(path), save_fragment(path, f), f.ports])
    code = """
import json, sys
from cffg.core import validate
from cffg.structure_fragments import load_fragment
from test_structure_fragments import attached, initial
for path, key, ports in json.loads(sys.argv[1]):
    f = load_fragment(path, expected_id=key)
    assert f.fragment_id == key and f.ports == ports
    model, arrays = attached(f, initial())
    validate(model)
    assert 'a/marker' in model.ports and set(model.blobs) == set(arrays)
"""
    from pathlib import Path

    paths = [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
    subprocess.run(
        [sys.executable, "-c", code, json.dumps(row)],
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(paths)},
    )


@pytest.mark.parametrize(
    "builder,sensor,selector,success",
    [
        (object_persistence, "detection", None, (0.02,)),
        (contact_causality, "motion", "contact", (0.05, 0.80)),
        (agency, "position_reading", "applied_action", (0.03, 0.90)),
    ],
)
def test_observed_conditional_and_missing_reading(builder, sensor, selector, success):
    t, arrays = attached(builder(), initial())
    assert {name for name, p in t.ports.items() if p.role is PortRole.DATA} == {
        "a/marker",
        f"b/{sensor}",
        *([f"b/{selector}"] if selector else []),
    }
    e = Engine(compile_template(t, arrays=arrays))
    false_hit = 0.03 if selector is None else 0.05
    for action, p in enumerate(success):
        base = {"a/marker": 0} | ({f"b/{selector}": action} if selector else {})
        for y in (0, 1, None):
            s = e.bind_data(e.init_state(), base | {f"b/{sensor}": y})
            state = e.run(s, 1).require_converged()
            likelihood = np.array([false_hit, 0.95])
            if y == 0:
                likelihood = 1 - likelihood
            elif y is None:
                likelihood = np.ones(2)
            joint = np.array([1 - p, p]) * likelihood
            np.testing.assert_allclose(
                e.marginals(state)["b/current"], joint / joint.sum(), atol=1e-6
            )
            assert float(e.bfe(state)) == pytest.approx(-np.log(joint.sum()), abs=1e-6)


def test_fragment_id_checks_port_semantics_and_bytes(tmp_path):
    f = agency()
    path = tmp_path / "entry"
    key = save_fragment(path, f)
    assert load_fragment(path).fragment_id == key == agency().fragment_id
    for other in (
        replace(f, source="revised"),
        replace(f, description=f.description + " Revised."),
        replace(f, arrays={k: a.astype("float32") for k, a in f.arrays.items()}),
        replace(f, arrays=f.arrays | {"transition": f.arrays["transition"][::-1]}),
        replace(f, outcomes=f.outcomes | {"applied_action": ("right", "left")}),
    ):
        assert other.fragment_id != key
    with pytest.raises(ValueError, match="fragment ID mismatch"):
        load_fragment(path, expected_id="changed")
    with np.load(path, allow_pickle=False) as raw:
        contents = dict(raw)
    contents["blob_0"] = contents["blob_0"][::-1]
    with path.open("wb") as out:
        np.savez(out, **contents)
    with pytest.raises(ValueError, match="array content mismatch"):
        load_fragment(path, expected_id=key)


def test_ports_reject_role_domain_and_order_without_mutating_host():
    f = object_persistence()
    host, _ = initial(3)
    before = to_json(host)
    for bonds, error in (
        ({"latent": "previous"}, "domain mismatch"),
        ({"marker": "previous"}, "GLUE"),
        ({"latent": "detection"}, "GLUE"),
        ({"absent": "previous"}, "GLUE"),
    ):
        with pytest.raises(ValueError, match=error):
            f.compose_step(host, bonds, name="invalid")
    assert to_json(host) == before
    with pytest.raises(ValueError, match="outcomes disagree"):
        f.compose_step(agency(), {"current": "previous"}, name="invalid")


def test_ten_persistence_slices_replay_with_a_reading_per_slice():
    f = object_persistence()
    model = initial()
    outlet = "latent"
    for _ in range(10):
        model = attached(f, model, outlet)
        outlet = "b/current"
    t, arrays = model
    assert len(t.factors) == 2 + 3 * 10
    assert len(arrays) == 2 + 2 * 10
    assert sum(p.role is PortRole.DATA for p in t.ports.values()) == 11
    e = Engine(compile_template(t, arrays=arrays))
    state = e.run(e.init_state(), 1).require_converged()
    p = 0.0
    for _ in range(10):
        p = 0.9 * p + 0.02 * (1 - p)
    np.testing.assert_allclose(e.marginals(state)[outlet], [1 - p, p], atol=1e-6)
