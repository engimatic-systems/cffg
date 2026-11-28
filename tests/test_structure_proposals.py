"""Structure proposals bind the graph, numeric inputs, operations and lineage."""

import copy
import hashlib
import json

import numpy as np
import pytest

from cffg import Engine, compile_template, load_bundle, save_bundle
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.compose import compose
from cffg.core.edits import (
    AddBlob,
    AddFactor,
    AddVariable,
    DeclarePort,
    RebindBlobPath,
    RemoveFactor,
)
from cffg.core.serialize import load, save, structure_hash, to_json
from cffg.structure_proposals import (
    ProposalError,
    compose_step,
    edit_step,
    record_proposal,
    replay_proposal,
)


def prior():
    t = Template(
        "prior",
        variables={"z": VariableSpec("z", 2)},
        factors={
            "initial": FactorSpec(
                "initial", "categorical", {"out": "z"}, param_blobs={"statistic": "p"}
            )
        },
        ports={"attach": PortSpec("attach", "z", PortRole.GLUE)},
        blobs={"p": BlobRef("p", "p.npy")},
    )
    return t, {"p": np.array([1.0, 0.0], dtype=np.float32)}


def make(steps, models, inputs=None, **kwargs):
    args = {
        "source_id": "original",
        "parent_id": "round-1",
        "proposal_source": "external",
    }
    return record_proposal(**(args | kwargs), steps=steps, models=models, inputs=inputs)


def test_ordered_composition_and_edits_from_portable_inputs(tmp_path):
    source, arrays = prior()
    fragment, weights = prior()
    weights["p"] = np.array([0.25, 0.75], dtype=np.float64)
    models = {"original": (source, arrays), "likelihood": (fragment, weights)}
    inputs = {"refit": np.array([0.625, 0.375], dtype=np.float32)}
    steps = [
        edit_step([RebindBlobPath("p", "fit.npy")], parameters={"p": "refit"}),
        compose_step(
            "likelihood",
            {"attach": "attach"},
            name="joint",
            scope_a="left",
            scope_b="right",
        ),
    ]
    record = make(steps, models, inputs)
    assert record["candidate_id"] == make(steps, models, inputs)["candidate_id"]
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(record))
    fresh = {}
    for key, (model, points) in models.items():
        # Both model bundles relocate their blob paths on load.
        save_bundle(tmp_path / f"{key}.cffgz", model, points)
        bundle = load_bundle(tmp_path / f"{key}.cffgz")
        fresh[key] = bundle.template, bundle.arrays
    np.save(tmp_path / "refit.npy", inputs["refit"])
    actual, values = replay_proposal(
        json.loads(path.read_text()),
        fresh,
        inputs={"refit": np.load(tmp_path / "refit.npy")},
        expected_id=record["candidate_id"],
    )
    direct = compose(
        source,
        fragment,
        {"attach": "attach"},
        name="joint",
        scope_a="left",
        scope_b="right",
    )
    assert (
        structure_hash(actual)
        == structure_hash(direct)
        == record["result"]["structure_sha256"]
    )
    for key, value in {"left/p": inputs["refit"], "right/p": weights["p"]}.items():
        assert (
            values[key].dtype == value.dtype
            and values[key].tobytes() == value.tobytes()
        )
    e = Engine(compile_template(actual, arrays=values))
    state = e.run(e.init_state(), 1).require_converged()
    np.testing.assert_allclose(e.marginals(state)["left/z"], [5 / 14, 9 / 14])
    assert not np.shares_memory(values["right/p"], fresh["likelihood"][1]["p"])
    assert source.blobs["p"].path == "p.npy"


def test_lineage_structure_and_numeric_content_have_distinct_identities(tmp_path):
    t, arrays = prior()
    models = {"original": (t, arrays), "alias": (t, arrays)}
    path = tmp_path / "model.json"
    save(t, path)
    record = make([], models)
    assert replay_proposal(record, {"original": (load(path), arrays)})[0] == t
    for alternative in (
        {"parent_id": "round-2"},
        {"source_id": "alias"},
        {"proposal_source": "builtin"},
    ):
        other = make([], models, **alternative)
        assert other["candidate_id"] != record["candidate_id"]
        assert other["result"] == record["result"]
    for value in (
        np.array([0, 1], np.float32),
        arrays["p"].astype("float64"),
        arrays["p"].reshape(1, 2),
    ):
        different = {"original": (t, {"p": value})}
        other = make([], different)
        assert (
            other["models"]["original"]["structure_sha256"]
            == record["result"]["structure_sha256"]
        )
        assert (
            other["models"]["original"]["arrays_sha256"]
            != record["result"]["arrays_sha256"]
        )
        with pytest.raises(
            ProposalError, match="models content identity mismatch.*original"
        ):
            replay_proposal(record, different)


def test_invalid_operations_and_missing_or_substituted_inputs_are_transactional():
    t, points = prior()
    before = to_json(t)
    models = {"original": (t, points), "fragment": (t, points)}
    step = edit_step([RebindBlobPath("p", "next.npy")], parameters={"p": "next"})
    edits = [step, compose_step("fragment", {"attach": "attach"}, name="joined")]
    inputs = {"next": np.array([0.1, 0.9])}
    record = make(edits, models, inputs)
    for args, error in (
        ((edits, {"original": (t, points)}, inputs), "missing model 'fragment'"),
        ((edits, models, {}), "missing numeric input 'next'"),
        (([], {"original": (t, {})}), "missing numeric inputs for blobs"),
        (([edit_step([RemoveFactor("absent")])], models), "step 0"),
        (
            ([edit_step([RebindBlobPath("p", "next.npy")])], models),
            "explicit numeric input",
        ),
        (
            ([compose_step("fragment", {"unknown": "attach"}, name="bad")], models),
            "no port",
        ),
    ):
        with pytest.raises(ProposalError, match=error):
            make(*args)
    with pytest.raises(ProposalError, match="inputs content identity mismatch.*next"):
        replay_proposal(record, models, inputs={"next": points["p"]})
    wrong_fragment = {"fragment": (t, {"p": points["p"][::-1]})}
    with pytest.raises(
        ProposalError, match="models content identity mismatch.*fragment"
    ):
        replay_proposal(record, models | wrong_fragment, inputs=inputs)
    changed_graph = copy.deepcopy(t)
    changed_graph.ports["attach"] = PortSpec("attach", "z", PortRole.DATA)
    with pytest.raises(
        ProposalError, match="models content identity mismatch.*original"
    ):
        replay_proposal(
            record, models | {"original": (changed_graph, points)}, inputs=inputs
        )
    with pytest.raises(ProposalError, match="candidate ID mismatch"):
        replay_proposal(record, models, inputs=inputs, expected_id="different record")
    tampered = copy.deepcopy(record)
    tampered["steps"].reverse()
    with pytest.raises(ProposalError, match="candidate ID mismatch"):
        replay_proposal(tampered, models, inputs=inputs)
    forged = copy.deepcopy(record)
    forged.pop("candidate_id")
    forged["result"]["arrays_sha256"] = "0" * 64
    # A self-consistent record digest cannot make a false result match replay.
    payload = json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()
    forged["candidate_id"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ProposalError, match="result content identity mismatch"):
        replay_proposal(forged, models, inputs=inputs)
    assert to_json(t) == before
    np.testing.assert_array_equal(points["p"], [1.0, 0.0])


def test_added_blob_uses_the_retained_numeric_input():
    template, points = prior()
    models = {"original": (template, points)}
    add = AddBlob(BlobRef("new", "unavailable.npy"))
    empty = edit_step([add])
    with pytest.raises(ProposalError, match="missing numeric inputs.*new"):
        make([empty], models)
    step = edit_step([add], parameters={"new": "snapshot"})
    # A scalar and a length-one array have the same raw bytes but different shapes.
    scalar = np.array(0.0, dtype=np.float32)
    record = make([step], models, inputs={"snapshot": scalar})
    result, arrays = replay_proposal(record, models, inputs={"snapshot": scalar})
    assert "new" in result.blobs and "new" not in template.blobs
    assert arrays["new"].shape == () and arrays["new"].dtype == scalar.dtype
    with pytest.raises(
        ProposalError, match="inputs content identity mismatch.*snapshot"
    ):
        replay_proposal(record, models, inputs={"snapshot": scalar.reshape(1)})


@pytest.mark.parametrize("copies", [1, 10])
def test_replay_scales_to_ten_times_the_edit_count(copies):
    t, arrays = prior()
    models = {"original": (t, arrays)}
    steps = []
    for i in range(copies):
        var = f"z{i}"
        steps.append(
            edit_step(
                [
                    AddVariable(VariableSpec(var, 2)),
                    AddFactor(
                        FactorSpec(
                            f"d{i}",
                            "categorical",
                            {"out": var},
                            param_blobs={"statistic": "p"},
                        )
                    ),
                    DeclarePort(PortSpec(var, var, PortRole.DATA)),
                ]
            )
        )
    result, points = replay_proposal(make(steps, models), models)
    assert len(result.factors) == copies + 1 and len(t.factors) == 1
    assert set(points) == {"p"}
