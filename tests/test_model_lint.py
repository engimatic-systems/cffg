"""Structural diagnostics use declarations and leave execution opt-in."""

from dataclasses import replace
import warnings

import numpy as np
import pytest
import jax

from cffg import Engine, compile_template
from cffg.core import (
    BlobRef, ConstraintKind, ConstraintSpec, FactorSpec, ModelError, PlateSpec, PortRole,
    PortSpec, Template, VariableSpec, lint,
)
from cffg.core.serialize import from_json, to_json


def pair():
    arrays = {"A": np.array([[0.8, 0.3], [0.2, 0.7]]), "p": np.array([0.6, 0.4])}
    template = Template(
        name="pair",
        variables={v: VariableSpec(v, 2) for v in ("s", "o")},
        factors={
            "center": FactorSpec("center", "categorical", {"out": "s"}, param_blobs={"statistic": "p"}),
            "camera": FactorSpec("camera", "matrix_cat", {"in": "s", "out": "o"}, param_blobs={"matrix": "A"}),
            "target": FactorSpec("target", "categorical", {"out": "o"}, param_blobs={"statistic": "p"}),
        },
        blobs={name: BlobRef(name, name + ".npy") for name in arrays},
    )
    return template, arrays


def test_lint_names_each_scoped_goal_pair_and_preserves_input():
    template = Template("many", plates={"trial": PlateSpec(pair()[0], (256,))})
    before = to_json(template)
    findings = lint(template)
    assert before == to_json(template)
    assert lint(from_json(before)) == findings
    names = sorted(f"trial/{i}/camera" for i in range(256))
    assert [f.owner for f in findings] == names
    assert all(f.code == "missing_p_substitution" and f.owner in f.message
               and f.owner.replace("camera", "target") in f.message
               and "no P-substitution epistemic term" in f.message
               and "Ordinary KL-control remains valid" in f.message for f in findings)


def test_lint_skips_substituted_pairs_data_ports_and_stochastic_parameter_edges():
    t, _ = pair()
    t.constraints.append(ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "camera", (("substituted_role", "out"),)))
    assert lint(t) == ()
    t.constraints.clear()
    t.factors.pop("target")
    t.ports["reading"] = PortSpec("reading", "o", PortRole.DATA)
    assert lint(t) == ()
    t.constraints.extend(ConstraintSpec(ConstraintKind.MEAN_FIELD, name) for name in ("camera", "center"))
    assert [(f.code, f.owner) for f in lint(t)] == [("unary_mean_field", "center")]
    t.variables["parameter"] = VariableSpec("parameter", 2)
    t.factors["center"] = replace(t.factors["center"], param_edges={"statistic": "parameter"}, param_blobs={})
    t.factors["base"] = FactorSpec("base", "dirichlet", {"out": "parameter"}, param_blobs={"counts": "p"})
    assert lint(t) == ()


def test_compile_emits_the_same_findings_only_after_success():
    t, arrays = pair()
    t.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "center"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = compile_template(t, arrays=arrays)
    with pytest.warns(UserWarning) as emitted:
        checked = compile_template(t, arrays=arrays, lint=True)
    assert [str(w.message) for w in emitted] == [f"[{f.code}] {f.message}" for f in lint(t)]
    assert quiet.edges == checked.edges
    assert quiet.member_of == checked.member_of
    plain, advised = Engine(quiet), Engine(checked)
    initial, checked_initial = plain.init_state(), advised.init_state()
    for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(checked_initial), strict=True):
        np.testing.assert_array_equal(a, b)
    assert plain.bfe(initial) == advised.bfe(checked_initial)
    del t.variables["o"]
    with warnings.catch_warnings(), pytest.raises(ModelError):
        warnings.simplefilter("error")
        compile_template(t, arrays=arrays, lint=True)


def test_delta_run_warning_depends_on_the_arena_and_hardening_phase():
    t, arrays = pair()
    for constrained in (False, True):
        if constrained:
            t.constraints.append(ConstraintSpec(ConstraintKind.DELTA, "s"))
        engine = Engine(compile_template(t, arrays=arrays))
        state = engine.init_state()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quiet = engine.run(state, 4, harden_last=0)
            engine.run(state, 4, harden_last=2, lint=True)
            if not constrained:
                engine.run(state, 4, harden_last=0, lint=True)
        if constrained:
            with pytest.warns(UserWarning, match=r"delta_hardening_disabled.*pair.*1 DELTA"):
                checked = engine.run(state, 4, harden_last=0, lint=True)
            assert checked.status == quiet.status
            np.testing.assert_array_equal(checked.state.messages, quiet.state.messages)
