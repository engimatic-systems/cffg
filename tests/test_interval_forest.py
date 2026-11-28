"""Public model, scalar and batched inference, and signed D13a tree ledger."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.colony import Colony
from cffg import Engine, compile_template, load_bundle, save_bundle
from cffg import intervals as iv
from cffg.compiler import CompileError
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.runtime_types import RunStatus
from test_neural_delta_derivation import Unit, interval_tree


def fixture(d=1, cap=3, dtype=np.float64):
    t = Template("piecewise_sensor")
    k = np.arange(d, dtype=dtype)
    a = {
        "knot": k,
        "offset": -k,
        "left": k / 16 + dtype(0.5),
        "right": k / 8 + dtype(2.0),
    }
    a["lower"] = np.zeros((d, cap), dtype)
    a["lower"][:, 0] = k - 2
    a["upper"] = np.zeros((d, cap), dtype)
    a["upper"][:, 0] = k + 2
    a["log_height"] = np.zeros((d, cap), dtype)
    a["log_height"][:, 0] = -np.log(4.0)
    a["count"] = np.ones(d, np.int32)
    a["tails"] = np.zeros((d, 2), bool)
    t.blobs = {b: BlobRef(b, b + ".npy") for b in a}
    t.variables = {
        e: VariableSpec(e, interval_dim=d, interval_capacity=cap) for e in ("x", "y")
    }
    t.factors["prior"] = FactorSpec(
        "prior",
        "interval_potential",
        {"out": "x"},
        param_blobs={b: b for b in ("lower", "upper", "log_height", "count", "tails")},
    )
    t.factors["sensor"] = FactorSpec(
        "sensor",
        "interval_activation",
        {"in": "x", "out": "y"},
        param_blobs={b: b for b in ("knot", "offset", "left", "right")},
    )
    t.ports["read"] = PortSpec("read", "y", PortRole.DATA)
    low, hi, log = (np.zeros((d, cap), dtype) for _ in range(3))
    low[:, 0] = -k - a["left"]
    hi[:, 0] = -k + a["right"] / 2
    obs = iv.IntervalMessage(low, hi, log, np.ones(d, np.int32))
    return t, a, obs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("d", [1, 32])
def test_signed_ledger_public_schedules_and_bundle(tmp_path, d, dtype):
    with jax.enable_x64(dtype == np.float64):
        t, a, obs = fixture(d=d, dtype=dtype)
        template = from_json(to_json(t))
        assert structure_hash(t) == structure_hash(template)
        path = tmp_path / "sensor.cffgz"
        save_bundle(path, t, a)
        bundle = load_bundle(path)
        exact = Engine(compile_template(bundle.template, arrays=bundle.arrays))
        flood = Engine(compile_template(template, arrays=a), schedule="flood")
        assert isinstance(exact, Engine) and exact.schedule.kind == "two_pass"
        assert flood.schedule.kind == "flood"
        tol = 128 * d * np.finfo(dtype).eps
        ref = interval_tree(Unit(0.5, 2), (-2, 2), (-0.5, 1), dtype)
        expected = -d * np.log(3 / 8)
        for engine in (exact, flood):
            base = engine.init_state()
            state = engine.bind_data(base, {"read": obs})
            if engine is flood:
                assert (
                    engine.run(state, 1, tol=tol).status == RunStatus.BUDGET_EXHAUSTED
                )
            run = engine.run(state, 5, tol=tol)
            got = run.require_converged()
            signed = engine.interval_ledger(got)
            np.testing.assert_allclose(
                [engine.bfe(got), run.fe_trace[-1], signed["total"]],
                expected,
                atol=tol,
                rtol=0,
            )
            assert set(signed["nodes"]) == {"prior", "sensor"}
            assert set(signed["boundaries"]) == {"read"}
            if d == 1:
                rows = [
                    signed["nodes"]["prior"],
                    signed["nodes"]["sensor"],
                    signed["boundaries"]["read"],
                    signed["edges"]["x"],
                    signed["edges"]["y"],
                ]
                np.testing.assert_allclose(rows, ref["terms"], atol=tol)
            belief = engine.interval_marginals(got)["y"]
            np.testing.assert_array_equal(belief.count, np.full(d, 2))
            first = engine.interval_marginals(got, ["x"])["x"]
            np.testing.assert_allclose(first.log_height[:, 0], -np.log(1.5), atol=tol)
            missing = engine.bind_data(got, {"read": None})
            np.testing.assert_allclose(
                engine.run(missing, 5, tol=tol).fe_trace[-1], 0.0, atol=tol
            )


def test_capacity_improper_and_zero_normalizers_are_distinct():
    t, a, obs = fixture(cap=1)
    engine = Engine(compile_template(t, arrays=a))
    state = engine.init_state()
    bad = engine.run(state, 1, tol=1e-8)
    assert (
        bad.failure.reason == "interval_capacity"
        and bad.state is state
        and not bad.fe_trace
    )
    t, a, obs = fixture()
    a["lower"][:] = 0
    a["upper"][:] = 0
    a["log_height"][:] = 0
    a["tails"][:] = True
    engine = Engine(compile_template(t, arrays=a))
    state = engine.init_state()
    bad = engine.run(state, 1, tol=1e-8)
    assert (
        bad.status == RunStatus.DIVERGED
        and bad.failure.reason == "interval_improper"
        and not bad.fe_trace
    )
    got = engine.run(engine.bind_data(state, {"read": obs}), 1, tol=1e-8)
    np.testing.assert_allclose(got.fe_trace, [-np.log(1.5)])
    zero = obs._replace(count=np.zeros(1, np.int32))
    impossible = engine.run(engine.bind_data(state, {"read": zero}), 1, tol=1e-8)
    assert impossible.status == RunStatus.INFEASIBLE and impossible.fe_trace == (
        np.inf,
    )


def test_colony_updates_keep_member_parameters_evidence_and_status_separate():
    t, a, obs = fixture(dtype=np.float32)
    other = {
        **a,
        "left": np.array([1.0], np.float32),
        "right": np.array([1.0], np.float32),
    }
    colony = Colony()
    ids = [colony.spawn(t, p, observations={"read": obs}) for p in (a, other)]
    colony.bind(ids[1], {"read": obs._replace(count=np.zeros(1, np.int32))})
    res = colony.run(2, tol=1e-6)
    assert (
        res[ids[0]].status is RunStatus.CONVERGED
        and res[ids[1]].status is RunStatus.INFEASIBLE
    )
    colony.bind(ids[1], {"read": obs})
    got = colony.run(2, tol=1e-6)
    assert len(colony.engines) == 1
    for j, p in zip(ids, (a, other)):
        ref = Engine(compile_template(t, arrays=p))
        result = ref.run(ref.bind_data(ref.init_state(), {"read": obs}), 2, tol=1e-6)
        np.testing.assert_allclose(got[j].fe_trace, result.fe_trace)
        np.testing.assert_allclose(
            colony.interval_marginals(j)["y"].log_height,
            ref.interval_marginals(result.state)["y"].log_height,
        )


def test_shape_domain_dtype_and_numeric_failures_are_explicit():
    t, a, obs = fixture()
    with pytest.raises(CompileError, match="sum-product"):
        compile_template(t, arrays=a, semiring="max")
    with pytest.raises(CompileError, match="shape"):
        compile_template(t, arrays={**a, "left": np.ones((1, 1))})
    with pytest.raises(CompileError, match="slope"):
        compile_template(t, arrays={**a, "left": np.zeros(1)})
    with pytest.raises(ValueError, match="capacity"):
        VariableSpec("bad", interval_dim=1)
    with jax.enable_x64(False):
        with pytest.raises(CompileError, match="unresolved"):
            compile_template(
                t,
                arrays={
                    **a,
                    "lower": np.array([[2.0**27, 0, 0]]),
                    "upper": np.array([[2.0**27 + 1, 0, 0]]),
                },
            )
        t, a, obs = fixture(dtype=np.float32)
        a["offset"][:] = 2.0**27
        engine = Engine(compile_template(t, arrays=a))
        state = engine.init_state()
        failure = engine.run(state, 2, tol=1e-5)
        assert (
            failure.status is RunStatus.DIVERGED
            and failure.failure.reason == "interval_numerical"
        )
        assert failure.state is state and not failure.fe_trace
        with pytest.raises(ValueError, match="IntervalMessage"):
            engine.bind_data(state, {"read": 1.0})
    x = t.variables["x"]
    t.variables["x"] = replace(x, interval_capacity=x.interval_capacity + 1)
    with pytest.raises(ValueError, match="domain"):
        compile_template(t, arrays=a)


def test_equality_branch_components_cycle_and_prebuilt_schedule():
    t, a, obs = fixture()
    t.variables["q"] = VariableSpec("q", interval_dim=1, interval_capacity=3)
    t.variables["qcopy"] = replace(t.variables["q"], name="qcopy")
    t.factors["sensor"] = replace(t.factors["sensor"], edges={"in": "q", "out": "y"})
    t.factors["copy"] = FactorSpec(
        "copy", "equality", {"a": "x", "b": "q", "c": "qcopy"}
    )
    t.ports["second_read"] = PortSpec("second_read", "qcopy", PortRole.DATA)
    # Disconnected unit-affine chain with its own normalizer.
    for v in ("independent",):
        t.variables[v] = replace(t.variables["x"], name=v)
        t.factors[v] = replace(t.factors["prior"], name=v, edges={"out": v})
        t.ports[v] = PortSpec(v, v, PortRole.DATA)
    e = Engine(compile_template(t, arrays=a))
    second = obs._replace(
        lower=np.array([[-0.25, 0, 0]]), upper=np.array([[0.5, 0, 0]])
    )
    state = e.bind_data(
        e.init_state(), {"read": obs, "second_read": second, "independent": second}
    )
    result = e.run(state, 1, tol=1e-8)
    np.testing.assert_allclose(result.fe_trace, [-2 * np.log(0.75 / 4)])
    assert Engine(e.arena, schedule=e.schedule).schedule == e.schedule
    with pytest.raises(ValueError, match="schedule"):
        Engine(e.arena, schedule=replace(e.schedule, fingerprint="wrong"))
    # Wire a two-node cycle; each variable still has valid FFG degree two.
    t, a, _ = fixture()
    del t.factors["prior"]
    t.ports.clear()
    t.factors["again"] = replace(
        t.factors["sensor"], name="again", edges={"in": "y", "out": "x"}
    )
    with pytest.raises(CompileError, match="acyclic"):
        compile_template(t, arrays=a)


def test_source_boundaries_are_member_specific_and_rebinding_cannot_merge_an_endpoint():
    with jax.enable_x64(False):
        t, a, window = fixture()
        moved = {**a, "offset": np.array([2.0**27])}
        colony = Colony()
        large, standard = (colony.spawn(t, p) for p in (moved, a))
        window.lower[0, 0] = 2.0**27 + 1
        window.upper[0, 0] = 2.0**27 + 32
        with pytest.raises(ValueError, match="distinct interval boundaries"):
            colony.bind(large, {"read": window})
        colony.bind(standard, {"read": window})
        t, a, _ = fixture()
        a["lower"][0, 0] = 1.0
        a["upper"][0, 0] = 2.0**27 + 1
        a["knot"][0] = 2.0**27
        with pytest.raises(CompileError, match="distinct interval boundaries"):
            compile_template(t, arrays=a)


def test_repeated_single_flood_updates_certify_the_same_static_messages():
    t, a, obs = fixture()
    engine = Engine(compile_template(t, arrays=a), schedule="flood")
    state = engine.bind_data(engine.init_state(), {"read": obs})
    for _ in range(5):
        run = engine.run(state, 1, tol=1e-6)
        state = run.state
        if run.status is RunStatus.CONVERGED:
            break
    assert run.status is RunStatus.CONVERGED
    np.testing.assert_allclose(engine.bfe(state), -np.log(3 / 8))
    assert engine.marginals(state) == {} and engine.gaussian_marginals(state) == {}
