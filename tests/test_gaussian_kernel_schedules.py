"""Custom Gaussian conditioning through scans and distinct-instance Colony."""

from dataclasses import replace

import equinox as eqx
import numpy as np
import pytest
from gaussian_kernel_fixture import (
    CUSTOM,
    chain_model,
    covariance_oracle,
    declaration,
    message,
    model,
)
from jax.scipy.special import logsumexp

from cffg import kernels
from cffg.colony import Colony
from cffg.compiler import compile_template
from cffg.engine import Engine
from cffg.schedule import ScanStage
from cffg.schedule_program import _batch_stage_fn, _flat_scan_fn


@pytest.fixture
def registered():
    kernel = kernels.register_kernel(declaration())
    yield kernel
    kernels.unregister_kernel(CUSTOM)


def test_observed_custom_chain_scan_transport_and_trace_reuse(registered):
    traces = []

    def counted(*args):
        traces.append(None)
        return message(*args)

    kernels.unregister_kernel(CUSTOM)
    kernels.register_kernel(
        replace(registered, binding=replace(registered.binding, message=counted))
    )
    t, arrays, observations = chain_model()
    arena = compile_template(t, arrays=arrays)
    eng = Engine(arena)
    assert eng.schedule.kind == "two_pass"
    assert eng.scan_registerized and all(eng.scan_registerized)
    initial = eng.bind_data(eng.init_state(), observations)
    first = eng.step(initial)
    count = len(traces)
    changed = eng.bind_data(initial, {"sensor1": np.array([0.9, 0.5])})
    second = eng.step(changed)
    assert len(traces) == count
    assert not np.array_equal(first.g_eta, second.g_eta)
    exact = eng.run(initial, 1).require_converged()
    flood = (
        Engine(arena, schedule="flood").run(initial, 50, tol=1e-11).require_converged()
    )
    np.testing.assert_allclose(exact.g_eta, flood.g_eta, atol=1e-11)
    np.testing.assert_allclose(exact.g_lam, flood.g_lam, atol=1e-11)
    np.testing.assert_allclose(eng.bfe(exact), eng.bfe(flood), atol=1e-11)

    args = (
        initial.messages,
        initial.g_eta,
        initial.g_lam,
        initial.ef_eta,
        initial.g_obs,
        initial.g_mask,
        initial.ef_obs,
        initial.ef_mask,
        initial.params,
    )
    current = args[:4]
    for stage in eng.schedule.stages:
        fn = (
            _flat_scan_fn(arena, stage, logsumexp)
            if isinstance(stage, ScanStage)
            else _batch_stage_fn(arena, stage, logsumexp)
        )
        current = eqx.filter_jit(fn)(*current, *args[4:])
    np.testing.assert_allclose(current[1], exact.g_eta, atol=1e-11)
    np.testing.assert_allclose(current[2], exact.g_lam, atol=1e-11)

    closed, _, _ = eqx.filter_make_jaxpr(eng._propagate)(*args)
    carries = []

    def visit(jaxpr):
        for equation in jaxpr.eqns:
            if equation.primitive.name == "scan":
                body = equation.params["jaxpr"].jaxpr
                start = equation.params["num_consts"]
                carries.extend(
                    v.aval.shape
                    for v in body.invars[start : start + equation.params["num_carry"]]
                )
            for value in equation.params.values():
                if hasattr(value, "jaxpr"):
                    visit(value.jaxpr)
                elif hasattr(value, "eqns"):
                    visit(value)

    visit(closed.jaxpr)
    assert carries
    assert all(shape in {(1, 1), (1, 3), (1, 1, 1), (1, 3, 3)} for shape in carries)
    assert initial.g_eta.shape not in carries and initial.g_lam.shape not in carries


def test_colony_shares_one_engine_for_distinct_parameters_and_observations(registered):
    t, arrays = model(observed=True)
    variants = [arrays, {**arrays, "center": arrays["center"] + 0.3}]
    observations = [np.array([0.1, 0.7]), np.array([-0.8, 0.3])]
    colony = Colony()
    ids = [
        colony.spawn(t, values, observations={"sensor": y})
        for values, y in zip(variants, observations)
    ]
    assert len(colony.engines) == 1
    results = colony.run(10, tol=1e-11)
    engine = next(iter(colony.engines.values()))
    for iid, values, y in zip(ids, variants, observations):
        state = results[iid].require_converged()
        mean, covariance, energy = covariance_oracle(values, (3, 2, 1), y, True)
        actual = engine.kernel_readouts(state)["joint"]
        np.testing.assert_allclose(actual["mean"], mean, atol=1e-11)
        np.testing.assert_allclose(actual["covariance"], covariance, atol=1e-11)
        np.testing.assert_allclose(engine.bfe(state), energy, atol=1e-11)
