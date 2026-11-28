"""Graph-to-box reconstruction on the actual P4 reactor, with a precision-form oracle."""

import json
from dataclasses import replace

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from cffg.bounded_actions import BoundStatus
from cffg.bounded_chain import BoundedChainRegion, BoundedCoordinate
from test_cstr_runtime import fixture as reactor


def model():
    template, arrays = reactor.model(horizon=2)
    actions = tuple(BoundedCoordinate(f"s{t}c", 3, -1., 1.) for t in (4, 5))
    return BoundedChainRegion(template, actions), arrays


def dense_oracle(reactor, arrays, data):
    # Independently freeze RK4 at prefix-filter means, multiply all normalized
    # factors in information form, and integrate the entire 28D latent vector.
    times, dim = 7, 4
    size = times * dim
    precision, information = np.zeros((size, size)), np.zeros(size)
    constant = 0.
    def factor(design, target, covariance):
        nonlocal constant
        weight = np.linalg.inv(covariance)
        precision[:] += design.T @ weight @ design
        information[:] += design.T @ weight @ target
        constant += (target @ weight @ target + np.linalg.slogdet(covariance)[1]
                     + len(target)*np.log(2*np.pi)) / 2

    slot = np.eye(size).reshape(times, dim, size)
    m, cov = arrays["m0"], arrays["P0"]
    factor(slot[0], m, cov)
    f = lambda x: reactor.cstr_rk4(x, arrays["theta"], reactor.CONFIGURATION)
    jacobian = jax.jacfwd(f)
    for t in range(times):
        kind = "past" if t < 4 else "now" if t == 4 else "goal"
        value, h, noise = data[f"e{t}"], arrays[f"G_{kind}"], arrays[f"R_{kind}"]
        if value is not None:
            factor(h @ slot[t], value, noise)
            gain = np.linalg.solve(noise + h @ cov @ h.T, h @ cov).T
            m = m + gain @ (value - h @ m)
            cov = cov - gain @ h @ cov
        if t + 1 < times:
            a, nxt = np.asarray(jacobian(m)), np.asarray(f(m))
            noise = arrays["Q" if t < 4 else "Qp"]
            factor(slot[t+1] - a @ slot[t], nxt-a@m, noise)
            m, cov = nxt, noise + a @ cov @ a.T
    cov = np.linalg.inv(precision)
    m = np.linalg.solve(precision, information)
    energy = constant - information @ m / 2 + (np.linalg.slogdet(precision)[1] - size*np.log(2*np.pi))/2
    return m, cov, energy


def test_reactor_joint_and_live_data_match_independent_precision():
    from test_bounded_action_derivation import conditional_integrals

    region, arrays = model()
    region = BoundedChainRegion.from_dict(json.loads(json.dumps(region.to_dict()))).scoped("plant")
    engine = region.compile(arrays={"plant/"+k: v for k, v in arrays.items()})
    binder, base = reactor.Binder(engine), engine.init_state()
    for goal in (0., .2):
        data = reactor.observations(horizon=2, goal=goal)
        result = engine.run(binder.bind(base, {"plant/"+k: v for k, v in data.items()}), 1, tol=1e-6)
        out = region.evaluate(engine, result)
        assert out.bounded.status == BoundStatus.SUCCESS
        m, cov, energy = dense_oracle(reactor, arrays, data)
        idx = [4*t+i for t, i in out.order]
        assert_allclose(out.gaussian_mean, m[idx], atol=2e-10)
        assert_allclose(out.gaussian_covariance, cov[np.ix_(idx, idx)], atol=2e-10)
        assert_allclose(out.gaussian_free_energy, energy, atol=2e-9)
        raw = conditional_integrals(m[idx[:2]], cov[np.ix_(idx[:2], idx[:2])], np.array([[-1., 1.]]*2))
        assert_allclose(out.bounded.mean[:2], raw[1:]/raw[0], atol=1e-9)
        assert_allclose(out.bounded.free_energy, energy-np.log(raw[0]), atol=2e-9)
    other = region.compile(arrays={"plant/"+k: v for k, v in arrays.items()})
    with pytest.raises(ValueError, match="identity"):
        region.evaluate(other, result)
    with pytest.raises(ValueError, match="identity"):
        region.evaluate(engine, replace(result, free_energies=(0.,)))
    failed = engine.run(base, 0, tol=1e-6)
    with pytest.raises(ValueError, match="certified"):
        region.evaluate(engine, failed)


def test_h15_reactor_rejects_before_compilation_and_state_aliases_fail():
    template, arrays = reactor.model()
    with pytest.raises(ValueError, match="one or two"):
        BoundedChainRegion(template, tuple(BoundedCoordinate(f"s{t}c", 3, -1., 1.) for t in range(4, 19)))
    region, arrays = model()
    first = region.actions[0]
    for edge, coordinate, match in (("s4", 3, "aliased"), ("s4c", 3, "duplicate"),
                                    ("s6", 3, "nonterminal"), ("e4", 0, "nonterminal"),
                                    ("s4o", 4, "nonterminal")):
        invalid = replace(region, actions=(first, replace(first, edge=edge, coordinate=coordinate)))
        with pytest.raises(ValueError, match=match):
            invalid.compile(arrays=arrays)


def test_region_rejects_open_ports_and_mutated_parameter_bindings(monkeypatch):
    from cffg.core import PortRole

    region, arrays = model()
    with monkeypatch.context() as m:
        port = region.template.ports["e6"]
        m.setitem(region.template.ports, "e6", replace(port, role=PortRole.GLUE))
        with pytest.raises(ValueError, match="DATA ports"):
            region.compile(arrays=arrays)
    engine = region.compile(arrays=arrays)
    data = reactor.observations(horizon=2)
    result = engine.run(reactor.Binder(engine).bind(engine.init_state(), data), 1, tol=1e-6)
    params = result.state.params
    prior = engine.directed_chain_plan.prior.group
    path = dict(prior.parameter_paths)["m"]
    with monkeypatch.context() as m:
        m.setitem(params, path, params[path].astype(np.float32))
        with pytest.raises(ValueError, match="dtype"):
            region.evaluate(engine, result)
    with monkeypatch.context() as m:
        m.setitem(params, path, params[path]+1.)
        with pytest.raises(ValueError, match="certificate"):
            region.evaluate(engine, result)


@pytest.mark.slow
def test_reactor_comparison_reads_selected_edges_and_times_both_arms(monkeypatch):
    import sys
    from pathlib import Path
    from cffg import Engine

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    monkeypatch.setitem(sys.modules, "cstr_runtime_fixture", reactor)
    import cstr_bounded_comparison as comparison

    def unexpected(*args):
        pytest.fail("two selected actions should only read their named edges")

    monkeypatch.setattr(Engine, "gaussian_marginals", unexpected)
    report = comparison.compare()
    clipped, bounded = (report["arms"][key] for key in ("clipped", "bounded"))
    assert_allclose(bounded["objective"] - clipped["objective"], -report["log_mass"], atol=1e-8)
    for arm in (clipped, bounded):
        assert 295 <= arm["applied_K"] <= 302
        assert np.isfinite([arm["first_readout_seconds"], arm["warm_readout_seconds"]]).all()
