"""Known-input Gaussian action and correlated state carry through the public facade."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from cffg import RollingAgent, RunStatus, save_bundle
from cffg.core import (BlobRef, FactorSpec, GaussianFilterSpec, GaussianPlanSpec,
                       ModelError, PortRole, PortSpec, Template, VariableSpec,
                       fan_out, observe_interior, validate)
from cffg.core.compose import scoped
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.runtime_types import InferenceError, RunResult


def window(plan):
    port = "target" if plan else "reading"
    t = Template(name=port)
    t.variables = {n: VariableSpec(n, dim=d) for n, d in (("x", 3), ("next", 3), ("y", 1))}
    def factor(name, kind, edges, **blobs):
        t.factors[name] = FactorSpec(name, kind, edges=edges, param_blobs=blobs)
    factor("prior", "gaussian", {"out": "x"}, mean="m", cov="P")
    factor("motion", "gaussian_linear", {"in": "x", "out": "next"},
           matrix="A", offset="b", cov="Q")
    factor("sensor", "gaussian_linear", {"in": "next", "out": "y"},
           matrix="H", offset="z", cov="R")
    arrays = dict(m=np.zeros(3), P=np.array([[1., .6, 0], [.6, 1., 0], [0, 0, 1.]]),
                  A=np.diag([1., 1., 0.]), b=np.zeros(3), Q=np.diag([.2, .2, 1.]),
                  H=np.array([[1., 0., 0.]]), z=np.zeros(1), R=np.array([[.3]]))
    args = (("x1", "x2", "u"), "prior", ("u",))
    if plan:
        arrays["A"][:, 2] = [1., .5, 0]
        t.agent = GaussianPlanSpec(*args, "x", (2,), (port,))
    else:
        arrays["B"] = np.array([[1.], [.5], [0.]])
        t.agent = GaussianFilterSpec(*args, "next", "motion", "B", port)
    t.ports[port] = PortSpec(port, "y", PortRole.DATA)
    t.blobs = {k: BlobRef(k, k + ".npy") for k in arrays}
    return t, arrays


def bundles(root):
    paths = [root / name for name in ("plan.cffgz", "filter.cffgz")]
    for plan, path in zip((True, False), paths, strict=True):
        save_bundle(path, *window(plan))
    return paths


def test_declaration_identity_scoping_and_dedicated_parameters():
    for plan in (True, False):
        t, _ = window(plan)
        assert from_json(to_json(t)).agent == t.agent
        scoped_t = scoped(t, "left")
        validate(scoped_t)
        assert scoped_t.agent.prior == "left/prior"
        assert scoped_t.agent.coordinates == t.agent.coordinates
        if plan:
            original = structure_hash(t)
            t.agent = replace(t.agent, indices=(1,))
            assert structure_hash(t) != original
            t.agent = replace(t.agent, action="y", indices=(0,))
            with pytest.raises(ModelError, match="cavity predictive"):
                validate(t)
            t.agent = replace(t.agent, action="x", indices=(True,))
            with pytest.raises(ModelError, match="indices"):
                validate(t)
        else:
            t.factors["motion"].param_blobs["offset"] = "m"
            with pytest.raises(ModelError, match="dedicated"):
                validate(t)


def tapped_window(plan, edge, *, indirect=False):
    t, arrays = window(plan)
    spec, t.agent = t.agent, None
    t.ports.clear()
    # Close the old noisy sensor with N(0, R), then observe the specified state exactly.
    t.factors["terminal"] = FactorSpec("terminal", "gaussian", {"out": "y"},
                                       param_blobs={"mean": "z", "cov": "R"})
    t, port = observe_interior(t, edge)
    if indirect:
        t, (branch,) = fan_out(t, port, 1)
        del t.ports[branch]
        t.blobs["mu_end"] = BlobRef("mu_end", "mu_end.npy")
        arrays["mu_end"] = np.zeros(3)
        t.factors["end"] = FactorSpec("end", "gaussian", {"out": branch},
                                      param_blobs={"mean": "mu_end", "cov": "Q"})
        for name, factor in t.factors.items():
            if factor.type == "equality":
                a, b, c = factor.edges.values()
                t.factors[name] = replace(factor, edges={"b": a, "c": b, "a": c})
    t.ports["context"] = replace(t.ports.pop(port), name="context")
    t.agent = (replace(spec, preview_ports=("context",)) if plan else
               replace(spec, observation_port="context"))
    return t, arrays


@pytest.mark.parametrize("indirect", [False, True])
def test_filter_rejects_a_tap_that_clamps_the_carried_posterior(indirect):
    t, _ = tapped_window(False, "next", indirect=indirect)
    for candidate in (t, from_json(to_json(t))):
        with pytest.raises(ModelError, match="observation 'context' clamps posterior 'next'"):
            validate(candidate)


def test_filter_carries_process_uncertainty_after_a_tap_on_the_previous_state(tmp_path):
    plan, filt = tmp_path / "plan.cffgz", tmp_path / "filter.cffgz"
    save_bundle(plan, *tapped_window(True, "x"))
    save_bundle(filt, *tapped_window(False, "x"))
    agent = RollingAgent.from_gaussian_bundles(plan, filt)
    decision = agent.propose({"context": [1., 2., .5]})
    np.testing.assert_array_equal(decision.command, [.5])
    mean, cov = agent.observe(decision.command, [1., 2., .5], tick=0)
    # After Q, the terminal scalar likelihood has variance R + R = .6.
    np.testing.assert_allclose(mean, [1.125, 2.25, 0], atol=2e-6)
    np.testing.assert_allclose(cov, np.diag([.15, .2, 1.]), atol=2e-6)


@pytest.mark.parametrize("x64", [pytest.param(True, marks=pytest.mark.slow), False])
def test_applied_affine_input_and_pending_or_committed_restart(tmp_path, x64):
    with jax.enable_x64(x64):
        paths = bundles(tmp_path)
        agent = RollingAgent.from_gaussian_bundles(*paths)
        preview = {"target": [2.]}
        decision = agent.propose(preview)
        np.testing.assert_allclose(decision.command, [.8], atol=2e-6)
        energy = .5 * (np.log(2 * np.pi * 2.5) + 4 / 2.5)
        np.testing.assert_allclose(decision.free_energy, energy, rtol=16 * np.finfo(decision.command.dtype).eps)
        decision.command[:] = 99
        with pytest.raises(ValueError, match="preview changed"):
            agent.propose({"target": [-1.]})

        path = tmp_path / "agent.npz"
        agent.checkpoint(path)
        other = RollingAgent.from_gaussian_bundles(*paths)
        other.restore(path)
        np.testing.assert_array_equal(other.propose(preview).command, agent.propose(preview).command)
        for item in (agent, other):
            with pytest.raises(ValueError, match="applied action"):
                item.observe([np.nan], [.5], tick=0)
            m, P = item.observe([-.5], [.5], tick=0)
            np.testing.assert_allclose(m, [.3, .15, 0], atol=2e-6)
            np.testing.assert_allclose(P, [[.24,.12,0],[.12,.96,0],[0,0,1.]], atol=2e-6)
            with pytest.raises(ValueError, match="outstanding"):
                item.observe([-.5], [.5], tick=0)
        agent.checkpoint(path)
        resumed = other
        resumed.restore(path)
        for item in (agent, resumed):
            item.propose({"target": None})
            item.observe([.25], None, tick=1)
        for lhs, rhs in zip(agent.belief, resumed.belief, strict=True):
            np.testing.assert_array_equal(lhs, rhs)
        np.testing.assert_allclose(agent.belief[0], [.55, .275, 0], atol=2e-6)


def test_different_control_map_and_corrupt_covariance_are_atomic(tmp_path):
    paths = bundles(tmp_path)
    original = RollingAgent.from_gaussian_bundles(*paths)
    path = tmp_path / "agent.npz"
    original.propose({"target": [1.]})
    original.checkpoint(path)
    t, arrays = window(False)
    arrays["B"] *= 2
    save_bundle(paths[1], t, arrays)
    other = RollingAgent.from_gaussian_bundles(*paths)
    with pytest.raises(ValueError, match="incompatible"):
        other.restore(path)
    with np.load(path) as saved:
        damaged = {k:v.copy() for k,v in saved.items()}
    damaged["cov"][1] = damaged["cov"][0]
    np.savez(path, **damaged)
    with pytest.raises(ValueError, match="incompatible"):
        original.restore(path)
    assert original.tick == other.tick == 0
    original.observe([0.], None, tick=0)


def test_unresolved_gaussian_prior_at_execution_precision(tmp_path):
    paths = bundles(tmp_path)
    t, arrays = window(True)
    arrays["P"][0, 1] = arrays["P"][1, 0] = 1. - 2.**-22
    save_bundle(paths[0], t, arrays)
    with jax.enable_x64(False), pytest.raises(ValueError, match="covariance is unresolved"):
        RollingAgent.from_gaussian_bundles(*paths)


def test_exhausted_plan_preserves_the_boundary_and_rejects_feedback(tmp_path, monkeypatch):
    agent = RollingAgent.from_gaussian_bundles(*bundles(tmp_path), n_sweeps=1)
    before = agent.belief
    def exhausted(state, budget, **kwargs):
        return RunResult(state, RunStatus.BUDGET_EXHAUSTED, budget, (), (), 1.)

    monkeypatch.setattr(agent.plan, "run", exhausted)
    with pytest.raises(InferenceError, match="inference budget_exhausted"):
        agent.propose({"target": [2.]})
    with pytest.raises(ValueError, match="outstanding"):
        agent.observe([-.5], [.5], tick=0)
    for current, expected in zip(agent.belief, before, strict=True):
        np.testing.assert_array_equal(current, expected)
    assert agent.tick == 0


@pytest.mark.slow
def test_public_gaussian_restart_in_a_fresh_interpreter(tmp_path):
    import os
    import subprocess
    import sys
    from textwrap import dedent

    paths = bundles(tmp_path)
    agent = RollingAgent.from_gaussian_bundles(*paths)
    agent.propose({"target": [2.]})
    agent.checkpoint(tmp_path / "pending.npz")
    agent.observe([-.5], [.5], tick=0)
    expected = agent.propose({"target": [.4]})

    script = dedent("""\
        import numpy as np
        from cffg import RollingAgent

        agent = RollingAgent.from_gaussian_bundles("plan.cffgz", "filter.cffgz")
        agent.restore("pending.npz")
        assert agent.propose({"target": [2.]}).tick == 0
        m, P = agent.observe([-.5], [.5], tick=0)
        decision = agent.propose({"target": [.4]})
        np.savez("next.npz", mean=m, cov=P, command=decision.command, energy=decision.free_energy)
    """)
    env = {**os.environ, "JAX_ENABLE_X64": str(int(jax.config.jax_enable_x64))}
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, check=True)
    with np.load(tmp_path / "next.npz") as result:
        mean, cov = agent.belief
        for name, reference in (("mean", mean), ("cov", cov), ("command", expected.command),
                                ("energy", expected.free_energy)):
            np.testing.assert_array_equal(result[name], reference)
