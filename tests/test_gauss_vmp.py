"""gaussian_prec plumbing (cff-7c5o): the engine's first mixed Gaussian/EF
client. A scalar Gaussian node N(x; m, τ⁻¹) whose precision τ lives on an
ef:gamma edge, under a mean-field q(x)q(τ) VMP bead.

The math (cffg.gauss.prec_*, DERIVATIONS.md D4) is proven separately; this file
checks the compiler/engine wiring: the PrecGroupPlan shape/param contract and
every rejection; the conjugate fixed point against the exact Gamma posterior;
BFE monotonicity under VMP; the mixed-ledger identity BFE = −log evidence via
1-D quadrature; an f32 smoke; and a gallery diagram.
"""

from math import log, pi

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from cffg.compiler import CompileError, PrecGroupPlan, compile_template
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.validate import ModelError
from cffg.engine import Engine


# --- template builder ---------------------------------------------------------


def gamma_gauss_prec(
    n: int = 3,
    a0: float = 2.0,
    b0: float = 1.5,
    xs=(1.0, -0.5, 2.0),
    m: float = 0.0,
    xcov: float = 1e-10,
) -> tuple[Template, dict]:
    """Gamma(a0, b0) prior on a shared precision τ, fanned out through an
    equality chain to n gaussian_prec nodes N(x_i; m, τ⁻¹); each x_i edge is
    pinned by a tight gaussian prior (mean xs[i], cov xcov) — effectively an
    observation x_i. Mirrors the EF gamma_poisson fan-out (tests/test_ef_engine)."""
    t = Template(name=f"gamma_gauss_prec_{n}")
    t.variables["tau0"] = VariableSpec("tau0", ef="gamma")
    t.factors["prior"] = FactorSpec(
        "prior", "gamma", edges={"out": "tau0"}, param_blobs={"shape": "a0", "rate": "b0"}
    )
    t.blobs["a0"] = BlobRef("a0", "blobs/a0.npy")
    t.blobs["b0"] = BlobRef("b0", "blobs/b0.npy")
    t.blobs["m"] = BlobRef("m", "blobs/m.npy")
    arrays: dict = {"a0": np.array(a0), "b0": np.array(b0), "m": np.array(m)}

    tail = "tau0"
    for i in range(n):
        if i < n - 1:  # branch τ through an equality node (FFG degree ≤ 2)
            br, nxt = f"tau_b{i}", f"tau{i + 1}"
            t.variables[br] = VariableSpec(br, ef="gamma")
            t.variables[nxt] = VariableSpec(nxt, ef="gamma")
            t.factors[f"eq{i}"] = FactorSpec(
                f"eq{i}", "equality", edges={"a": tail, "b": br, "c": nxt}
            )
            prec, tail = br, nxt
        else:
            prec = tail
        xv = f"x{i}"
        t.variables[xv] = VariableSpec(xv, dim=1)
        t.factors[f"gp{i}"] = FactorSpec(
            f"gp{i}", "gaussian_prec", edges={"out": xv, "prec": prec}, param_blobs={"mean": "m"}
        )
        t.factors[f"xprior{i}"] = FactorSpec(
            f"xprior{i}", "gaussian", edges={"out": xv},
            param_blobs={"mean": f"xm{i}", "cov": "xcov"},
        )
        t.blobs[f"xm{i}"] = BlobRef(f"xm{i}", f"blobs/xm{i}.npy")
        arrays[f"xm{i}"] = np.array([xs[i]])
    t.blobs["xcov"] = BlobRef("xcov", "blobs/xcov.npy")
    arrays["xcov"] = np.array([[xcov]])
    return t, arrays


def _exact_post(a0, b0, xs, m):
    n = len(xs)
    return a0 + n / 2.0, b0 + sum((x - m) ** 2 for x in xs) / 2.0


# --- (a) compile plan shapes + rejections -------------------------------------


def test_prec_group_plan_shapes():
    t, arrays = gamma_gauss_prec(3)
    a = compile_template(t, arrays=arrays)
    assert len(a.prec_groups) == 1
    g = a.prec_groups[0]
    assert isinstance(g, PrecGroupPlan)
    assert g.members == ("gp0", "gp1", "gp2")
    assert g.x_gather_both.shape == (3, 2)
    assert g.tau_gather_both.shape == (3, 2)
    assert g.x_scatter.shape == (3,) and g.tau_scatter.shape == (3,)
    # marginal gather covers BOTH slots (own + opposite), and own == scatter.
    np.testing.assert_array_equal(g.x_gather_both[:, 0], g.x_scatter)
    np.testing.assert_array_equal(g.tau_gather_both[:, 0], g.tau_scatter)
    assert not np.array_equal(g.x_gather_both[:, 0], g.x_gather_both[:, 1])
    # scalar means stacked [M, 1]; all share the one m blob.
    means = a.log_params[g.name + ":m"]
    assert means.shape == (3, 1)
    np.testing.assert_allclose(means, 0.0)
    # the τ chain and the x edges are interior; the x edges land in gauss_interior.
    (fam, pairs) = a.ef_interior[0]
    assert fam == "gamma" and pairs.shape == (5, 2)  # tau0, tau_b0, tau1, tau_b1, tau2
    assert any(d == 1 for d, _ in a.gauss_interior)  # x edges


def test_wrong_ef_family_on_prec_rejected():
    t, arrays = gamma_gauss_prec(1)
    t.variables["tau0"] = VariableSpec("tau0", ef="beta")  # not gamma
    with pytest.raises(ModelError, match="needs an ef='gamma' edge"):
        compile_template(t, arrays=arrays)


def test_non_gaussian_out_rejected():
    """The out edge must be Gaussian (family check), not e.g. discrete."""
    t, arrays = gamma_gauss_prec(1)
    t.variables["x0"] = VariableSpec("x0", cardinality=2)  # discrete, not gaussian
    with pytest.raises(ModelError, match="needs a gaussian edge"):
        compile_template(t, arrays=arrays)


def test_vector_out_rejected():
    t, arrays = gamma_gauss_prec(1)
    t.variables["x0"] = VariableSpec("x0", dim=2)  # vector precision-mean prior
    arrays["xm0"] = np.array([1.0, 2.0])
    arrays["xcov"] = np.eye(2)
    with pytest.raises(CompileError, match="gaussian_prec is scalar"):
        compile_template(t, arrays=arrays)


def test_data_port_on_out_rejected():
    """An observed gaussian_prec output must route through an emission, not a
    Gaussian DATA clamp on the node's own out edge."""
    t, arrays = gamma_gauss_prec(1)
    del t.factors["xprior0"]  # free the x0 edge (degree drops to 1)
    del t.blobs["xm0"]
    t.ports["obs0"] = PortSpec("obs0", "x0", PortRole.DATA)
    with pytest.raises(CompileError, match="observed gaussian_prec outputs"):
        compile_template(t, arrays=arrays)


def test_data_port_on_prec_rejected():
    """A DATA port on the ef:gamma prec edge is a clamp, not conditioning —
    rejected by the EF DATA-port check."""
    t, arrays = gamma_gauss_prec(1)
    del t.factors["prior"]  # free tau0 (degree drops to 1)
    del t.blobs["a0"]
    del t.blobs["b0"]
    t.ports["taup"] = PortSpec("taup", "tau0", PortRole.DATA)
    with pytest.raises(CompileError, match="EF DATA ports must clamp"):
        compile_template(t, arrays=arrays)


def test_mean_field_bead_rejected():
    t, arrays = gamma_gauss_prec(1)
    t.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "gp0"))
    with pytest.raises(CompileError, match="hybrid beads are undefined"):
        compile_template(t, arrays=arrays)


# --- (b) conjugacy fixed point ------------------------------------------------


def test_conjugate_fixed_point():
    a0, b0, m = 2.0, 1.5, 0.3
    xs = (1.0, -0.5, 2.0, 0.7)
    t, arrays = gamma_gauss_prec(4, a0, b0, xs, m)
    eng = Engine(compile_template(t, arrays=arrays))
    st, deltas = eng.run(
        eng.init_state(), 40, tol=1e-13
    ).require_converged_with_trace()
    assert np.isfinite(deltas[-1]) and deltas[-1] < 1e-12  # converged, not budget-exhausted
    post = eng.ef_marginals(st)
    a_n, b_n = _exact_post(a0, b0, xs, m)
    for name, mrg in post.items():  # every τ edge is the same variable
        assert mrg.family == "gamma" and mrg.proper, name
        np.testing.assert_allclose(mrg.params, (a_n, b_n), rtol=1e-6)


# --- (c) monotone BFE ---------------------------------------------------------


def test_bfe_monotone_decrease():
    """The ticket's 'VMP monotonically decreases BFE' acceptance. A single
    gaussian_prec node with a genuinely uncertain x (loose prior, cov ~ 1.0):
    synchronous flooding updates q(x) and q(τ) together and the mean-field VFE
    is non-increasing every sweep, to machine precision — flooding is monotone
    when there is one mean-field bead pair (no cross-node Jacobi coupling; see
    test_bfe_multinode_flooding_oscillates for the multi-node story)."""
    a0, b0, m = 3.0, 2.0, 0.0
    t, arrays = gamma_gauss_prec(1, a0, b0, (1.0,), m, xcov=1.0)
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.init_state()
    trace = []
    for _ in range(30):
        st = eng.step(st)
        trace.append(eng.bfe(st))
    trace = np.array(trace)
    assert np.all(np.isfinite(trace))  # one bead pair: proper from the first sweep
    diffs = np.diff(trace)
    assert np.all(diffs <= 1e-9), f"non-monotone BFE step: {diffs.max()}\ntrace={trace}"
    assert trace[-1] < trace[0]  # it genuinely descends (x/τ both uncertain)


def test_bfe_multinode_flooding_oscillates():
    """Honest record (cff-7c5o): with several gaussian_prec nodes sharing one
    precision through an equality fan-out, the early sweeps are +inf while the
    τ chain is still improper (correct loudness), then the ledger goes finite.
    The finite tail CONVERGES to the mean-field fixed point but is NOT strictly
    monotone: synchronous flooding is Jacobi iteration over the mean-field
    coordinates, which oscillates. Sequential (coordinate) VMP would be
    monotone; the engine only floods. The step is small and bounded, and both
    the fixed point and the −log-evidence ledger are unaffected."""
    a0, b0, m = 3.0, 2.0, 0.0
    xs = (1.0, -0.5, 2.0, 0.7)
    t, arrays = gamma_gauss_prec(4, a0, b0, xs, m, xcov=1.0)
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.init_state()
    trace = []
    for _ in range(60):
        st = eng.step(st, damp=0.5)  # damping shrinks but never kills the oscillation
        trace.append(eng.bfe(st))
    trace = np.array(trace)
    assert np.isinf(trace[0]) and trace[0] > 0  # improper τ chain: loud
    finite = np.isfinite(trace)
    assert finite[-1] and finite.sum() >= 10
    tail = trace[finite]
    # Converges (last sweeps flat) even though not monotone en route.
    assert abs(tail[-1] - tail[-10]) < 1e-3
    diffs = np.diff(tail)
    assert diffs.max() > 1e-9  # genuinely non-monotone — the reported finding
    assert diffs.max() < 5e-2  # but small and bounded (a flooding artifact)


# --- (d) mixed-ledger identity ------------------------------------------------


def _neg_log_evidence_quad(a0, b0, x_obs, m):
    """−log ∫ Gamma(τ; a0, b0) N(x_obs; m, τ⁻¹) dτ by 1-D quadrature."""
    from scipy.integrate import quad
    from scipy.special import gammaln

    log_c = a0 * log(b0) - gammaln(a0) - 0.5 * log(2.0 * pi)

    def integrand(tau):
        log_f = (
            log_c
            + (a0 - 0.5) * log(tau)
            - tau * (b0 + 0.5 * (x_obs - m) ** 2)
        )
        return np.exp(log_f)

    val, _ = quad(integrand, 0.0, np.inf)
    return -log(val)


def test_mixed_ledger_identity():
    a0, b0, m, x_obs = 2.5, 1.5, 0.4, 1.3
    t, arrays = gamma_gauss_prec(1, a0, b0, (x_obs,), m)
    eng = Engine(compile_template(t, arrays=arrays))
    st, deltas = eng.run(
        eng.init_state(), 30, tol=1e-13
    ).require_converged_with_trace()
    assert np.isfinite(eng.bfe(st))
    np.testing.assert_allclose(
        eng.bfe(st), _neg_log_evidence_quad(a0, b0, x_obs, m), rtol=1e-6
    )


# --- (e) f32 smoke ------------------------------------------------------------


def test_f32_smoke():
    import jax.numpy as jnp
    from cffg.engine import _replace

    t, arrays = gamma_gauss_prec(1)
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.init_state()
    # Cast every float array to float32; one sweep + bfe must stay finite.
    st = _replace(
        st,
        messages=st.messages.astype(jnp.float32),
        params={k: v.astype(jnp.float32) for k, v in st.params.items()},
        g_eta=st.g_eta.astype(jnp.float32),
        g_lam=st.g_lam.astype(jnp.float32),
        g_obs=st.g_obs.astype(jnp.float32),
        g_mask=st.g_mask.astype(jnp.float32),
        ef_eta=st.ef_eta.astype(jnp.float32),
        ef_obs=st.ef_obs.astype(jnp.float32),
        ef_mask=st.ef_mask.astype(jnp.float32),
    )
    st = eng.step(st)
    assert np.all(np.isfinite(np.asarray(st.ef_eta)))
    assert np.isfinite(eng.bfe(st))


# --- (f) viz gallery ----------------------------------------------------------


def test_gaussian_prec_glyph_and_diagram(viz_gallery):
    from cffg.viz.diagram import GLYPHS

    assert GLYPHS["gaussian_prec"] == "𝒩τ" and len(GLYPHS["gaussian_prec"]) <= 3
    t, _ = gamma_gauss_prec(3)
    viz_gallery(t, tag="gaussian_prec: shared τ over 3 pinned scalar nodes")
