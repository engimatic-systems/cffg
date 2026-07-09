"""Hybrid Gaussian-mixture plumbing (cff-pksr): compiler + engine wiring for
`gaussian_mixture` and `gaussian_mixture_linear`, checked against direct calls
to cffg.mixg (the math is proven separately in tests/test_mixg.py).

Coverage: the MixGroupPlan shape/param contract and every compile-time
rejection; a single-node fixed point where each written message and the BFE
match hand-assembled mixg terms; the K=1 reduction to plain gaussian_linear;
the vmap/colony contract; and a rendered diagram for the gallery.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import gauss, mixg
from cffg.compiler import CompileError, MixGroupPlan, compile_template
from cffg.core.ir import (
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
from cffg.engine import Engine, _replace
from cffg.rules import NODE_ENERGIES

jax.config.update("jax_enable_x64", True)


# --- template builders --------------------------------------------------------


def _blobs(*names):
    return {n: BlobRef(n, f"{n}.npy") for n in names}


def mixlin_template(name="mixlin", *, K=2, do=1, di=1, seed=0):
    """x: gaussian prior -- gaussian_mixture_linear --> y (dangling glue);
    selector s driven by a categorical prior. y vacuous (a GLUE half-edge)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(K, do, di))
    b = rng.normal(size=(K, do))
    # SPD covariances per regime.
    V = np.stack([np.eye(do) * (0.3 + 0.4 * k) for k in range(K)])
    mx = rng.normal(size=(di,))
    Vx = np.eye(di) * 0.8
    c = np.full(K, 1.0 / K)
    t = Template(
        name=name,
        variables={
            "x": VariableSpec("x", dim=di),
            "y": VariableSpec("y", dim=do),
            "s": VariableSpec("s", cardinality=K),
        },
        factors={
            "xprior": FactorSpec(
                "xprior", "gaussian", edges={"out": "x"},
                param_blobs={"mean": "mx", "cov": "Vx"},
            ),
            "mix": FactorSpec(
                "mix", "gaussian_mixture_linear",
                edges={"out": "y", "in": "x", "selector": "s"},
                param_blobs={"matrix": "A", "offset": "b", "cov": "V"},
            ),
            "sprior": FactorSpec(
                "sprior", "categorical", edges={"out": "s"},
                param_blobs={"statistic": "c"},
            ),
        },
        ports={"yout": PortSpec("yout", "y", PortRole.GLUE)},
        blobs=_blobs("mx", "Vx", "A", "b", "V", "c"),
    )
    arrays = {"mx": mx, "Vx": Vx, "A": A, "b": b, "V": V, "c": c}
    return t, arrays


# --- (a) compile plan: shapes, params, and every rejection --------------------


def test_mix_group_plan_shapes_and_params():
    t, arrays = mixlin_template(K=3, do=2, di=1)
    arena = compile_template(t, arrays=arrays)

    assert len(arena.mix_groups) == 1
    g = arena.mix_groups[0]
    assert isinstance(g, MixGroupPlan)
    assert g.type == "gaussian_mixture_linear"
    assert g.dims == (2, 1)
    assert g.k_sel == 3
    assert g.members == ("mix",)
    # Gaussian roles (out, in) -> [M, 2]; selector -> [M].
    assert g.g_gather.shape == (1, 2) and g.g_scatter.shape == (1, 2)
    assert g.s_gather.shape == (1,) and g.s_scatter.shape == (1,)
    # read-opposite/write-own: gather != scatter on every slot.
    assert not np.array_equal(g.g_gather, g.g_scatter)
    assert not np.array_equal(g.s_gather, g.s_scatter)

    # Per-regime parameter stacks keyed under the group name.
    W = arena.log_params[g.name + ":W"]
    A = arena.log_params[g.name + ":A"]
    b = arena.log_params[g.name + ":b"]
    assert W.shape == (1, 3, 2, 2)
    assert A.shape == (1, 3, 2, 1)
    assert b.shape == (1, 3, 2)
    # W is the precision of each regime's covariance.
    np.testing.assert_allclose(W[0, 0], np.linalg.inv(arrays["V"][0]), atol=1e-12)

    # The Gaussian out/in edges land in gauss_interior only when interior; here
    # x is interior (degree 2), the selector is an interior discrete edge.
    assert any(d == 1 for d, _ in arena.gauss_interior)  # x edge
    assert arena.interior_slot_pairs.shape[0] >= 1  # selector edge


def test_gaussian_mixture_prior_plan():
    K, d = 2, 2
    rng = np.random.default_rng(1)
    mean = rng.normal(size=(K, d))
    V = np.stack([np.eye(d) * (0.5 + k) for k in range(K)])
    t = Template(
        name="mixprior",
        variables={
            "y": VariableSpec("y", dim=d),
            "s": VariableSpec("s", cardinality=K),
        },
        factors={
            "mix": FactorSpec(
                "mix", "gaussian_mixture",
                edges={"out": "y", "selector": "s"},
                param_blobs={"mean": "m", "cov": "V"},
            ),
            "sprior": FactorSpec(
                "sprior", "categorical", edges={"out": "s"},
                param_blobs={"statistic": "c"},
            ),
        },
        ports={"yout": PortSpec("yout", "y", PortRole.GLUE)},
        blobs=_blobs("m", "V", "c"),
    )
    arena = compile_template(
        t, arrays={"m": mean, "V": V, "c": np.full(K, 1.0 / K)}
    )
    g = arena.mix_groups[0]
    assert g.type == "gaussian_mixture" and g.dims == (d,) and g.k_sel == K
    assert g.g_gather.shape == (1, 1)  # only the out edge
    assert arena.log_params[g.name + ":m"].shape == (1, K, d)
    assert arena.log_params[g.name + ":W"].shape == (1, K, d, d)


def test_non_spd_covariance_rejected():
    t, arrays = mixlin_template(K=2, do=1, di=1)
    arrays["V"] = arrays["V"].copy()
    arrays["V"][1] = np.array([[-1.0]])  # regime 1 not SPD
    with pytest.raises(CompileError, match="positive definite"):
        compile_template(t, arrays=arrays)


def test_non_symmetric_covariance_rejected():
    t, arrays = mixlin_template(K=2, do=2, di=1)
    arrays["V"] = arrays["V"].copy()
    arrays["V"][0] = np.array([[1.0, 0.3], [0.0, 1.0]])  # asymmetric
    with pytest.raises(CompileError, match="not symmetric"):
        compile_template(t, arrays=arrays)


def test_data_port_on_mixture_output_rejected():
    t, arrays = mixlin_template(K=2, do=1, di=1)
    t.ports["yout"] = PortSpec("yout", "y", PortRole.DATA)
    with pytest.raises(CompileError, match="observed mixture outputs are not"):
        compile_template(t, arrays=arrays)


def test_mean_field_bead_on_hybrid_rejected():
    t, arrays = mixlin_template(K=2)
    t.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "mix"))
    with pytest.raises(CompileError, match="hybrid beads are undefined"):
        compile_template(t, arrays=arrays)


def test_p_substitution_bead_on_hybrid_rejected():
    t, arrays = mixlin_template(K=2)
    t.constraints.append(
        ConstraintSpec(
            ConstraintKind.P_SUBSTITUTION, "mix",
            detail=(("substituted_role", "out"),),
        )
    )
    with pytest.raises(CompileError, match="hybrid beads are undefined"):
        compile_template(t, arrays=arrays)


def test_wrong_family_edge_rejected():
    # The 'in' role must be a Gaussian edge; make it discrete.
    t, arrays = mixlin_template(K=2, do=1, di=1)
    t.variables["x"] = VariableSpec("x", cardinality=3)
    # xprior was a gaussian leaf on x; drop it and expose x as a glue port so
    # only the family mismatch (not degree) is under test.
    del t.factors["xprior"]
    t.ports["xin"] = PortSpec("xin", "x", PortRole.GLUE)
    with pytest.raises(ModelError, match="role 'in' needs a gaussian edge"):
        compile_template(t, arrays=arrays)


# --- (b) single-node fixed point: messages and BFE vs hand-assembled mixg -----


def _converge(arena):
    engine = Engine(arena)
    state, deltas = engine.run(engine.init_state(), 60, tol=1e-13)
    assert deltas[-1] < 1e-11  # tree fragment: exact convergence
    return engine, state


def test_written_messages_match_mixg_calls():
    t, arrays = mixlin_template(K=3, do=2, di=1, seed=7)
    arena = compile_template(t, arrays=arrays)
    engine, state = _converge(arena)
    g = arena.mix_groups[0]
    do, di = g.dims
    W = np.asarray(state.params[g.name + ":W"])
    A = np.asarray(state.params[g.name + ":A"])
    b = np.asarray(state.params[g.name + ":b"])
    ge, gl = np.asarray(state.g_eta), np.asarray(state.g_lam)
    msgs = np.asarray(state.messages)

    mu_e = msgs[g.s_gather][:, : g.k_sel]
    e_o, l_o = ge[g.g_gather[:, 0]][:, :do], gl[g.g_gather[:, 0]][:, :do, :do]
    e_i, l_i = ge[g.g_gather[:, 1]][:, :di], gl[g.g_gather[:, 1]][:, :di, :di]

    # → out (forward collapse)
    eta_out, lam_out = mixg.mixlin_forward(e_i, l_i, mu_e, A, b, W)
    np.testing.assert_allclose(ge[g.g_scatter[:, 0]][:, :do], eta_out, atol=1e-10)
    np.testing.assert_allclose(gl[g.g_scatter[:, 0]][:, :do, :do], lam_out, atol=1e-10)
    # → in (backward collapse, the weight trap)
    eta_in, lam_in = mixg.mixlin_backward(e_o, l_o, mu_e, A, b, W)
    np.testing.assert_allclose(ge[g.g_scatter[:, 1]][:, :di], eta_in, atol=1e-10)
    np.testing.assert_allclose(gl[g.g_scatter[:, 1]][:, :di, :di], lam_in, atol=1e-10)
    # → selector (exact)
    to_sel = mixg.mixlin_to_selector(e_o, l_o, e_i, l_i, A, b, W)
    np.testing.assert_allclose(msgs[g.s_scatter][:, : g.k_sel], to_sel, atol=1e-10)


def test_bfe_equals_hand_assembled_terms():
    t, arrays = mixlin_template(K=3, do=2, di=1, seed=3)
    arena = compile_template(t, arrays=arrays)
    engine, state = _converge(arena)
    ge, gl = state.g_eta, state.g_lam
    msgs = state.messages
    g = arena.mix_groups[0]
    do, di = g.dims

    # Hybrid node energy (mixg).
    W = state.params[g.name + ":W"]
    A, b = state.params[g.name + ":A"], state.params[g.name + ":b"]
    mu_e = msgs[g.s_gather][:, : g.k_sel]
    e_o, l_o = ge[g.g_gather[:, 0]][:, :do], gl[g.g_gather[:, 0]][:, :do, :do]
    e_i, l_i = ge[g.g_gather[:, 1]][:, :di], gl[g.g_gather[:, 1]][:, :di, :di]
    mix_term = mixg.mixlin_energy(e_o, l_o, e_i, l_i, mu_e, A, b, W).sum()

    # Gaussian prior node energy (xprior). Its incoming is mix's message to x.
    gp = next(gg for gg in arena.gauss_groups if gg.type == "gaussian")
    (d,) = gp.dims
    prior_term = gauss.prior_energy(
        ge[gp.gather[:, 0]][:, :d], gl[gp.gather[:, 0]][:, :d, :d],
        state.params[gp.name + ":m"], state.params[gp.name + ":W"],
    ).sum()

    # Categorical node energy (sprior).
    cg = next(gg for gg in arena.groups if gg.type == "categorical")
    cat_term = NODE_ENERGIES["categorical"](
        msgs[cg.gather], state.params[cg.name], cg.cards
    ).sum()

    # Interior edge entropies: Gaussian x edge + discrete selector edge.
    gauss_H = jnp.asarray(0.0)
    for dd, pairs in arena.gauss_interior:
        gauss_H = gauss_H + gauss.edge_entropy(
            gl[pairs[:, 0], :dd, :dd], gl[pairs[:, 1], :dd, :dd]
        ).sum()
    pairs = arena.interior_slot_pairs
    lq = msgs[pairs[:, 0]] + msgs[pairs[:, 1]]
    lq = lq - jax.scipy.special.logsumexp(lq, axis=-1, keepdims=True)
    q = jnp.exp(lq)
    disc_H = -jnp.where(q > 1e-12, q * lq, 0.0).sum()

    hand = float(mix_term + prior_term + cat_term + gauss_H + disc_H)
    np.testing.assert_allclose(engine.bfe(state), hand, atol=1e-9)


# --- (c) K=1 reduces to plain gaussian_linear ---------------------------------


def _linear_chain(mix: bool):
    """x: gaussian prior -- (mix K=1 | gaussian_linear) --> y: gaussian prior.
    Both variables fully connected; compare the two engines' marginals."""
    rng = np.random.default_rng(11)
    A = rng.normal(size=(1, 1))
    b = rng.normal(size=(1,))
    V = np.array([[0.4]])
    mx, Vx = np.array([0.2]), np.array([[0.9]])
    my, Vy = np.array([-0.5]), np.array([[0.6]])
    variables = {
        "x": VariableSpec("x", dim=1),
        "y": VariableSpec("y", dim=1),
    }
    factors = {
        "xprior": FactorSpec(
            "xprior", "gaussian", edges={"out": "x"},
            param_blobs={"mean": "mx", "cov": "Vx"},
        ),
        "yprior": FactorSpec(
            "yprior", "gaussian", edges={"out": "y"},
            param_blobs={"mean": "my", "cov": "Vy"},
        ),
    }
    blobs = _blobs("mx", "Vx", "my", "Vy", "A", "b", "V")
    arrays = {"mx": mx, "Vx": Vx, "my": my, "Vy": Vy}
    if mix:
        variables["s"] = VariableSpec("s", cardinality=1)
        factors["link"] = FactorSpec(
            "link", "gaussian_mixture_linear",
            edges={"out": "y", "in": "x", "selector": "s"},
            param_blobs={"matrix": "A", "offset": "b", "cov": "V"},
        )
        factors["sprior"] = FactorSpec(
            "sprior", "categorical", edges={"out": "s"},
            param_blobs={"statistic": "c"},
        )
        blobs.update(_blobs("c"))
        arrays.update({"A": A[None], "b": b[None], "V": V[None], "c": np.array([1.0])})
    else:
        factors["link"] = FactorSpec(
            "link", "gaussian_linear",
            edges={"out": "y", "in": "x"},
            param_blobs={"matrix": "A", "offset": "b", "cov": "V"},
        )
        arrays.update({"A": A, "b": b, "V": V})
    t = Template(name="chain", variables=variables, factors=factors, blobs=blobs)
    return compile_template(t, arrays=arrays)


def test_k1_reduces_to_gaussian_linear():
    eng_m, st_m = _converge(_linear_chain(mix=True))
    eng_l, st_l = _converge(_linear_chain(mix=False))
    mm, ml = eng_m.gaussian_marginals(st_m), eng_l.gaussian_marginals(st_l)
    for v in ("x", "y"):
        np.testing.assert_allclose(mm[v][0], ml[v][0], atol=1e-9)
        np.testing.assert_allclose(mm[v][1], ml[v][1], atol=1e-9)
    np.testing.assert_allclose(eng_m.bfe(st_m), eng_l.bfe(st_l), atol=1e-9)


# --- (d) vmap / colony contract -----------------------------------------------


def test_vmap_batch_equals_sequential():
    t, arrays = mixlin_template(K=3, do=2, di=1, seed=5)
    engine = Engine(compile_template(t, arrays=arrays))
    # Two instances that differ only in their (learned) parameter blobs.
    t2, arrays2 = mixlin_template(K=3, do=2, di=1, seed=9)
    engine2 = Engine(compile_template(t2, arrays=arrays2))
    states = [engine.init_state(), engine2.init_state()]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

    def one(st):
        msgs, g_eta, g_lam, ef_eta = engine._step(
            st.messages, st.g_eta, st.g_lam, st.g_obs, st.g_mask,
            st.ef_eta, st.ef_obs, st.ef_mask, st.params,
        )
        return _replace(st, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta)

    step_v = eqx.filter_vmap(one)
    for _ in range(60):
        batched = step_v(batched)
    solo0, _ = engine.run(states[0], 60)
    solo1, _ = engine2.run(states[1], 60)
    for b, solo in enumerate((solo0, solo1)):
        np.testing.assert_allclose(
            np.asarray(batched.g_eta[b]), np.asarray(solo.g_eta), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(batched.messages[b]), np.asarray(solo.messages), atol=1e-10
        )


# --- (e) viz ------------------------------------------------------------------


def test_hybrid_glyphs_and_diagram(viz_gallery):
    from cffg.viz import derive
    from cffg.viz.diagram import GLYPHS

    assert GLYPHS["gaussian_mixture"] == "𝒩M"
    assert GLYPHS["gaussian_mixture_linear"] == "𝒩MA"
    t, _ = mixlin_template(K=3, do=2, di=1)
    d = derive(t)
    assert d.nodes["mix"].glyph == "𝒩MA"
    viz_gallery(t, tag="gaussian_mixture_linear switching node")
