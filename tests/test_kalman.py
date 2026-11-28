"""Kalman tier (cff-4e5z): the continuous mirror of the HMM parity tier.

BP on the LGSSM tree must reproduce the RTS smoother exactly, and the Bethe
ledger must equal −log p(y) — the continuous tree identity. Also pins the
serialization round-trip and the vmap/colony contract for Gaussian arenas.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.compiler import CompileError, compile_template
from cffg.core.serialize import from_json, to_json
from cffg.engine import Engine, _replace
from cffg.models.kalman import kalman_smoother, lgssm

T, N, M = 5, 2, 1
F = np.array([[1.0, 0.3], [0.0, 0.9]])
Q = np.array([[0.05, 0.01], [0.01, 0.1]])
H = np.array([[1.0, 0.5]])
R = np.array([[0.2]])
M0 = np.array([0.5, -0.3])
P0 = np.array([[1.0, 0.2], [0.2, 0.8]])

YS = [np.array([0.9]), np.array([1.4]), np.array([0.7]), np.array([-0.2]), np.array([0.3])]


def run_engine(ys):
    t, arrays = lgssm(T, F, Q, H, R, M0, P0)
    engine = Engine(compile_template(t, arrays=arrays))
    state = engine.bind_data(engine.init_state(), {f"y{k}": ys[k] for k in range(T)})
    state, deltas = engine.run(
        state, 3 * T + 10, tol=1e-12
    ).require_converged_with_trace()
    assert deltas[-1] < 1e-12  # tree: exact convergence, not near-convergence
    return engine, state


def test_smoothed_marginals_match_rts():
    engine, state = run_engine(YS)
    means, covs, _ = kalman_smoother(YS, F, Q, H, R, M0, P0)
    marg = engine.gaussian_marginals(state)
    for k in range(T):
        mean, cov = marg[f"x{k}"]
        np.testing.assert_allclose(mean, means[k], atol=1e-8)
        np.testing.assert_allclose(cov, covs[k], atol=1e-8)
        if k < T - 1:  # every leg of the equality carries the same marginal
            np.testing.assert_allclose(marg[f"x{k}o"][0], means[k], atol=1e-8)


def test_bfe_is_negative_log_likelihood():
    engine, state = run_engine(YS)
    _, _, loglik = kalman_smoother(YS, F, Q, H, R, M0, P0)
    np.testing.assert_allclose(engine.bfe(state), -loglik, atol=1e-7)


def test_missing_observation():
    ys = list(YS)
    ys[2] = None
    engine, state = run_engine(ys)
    means, covs, loglik = kalman_smoother(ys, F, Q, H, R, M0, P0)
    marg = engine.gaussian_marginals(state)
    for k in range(T):
        np.testing.assert_allclose(marg[f"x{k}"][0], means[k], atol=1e-8)
        np.testing.assert_allclose(marg[f"x{k}"][1], covs[k], atol=1e-8)
    np.testing.assert_allclose(engine.bfe(state), -loglik, atol=1e-7)


def test_predictive_readout_on_observed_edge():
    # The observed edge's exterior marginal is the *cavity* predictive: the
    # forward message excludes y0's own likelihood (BP semantics), i.e. the
    # distribution of y0 given every OTHER observation — the innovation
    # quantity a model-scoring loop wants.
    engine, state = run_engine(YS)
    means_loo, covs_loo, _ = kalman_smoother([None, *YS[1:]], F, Q, H, R, M0, P0)
    mean_y, cov_y = engine.gaussian_marginals(state)["y0"]
    np.testing.assert_allclose(mean_y, H @ means_loo[0], atol=1e-8)
    np.testing.assert_allclose(cov_y, H @ covs_loo[0] @ H.T + R, atol=1e-8)


@pytest.mark.slow
def test_improper_edge_readout_is_explicit_nan():
    engine, state = run_engine(YS)
    bad = _replace(state, g_lam=-jnp.ones_like(state.g_lam))
    mean, cov = engine.gaussian_marginals(bad)["x0"]
    assert np.isnan(mean).all() and np.isnan(cov).all()


@pytest.mark.slow
def test_serialization_round_trip_compiles_and_agrees():
    t, arrays = lgssm(T, F, Q, H, R, M0, P0)
    back = from_json(to_json(t))
    engine = Engine(compile_template(back, arrays=arrays))
    state = engine.bind_data(engine.init_state(), {f"y{k}": YS[k] for k in range(T)})
    state = engine.run(state, 3 * T + 10).state
    means, _, _ = kalman_smoother(YS, F, Q, H, R, M0, P0)
    np.testing.assert_allclose(
        engine.gaussian_marginals(state)["x3"][0], means[3], atol=1e-8
    )


@pytest.mark.slow
def test_vmap_batch_equals_sequential():
    # Colony contract: Gaussian instances sharing a structure batch too.
    t, arrays = lgssm(T, F, Q, H, R, M0, P0)
    engine = Engine(compile_template(t, arrays=arrays))
    seqs = [YS, [y + 0.5 for y in YS], [None, *YS[1:]]]
    states = [
        engine.bind_data(engine.init_state(), {f"y{k}": s[k] for k in range(T)})
        for s in seqs
    ]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

    def one(st):
        msgs, g_eta, g_lam, ef_eta, _, _, *_ = engine._step(
            st.messages, st.g_eta, st.g_lam, st.g_obs, st.g_mask,
            st.ef_eta, st.ef_obs, st.ef_mask, st.params,
        )
        return _replace(st, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta)

    step_v = eqx.filter_vmap(one)
    for _ in range(3 * T + 10):
        batched = step_v(batched)
    for b in range(len(seqs)):
        solo = engine.run(states[b], 3 * T + 10).state
        np.testing.assert_allclose(
            np.asarray(batched.g_eta[b]), np.asarray(solo.g_eta), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(batched.g_lam[b]), np.asarray(solo.g_lam), atol=1e-10
        )


def test_lgssm_renders_at_every_level(viz_gallery):
    # The viz pipeline is domain-agnostic (it never reads cardinality), so the
    # Gaussian chain renders today; Gaussian-specific glyphs are cff-bxk4.
    import xml.etree.ElementTree as ET

    from cffg.viz import render_svg

    t, _ = lgssm(3, F, Q, H, R, M0, P0)
    for level in (0, 4):
        svg = render_svg(t, level=level)
        ET.fromstring(svg)
        viz_gallery(svg, tag=f"LGSSM T=3, level {level}")


def domain_lgssm(steps=3):
    state = np.eye(4)
    template, _ = lgssm(steps, state, .1 * state, state[[0, 2]], np.eye(2), np.zeros(4), state)
    return template


def _disjoint(left, right):
    return (min(left[2], right[2]) <= max(left[0], right[0])
            or min(left[3], right[3]) <= max(left[1], right[1]))


def _check_domain_geometry(scene, path):
    from cffg.viz.scene import Path, Terminal

    extent = path.domain_label.bounds
    assert np.isfinite(extent).all()
    x, y, u, v = extent
    assert 0 < x < u < scene.width and 0 < y < v < scene.height
    for box in scene.boxes().values():
        assert _disjoint(extent, (box.x - box.w / 2, box.y - box.h / 2,
                                  box.x + box.w / 2, box.y + box.h / 2))
    for wire in scene.by_ref(path.ref):
        if isinstance(wire, Path):
            for (a, b), (c, d) in zip(wire.points, wire.points[1:]):
                assert _disjoint(extent, (min(a, c) - .65, min(b, d) - .65,
                                          max(a, c) + .65, max(b, d) + .65))
    for item in scene.items:
        if isinstance(item, Terminal) and item.ring_radius is not None:
            r = item.ring_radius + .65
            assert _disjoint(extent, (item.x - r, item.y - r, item.x + r, item.y + r))


def test_lgssm_domain_metadata_and_geometry_at_every_level(viz_gallery):
    import xml.etree.ElementTree as ET
    from cffg.viz import build_scene, derive, to_svg
    from cffg.viz.diagram import TerminalEnd, TerminalKind
    from cffg.viz.scene import Path, Terminal

    template = domain_lgssm()
    diagram = derive(template)
    expected = {f"var:{name}": (name, spec.dim) for name, spec in template.variables.items()}
    for ref, (name, dimension) in expected.items():
        wire = diagram.wires[ref]
        assert (wire.family, wire.dimension, wire.label) == ("gaussian", dimension, name)
    assert all(not wire.dimension for wire in diagram.wires.values() if wire.id.startswith("blob:"))
    refs = {f"factor:{name}" for name in template.factors} | set(diagram.wires)
    refs |= {end.ref for wire in diagram.wires.values() for end in wire.ends if isinstance(end, TerminalEnd)}

    for level in range(5):
        scene = build_scene(template, level=level)
        assert {item.ref for item in scene.items} == refs
        marked = [item for item in scene.items if isinstance(item, Path) and item.domain_label]
        assert sorted(item.ref for item in marked) == sorted(expected)
        for path in marked:
            name, dimension = expected[path.ref]
            label = path.domain_label
            assert (label.ref, label.text, path.label) == (path.ref, {2: "ℝ²", 4: "ℝ⁴"}[dimension], name)
            assert label in scene.labels()
            _check_domain_geometry(scene, path)

        svg = to_svg(scene)
        assert svg == to_svg(build_scene(template, level=level))
        root = ET.fromstring(svg)
        for name in template.ports:
            mark, = [t for t in scene.by_ref(f"port:{name}") if isinstance(t, Terminal)]
            assert (mark.kind, mark.dimension, mark.label) == (TerminalKind.DATA, 2, name)
            group, = [g for g in root.findall(".//{*}g") if g.findtext("{*}title") == mark.ref]
            outer, inner = group.findall("{*}circle")
            assert outer.get("fill") == "none" and inner.get("fill") != "none"
            assert float(outer.get("r")) == mark.ring_radius > float(inner.get("r"))
            assert (inner.get("cx"), inner.get("cy")) == (outer.get("cx"), outer.get("cy"))
            assert any(node.text == "∂" and node.get("fill") == "#ffffff" for node in group.findall("{*}text"))
        if level in (0, 4):
            viz_gallery(svg, tag=f"LGSSM ℝ⁴/ℝ², level {level}")


@pytest.mark.parametrize("level", [0, 4])
def test_lgssm_growth_retains_early_domain_label_and_terminal_coordinates(level):
    from cffg.viz import build_scene, derive
    from cffg.viz.diagram import TerminalEnd
    from cffg.viz.scene import Path, Terminal

    old, new = [build_scene(domain_lgssm(steps), level=level) for steps in (3, 4)]
    factors = {"prior", "eq0", "trans1", "emis0", "eq1", "trans2", "emis1"}
    for name in factors:
        a, b = [scene.boxes()[f"factor:{name}"] for scene in (old, new)]
        assert (a.x, a.y) == (b.x, b.y), name
    for name in ("x0", "x0c", "x0o", "y0", "x1", "x1c", "x1o", "y1"):
        a, b = [next(item for item in scene.by_ref(f"var:{name}") if isinstance(item, Path))
                for scene in (old, new)]
        assert a.points == b.points and a.domain_label == b.domain_label, name
    for wire in derive(domain_lgssm()).wires.values():
        ends = [end for end in wire.ends if isinstance(end, TerminalEnd)]
        if not ends or wire.node_ends()[0].factor not in factors:
            continue
        positions = []
        for scene in (old, new):
            path = next(item for item in scene.by_ref(wire.id) if isinstance(item, Path))
            mark = next(t for t in scene.by_ref(ends[0].ref)
                        if isinstance(t, Terminal) and (t.x, t.y) == path.points[-1])
            positions.append((mark.x, mark.y))
        assert positions[0] == positions[1], wire.id


def test_lgssm_with_glue_ports_renders(viz_gallery):
    # A single Kalman slice the way the rollout machinery would see it:
    # glue in/out plus the emission branch — the composable unit.
    import xml.etree.ElementTree as ET

    from cffg.core.ir import PortRole, PortSpec
    from cffg.viz import build_scene, to_svg
    from cffg.viz.scene import EllipsisMark, Path, Terminal

    t, _ = lgssm(2, F, Q, H, R, M0, P0)
    del t.factors["prior"]
    t.ports["head"] = PortSpec("head", "x0", PortRole.GLUE)
    scene = build_scene(t, level=4)
    assert any(isinstance(item, EllipsisMark) for item in scene.by_ref("port:head"))
    assert all(not isinstance(item, Terminal) for item in scene.by_ref("port:head"))
    assert sum(isinstance(item, Path) and item.domain_label is not None for item in scene.by_ref("var:x0")) == 1
    svg = to_svg(scene)
    ET.fromstring(svg)
    viz_gallery(svg, tag="Kalman slice with glue port")


def test_data_port_elsewhere_rejected():
    import pytest

    from cffg.core.ir import FactorSpec, PortRole, PortSpec, Template, VariableSpec

    t = Template(name="bad")
    t.variables["x"] = VariableSpec("x", dim=1)
    t.factors["prior"] = FactorSpec(
        "prior", "gaussian", edges={"out": "x"}, param_blobs={"mean": "m", "cov": "v"}
    )
    from cffg.core.ir import BlobRef

    t.blobs = {"m": BlobRef("m", "m.npy"), "v": BlobRef("v", "v.npy")}
    t.ports["x"] = PortSpec("x", "x", PortRole.DATA)
    with pytest.raises(CompileError, match="registered conditioning role"):
        compile_template(t, arrays={"m": np.zeros(1), "v": np.eye(1)})


def test_bad_covariance_rejected():
    import pytest

    t, arrays = lgssm(2, F, Q, H, R, M0, P0)
    arrays["R"] = np.array([[-1.0]])
    with pytest.raises(CompileError, match="positive definite"):
        compile_template(t, arrays=arrays)
