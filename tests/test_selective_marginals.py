"""Shape-batched readouts against independent, one-edge-at-a-time oracles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.benchmarks.scheduling import hmm_chain
from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.compiler import compile_template
from cffg.core.ir import BlobRef, FactorSpec, PortRole, PortSpec, VariableSpec
from cffg.engine import Engine, _replace
from cffg.models.kalman import kalman_smoother, lgssm


def parts(value):
    return value if isinstance(value, tuple) else (value,)


def model():
    t, arrays = lgssm(5, np.eye(2), np.eye(2), np.ones((1, 2)), np.eye(1), np.zeros(2), np.eye(2))
    for i, card in enumerate((3, 2, 3, 3, 2, 2)):
        name = f"z{i}"
        key = f"prior-{name}"
        arrays[key] = np.full(card, 1 / card)
        t.blobs[key] = BlobRef(key, f"{key}.npy")
        t.variables[name] = VariableSpec(name, card)
        t.factors[key] = FactorSpec(key, "categorical", {"out": name}, param_blobs={"statistic": key})
        t.ports[name] = PortSpec(name, name, PortRole.GLUE)
    return t, arrays


@pytest.mark.parametrize("x64", [False, True])
def test_mixed_readouts_match_scalar_products_and_report_empty_support(x64):
    with jax.enable_x64(x64):
        template, arrays = model()
        engine = Engine(compile_template(template, arrays=arrays))
        view = engine.arena
        state = engine.init_state()
        dtype = np.float64 if x64 else np.float32
        tol = 128 * np.finfo(dtype).eps
        h, p, logs = (np.array(x, dtype=dtype) for x in (state.g_eta, state.g_lam, state.messages))
        cases = (
            ([0, 0, 0], [0, 0, 0]),              # Two vacuous messages.
            ([0, 0], [-1, -3]),                  # One vacuous message.
            ([0, NEG, -np.inf], [0, 0, 0]),       # Exactly one supported category.
            ([NEG, 0, 0], [0, NEG, NEG]),        # Disjoint finite-sentinel supports.
            ([-np.inf, 0], [0, -np.inf]),         # Disjoint literal supports.
            ([NEG / 2, NEG], [NEG / 2, 0]),      # Two supported values can sum to NEG.
        )
        expected_disc, expected_gauss = {}, {}
        rng = np.random.default_rng(174)
        for edge in view.edges.values():
            slots = (edge.slot0, edge.slot1)
            if not edge.dim:
                pair = np.asarray(cases[int(edge.name[1:])], dtype=dtype)
                logs[view.message_indices(slots, edge.card)] = pair
                supported = np.all((pair != dtype(NEG)) & np.isfinite(pair), axis=0)
                probability = np.zeros(edge.card, dtype=dtype)
                if supported.any():
                    logp = pair[:, supported].astype(np.float64).sum(axis=0)
                    mass = np.exp(logp - logp.max())
                    probability[supported] = mass / mass.sum()
                else:
                    probability[:] = np.nan
                expected_disc[edge.name] = probability
                continue
            dim = edge.dim
            a = rng.normal(size=(dim, dim))
            precision = (a.T @ a + np.eye(dim)).astype(dtype)
            information = rng.normal(size=dim).astype(dtype)
            if edge.name == "x0":
                precision[:] = 0
            elif edge.name == "x0c":
                precision[:] = 1                      # Singular with positive diagonal.
            elif edge.name == "x0o":
                precision[:] = [[1, 2], [2, 1]]       # Indefinite with positive diagonal.
            elif edge.name == "x1":
                precision[:] = [[1, .4], [0, 1]]
            elif edge.name == "x1c":
                information[0] = np.nan
            elif edge.name == "y0":
                precision[:] = np.inf
            p[view.g_lam_indices(slots, dim)] = [precision, np.zeros_like(precision)]
            h[view.g_eta_indices(slots, dim)] = [information, np.zeros_like(information)]
            if edge.name in {"x0", "x0c", "x0o", "x1", "x1c", "y0"}:
                expected_gauss[edge.name] = (np.full(dim, np.nan), np.full((dim, dim), np.nan))
            else:
                cov = np.linalg.inv(precision.astype(np.float64))
                expected_gauss[edge.name] = (cov @ information, cov)
        state = _replace(state, g_eta=jnp.asarray(h), g_lam=jnp.asarray(p), messages=jnp.asarray(logs))
        for read, expected in ((engine.marginals, expected_disc), (engine.gaussian_marginals, expected_gauss)):
            full = read(state)
            assert list(full) == list(expected)
            names = list(expected)[::-2]
            subset = read(state, iter(names + names))
            assert list(subset) == names
            for name, reference in expected.items():
                for actual, wanted in zip(parts(full[name]), parts(reference), strict=True):
                    assert actual.dtype == dtype and actual.shape == wanted.shape
                    np.testing.assert_allclose(actual, wanted, atol=tol, rtol=tol, equal_nan=True)
                    np.testing.assert_array_equal(actual[wanted == 0], 0.)
                if name in subset:
                    for actual, wanted in zip(parts(subset[name]), parts(full[name]), strict=True):
                        np.testing.assert_array_equal(actual, wanted)


def test_selection_checks_domain_before_computation_and_only_gathers_requested_rows(monkeypatch):
    template, arrays = model()
    colony = Colony()
    iid = colony.spawn(template, arrays)
    inst = colony.instances[iid]
    view = colony.engines[inst.key].arena
    for read, valid, wrong, gather in (
        (colony.marginals, "z0", "x0", "message_indices"),
        (colony.gaussian_marginals, "x0", "z0", "g_lam_indices"),
    ):
        original = getattr(type(view), gather)
        calls = []
        def spy(self, slots, width):
            calls.append(tuple(np.asarray(slots).ravel()))
            return original(self, slots, width)
        with monkeypatch.context() as patch:
            patch.setattr(type(view), gather, spy)
            with pytest.raises(ValueError, match=wrong):
                read(iid, [valid, wrong])
            with pytest.raises(KeyError, match="missing"):
                read(iid, [valid, "missing"])
            assert read(iid, []) == {}
            assert calls == []
            assert list(read(iid, [valid, valid])) == [valid]
            assert calls == [(view.edges[valid].slot0, view.edges[valid].slot1)]


def test_256_step_chains_read_every_edge():
    n = 256
    t = hmm_chain(n, k=2)
    observation = np.array([[.75, .25], [.25, .75]])
    engine = Engine(compile_template(t, arrays={"d": np.ones(2)/2, "A": observation, "B": np.eye(2)}))
    seeded = engine.bind_data(engine.init_state(), {f"y{n}": 1})
    state = engine.run(seeded, 1, tol=1e-9).require_converged()
    all_rows = engine.marginals(state)
    assert list(all_rows) == list(engine.arena.edges)
    for name, p in all_rows.items():
        if name.startswith("z"):
            np.testing.assert_allclose(p, [.25, .75], atol=1e-12, rtol=1e-12)

    f, q, h, r = .85 * np.eye(2), .2 * np.eye(2), np.array([[1., -.2]]), np.eye(1)
    m0, p0 = np.zeros(2), np.eye(2)
    ys = np.random.default_rng(52).standard_normal((n, 1))
    t, arrays = lgssm(n, f, q, h, r, m0, p0)
    engine = Engine(compile_template(t, arrays=arrays))
    seeded = engine.bind_data(engine.init_state(), {f"y{i}": y for i, y in enumerate(ys)})
    state = engine.run(seeded, 1, tol=1e-9).require_converged()
    all_rows = engine.gaussian_marginals(state)
    assert list(all_rows) == list(engine.arena.edges)
    means, covariances, _ = kalman_smoother(ys, f, q, h, r, m0, p0)
    np.testing.assert_allclose([all_rows[f"x{i}"][0] for i in range(n)], means, atol=1e-10)
    np.testing.assert_allclose([all_rows[f"x{i}"][1] for i in range(n)], covariances, atol=1e-10)
