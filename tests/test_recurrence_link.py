"""PG/JJ recurrence-link rules, engine integration, and hostile inputs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from cffg import efrules, recurrence
from cffg.compiler import CompileError, RecurrencePlan, compile_template
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
from cffg.core.serialize import from_json, to_dict, to_json
from cffg.core.validate import ModelError
from cffg.engine import Engine, RunStatus, _replace

jax.config.update("jax_enable_x64", True)


def _template(
    *,
    weights,
    offsets,
    p_in=None,
    p_out=None,
    mean=None,
    covariance=None,
    drive_prior=True,
):
    weights = np.asarray(weights)
    card, sticks, dimension = weights.shape
    assert sticks == card - 1
    p_in = np.full(card, 1.0 / card) if p_in is None else np.asarray(p_in)
    p_out = np.full(card, 1.0 / card) if p_out is None else np.asarray(p_out)
    mean = np.zeros(dimension) if mean is None else np.asarray(mean)
    covariance = (
        np.eye(dimension) if covariance is None else np.asarray(covariance)
    )
    variables = {
        "z_out": VariableSpec("z_out", cardinality=card),
        "z_in": VariableSpec("z_in", cardinality=card),
        "drive": VariableSpec("drive", dim=dimension),
    }
    factors = {
        "out_prior": FactorSpec(
            "out_prior",
            "categorical",
            edges={"out": "z_out"},
            param_blobs={"statistic": "p_out"},
        ),
        "in_prior": FactorSpec(
            "in_prior",
            "categorical",
            edges={"out": "z_in"},
            param_blobs={"statistic": "p_in"},
        ),
        "recurrence": FactorSpec(
            "recurrence",
            "recurrence_link",
            edges={"out": "z_out", "in": "z_in", "drive": "drive"},
            param_blobs={"weights": "weights", "offsets": "offsets"},
        ),
    }
    blobs = {
        name: BlobRef(name, f"{name}.npy")
        for name in ("p_out", "p_in", "weights", "offsets")
    }
    arrays = {
        "p_out": p_out,
        "p_in": p_in,
        "weights": weights,
        "offsets": np.asarray(offsets),
    }
    ports = {}
    if drive_prior:
        factors["drive_prior"] = FactorSpec(
            "drive_prior",
            "gaussian",
            edges={"out": "drive"},
            param_blobs={"mean": "mean", "cov": "covariance"},
        )
        blobs.update(
            {
                "mean": BlobRef("mean", "mean.npy"),
                "covariance": BlobRef("covariance", "covariance.npy"),
            }
        )
        arrays.update({"mean": mean, "covariance": covariance})
    else:
        ports["drive"] = PortSpec("drive", "drive", PortRole.GLUE)
    return (
        Template(
            name="recurrence-link",
            variables=variables,
            factors=factors,
            ports=ports,
            blobs=blobs,
        ),
        arrays,
    )


def _stick_probabilities(points, weights, offsets):
    logits = np.einsum("ild,nd->nil", weights, points) + offsets[None]
    sticks = 1.0 / (1.0 + np.exp(-logits))
    remaining = np.ones(sticks.shape[:-1])
    probabilities = []
    for index in range(sticks.shape[-1]):
        probabilities.append(remaining * sticks[..., index])
        remaining = remaining * (1.0 - sticks[..., index])
    probabilities.append(remaining)
    return np.stack(probabilities, axis=-1)


def _numpy_coordinates(q_out, q_in, mean, covariance, weights, offsets):
    logit_mean = np.einsum("ild,d->il", weights, mean) + offsets
    variance = np.einsum("ild,de,ile->il", weights, covariance, weights)
    xi = np.sqrt(logit_mean**2 + variance)
    omega = np.empty_like(xi)
    nonzero = xi > 0
    omega[nonzero] = np.tanh(xi[nonzero] / 2.0) / (2.0 * xi[nonzero])
    omega[~nonzero] = 0.25
    log_sigmoid = -np.logaddexp(0.0, -xi)
    common = log_sigmoid - xi / 2.0
    success = common + logit_mean / 2.0
    failure = common - logit_mean / 2.0
    prefix = np.concatenate(
        [np.zeros((failure.shape[0], 1)), np.cumsum(failure, axis=-1)],
        axis=-1,
    )
    scores = np.concatenate(
        [prefix[:, :-1] + success, prefix[:, -1:]], axis=-1
    )
    at_risk = np.cumsum(q_out[::-1])[::-1][:-1]
    kappa = q_out[:-1] - at_risk / 2.0
    pg_weight = q_in[:, None] * at_risk[None] * omega
    linear_weight = q_in[:, None] * kappa[None] - pg_weight * offsets
    eta = np.einsum("il,ild->d", linear_weight, weights)
    precision = np.einsum("il,ild,ile->de", pg_weight, weights, weights)
    expected = np.einsum("i,j,ij->", q_in, q_out, scores)
    return scores, eta, precision, expected


def _exact_scalar_posterior(arrays):
    nodes, weights = hermgauss(160)
    mean = float(arrays["mean"][0])
    variance = float(arrays["covariance"][0, 0])
    points = (np.sqrt(2.0 * variance) * nodes + mean)[:, None]
    quadrature = weights / np.sqrt(np.pi)
    transition = _stick_probabilities(
        points, arrays["weights"], arrays["offsets"]
    )
    joint = (
        quadrature[:, None, None]
        * arrays["p_in"][None, :, None]
        * arrays["p_out"][None, None, :]
        * transition
    )
    normalizer = joint.sum()
    joint /= normalizer
    q_out = joint.sum(axis=(0, 1))
    q_in = joint.sum(axis=(0, 2))
    q_x = joint.sum(axis=(1, 2))
    x = points[:, 0]
    x_mean = q_x @ x
    x_variance = q_x @ np.square(x) - x_mean**2
    return q_out, q_in, x_mean, x_variance, -np.log(normalizer)


def _run(template, arrays, *, schedule="auto", dtype=None, iterations=250):
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    state = engine.init_state()
    if dtype is not None:
        state = jax.tree_util.tree_map(
            lambda value: (
                value.astype(dtype)
                if jnp.issubdtype(value.dtype, jnp.inexact)
                else value
            ),
            state,
        )
    result = engine.run(
        state,
        iterations,
        tol=1e-5 if dtype == jnp.float32 else 1e-9,
    )
    return engine, result


def test_rule_matches_independent_numpy_coordinates():
    q_out = np.array([0.55, 0.3, 0.15])
    q_in = np.array([0.2, 0.5, 0.3])
    mean = np.array([0.25, -0.1])
    covariance = np.array([[0.8, 0.1], [0.1, 0.5]])
    weights = np.array(
        [
            [[0.5, -0.2], [0.1, 0.4]],
            [[-0.3, 0.6], [0.7, -0.1]],
            [[0.2, 0.3], [-0.5, 0.2]],
        ]
    )
    offsets = np.array([[0.1, -0.2], [-0.3, 0.25], [0.05, 0.4]])
    precision = np.linalg.inv(covariance)
    eta = precision @ mean
    slots_out = np.stack([np.log(q_out), np.zeros_like(q_out)])[None]
    slots_in = np.stack([np.log(q_in), np.zeros_like(q_in)])[None]

    result = recurrence.update(
        jnp.asarray(slots_out),
        jnp.asarray(slots_in),
        jnp.asarray(eta[None]),
        jnp.asarray(precision[None]),
        jnp.asarray(weights[None]),
        jnp.asarray(offsets[None]),
    )
    scores, expected_eta, expected_precision, _ = _numpy_coordinates(
        q_out, q_in, mean, covariance, weights, offsets
    )
    expected_out = efrules.CATEGORICAL.normalize(jnp.asarray(q_in @ scores))
    expected_in = efrules.CATEGORICAL.normalize(jnp.asarray(scores @ q_out))
    np.testing.assert_allclose(result.to_out[0], expected_out, atol=1e-12)
    np.testing.assert_allclose(result.to_in[0], expected_in, atol=1e-12)
    np.testing.assert_allclose(result.drive_eta[0], expected_eta, atol=1e-12)
    np.testing.assert_allclose(
        result.drive_precision[0], expected_precision, atol=1e-12
    )
    assert int(result.status[0]) == 0


def test_engine_weight_stats_match_independent_soft_belief_algebra():
    weights = np.array(
        [
            [[0.5, -0.2], [0.1, 0.4]],
            [[-0.3, 0.6], [0.7, -0.1]],
            [[0.2, 0.3], [-0.5, 0.2]],
        ]
    )
    offsets = np.array([[0.1, -0.2], [-0.3, 0.25], [0.05, 0.4]])
    template, arrays = _template(
        weights=weights,
        offsets=offsets,
        p_in=np.array([0.2, 0.5, 0.3]),
        p_out=np.array([0.55, 0.3, 0.15]),
        mean=np.array([0.25, -0.1]),
        covariance=np.array([[0.8, 0.1], [0.1, 0.5]]),
    )
    engine, result = _run(template, arrays)
    state = result.require_converged()
    belief = engine.gaussian_node_stats(state)["recurrence"]
    actual = engine.recurrence_weight_stats(state)["recurrence"]

    q_out = belief["out_probability"]
    q_in = belief["in_probability"]
    feature_mean = np.append(belief["drive_mean"], 1.0)
    feature_second = np.block(
        [
            [belief["drive_second"], belief["drive_mean"][:, None]],
            [belief["drive_mean"][None, :], np.ones((1, 1))],
        ]
    )
    coefficients = np.concatenate([weights, offsets[..., None]], axis=-1)
    xi = np.sqrt(
        np.einsum(
            "ilp,pq,ilq->il", coefficients, feature_second, coefficients
        )
    )
    omega = np.empty_like(xi)
    nonzero = xi > 0
    omega[nonzero] = np.tanh(xi[nonzero] / 2.0) / (2.0 * xi[nonzero])
    omega[~nonzero] = 0.25
    at_risk = np.cumsum(q_out[::-1])[::-1][:-1]
    kappa = q_out[:-1] - 0.5 * at_risk
    exposure = q_in[:, None] * at_risk[None]
    linear = (
        q_in[:, None, None] * kappa[None, :, None] * feature_mean
    )
    quadratic = (
        exposure[..., None, None]
        * omega[..., None, None]
        * feature_second
    )

    assert bool(actual["proper"])
    np.testing.assert_allclose(actual["omega"], omega, atol=1e-11)
    np.testing.assert_allclose(actual["exposure"], exposure, atol=1e-11)
    np.testing.assert_allclose(actual["linear"], linear, atol=1e-11)
    np.testing.assert_allclose(actual["quadratic"], quadratic, atol=1e-11)

    extreme = recurrence.weight_stats(
        np.array([[0.5, 0.5]]),
        np.array([[0.5, 0.5]]),
        np.zeros((1, 1)),
        np.zeros((1, 1, 1)),
        np.zeros((1, 2, 1, 1)),
        np.array([[[0.0], [1.0e4]]]),
    )
    assert extreme.omega[0, 0, 0] == pytest.approx(0.25)
    assert np.isfinite(extreme.omega[0, 1, 0])
    invalid = recurrence.weight_stats(
        np.array([[0.4, 0.4]]),
        np.array([[0.5, 0.5]]),
        np.zeros((1, 1)),
        np.ones((1, 1, 1)),
        np.zeros((1, 2, 1, 1)),
        np.zeros((1, 2, 1)),
    )
    assert not bool(invalid.proper[0])
    assert np.isnan(invalid.linear[0]).all()


def test_jj_expected_log_is_a_lower_bound():
    q_out = np.array([0.65, 0.35])
    q_in = np.array([0.4, 0.6])
    mean = np.array([0.3])
    variance = np.array([[0.7]])
    weights = np.array([[[1.2]], [[-0.8]]])
    offsets = np.array([[0.2], [-0.1]])
    precision = np.linalg.inv(variance)
    eta = precision @ mean
    bounded = float(
        recurrence.expected_log_bound(
            jnp.asarray(q_out[None]),
            jnp.asarray(q_in[None]),
            jnp.asarray(eta[None]),
            jnp.asarray(precision[None]),
            jnp.asarray(weights[None]),
            jnp.asarray(offsets[None]),
        )[0]
    )

    nodes, gh_weights = hermgauss(200)
    points = (np.sqrt(2.0 * variance[0, 0]) * nodes + mean[0])[:, None]
    transition = _stick_probabilities(points, weights, offsets)
    exact = np.einsum(
        "n,i,j,nij->",
        gh_weights / np.sqrt(np.pi),
        q_in,
        q_out,
        np.log(transition),
    )
    assert bounded <= exact + 1e-12
    assert exact - bounded > 1e-4


def test_compiler_plan_params_and_serialization():
    weights = np.arange(18, dtype=float).reshape(3, 2, 3) / 20.0
    offsets = np.arange(6, dtype=float).reshape(3, 2) / 10.0
    template, arrays = _template(weights=weights, offsets=offsets)
    round_tripped = from_json(to_json(template))
    assert to_dict(round_tripped) == to_dict(template)

    arena = compile_template(round_tripped, arrays=arrays)
    assert len(arena.recurrence_groups) == 1
    group = arena.recurrence_groups[0]
    assert isinstance(group, RecurrencePlan)
    assert group.cards == (3, 3) and group.dim == 3
    assert group.d_gather_both.shape == (1, 2, 2)
    assert group.d_scatter.shape == (1, 2)
    assert group.drive_gather_both.shape == (1, 2)
    assert group.drive_scatter.shape == (1,)
    assert arena.log_params[group.name + ":weights"].shape == (1, 3, 2, 3)
    assert arena.log_params[group.name + ":offsets"].shape == (1, 3, 2)
    schedule = Engine(arena).schedule
    assert schedule.kind == "mixed"
    assert "recurrence-link VMP groups" in schedule.reason


def test_compiler_rejects_invalid_shape_domain_beads_and_semiring():
    template, arrays = _template(
        weights=np.ones((2, 1, 1)), offsets=np.zeros((2, 1))
    )
    arrays["weights"] = np.ones((1, 2, 1))
    with pytest.raises(CompileError, match="parameter shape"):
        compile_template(template, arrays=arrays)

    template, arrays = _template(
        weights=np.ones((2, 1, 1)), offsets=np.zeros((2, 1))
    )
    template.variables["drive"] = VariableSpec("drive", cardinality=2)
    with pytest.raises(ModelError, match="role 'drive' needs a gaussian edge"):
        compile_template(template, arrays=arrays)

    template, arrays = _template(
        weights=np.ones((2, 1, 1)), offsets=np.zeros((2, 1))
    )
    template.constraints.append(
        ConstraintSpec(ConstraintKind.MEAN_FIELD, "recurrence")
    )
    with pytest.raises(CompileError, match="hybrid beads are undefined"):
        compile_template(template, arrays=arrays)

    template, arrays = _template(
        weights=np.ones((2, 1, 1)), offsets=np.zeros((2, 1))
    )
    with pytest.raises(CompileError, match="Gaussian/hybrid groups"):
        compile_template(template, arrays=arrays, semiring="max")

    template, arrays = _template(
        weights=np.ones((2, 1, 1)), offsets=np.zeros((2, 1))
    )
    template.constraints.append(
        ConstraintSpec(
            ConstraintKind.P_SUBSTITUTION,
            "recurrence",
            (("substituted_role", "out"),),
        )
    )
    with pytest.raises(CompileError, match="P-substitution on recurrence_link"):
        compile_template(template, arrays=arrays)

    template, arrays = _template(
        weights=np.ones((2, 1, 1)), offsets=np.zeros((2, 1))
    )
    template.variables["z_out"] = VariableSpec("z_out", cardinality=3)
    with pytest.raises(CompileError, match="equal out/in cardinality"):
        compile_template(template, arrays=arrays)

    template, arrays = _template(
        weights=np.ones((2, 1, 1)), offsets=np.zeros((2, 1))
    )
    arrays["offsets"][0, 0] = np.inf
    with pytest.raises(CompileError, match="must be finite"):
        compile_template(template, arrays=arrays)


def test_engine_tracks_dense_quadrature_and_upper_bounds_evidence():
    template, arrays = _template(
        weights=np.array([[[0.6]], [[-0.6]]]),
        offsets=np.zeros((2, 1)),
        p_in=np.array([0.6, 0.4]),
        p_out=np.array([0.8, 0.2]),
    )
    engine, result = _run(template, arrays)
    assert result.status is RunStatus.CONVERGED
    exact_out, exact_in, exact_mean, exact_variance, exact_evidence = (
        _exact_scalar_posterior(arrays)
    )
    marginals = engine.marginals(result.state)
    actual_mean, actual_covariance = engine.gaussian_marginals(result.state)[
        "drive"
    ]
    np.testing.assert_allclose(marginals["z_out"], exact_out, atol=0.015)
    np.testing.assert_allclose(marginals["z_in"], exact_in, atol=0.015)
    np.testing.assert_allclose(actual_mean, [exact_mean], atol=0.005)
    np.testing.assert_allclose(
        actual_covariance, [[exact_variance]], atol=0.1
    )
    bounded_vfe = engine.bfe(result.state)
    assert bounded_vfe >= exact_evidence - 1e-9
    assert bounded_vfe - exact_evidence < 0.06


def test_flood_and_mixed_schedules_reach_the_same_bound_root():
    template, arrays = _template(
        weights=np.array([[[0.4]], [[-0.3]]]),
        offsets=np.array([[0.1], [-0.2]]),
        p_in=np.array([0.55, 0.45]),
        p_out=np.array([0.7, 0.3]),
    )
    mixed_engine, mixed = _run(template, arrays, schedule="auto")
    flood_engine, flood = _run(template, arrays, schedule="flood")
    assert mixed.status is RunStatus.CONVERGED
    assert flood.status is RunStatus.CONVERGED
    for edge in ("z_out", "z_in"):
        np.testing.assert_allclose(
            mixed_engine.marginals(mixed.state)[edge],
            flood_engine.marginals(flood.state)[edge],
            atol=1e-7,
        )
    mixed_drive = mixed_engine.gaussian_marginals(mixed.state)["drive"]
    flood_drive = flood_engine.gaussian_marginals(flood.state)["drive"]
    for mixed_value, flood_value in zip(mixed_drive, flood_drive):
        np.testing.assert_allclose(mixed_value, flood_value, atol=1e-7)
    assert mixed_engine.bfe(mixed.state) == pytest.approx(
        flood_engine.bfe(flood.state), abs=1e-8
    )


def test_exact_zero_support_is_preserved():
    template, arrays = _template(
        weights=np.array([[[0.5]], [[-0.5]]]),
        offsets=np.zeros((2, 1)),
        p_in=np.array([1.0, 0.0]),
        p_out=np.array([0.75, 0.25]),
    )
    engine, result = _run(template, arrays)
    assert result.status is RunStatus.CONVERGED
    np.testing.assert_array_equal(
        engine.marginals(result.state)["z_in"], [1.0, 0.0]
    )
    learning = engine.recurrence_weight_stats(result.state)["recurrence"]
    np.testing.assert_array_equal(learning["exposure"][1], 0.0)
    np.testing.assert_array_equal(learning["linear"][1], 0.0)
    np.testing.assert_array_equal(learning["quadratic"][1], 0.0)
    zero_at_risk = recurrence.weight_stats(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros((1, 1)),
        np.ones((1, 1, 1)),
        np.zeros((1, 3, 2, 1)),
        np.zeros((1, 3, 2)),
    )
    np.testing.assert_array_equal(zero_at_risk.exposure[0, :, 1], 0.0)
    np.testing.assert_array_equal(zero_at_risk.linear[0, :, 1], 0.0)
    np.testing.assert_array_equal(zero_at_risk.quadratic[0, :, 1], 0.0)
    assert np.isfinite(engine.bfe(result.state))


def test_vacuous_and_improper_drive_are_loud_after_transient_flooding():
    template, arrays = _template(
        weights=np.array([[[0.5]], [[-0.5]]]),
        offsets=np.zeros((2, 1)),
        drive_prior=False,
    )
    engine, result = _run(template, arrays, iterations=20)
    assert result.status is RunStatus.DIVERGED
    assert not np.isfinite(engine.bfe(result.state))
    learning = engine.recurrence_weight_stats(result.state)["recurrence"]
    assert not bool(learning["proper"])
    assert np.isnan(learning["quadratic"]).all()

    template, arrays = _template(
        weights=np.array([[[0.5]], [[-0.5]]]),
        offsets=np.zeros((2, 1)),
    )
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()
    prior = next(
        group for group in engine.arena.gauss_groups if "drive_prior" in group.members
    )
    params = dict(state.params)
    params[prior.name + ":W"] = -jnp.ones_like(params[prior.name + ":W"])
    result = engine.run(_replace(state, params=params), 30, tol=1e-8)
    assert result.status is RunStatus.DIVERGED
    assert result.inner_status is RunStatus.DIVERGED
    assert not np.isfinite(engine.bfe(result.state))


def test_float32_k32_d16_converges_without_cubic_plan_storage():
    card, dimension = 32, 16
    rng = np.random.default_rng(7)
    weights = rng.normal(scale=0.01, size=(card, card - 1, dimension))
    uniform_offsets = -np.log(np.arange(card - 1, 0, -1, dtype=float))
    offsets = np.broadcast_to(uniform_offsets, (card, card - 1)).copy()
    template, arrays = _template(
        weights=weights,
        offsets=offsets,
        covariance=np.eye(dimension) * 0.8,
    )
    arrays = {key: value.astype(np.float32) for key, value in arrays.items()}
    engine, result = _run(
        template,
        arrays,
        dtype=jnp.float32,
        iterations=200,
    )
    assert result.status is RunStatus.CONVERGED
    group = engine.arena.recurrence_groups[0]
    assert group.d_gather_both.size + group.drive_gather_both.size < card**2
    assert result.state.params[group.name + ":weights"].shape == (
        1,
        card,
        card - 1,
        dimension,
    )
    assert np.isfinite(engine.bfe(result.state))
    assert np.all(np.isfinite(engine.marginals(result.state)["z_out"]))
    drive_mean, drive_covariance = engine.gaussian_marginals(result.state)[
        "drive"
    ]
    assert np.all(np.isfinite(drive_mean))
    assert np.all(np.isfinite(drive_covariance))
    learning = engine.recurrence_weight_stats(result.state)["recurrence"]
    assert bool(learning["proper"])
    assert learning["omega"].dtype == np.float32
    assert learning["linear"].shape == (card, card - 1, dimension + 1)
    assert learning["quadratic"].shape == (
        card,
        card - 1,
        dimension + 1,
        dimension + 1,
    )


def test_vmap_step_matches_sequential():
    template, arrays = _template(
        weights=np.array([[[0.4]], [[-0.2]]]),
        offsets=np.array([[0.1], [-0.15]]),
    )
    engine = Engine(compile_template(template, arrays=arrays))
    states = [engine.init_state(), engine.init_state()]
    batched = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)

    def one(state):
        messages, eta, precision, ef_eta, _, _ = engine._step(
            state.messages,
            state.g_eta,
            state.g_lam,
            state.g_obs,
            state.g_mask,
            state.ef_eta,
            state.ef_obs,
            state.ef_mask,
            state.params,
        )
        return _replace(
            state,
            messages=messages,
            g_eta=eta,
            g_lam=precision,
            ef_eta=ef_eta,
        )

    batched_result = eqx.filter_vmap(one)(batched)
    sequential = [one(state) for state in states]
    for row, state in enumerate(sequential):
        np.testing.assert_allclose(batched_result.messages[row], state.messages)
        np.testing.assert_allclose(batched_result.g_eta[row], state.g_eta)
        np.testing.assert_allclose(batched_result.g_lam[row], state.g_lam)
