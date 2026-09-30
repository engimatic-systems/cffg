"""Factor-local learning statistics and one-step mixture predictives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from cffg import conj, gauss, mixg
from cffg.compiler import NEG, compile_template
from cffg.core.ir import (
    BlobRef,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.engine import Engine, _replace

jax.config.update("jax_enable_x64", True)


def _blobs(*names):
    return {name: BlobRef(name, f"{name}.npy") for name in names}


def _linear_template(*, mixture=True, regimes=2, dimension=1):
    means = np.linspace(-0.2, 0.3, dimension)
    x_covariance = np.eye(dimension) * 0.7
    y_mean = np.linspace(0.4, -0.1, dimension)
    y_covariance = np.eye(dimension) * 0.9
    matrices = np.stack(
        [np.eye(dimension) * (0.45 + 0.2 * k) for k in range(regimes)]
    )
    offsets = np.stack(
        [np.full(dimension, -0.15 + 0.25 * k) for k in range(regimes)]
    )
    covariance = np.stack(
        [np.eye(dimension) * (0.35 + 0.1 * k) for k in range(regimes)]
    )
    variables = {
        "x": VariableSpec("x", dim=dimension),
        "y": VariableSpec("y", dim=dimension),
    }
    factors = {
        "x_prior": FactorSpec(
            "x_prior",
            "gaussian",
            edges={"out": "x"},
            param_blobs={"mean": "x_mean", "cov": "x_covariance"},
        ),
        "y_prior": FactorSpec(
            "y_prior",
            "gaussian",
            edges={"out": "y"},
            param_blobs={"mean": "y_mean", "cov": "y_covariance"},
        ),
    }
    blobs = _blobs(
        "x_mean",
        "x_covariance",
        "y_mean",
        "y_covariance",
        "matrix",
        "offset",
        "covariance",
    )
    arrays = {
        "x_mean": means,
        "x_covariance": x_covariance,
        "y_mean": y_mean,
        "y_covariance": y_covariance,
        "matrix": matrices if mixture else matrices[0],
        "offset": offsets if mixture else offsets[0],
        "covariance": covariance if mixture else covariance[0],
    }
    if mixture:
        variables["selector"] = VariableSpec(
            "selector", cardinality=regimes
        )
        factors["transition"] = FactorSpec(
            "transition",
            "gaussian_mixture_linear",
            edges={"out": "y", "in": "x", "selector": "selector"},
            param_blobs={
                "matrix": "matrix",
                "offset": "offset",
                "cov": "covariance",
            },
        )
        factors["selector_prior"] = FactorSpec(
            "selector_prior",
            "categorical",
            edges={"out": "selector"},
            param_blobs={"statistic": "probability"},
        )
        blobs["probability"] = BlobRef("probability", "probability.npy")
        arrays["probability"] = np.arange(1, regimes + 1, dtype=float)
        arrays["probability"] /= arrays["probability"].sum()
    else:
        factors["transition"] = FactorSpec(
            "transition",
            "gaussian_linear",
            edges={"out": "y", "in": "x"},
            param_blobs={
                "matrix": "matrix",
                "offset": "offset",
                "cov": "covariance",
            },
        )
    return Template(
        name="linear-readout",
        variables=variables,
        factors=factors,
        blobs=blobs,
    ), arrays


def _prior_mixture_template(*, regimes=3, dimension=2):
    component_mean = np.arange(regimes * dimension, dtype=float).reshape(
        regimes, dimension
    ) / 4.0
    component_covariance = np.stack(
        [np.eye(dimension) * (0.3 + 0.2 * k) for k in range(regimes)]
    )
    probability = np.arange(regimes, 0, -1, dtype=float)
    probability /= probability.sum()
    template = Template(
        name="prior-mixture-readout",
        variables={
            "y": VariableSpec("y", dim=dimension),
            "selector": VariableSpec("selector", cardinality=regimes),
        },
        factors={
            "mixture": FactorSpec(
                "mixture",
                "gaussian_mixture",
                edges={"out": "y", "selector": "selector"},
                param_blobs={"mean": "mean", "cov": "covariance"},
            ),
            "selector_prior": FactorSpec(
                "selector_prior",
                "categorical",
                edges={"out": "selector"},
                param_blobs={"statistic": "probability"},
            ),
        },
        ports={"y": PortSpec("y", "y", PortRole.GLUE)},
        blobs=_blobs("mean", "covariance", "probability"),
    )
    return template, {
        "mean": component_mean,
        "covariance": component_covariance,
        "probability": probability,
    }


def _recurrence_template():
    regimes = 3
    dimension = 2
    template = Template(
        name="recurrence-readout",
        variables={
            "out": VariableSpec("out", cardinality=regimes),
            "in": VariableSpec("in", cardinality=regimes),
            "drive": VariableSpec("drive", dim=dimension),
        },
        factors={
            "out_prior": FactorSpec(
                "out_prior",
                "categorical",
                edges={"out": "out"},
                param_blobs={"statistic": "out_probability"},
            ),
            "in_prior": FactorSpec(
                "in_prior",
                "categorical",
                edges={"out": "in"},
                param_blobs={"statistic": "in_probability"},
            ),
            "drive_prior": FactorSpec(
                "drive_prior",
                "gaussian",
                edges={"out": "drive"},
                param_blobs={"mean": "drive_mean", "cov": "drive_covariance"},
            ),
            "recurrence": FactorSpec(
                "recurrence",
                "recurrence_link",
                edges={"out": "out", "in": "in", "drive": "drive"},
                param_blobs={"weights": "weights", "offsets": "offsets"},
            ),
        },
        blobs=_blobs(
            "out_probability",
            "in_probability",
            "drive_mean",
            "drive_covariance",
            "weights",
            "offsets",
        ),
    )
    return template, {
        "out_probability": np.array([0.2, 0.5, 0.3]),
        "in_probability": np.array([0.6, 0.25, 0.15]),
        "drive_mean": np.array([0.15, -0.25]),
        "drive_covariance": np.array([[0.7, 0.1], [0.1, 0.5]]),
        "weights": np.array(
            [
                [[0.2, -0.1], [0.3, 0.15]],
                [[-0.25, 0.2], [0.1, -0.3]],
                [[0.15, 0.2], [-0.2, 0.1]],
            ]
        ),
        "offsets": np.array([[0.1, -0.2], [-0.1, 0.2], [0.0, 0.15]]),
    }


def _converge(template, arrays):
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.run(
        engine.init_state(), 80, tol=1e-12
    ).require_converged()
    return engine, state


def _incoming_gaussian(arena, state, group, role, dimension):
    eta = np.asarray(state.g_eta)[
        arena.g_eta_indices(group.g_gather[:, role], dimension)
    ][0]
    precision = np.asarray(state.g_lam)[
        arena.g_lam_indices(group.g_gather[:, role], dimension)
    ][0]
    return eta, precision


def _numpy_linear_stats(engine, state):
    group = engine.arena.mix_groups[0]
    dimension = group.dims[0]
    eta_y, precision_y = _incoming_gaussian(
        engine.arena, state, group, 0, dimension
    )
    eta_x, precision_x = _incoming_gaussian(
        engine.arena, state, group, 1, dimension
    )
    log_selector = np.asarray(state.messages)[
        engine.arena.message_indices(group.s_gather, group.k_sel)
    ][0]
    matrix = np.asarray(state.params[group.name + ":A"])[0]
    offset = np.asarray(state.params[group.name + ":b"])[0]
    precision = np.asarray(state.params[group.name + ":W"])[0]
    means = []
    covariances = []
    scores = []
    for regime in range(group.k_sel):
        a = matrix[regime]
        b = offset[regime]
        w = precision[regime]
        joint_precision = np.block(
            [
                [w + precision_y, -w @ a],
                [-a.T @ w, a.T @ w @ a + precision_x],
            ]
        )
        joint_eta = np.concatenate(
            [w @ b + eta_y, -a.T @ w @ b + eta_x]
        )
        covariance = np.linalg.inv(joint_precision)
        mean = covariance @ joint_eta
        _, logdet_joint = np.linalg.slogdet(joint_precision)
        _, logdet_w = np.linalg.slogdet(w)
        log_z = (
            0.5 * joint_eta @ mean
            + 0.5 * joint_eta.size * np.log(2.0 * np.pi)
            - 0.5 * logdet_joint
        )
        node_constant = (
            -0.5 * b @ w @ b
            + 0.5 * logdet_w
            - 0.5 * dimension * np.log(2.0 * np.pi)
        )
        means.append(mean)
        covariances.append(covariance)
        scores.append(log_selector[regime] + log_z + node_constant)
    responsibility = np.exp(np.asarray(scores) - np.max(scores))
    responsibility /= responsibility.sum()
    syy = []
    syx = []
    sxx = []
    for weight, mean, covariance in zip(
        responsibility, means, covariances, strict=True
    ):
        y_mean, x_mean = mean[:dimension], mean[dimension:]
        yy = covariance[:dimension, :dimension] + np.outer(y_mean, y_mean)
        yx = covariance[:dimension, dimension:] + np.outer(y_mean, x_mean)
        xx = covariance[dimension:, dimension:] + np.outer(x_mean, x_mean)
        syy.append(weight * yy)
        syx.append(
            weight * np.concatenate([yx, y_mean[:, None]], axis=1)
        )
        sxx.append(
            weight
            * np.block(
                [
                    [xx, x_mean[:, None]],
                    [x_mean[None, :], np.ones((1, 1))],
                ]
            )
        )
    return responsibility, np.asarray(syy), np.asarray(syx), np.asarray(sxx)


def test_linear_mixture_stats_match_independent_d3_joint():
    template, arrays = _linear_template(regimes=3, dimension=2)
    engine, state = _converge(template, arrays)
    actual = engine.gaussian_node_stats(state)["transition"]
    responsibility, syy, syx, sxx = _numpy_linear_stats(engine, state)

    assert bool(actual["proper"])
    np.testing.assert_allclose(
        actual["responsibility"], responsibility, atol=1e-11
    )
    np.testing.assert_allclose(actual["Syy"], syy, atol=1e-11)
    np.testing.assert_allclose(actual["Syx"], syx, atol=1e-11)
    np.testing.assert_allclose(actual["Sxx"], sxx, atol=1e-11)
    np.testing.assert_allclose(actual["Sxx"][:, -1, -1], responsibility)
    np.testing.assert_allclose(actual["Syy"].sum(axis=0), syy.sum(axis=0))


@pytest.mark.slow
def test_single_regime_stats_reduce_to_gaussian_linear():
    mixture_template, mixture_arrays = _linear_template(
        mixture=True, regimes=1, dimension=2
    )
    linear_template, linear_arrays = _linear_template(
        mixture=False, regimes=1, dimension=2
    )
    mixture_engine, mixture_state = _converge(
        mixture_template, mixture_arrays
    )
    linear_engine, linear_state = _converge(linear_template, linear_arrays)

    mixture = mixture_engine.gaussian_node_stats(mixture_state)["transition"]
    linear = linear_engine.gaussian_node_stats(linear_state)["transition"]
    np.testing.assert_allclose(mixture["responsibility"], [1.0])
    for key in ("Syy", "Syx", "Sxx"):
        np.testing.assert_allclose(mixture[key][0], linear[key], atol=1e-10)


def test_linear_predictive_matches_components_and_forward_message():
    template, arrays = _linear_template(regimes=3, dimension=2)
    engine, state = _converge(template, arrays)
    group = engine.arena.mix_groups[0]
    predictive = engine.gaussian_mixture_predictives(state)["transition"]
    eta_x, precision_x = _incoming_gaussian(
        engine.arena, state, group, 1, 2
    )
    x_covariance = np.linalg.inv(precision_x)
    x_mean = x_covariance @ eta_x
    matrix = np.asarray(state.params[group.name + ":A"])[0]
    offset = np.asarray(state.params[group.name + ":b"])[0]
    noise_covariance = np.linalg.inv(
        np.asarray(state.params[group.name + ":W"])[0]
    )
    component_mean = np.einsum("kij,j->ki", matrix, x_mean) + offset
    component_covariance = noise_covariance + np.einsum(
        "kij,jl,kml->kim", matrix, x_covariance, matrix
    )
    weights = arrays["probability"]
    collapsed_mean = weights @ component_mean
    centered = component_mean - collapsed_mean
    collapsed_covariance = np.einsum(
        "k,kij->ij", weights, component_covariance
    ) + np.einsum("k,ki,kj->ij", weights, centered, centered)

    assert bool(predictive["proper"])
    np.testing.assert_allclose(predictive["weight"], weights, atol=1e-11)
    np.testing.assert_allclose(
        predictive["component_mean"], component_mean, atol=1e-11
    )
    np.testing.assert_allclose(
        predictive["component_covariance"], component_covariance, atol=1e-11
    )
    np.testing.assert_allclose(predictive["mean"], collapsed_mean, atol=1e-11)
    np.testing.assert_allclose(
        predictive["covariance"], collapsed_covariance, atol=1e-11
    )

    eta_out = np.asarray(state.g_eta)[
        engine.arena.g_eta_indices(group.g_scatter[:, 0], 2)
    ][0]
    precision_out = np.asarray(state.g_lam)[
        engine.arena.g_lam_indices(group.g_scatter[:, 0], 2)
    ][0]
    forward_mean, forward_covariance = gauss.moments(eta_out, precision_out)
    np.testing.assert_allclose(predictive["mean"], forward_mean, atol=1e-10)
    np.testing.assert_allclose(
        predictive["covariance"], forward_covariance, atol=1e-10
    )

    output_eta_indices = engine.arena.g_eta_indices(
        group.g_gather[:, 0], 2
    )[0]
    output_precision_indices = engine.arena.g_lam_indices(
        group.g_gather[:, 0], 2
    )[0]
    changed = _replace(
        state,
        g_eta=state.g_eta.at[output_eta_indices].set(
            jnp.asarray([40.0, -30.0])
        ),
        g_lam=state.g_lam.at[output_precision_indices].set(
            jnp.eye(2) * 20.0
        ),
    )
    changed_predictive = engine.gaussian_mixture_predictives(changed)[
        "transition"
    ]
    for key in (
        "weight",
        "component_mean",
        "component_covariance",
        "mean",
        "covariance",
    ):
        np.testing.assert_array_equal(changed_predictive[key], predictive[key])

    selector_indices = engine.arena.message_indices(
        group.s_gather, group.k_sel
    )[0]
    changed_selector = state.messages.at[selector_indices].set(
        jnp.asarray([0.0, -2.0, -4.0])
    )
    selector_predictive = engine.gaussian_mixture_predictives(
        _replace(state, messages=changed_selector)
    )["transition"]
    assert not np.allclose(selector_predictive["weight"], predictive["weight"])
    assert not np.allclose(selector_predictive["mean"], predictive["mean"])

    input_eta_indices = engine.arena.g_eta_indices(
        group.g_gather[:, 1], 2
    )[0]
    changed_input = state.g_eta.at[input_eta_indices].add(
        jnp.asarray([0.8, -0.4])
    )
    input_predictive = engine.gaussian_mixture_predictives(
        _replace(state, g_eta=changed_input)
    )["transition"]
    assert not np.allclose(
        input_predictive["component_mean"], predictive["component_mean"]
    )


def test_scalar_predictive_matches_dense_gauss_hermite_quadrature():
    template, arrays = _linear_template(regimes=3, dimension=1)
    engine, state = _converge(template, arrays)
    group = engine.arena.mix_groups[0]
    predictive = engine.gaussian_mixture_predictives(state)["transition"]
    eta_x, precision_x = _incoming_gaussian(
        engine.arena, state, group, 1, 1
    )
    x_variance = 1.0 / precision_x[0, 0]
    x_mean = x_variance * eta_x[0]
    nodes, quadrature_weight = hermgauss(160)
    points = np.sqrt(2.0 * x_variance) * nodes + x_mean
    quadrature_weight /= np.sqrt(np.pi)
    matrix = arrays["matrix"][:, 0, 0]
    offset = arrays["offset"][:, 0]
    noise_variance = arrays["covariance"][:, 0, 0]
    conditional_mean = matrix[:, None] * points[None] + offset[:, None]
    component_mean = conditional_mean @ quadrature_weight
    component_second = (
        conditional_mean**2 + noise_variance[:, None]
    ) @ quadrature_weight
    weight = arrays["probability"]
    quadrature_mean = weight @ component_mean
    quadrature_variance = weight @ component_second - quadrature_mean**2

    np.testing.assert_allclose(predictive["mean"], [quadrature_mean], atol=1e-11)
    np.testing.assert_allclose(
        predictive["covariance"], [[quadrature_variance]], atol=1e-11
    )


def test_prior_mixture_stats_and_predictive_name_weighting_explicitly():
    template, arrays = _prior_mixture_template()
    engine, state = _converge(template, arrays)
    stats = engine.gaussian_node_stats(state)["mixture"]
    predictive = engine.gaussian_mixture_predictives(state)["mixture"]

    assert bool(stats["proper"])
    assert set(stats) == {
        "responsibility",
        "weighted_mean",
        "weighted_second",
        "proper",
    }
    np.testing.assert_allclose(predictive["weight"], arrays["probability"])
    np.testing.assert_allclose(predictive["component_mean"], arrays["mean"])
    np.testing.assert_allclose(
        predictive["component_covariance"], arrays["covariance"]
    )
    expected_weighted_mean = arrays["probability"][:, None] * arrays["mean"]
    expected_weighted_second = arrays["probability"][:, None, None] * (
        arrays["covariance"]
        + np.einsum("ki,kj->kij", arrays["mean"], arrays["mean"])
    )
    np.testing.assert_allclose(stats["responsibility"], arrays["probability"])
    np.testing.assert_allclose(stats["weighted_mean"], expected_weighted_mean)
    np.testing.assert_allclose(
        stats["weighted_second"], expected_weighted_second
    )


def test_exact_support_improper_inputs_float32_and_uncomfortable_scale():
    regimes = 32
    dimension = 16
    rng = np.random.default_rng(11)
    matrix = rng.normal(scale=0.04, size=(1, regimes, dimension, dimension))
    offset = rng.normal(scale=0.1, size=(1, regimes, dimension))
    precision = np.broadcast_to(
        np.eye(dimension, dtype=np.float32),
        (1, regimes, dimension, dimension),
    )
    eta = np.zeros((1, dimension), dtype=np.float32)
    incoming_precision = np.eye(dimension, dtype=np.float32)[None]
    selector = np.zeros((1, regimes), dtype=np.float32)
    selector[:, ::5] = NEG
    stats = mixg.mixlin_stats(
        eta,
        incoming_precision,
        eta,
        incoming_precision,
        selector,
        matrix.astype(np.float32),
        offset.astype(np.float32),
        precision,
    )
    predictive = mixg.mixlin_predictive(
        eta,
        incoming_precision,
        selector,
        matrix.astype(np.float32),
        offset.astype(np.float32),
        precision,
    )

    assert bool(stats.proper[0]) and bool(predictive.proper[0])
    assert stats.syy.shape == (1, regimes, dimension, dimension)
    assert stats.syx.shape == (1, regimes, dimension, dimension + 1)
    assert stats.sxx.shape == (1, regimes, dimension + 1, dimension + 1)
    assert stats.syy.dtype == jnp.float32
    np.testing.assert_array_equal(stats.responsibility[0, ::5], 0.0)
    np.testing.assert_array_equal(stats.syy[0, ::5], 0.0)
    np.testing.assert_array_equal(stats.syx[0, ::5], 0.0)
    np.testing.assert_array_equal(stats.sxx[0, ::5], 0.0)
    np.testing.assert_array_equal(predictive.weight[0, ::5], 0.0)

    empty = mixg.mixlin_predictive(
        eta,
        incoming_precision,
        np.full_like(selector, NEG),
        matrix.astype(np.float32),
        offset.astype(np.float32),
        precision,
    )
    assert not bool(empty.proper[0])
    assert np.isnan(empty.mean[0]).all()
    empty_stats = mixg.mixlin_stats(
        eta,
        incoming_precision,
        eta,
        incoming_precision,
        np.full_like(selector, NEG),
        matrix.astype(np.float32),
        offset.astype(np.float32),
        precision,
    )
    assert not bool(empty_stats.proper[0])
    assert np.isnan(empty_stats.syy[0]).all()
    vacuous = mixg.mixlin_predictive(
        eta,
        np.zeros_like(incoming_precision),
        selector,
        matrix.astype(np.float32),
        offset.astype(np.float32),
        precision,
    )
    assert not bool(vacuous.proper[0])
    assert np.isnan(vacuous.covariance[0]).all()
    bad_precision = jnp.asarray(precision).at[:, 0].set(
        -jnp.eye(dimension, dtype=jnp.float32)
    )
    invalid = mixg.mixlin_stats(
        eta,
        incoming_precision,
        eta,
        incoming_precision,
        selector,
        matrix.astype(np.float32),
        offset.astype(np.float32),
        bad_precision,
    )
    assert not bool(invalid.proper[0])
    assert np.isnan(invalid.responsibility[0]).all()
    invalid_predictive = mixg.mixlin_predictive(
        eta,
        incoming_precision,
        selector,
        matrix.astype(np.float32),
        offset.astype(np.float32),
        bad_precision,
    )
    assert not bool(invalid_predictive.proper[0])
    assert np.isnan(invalid_predictive.component_mean[0]).all()

    improper_joint = mixg.mixlin_stats(
        eta,
        -2.0 * incoming_precision,
        eta,
        incoming_precision,
        selector,
        matrix.astype(np.float32),
        offset.astype(np.float32),
        precision,
    )
    assert not bool(improper_joint.proper[0])
    assert np.isnan(improper_joint.sxx[0]).all()


def test_prior_predictive_is_proper_without_a_gaussian_cavity():
    template, arrays = _prior_mixture_template(regimes=2, dimension=1)
    engine = Engine(compile_template(template, arrays=arrays))
    predictive = engine.gaussian_mixture_predictives(engine.init_state())[
        "mixture"
    ]
    assert bool(predictive["proper"])
    np.testing.assert_allclose(predictive["weight"], [0.5, 0.5])


def test_recurrence_readout_matches_edge_beliefs_and_preserves_support():
    template, arrays = _recurrence_template()
    engine, state = _converge(template, arrays)
    group = engine.arena.recurrence_groups[0]
    stats = engine.gaussian_node_stats(state)["recurrence"]
    pair = engine.recurrence_pair_beliefs(state)["recurrence"]

    out_messages = np.asarray(state.messages)[
        engine.arena.message_indices(group.d_gather_both[:, 0, :], 3)
    ][0]
    in_messages = np.asarray(state.messages)[
        engine.arena.message_indices(group.d_gather_both[:, 1, :], 3)
    ][0]

    def normalized(messages):
        score = messages.sum(axis=0)
        probability = np.exp(score - score.max())
        return probability / probability.sum()

    drive_eta = np.asarray(state.g_eta)[
        engine.arena.g_eta_indices(group.drive_gather_both, 2)
    ].sum(axis=1)[0]
    drive_precision = np.asarray(state.g_lam)[
        engine.arena.g_lam_indices(group.drive_gather_both, 2)
    ].sum(axis=1)[0]
    drive_covariance = np.linalg.inv(drive_precision)
    drive_mean = drive_covariance @ drive_eta

    assert bool(stats["proper"])
    for key, value in pair.items():
        np.testing.assert_array_equal(stats[key], value)
    np.testing.assert_allclose(
        pair["pair_probability"].sum(axis=0),
        pair["destination_probability"],
        atol=1e-11,
    )
    np.testing.assert_allclose(
        pair["pair_probability"].sum(axis=1),
        pair["source_probability"],
        atol=1e-11,
    )
    np.testing.assert_allclose(
        stats["out_probability"], normalized(out_messages), atol=1e-11
    )
    np.testing.assert_allclose(
        stats["in_probability"], normalized(in_messages), atol=1e-11
    )
    np.testing.assert_allclose(stats["drive_mean"], drive_mean, atol=1e-11)
    np.testing.assert_allclose(
        stats["drive_covariance"], drive_covariance, atol=1e-11
    )
    np.testing.assert_allclose(
        stats["drive_second"],
        drive_covariance + np.outer(drive_mean, drive_mean),
        atol=1e-11,
    )

    out_indices = engine.arena.message_indices(
        group.d_gather_both[:, 0, :], 3
    )[0]
    supported_messages = state.messages.at[out_indices[1, 1]].set(NEG)
    supported = engine.gaussian_node_stats(
        _replace(state, messages=supported_messages)
    )["recurrence"]
    assert bool(supported["proper"])
    assert not bool(supported["out_support"][1])
    assert supported["out_probability"][1] == 0.0

    empty_messages = state.messages.at[out_indices[1]].set(NEG)
    empty = engine.gaussian_node_stats(
        _replace(state, messages=empty_messages)
    )["recurrence"]
    assert not bool(empty["proper"])
    assert np.isnan(empty["out_probability"]).all()

    drive_precision_indices = engine.arena.g_lam_indices(
        group.drive_gather_both, 2
    )
    vacuous_precision = state.g_lam.at[drive_precision_indices].set(0.0)
    vacuous = engine.gaussian_node_stats(
        _replace(state, g_lam=vacuous_precision)
    )["recurrence"]
    assert not bool(vacuous["proper"])
    assert np.isnan(vacuous["drive_mean"]).all()
    assert np.isnan(vacuous["drive_covariance"]).all()


def test_soft_regime_stats_beat_hard_assignments_for_overlapping_dynamics():
    """A low-separation E/M step retains information hard labels discard."""
    rng = np.random.default_rng(0)
    sample_count = 50
    true_matrix = np.array([0.7, 1.0])
    true_offset = np.array([-0.1, 0.1])
    true_variance = 0.45**2
    regime = rng.integers(0, 2, sample_count)
    x = rng.normal(size=sample_count)
    y = (
        true_matrix[regime] * x
        + true_offset[regime]
        + rng.normal(scale=np.sqrt(true_variance), size=sample_count)
    )

    observation_precision = 1.0e5
    cavity_precision = np.broadcast_to(
        np.array([[[observation_precision]]]),
        (sample_count, 1, 1),
    )
    matrix = np.broadcast_to(
        true_matrix[None, :, None, None], (sample_count, 2, 1, 1)
    )
    offset = np.broadcast_to(
        true_offset[None, :, None], (sample_count, 2, 1)
    )
    noise_precision = np.broadcast_to(
        np.array([[[[1.0 / true_variance]], [[1.0 / true_variance]]]]),
        (sample_count, 2, 1, 1),
    )
    stats = mixg.mixlin_stats(
        (observation_precision * y)[:, None],
        cavity_precision,
        (observation_precision * x)[:, None],
        cavity_precision,
        np.full((sample_count, 2), -np.log(2.0)),
        matrix,
        offset,
        noise_precision,
    )
    assert np.asarray(stats.proper).all()
    responsibility = np.asarray(stats.responsibility)
    hard_assignment = np.eye(2)[responsibility.argmax(axis=1)]
    augmented_x = np.stack([x, np.ones(sample_count)], axis=1)

    def update_soft(regime_index):
        posterior = conj.vague_mniw(1, 1).update(
            syy=np.asarray(stats.syy).sum(axis=0)[regime_index],
            syx=np.asarray(stats.syx).sum(axis=0)[regime_index],
            sxx=np.asarray(stats.sxx).sum(axis=0)[regime_index],
            n=responsibility[:, regime_index].sum(),
        )
        return posterior.rollover()

    def update_hard(regime_index):
        weight = hard_assignment[:, regime_index]
        posterior = conj.vague_mniw(1, 1).update(
            syy=np.array([[np.sum(weight * y * y)]]),
            syx=np.array(
                [np.sum((weight * y)[:, None] * augmented_x, axis=0)]
            ),
            sxx=np.einsum(
                "n,ni,nj->ij", weight, augmented_x, augmented_x
            ),
            n=weight.sum(),
        )
        return posterior.rollover()

    soft = [update_soft(index) for index in range(2)]
    hard = [update_hard(index) for index in range(2)]

    def parameter_error(parameters):
        return sum(
            (parameters[index][0][0, 0] - true_matrix[index]) ** 2
            + (parameters[index][1][0] - true_offset[index]) ** 2
            + (parameters[index][2][0, 0] - true_variance) ** 2
            for index in range(2)
        )

    assert parameter_error(soft) < 0.1 * parameter_error(hard)
