"""Observed-only rSLDS initialization, fitting cadence, and hostile inputs."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.engine import Engine, RunResult, RunStatus
from cffg.learning_checkpoint import D10_RECURRENCE_STATISTIC_IDENTITY
from cffg.models import (
    PCAARHMMInitializer,
    RSLDSInitialization,
    fit_rslds,
    vague_rslds_fit_priors,
)
from cffg.models.coordinate_fit import CoordinateResidual
from cffg.models.rslds import EMISSION_LEARNABLE, RSLDSLearning
from cffg.models.rslds_fit import _point_parameters, _run_inner


def _observations(steps: int, dtype=np.float32) -> np.ndarray:
    time = np.linspace(0.0, 2.5 * np.pi, steps, dtype=dtype)
    return np.stack(
        (
            np.sin(time)
            + np.asarray(0.2, dtype=dtype) * np.sin(3.0 * time)
            + np.asarray(0.4, dtype=dtype),
            np.cos(time)
            - np.asarray(0.15, dtype=dtype) * np.sin(2.0 * time)
            - np.asarray(0.2, dtype=dtype),
        ),
        axis=1,
    )


def _nascar_observations(steps: int = 240, seed: int = 101) -> np.ndarray:
    """Pinned compact Linderman NASCAR pilot observations."""
    rng = np.random.RandomState(seed)
    rotation = []
    for angle in (np.pi / 24.0, np.pi / 48.0):
        matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        basis = np.linalg.qr(rng.randn(2, 2))[0]
        rotation.append(basis @ matrix @ basis.T)
    centers = (np.array([2.0, 0.0]), np.array([-2.0, 0.0]))
    dynamics_matrix = np.stack((*rotation, np.eye(2), np.eye(2)))
    dynamics_offset = np.stack(
        (
            *(
                -(matrix - np.eye(2)) @ center
                for matrix, center in zip(rotation, centers)
            ),
            np.array([0.1, 0.0]),
            np.array([-0.35, 0.0]),
        )
    )
    stick_weights = 100.0 * np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]
    )
    stick_offsets = 100.0 * np.array([-2.0, -2.0, 0.0]) + np.array(
        [-1.4985019321609645, -0.9990012881073119, -2.161965419040482e-12]
    )
    emission = rng.randn(10, 2)
    regimes = np.empty(steps, dtype=np.int32)
    states = np.empty((steps, 2))
    observations = np.empty((steps, 10))
    regimes[0] = rng.choice(4, p=np.full(4, 0.25))
    states[0] = rng.multivariate_normal(np.array([0.0, 1.0]), 1e-3 * np.eye(2))
    observations[0] = rng.multivariate_normal(
        emission @ states[0], 1e-5 * np.eye(10)
    )
    for step in range(1, steps):
        logits = stick_weights @ states[step - 1] + stick_offsets
        accepts = np.exp(-np.logaddexp(0.0, -logits))
        probabilities = np.empty(4)
        remaining = 1.0
        for stick, probability in enumerate(accepts):
            probabilities[stick] = remaining * probability
            remaining *= 1.0 - probability
        probabilities[-1] = remaining
        probabilities /= probabilities.sum()
        regimes[step] = rng.choice(4, p=probabilities)
        regime = regimes[step]
        states[step] = rng.multivariate_normal(
            dynamics_matrix[regime] @ states[step - 1]
            + dynamics_offset[regime],
            1e-4 * np.eye(2),
        )
        observations[step] = rng.multivariate_normal(
            emission @ states[step], 1e-5 * np.eye(10)
        )
    return observations


def _fit(
    observations: np.ndarray,
    *,
    outer_iterations: int = 2,
    inner_iterations: int = 100,
    inner_tolerance: float = 1e-5,
    damp: float = 0.0,
    initializer=PCAARHMMInitializer(),
    outer_tolerance: float = 2.0,
):
    return fit_rslds(
        observations,
        regimes=2,
        latent_dimension=1,
        initializer=initializer,
        priors=vague_rslds_fit_priors(2, 1, observations.shape[1]),
        max_outer_iterations=outer_iterations,
        max_inner_iterations=inner_iterations,
        inner_tolerance=inner_tolerance,
        outer_tolerance=outer_tolerance,
        key=jax.random.key(7),
        damp=damp,
    )


def test_pca_arhmm_initializer_is_seeded_finite_and_dtype_scaled():
    observations = _observations(8)
    original = observations.copy()
    initializer = PCAARHMMInitializer()

    first = initializer(
        observations,
        regimes=2,
        latent_dimension=1,
        key=jax.random.key(3),
    )
    second = initializer(
        observations,
        regimes=2,
        latent_dimension=1,
        key=jax.random.key(3),
    )

    np.testing.assert_array_equal(observations, original)
    for name, value in first.arrays().items():
        assert value.dtype == np.float32, name
        assert np.all(np.isfinite(value)), name
        np.testing.assert_array_equal(value, second.arrays()[name])
    for covariance in (
        first.dynamics_covariance,
        first.emission_covariance,
        first.initial_state_covariance,
    ):
        np.linalg.cholesky(covariance)
    np.testing.assert_allclose(
        first.emission_covariance,
        np.eye(2, dtype=np.float32) * first.emission_covariance[0, 0],
    )


def test_fit_replaces_one_sequence_coordinates_and_reuses_one_engine(monkeypatch):
    observations = _observations(8)
    engine_ids = []
    run_parameter_snapshots = []
    candidate_points = []
    candidate_models = []
    candidate_emissions = []
    original_run = Engine.run

    def tracked_run(self, *args, **kwargs):
        engine_ids.append(id(self))
        run_parameter_snapshots.append(
            {name: np.asarray(value).copy() for name, value in args[0].params.items()}
        )
        return original_run(self, *args, **kwargs)

    def tracked_points(learning):
        points = _point_parameters(learning)
        candidate_points.append(
            {name: np.asarray(value).copy() for name, value in points.items()}
        )
        candidate_models.append(RSLDSLearning._from_runtime(learning))
        candidate_emissions.append(
            {
                name: np.asarray(value).copy()
                for name, value in learning.params[EMISSION_LEARNABLE].items()
            }
        )
        return points

    monkeypatch.setattr(Engine, "run", tracked_run)
    monkeypatch.setattr("cffg.models.rslds_fit._point_parameters", tracked_points)
    priors = vague_rslds_fit_priors(2, 1, 2)
    result = fit_rslds(
        observations,
        regimes=2,
        latent_dimension=1,
        initializer=PCAARHMMInitializer(),
        priors=priors,
        max_outer_iterations=2,
        max_inner_iterations=100,
        inner_tolerance=1e-5,
        outer_tolerance=2.0,
        key=jax.random.key(7),
    )

    assert result.status is RunStatus.CONVERGED
    assert result.inner_status is RunStatus.CONVERGED
    assert (
        result.model_posterior.recurrence_statistic_identity
        == D10_RECURRENCE_STATISTIC_IDENTITY
    )
    assert result.iterations == 2
    assert len(result.residuals) == 2
    assert len(set(engine_ids)) == 1
    assert len(engine_ids) == 2
    assert [phase.coordinate_replaced for phase in result.phases] == [
        True,
        False,
    ]
    assert [
        (
            phase.evaluated_coordinate,
            phase.proposed_coordinate,
            phase.installed_coordinate,
            phase.certified_coordinate,
            phase.returned_coordinate,
        )
        for phase in result.phases
    ] == [
        ("coordinate/0", "coordinate/1", "coordinate/1", None, None),
        (
            "coordinate/1",
            "coordinate/2",
            "coordinate/1",
            "coordinate/1",
            "coordinate/1",
        ),
    ]
    assert result.returned_coordinate == "coordinate/1"
    assert result.certified_coordinate == result.returned_coordinate
    for phase, selected_residual in zip(result.phases, result.residuals):
        assert phase.fixed_point_residual == selected_residual
        assert phase.runtime_residual is not None
        assert phase.runtime_residual.maximum == selected_residual
        assert phase.legacy_fixed_point_residual is not None
        assert np.isfinite(phase.legacy_fixed_point_residual)
        assert set(phase.runtime_residual.per_family) == {
            "initial",
            "dynamics",
            "recurrence",
            "emission",
        }
        assert phase.functional_transition_residual is not None
        assert np.isfinite(phase.functional_transition_residual.maximum)
        assert phase.proposed_parameter_scales is not None
    assert (
        result.phases[-1].parameter_scales_after
        == result.phases[-1].parameter_scales_before
    )
    assert len(candidate_points) == 2
    for name, value in result.parameters.items():
        np.testing.assert_array_equal(value, candidate_points[0][name])
    assert any(
        not np.array_equal(candidate_points[0][name], candidate_points[1][name])
        for name in candidate_points[0]
    )
    for name, value in result.model_posterior.state_dict(result.engine).items():
        np.testing.assert_array_equal(
            value, candidate_models[0].state_dict(result.engine)[name]
        )
    for name, value in result.emission_posterior.items():
        np.testing.assert_array_equal(value, candidate_emissions[0][name])
    for name, value in result.state.params.items():
        if name.startswith("d10:"):
            continue
        np.testing.assert_array_equal(value, run_parameter_snapshots[-1][name])
    assert result.last_inference is not None
    for name, value in result.state.params.items():
        np.testing.assert_array_equal(
            value, result.last_inference.state.params[name]
        )
    assert result.last_inference is not None
    for value, expected in zip(
        jax.tree_util.tree_leaves(result.state),
        jax.tree_util.tree_leaves(result.last_inference.state),
    ):
        np.testing.assert_array_equal(value, expected)
    assert result.model_posterior.initial_alpha.sum() == pytest.approx(
        priors.initial_alpha.sum() + 1.0
    )
    assert result.emission_posterior["nu"] == pytest.approx(
        priors.emission.nu + observations.shape[0]
    )
    dynamics_evidence = sum(
        posterior.nu - prior.nu
        for posterior, prior in zip(result.model_posterior.dynamics, priors.dynamics)
    )
    assert dynamics_evidence == pytest.approx(observations.shape[0] - 1, abs=2e-5)
    assert set(result.parameters) == {
        "initial_regime_probability",
        "dynamics_matrix",
        "dynamics_offset",
        "dynamics_covariance",
        "recurrence_weights",
        "recurrence_offsets",
        "emission_matrix",
        "emission_offset",
        "emission_covariance",
    }
    assert all(np.all(np.isfinite(value)) for value in result.parameters.values())
    assert result.initialization.initial_state_mean.shape == (1,)
    assert result.initialization.initial_state_covariance.shape == (1, 1)
    assert set(result.timings) == {
        "initialization",
        "compile",
        "inference",
        "coordinate",
    }
    assert all(value >= 0.0 for value in result.timings.values())
    assert all(
        value.dtype == jnp.float32
        for value in result.state.params.values()
        if jnp.issubdtype(value.dtype, jnp.inexact)
    )
    initial = result.engine.init_state()
    assert {
        key: value.shape for key, value in result.state.params.items()
    } == {key: value.shape for key, value in initial.params.items()}


def test_fit_uses_runtime_map_residual_as_its_outer_certificate(monkeypatch):
    forced = CoordinateResidual(
        maximum=3.0,
        root_mean_square=3.0,
        per_family={"forced": 3.0},
        per_array={"forced/value": 3.0},
        vector=(3.0,),
    )
    monkeypatch.setattr("cffg.models.rslds_fit.coordinate_residual", lambda *_: forced)

    result = _fit(_observations(8), outer_iterations=2)

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.residuals == (3.0, 3.0)
    assert result.returned_coordinate == "coordinate/2"
    assert result.certified_coordinate is None
    assert result.phases[-1].evaluated_coordinate == result.returned_coordinate
    assert result.phases[-1].installed_coordinate == result.returned_coordinate
    assert result.phases[-1].returned_coordinate == result.returned_coordinate
    assert result.phases[-1].proposed_coordinate is None
    for phase in result.phases[:-1]:
        assert phase.fixed_point_residual == 3.0
        assert phase.legacy_fixed_point_residual is not None
        assert phase.runtime_residual is forced


def test_rebound_resets_the_two_hit_rule_before_certifying(monkeypatch):
    residuals = iter((0.05, 0.2, 0.05, 0.05))

    def forced_residual(*_):
        value = next(residuals)
        return CoordinateResidual(
            maximum=value,
            root_mean_square=value,
            per_family={"forced": value},
            per_array={"forced/value": value},
            vector=(value,),
        )

    monkeypatch.setattr("cffg.models.rslds_fit.coordinate_residual", forced_residual)

    result = _fit(
        _observations(8),
        outer_iterations=4,
        outer_tolerance=0.1,
    )

    assert result.status is RunStatus.CONVERGED
    assert result.residuals == pytest.approx((0.05, 0.2, 0.05, 0.05))
    assert [phase.coordinate_replaced for phase in result.phases] == [
        True,
        True,
        True,
        False,
    ]
    assert result.certified_coordinate == "coordinate/3"
    assert result.returned_coordinate == "coordinate/3"
    assert result.phases[-1].proposed_coordinate == "coordinate/4"
    assert result.phases[-1].installed_coordinate == "coordinate/3"


def test_nonfinite_proposed_coordinate_diverges_without_installing_it(monkeypatch):
    def nonfinite_points(learning):
        points = _point_parameters(learning)
        points["recurrence_offsets"][0, 0] = np.nan
        return points

    monkeypatch.setattr("cffg.models.rslds_fit._point_parameters", nonfinite_points)

    result = _fit(_observations(8), outer_iterations=1)

    assert result.status is RunStatus.DIVERGED
    assert "non-finite" in result.error
    assert result.certified_coordinate is None
    assert result.returned_coordinate == "coordinate/0"
    assert result.phases[-1].evaluated_coordinate == "coordinate/0"
    assert result.phases[-1].proposed_coordinate == "coordinate/1"
    assert result.phases[-1].installed_coordinate == "coordinate/0"
    assert not result.phases[-1].coordinate_replaced
    assert all(np.all(np.isfinite(value)) for value in result.parameters.values())


@dataclasses.dataclass(frozen=True)
class _ZeroSupportInitializer:
    base: PCAARHMMInitializer = PCAARHMMInitializer()

    def __call__(self, observations, *, regimes, latent_dimension, key):
        initialized = self.base(
            observations,
            regimes=regimes,
            latent_dimension=latent_dimension,
            key=key,
        )
        return dataclasses.replace(
            initialized,
            initial_regime_probability=np.array([1.0, 0.0], dtype=observations.dtype),
        )


def test_exact_zero_initial_regime_support_is_not_floored(monkeypatch):
    inner_results = []
    original_run = Engine.run

    def tracked_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        inner_results.append((self, result))
        return result

    monkeypatch.setattr(Engine, "run", tracked_run)
    _fit(_observations(8), outer_iterations=1, initializer=_ZeroSupportInitializer())

    engine, first = inner_results[0]
    marginal = next(
        iter(engine.structured_hybrid_beliefs(first.state).values())
    )["singleton_probability"][0]
    assert marginal[1] == 0.0


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("dynamics", "gaussian_mixture_linear statistic is improper"),
        ("emission", "gaussian_linear statistic is improper"),
        ("recurrence", "recurrence_link statistic is improper"),
    ),
)
def test_improper_statistics_diverge_before_parameter_write(
    monkeypatch, kind, message
):
    observations = _observations(8)
    initializer = PCAARHMMInitializer()
    initialized = initializer(
        observations,
        regimes=2,
        latent_dimension=1,
        key=jax.random.key(7),
    )
    injected_sources = []
    original_readout = Engine.learning_readout

    def improper_readout(self, state, readout_name):
        rows = original_readout(self, state, readout_name)
        if kind == "recurrence":
            if readout_name != "recurrence_weight_stats":
                return rows
            name = next(name for name in rows if name.startswith("switch"))
        else:
            if readout_name != "gaussian_node_stats":
                return rows
            name = next(name for name in rows if name.startswith(kind))
        injected_sources.append(name)
        rows[name] = {
            **rows[name],
            "proper": np.asarray(False),
            "drive_ok": np.asarray(False),
            "drive_min_eigenvalue": np.asarray(0.5),
            "drive_symmetry_error": np.asarray(0.25),
            "drive_tolerance": np.asarray(0.01),
        }
        return rows

    monkeypatch.setattr(Engine, "learning_readout", improper_readout)
    result = _fit(observations, outer_iterations=1, initializer=initializer)

    assert result.status is RunStatus.DIVERGED
    assert result.inner_status is RunStatus.CONVERGED
    assert message in result.error
    if kind == "recurrence":
        assert f"source {injected_sources[0]!r}" in result.error
        assert "drive_ok=false" in result.error
        assert "min_eigenvalue=0.5" in result.error
        assert "symmetry_error=0.25" in result.error
        assert "tolerance=0.01" in result.error
    for name, value in result.parameters.items():
        np.testing.assert_array_equal(value, initialized.arrays()[name])


def test_nonfinite_inner_free_energy_diverges_before_parameter_write(monkeypatch):
    observations = _observations(8)
    initializer = PCAARHMMInitializer()
    initialized = initializer(
        observations,
        regimes=2,
        latent_dimension=1,
        key=jax.random.key(7),
    )
    original_run = Engine.run

    def nonfinite_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        return dataclasses.replace(
            result,
            status=RunStatus.CONVERGED,
            free_energies=(float("nan"),),
        )

    monkeypatch.setattr(Engine, "run", nonfinite_run)
    result = _fit(observations, outer_iterations=1, initializer=initializer)

    assert result.status is RunStatus.DIVERGED
    assert result.inner_status is RunStatus.DIVERGED
    assert result.error == "inner Bethe free energy is missing or non-finite"
    for name, value in result.parameters.items():
        np.testing.assert_array_equal(value, initialized.arrays()[name])


def test_inner_budget_status_is_preserved():
    result = _fit(_observations(8), outer_iterations=2, inner_iterations=1)

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.inner_status is RunStatus.BUDGET_EXHAUSTED
    assert result.error == "inner inference budget_exhausted"


def test_deterministic_retry_rejects_mu3_engines_before_running():
    engine = object.__new__(Engine)
    engine.arena = dataclasses.make_dataclass("Arena", [("mu3_groups", tuple)])(
        (object(),)
    )

    with pytest.raises(ValueError, match="do not support stochastic mu3"):
        _run_inner(
            engine,
            object(),
            max_iterations=1,
            tolerance=1e-5,
            damping=0.0,
            max_retries=0,
        )


def test_inner_retry_schedule_retries_finite_exhaustion():
    class FakeEngine:
        arena = dataclasses.make_dataclass("Arena", [("mu3_groups", tuple)])(())

        def __init__(self):
            self.dampings = []

        def run(self, state, iterations, *, tol, damp):
            del tol
            self.dampings.append(damp)
            status = (
                RunStatus.BUDGET_EXHAUSTED
                if len(self.dampings) < 3
                else RunStatus.CONVERGED
            )
            return RunResult(
                state=state,
                status=status,
                iterations=iterations,
                message_deltas=(0.25,),
                free_energies=(1.0,),
                residual=0.25,
            )

        def learning_readout(self, state, name):
            del state
            assert name == "gaussian_node_stats"
            return {"drive": {"proper": True}}

    engine = FakeEngine()
    result, attempts = _run_inner(
        engine,
        jnp.zeros(1),
        max_iterations=8,
        tolerance=1e-4,
        damping=0.9,
        max_retries=2,
    )
    assert result.status is RunStatus.CONVERGED
    assert engine.dampings == pytest.approx([0.9, 0.8, 0.6])
    assert [attempt.status for attempt in attempts] == [
        RunStatus.BUDGET_EXHAUSTED,
        RunStatus.BUDGET_EXHAUSTED,
        RunStatus.CONVERGED,
    ]


def test_float32_fitter_survives_tenfold_horizon():
    result = _fit(
        _observations(80),
        outer_iterations=1,
        inner_iterations=200,
        inner_tolerance=0.1,
        damp=0.0,
    )

    assert result.inner_status is RunStatus.CONVERGED
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert len(result.free_energies) == 2
    assert np.all(np.isfinite(result.free_energies))
    assert np.isfinite(result.residual)
    assert all(np.all(np.isfinite(value)) for value in result.parameters.values())


@pytest.mark.slow
def test_structured_nascar_coordinates_avoid_the_legacy_finite_cycle():
    observations = _nascar_observations()
    key = jax.random.split(
        jax.random.fold_in(jax.random.key(9001), 101), 3
    )[0]
    result = fit_rslds(
        observations,
        regimes=4,
        latent_dimension=2,
        initializer=PCAARHMMInitializer(),
        priors=vague_rslds_fit_priors(4, 2, 10),
        max_outer_iterations=5,
        max_inner_iterations=2_000,
        inner_tolerance=1e-4,
        key=key,
        damp=0.0,
        max_inner_retries=2,
    )

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.inner_status is RunStatus.CONVERGED
    assert sum(phase.coordinate_replaced for phase in result.phases) == 5
    fourth = result.phases[3]
    assert [attempt.damping for attempt in fourth.attempts] == [0.0]
    completed = fourth.attempts[0]
    assert completed.status is RunStatus.CONVERGED
    assert completed.recurrence_status is RunStatus.CONVERGED
    assert completed.gaussian_beliefs_proper
    assert completed.state_finite
    assert np.all(np.isfinite(completed.free_energy_tail))
    assert all(np.isfinite(tuple(fourth.parameter_scales_before.values())))
    assert all(np.isfinite(tuple(fourth.parameter_scales_after.values())))
    terminal = result.phases[-1]
    assert not terminal.coordinate_replaced
    assert terminal.attempts[-1].status is RunStatus.CONVERGED
    assert terminal.attempts[-1].recurrence_status is RunStatus.CONVERGED
    assert result.last_inference is not None
    assert result.last_inference.status is RunStatus.CONVERGED
