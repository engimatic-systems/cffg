"""Observed-only rSLDS initialization, fitting cadence, and hostile inputs."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.engine import Engine, RunStatus
from cffg.models import (
    PCAARHMMInitializer,
    RSLDSInitialization,
    fit_rslds,
    vague_rslds_fit_priors,
)


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
        outer_tolerance=2.0,
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
    original_run = Engine.run

    def tracked_run(self, *args, **kwargs):
        engine_ids.append(id(self))
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(Engine, "run", tracked_run)
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
    assert result.iterations == 2
    assert len(result.residuals) == 2
    assert len(set(engine_ids)) == 1
    assert len(engine_ids) == 3
    assert [phase.coordinate_replaced for phase in result.phases] == [
        True,
        True,
        False,
    ]
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
    marginal = engine.marginals(first.state)["z0"]
    assert marginal[1] == 0.0


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("dynamics", "dynamics beliefs must all be proper"),
        ("emission", "gaussian_linear statistic is improper"),
        ("recurrence", "recurrence beliefs must all be proper"),
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
    if kind == "recurrence":
        original_readout = Engine.recurrence_weight_stats

        def improper_readout(self, state):
            rows = original_readout(self, state)
            name = next(name for name in rows if name.startswith("switch"))
            rows[name] = {**rows[name], "proper": np.asarray(False)}
            return rows

        monkeypatch.setattr(Engine, "recurrence_weight_stats", improper_readout)
    else:
        original_readout = Engine.gaussian_node_stats

        def improper_readout(self, state):
            rows = original_readout(self, state)
            name = next(name for name in rows if name.startswith(kind))
            rows[name] = {**rows[name], "proper": np.asarray(False)}
            return rows

        monkeypatch.setattr(Engine, "gaussian_node_stats", improper_readout)
    result = _fit(observations, outer_iterations=1, initializer=initializer)

    assert result.status is RunStatus.DIVERGED
    assert result.inner_status is RunStatus.CONVERGED
    assert message in result.error
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


def test_float32_fitter_survives_tenfold_horizon():
    result = _fit(
        _observations(80),
        outer_iterations=1,
        inner_iterations=200,
        inner_tolerance=0.1,
        damp=0.9,
    )

    assert result.inner_status is RunStatus.CONVERGED
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert len(result.free_energies) == 2
    assert np.all(np.isfinite(result.free_energies))
    assert np.isfinite(result.residual)
    assert all(np.all(np.isfinite(value)) for value in result.parameters.values())


def test_repeated_nascar_coordinates_globalize_the_finite_cycle():
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
        damp=0.9,
        max_inner_retries=2,
    )

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.inner_status is RunStatus.CONVERGED
    assert sum(phase.coordinate_replaced for phase in result.phases) == 5
    fourth = result.phases[3]
    assert [attempt.damping for attempt in fourth.attempts] == pytest.approx(
        [0.9, 0.8, 0.6]
    )
    assert [attempt.status for attempt in fourth.attempts] == [
        RunStatus.BUDGET_EXHAUSTED,
        RunStatus.BUDGET_EXHAUSTED,
        RunStatus.CONVERGED,
    ]
    exhausted = fourth.attempts[0]
    assert exhausted.recurrence_status is RunStatus.CONVERGED
    assert exhausted.gaussian_beliefs_proper
    assert exhausted.state_finite
    assert exhausted.message_delta_tail[-1] > 1e-2
    assert np.all(np.isfinite(exhausted.free_energy_tail))
    assert all(np.isfinite(tuple(fourth.parameter_scales_before.values())))
    assert all(np.isfinite(tuple(fourth.parameter_scales_after.values())))
    terminal = result.phases[-1]
    assert not terminal.coordinate_replaced
    assert terminal.attempts[-1].status is RunStatus.CONVERGED
    assert terminal.attempts[-1].recurrence_status is RunStatus.CONVERGED
    assert result.last_inference is not None
    assert result.last_inference.status is RunStatus.CONVERGED
