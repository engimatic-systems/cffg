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
    assert len(engine_ids) == 2
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


def test_exact_zero_initial_regime_support_is_not_floored():
    result = _fit(
        _observations(8), outer_iterations=1, initializer=_ZeroSupportInitializer()
    )

    assert result.last_inference is not None
    marginal = result.engine.marginals(result.last_inference.state)["z0"]
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
    assert len(result.free_energies) == 1
    assert np.isfinite(result.free_energies[0])
    assert np.isfinite(result.residual)
    assert all(np.all(np.isfinite(value)) for value in result.parameters.values())
