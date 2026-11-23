"""Causal heat-transfer estimators for the PCGym pilot.

A call to update consumes one realized, normalized measurement pair and the actual
intervening coolant command. Both filters use only the new temperature for evidence.
The previous noisy Ca and T enter the transition as plug-in state estimates. Their
sensor variances propagate through the RK4 temperature Jacobian. Adjacent pairs
share a measurement, so multiplying these local Gaussian likelihoods approximates
the joint time-series likelihood. The planner keeps its own concentration bias.
"""

from math import exp, isfinite, log, pi

import jax
import jax.numpy as jnp
import numpy as np

import cstr_runtime_fixture as fixture


NOMINAL = 50_000.
LOW, HIGH = 40_000., 60_000.
INITIAL_SD = .10  # standard deviation of log(UA / nominal)
MOTION_SD = .012  # per realized transition, in the same coordinates
GRID_STEP = 250.
HAZARD = .002
SENSOR_FRACTION = .001


def method_config():
    """JSON-ready frozen specification shared by experiment and diagnostics."""
    return {
        "model": "retained cstr_rk4; fixed non-UA parameters, zero concentration bias in the one-step predictor",
        "observation": "normalized [Ca, T]; only the new T is scored",
        "ua_bounds": [LOW, HIGH], "initial_ua": NOMINAL,
        "sensor_fraction_sd": SENSOR_FRACTION,
        "conditional_temperature_variance": (
            "new sensor variance at predicted T plus propagated independent previous Ca and T "
            "sensor variances at measured values; all variances in normalized coordinates"
        ),
        "approximations": (
            "First-order propagation around a noisy previous measurement, including the reset sample. "
            "Neighboring pairs reuse that measurement and ignore temporal error correlations. "
            "The retained bias adds only to the concentration output; this likelihood omits latent bias "
            "uncertainty and other model discrepancy."
        ),
        "learner": {
            "method": "scalar log(UA / initial_ua) random walk and extended Kalman temperature update",
            "initial_log_sd": INITIAL_SD, "motion_log_sd": MOTION_SD,
            "bounds": "project posterior location; retain Joseph-form local variance",
            "ua_readout": "initial_ua * exp(posterior log location)",
        },
        "reference": {
            "method": "exact log-space filter on an evenly spaced physical UA grid for the stated likelihood",
            "grid_step": GRID_STEP, "initial_ua_sd": NOMINAL * INITIAL_SD,
            "prior": "discrete Gaussian centered at initial_ua, normalized on grid",
            "motion": "Gaussian in physical UA with sd = current UA * motion_log_sd, normalized per source row",
            "motion_log_sd": MOTION_SD, "uniform_hazard_per_transition": HAZARD,
            "ua_readout": "posterior arithmetic mean on grid",
        },
    }


def _finite(value, name):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be real and finite")
    return value.astype(np.float64, copy=False)


def _observation(value):
    obs = _finite(value, "observation")
    if obs.shape != (2,):
        raise ValueError("observation must contain exactly normalized [Ca, T]")
    return obs


def _control(value):
    u = _finite(value, "applied command")
    if u.shape not in ((), (1,)) or abs(u.item()) > 1:
        raise ValueError("applied command must be a scalar in [-1, 1]")
    return u.item()


def _mean(obs, applied, z):
    theta = jnp.asarray(fixture.PARAMETERS).at[2].set(log(NOMINAL) + z)
    state = jnp.concatenate((obs, jnp.stack((jnp.zeros_like(applied), applied))))
    return fixture.cstr_rk4(state, theta, fixture.CONFIGURATION)[:2]


_forecast = jax.jit(_mean)


def predict(obs, applied, ua):
    """Return the next normalized [Ca, T] before the next measurement."""
    value = _finite(ua, "UA")
    if value.shape or value.item() <= 0:
        raise ValueError("UA must be a positive scalar")
    mean = _forecast(_observation(obs), _control(applied), log(value.item()) - log(NOMINAL))
    return _finite(mean, "one-step prediction")


def _temperature(obs, applied, z):
    mean = _mean(obs, applied, z)
    return mean[1], mean


def _sensor_sd(obs):
    # sigma(normalized x) = 2 * fraction * |physical x| / (high - low).
    low = jnp.asarray(fixture.CONFIGURATION["x_lo"])
    span = jnp.asarray(fixture.CONFIGURATION["x_hi"]) - low
    return SENSOR_FRACTION * jnp.abs(obs + 1 + 2 * low / span)


_temperature_jacobian = jax.jacfwd(_temperature, argnums=(0, 2), has_aux=True)


def _likelihood_model(obs, applied, z):
    (dx, dz), mean = _temperature_jacobian(obs, applied, z)
    variance = jnp.sum((dx * _sensor_sd(obs)) ** 2) + _sensor_sd(mean)[1] ** 2
    return mean[1], dz, variance


_single_model = jax.jit(_likelihood_model)
_grid_model = jax.jit(jax.vmap(_likelihood_model, in_axes=(None, None, 0)))


def _log_density(innovation, variance):
    return -0.5 * (log(2 * pi) + np.log(variance) + np.square(innovation) / variance)


class Learner:
    """Continuous scalar point estimate with local covariance in log coordinates."""

    def __init__(self, *, motion_sd=MOTION_SD):
        if not isfinite(motion_sd) or motion_sd < 0:
            raise ValueError("Motion standard deviation must be finite and nonnegative")
        self.motion_sd = float(motion_sd)
        self._z = 0.
        self._variance = INITIAL_SD**2
        self.n_updates = 0

    @property
    def ua(self):
        return NOMINAL * exp(self._z)

    def update(self, prev_obs, applied, next_obs):
        observed = _observation(next_obs)[1]
        model = _single_model(_observation(prev_obs), _control(applied), self._z)
        expected, derivative, noise = _finite(model, "temperature likelihood")
        if noise <= 0:
            raise ArithmeticError("nonpositive conditional temperature variance")

        prior = self._variance + self.motion_sd**2
        innovation = observed - expected
        total = noise + derivative**2 * prior
        gain = prior * derivative / total
        z = float(np.clip(self._z + gain * innovation, log(LOW / NOMINAL), log(HIGH / NOMINAL)))
        variance = (1 - gain * derivative)**2 * prior + gain**2 * noise
        score = float(_log_density(innovation, total))
        if not all(map(isfinite, (z, variance, total, score))) or variance <= 0:
            raise ArithmeticError("invalid scalar parameter update")
        before = self.ua
        self._z, self._variance = z, float(variance)
        self.n_updates += 1
        return dict(n_updates=self.n_updates, ua_before=before, ua_after=self.ua,
                    temperature_prediction=float(expected), temperature_innovation=float(innovation),
                    temperature_variance=float(total), conditional_temperature_variance=float(noise),
                    temperature_log_ua_derivative=float(derivative), log_predictive_density=score,
                    log_ua_variance=self._variance)


class GridReference:
    """Discrete Bayesian reference with a local transition and a uniform restart."""

    def __init__(self):
        self._grid = np.arange(LOW, HIGH + GRID_STEP, GRID_STEP)
        self._z = np.log(self._grid / NOMINAL)
        centered = self._grid - NOMINAL
        prior = -0.5 * (centered / (NOMINAL * INITIAL_SD))**2
        self._log_weights = prior - np.logaddexp.reduce(prior)
        displacement = self._grid[None, :] - self._grid[:, None]
        local = -0.5 * (displacement / (MOTION_SD * self._grid[:, None]))**2
        local -= np.logaddexp.reduce(local, axis=1, keepdims=True)
        self._log_transition = np.logaddexp(np.log1p(-HAZARD) + local, log(HAZARD / len(prior)))
        self.n_updates = 0

    @property
    def ua(self):
        return float(NOMINAL + np.exp(self._log_weights) @ (self._grid - NOMINAL))

    def update(self, prev_obs, applied, next_obs):
        observed = _observation(next_obs)[1]
        model = _grid_model(_observation(prev_obs), _control(applied), self._z)
        expected, _, noise = _finite(model, "grid temperature likelihoods")
        if np.any(noise <= 0):
            raise ArithmeticError("nonpositive conditional temperature variance")

        prior = np.logaddexp.reduce(self._log_weights[:, None] + self._log_transition, axis=0)
        prior -= np.logaddexp.reduce(prior)
        innovations = observed - expected
        joint = prior + _log_density(innovations, noise)
        peak = np.max(joint)
        relative = joint - peak
        log_mass = np.logaddexp.reduce(relative)
        score = peak + log_mass
        weights = np.exp(prior)
        mean = weights @ expected
        variance = weights @ (noise + (expected - mean)**2)
        if not np.isfinite(joint).all() or not all(map(isfinite, (score, mean, variance))) or variance <= 0:
            raise ArithmeticError("invalid grid parameter update")
        before = self.ua
        self._log_weights = relative - log_mass
        self.n_updates += 1
        return dict(n_updates=self.n_updates, ua_before=before, ua_after=self.ua,
                    temperature_prediction=float(mean), temperature_innovation=float(observed - mean),
                    temperature_variance=float(variance), log_predictive_density=float(score))


def warmup():
    """Compile the predictor and both likelihood kernels using disposable state."""
    obs, applied = np.zeros(2), 0.
    following = predict(obs, applied, NOMINAL)
    Learner().update(obs, applied, following)
    GridReference().update(obs, applied, following)
