"""Joint Laplace control with NumPyro's MAP guide and its observation-dependent ELBO."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
from scipy.linalg import cho_factor, cho_solve

from cstr_runtime_fixture import HORIZON, PAST, cstr_rk4


_SITE_COUNT = PAST + HORIZON + 1
_SHAPE = (_SITE_COUNT, 4)
_GROUPS = (("past", slice(0, PAST)), ("now", slice(PAST, PAST + 1)),
           ("goal", slice(PAST + 1, _SITE_COUNT)))


def _pack(data):
    observations, masks = [], []
    for kind, sites in _GROUPS:
        width = 3 if kind == "past" else 2
        values = np.zeros((sites.stop - sites.start, width))
        present = np.zeros(len(values), dtype=bool)
        for row, index in enumerate(range(sites.start, sites.stop)):
            value = data[f"e{index}"]
            if value is not None:
                if np.shape(value) != (width,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {width}-dimensional observation at e{index}")
                values[row] = value
                present[row] = True
        observations.append(jnp.asarray(values))
        masks.append(jnp.asarray(present))
    return tuple(observations), tuple(masks)


def _model(arrays, config, affine):
    mean = jnp.asarray(arrays["m0"])
    prior = dist.MultivariateNormal(mean, covariance_matrix=arrays["P0"])
    noise = np.stack([arrays["Q"]] * PAST + [arrays["Qp"]] * HORIZON)
    process_chol = jnp.asarray(np.linalg.cholesky(noise))
    initial_theta = jnp.asarray(arrays["theta"])
    sensors = [(jnp.asarray(arrays[f"G_{kind}"]), jnp.asarray(arrays[f"z_{kind}"]),
                jnp.asarray(np.linalg.cholesky(arrays[f"R_{kind}"]))) for kind, _ in _GROUPS]
    if affine is None:
        transition = jax.vmap(lambda state, theta: cstr_rk4(state, theta, config), in_axes=(0, None))
    else:
        matrix, offset = map(jnp.asarray, affine)

        def transition(states, theta):
            return states @ matrix.T + offset

    def model(observations, masks, zero_residual=False, *, theta=initial_theta):
        # Lebesgue measure on the trajectory. Its complete normalized density comes
        # from one prior and nineteen conditional transition distributions below.
        states = numpyro.sample("states", dist.ImproperUniform(dist.constraints.real, (), _SHAPE))
        numpyro.sample("prior", prior, obs=states[0])
        predicted = transition(states[:-1], theta)
        # At zero residual the ELBO Hessian gives the positive Gauss-Newton
        # metric. Stop-gradient preserves all derivatives of the physical mean.
        predicted = predicted + jax.lax.stop_gradient(jnp.where(
            zero_residual, states[1:] - predicted, 0.))
        numpyro.sample("transition", dist.MultivariateNormal(predicted, scale_tril=process_chol)
                       .to_event(1), obs=states[1:])
        for (kind, sites), (sensor, offset, chol), values, mask in zip(
                _GROUPS, sensors, observations, masks, strict=True):
            expected = states[sites] @ sensor.T + offset
            likelihood = dist.MultivariateNormal(expected, scale_tril=chol)
            numpyro.sample(kind, likelihood.mask(mask).to_event(1), obs=values)

    return model


class Adapter:
    def __init__(self, arrays, config, *, affine=None):
        if not jax.config.x64_enabled or jax.default_backend() != "cpu":
            raise ValueError("NumPyro control requires JAX_ENABLE_X64=1 and JAX_PLATFORMS=cpu")
        self._theta = np.array(arrays["theta"], dtype=np.float64, copy=True)
        self.model = _model(arrays, config, affine)
        self.guide = AutoLaplaceApproximation(
            self.model, init_loc_fn=init_to_value(values={"states": jnp.zeros(_SHAPE)}))
        self.key = jax.random.key(0)
        empty = _pack({f"e{i}": None for i in range(_SITE_COUNT)})
        numpyro.handlers.seed(self.guide, self.key)(*empty)
        objective = Trace_ELBO()

        def loss(loc, observations, masks, zero_residual=False, *, theta=self._theta):
            # get_posterior() on this NumPyro guide captures its prototype data.
            # Evaluate its same ELBO/Hessian with the live window as JIT operands.
            return objective.loss(self.key, {"auto_loc": loc}, self.model, self.guide,
                                  observations, masks, zero_residual, theta=theta)

        self.loss = jax.jit(loss)
        self.value_grad = jax.jit(jax.value_and_grad(loss))
        self.curvature = jax.jit(jax.hessian(loss))

    def install(self, ua):
        """Install physical UA for subsequent windows. The RK4 transition stores log UA."""
        value = np.asarray(ua)
        if (value.ndim != 0 or not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value)
                or not np.isfinite(value) or value <= 0):
            raise ValueError("UA must be a positive finite scalar")
        log_ua = np.log(float(value))
        if log_ua != self._theta[2]:
            theta = self._theta.copy()
            theta[2] = log_ua
            self._theta = theta

    def _fit(self, observations, masks):
        theta = jnp.asarray(self._theta)
        loc = np.zeros(np.prod(_SHAPE))
        gauss_newton_steps = 0
        backtracks = 0
        for iteration in range(101):
            value, gradient = self.value_grad(loc, observations, masks, theta=theta)
            value, gradient = float(value), np.asarray(gradient)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise ValueError("Nonfinite MAP objective or gradient")

            hessian = self.curvature(loc, observations, masks, False, theta=theta)
            try:
                factor = cho_factor(np.asarray(hessian), lower=True)
                positive = True
            except np.linalg.LinAlgError:
                metric = self.curvature(loc, observations, masks, True, theta=theta)
                factor = cho_factor(np.asarray(metric), lower=True)
                gauss_newton_steps += 1
                positive = False
            step = cho_solve(factor, gradient)
            squared_decrement = float(gradient @ step)
            decrement = float(np.sqrt(squared_decrement))
            diagnostics = dict(iterations=iteration, objective=value,
                               gradient_max=float(np.max(abs(gradient))),
                               newton_decrement=decrement, hessian_positive=positive,
                               gauss_newton_steps=gauss_newton_steps, backtracks=backtracks)
            if positive and decrement <= 1e-6:
                return loc, factor, diagnostics
            if iteration == 100:
                break
            # Positive curvature gives a descent direction. Backtrack on the
            # actual guide ELBO, allowing only roundoff at numerical convergence.
            slack = 16 * np.finfo(loc.dtype).eps * max(abs(value), 1.)
            for attempt in range(24):
                rate = 0.5 ** attempt
                candidate = loc - rate * step
                trial = float(self.loss(candidate, observations, masks, theta=theta))
                if np.isfinite(trial) and trial <= value - 1e-4 * rate * squared_decrement + slack:
                    loc = candidate
                    backtracks += attempt
                    break
            else:
                diagnostics["stop_reason"] = "line_search_exhausted"
                return loc, None, diagnostics
        diagnostics["stop_reason"] = "iteration_budget"
        return loc, None, diagnostics

    def solve(self, data, *, details=False):
        try:
            center, factor, info = self._fit(*_pack(data))
        except (ValueError, np.linalg.LinAlgError) as error:
            return dict(status="invalid_fit", method="numpyro_joint_laplace", action=None, error=str(error))
        info["method"] = "numpyro_joint_laplace"
        if factor is None:
            return dict(status="unconverged", action=None, **info)
        state = slice(4 * PAST, 4 * PAST + 4)
        answer = dict(status="ok", action=float(center[state][-1]), **info)
        if details:
            covariance_columns = cho_solve(factor, np.eye(center.size)[:, state])
            answer.update(mean=center[state], cov=covariance_columns[state],
                          next_action_cov=float(covariance_columns[state.stop + 3, -1]))
        return answer
