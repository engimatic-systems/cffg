"""One physical extraction interval for joint concentration and log(Kla) inference.

The deterministic mean leaves log(Kla) unchanged. The filter adds its process
covariance after this map. States follow PCGym's interleaved X1,Y1,...,X5,Y5
ordering; the two inputs are applied liquid and gas flows in physical units.
"""
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import DOP853

from plant import DT


def _rhs(x, u, kla):
    liquid, gas = x[::2], x[1::2]
    x_up = jnp.concatenate((jnp.array([.6], dtype=x.dtype), liquid[:-1]))
    y_up = jnp.concatenate((gas[1:], jnp.array([.05], dtype=x.dtype)))
    transfer = kla * (liquid - gas * gas)
    dx = (u[0] / 5.) * (x_up - liquid) - transfer
    dy = (u[1] / 5.) * (y_up - gas) + transfer
    return jnp.stack((dx, dy), axis=1).ravel()


def _advance(zu, steps):
    x, log_kla, u = zu[:10], zu[10], zu[11:]
    kla = jnp.exp(log_kla)
    h = DT / steps

    def substep(_, current):
        slopes = [_rhs(current, u, kla)]
        for row in DOP853.A[1:]:
            increment = sum(row[j] * slopes[j] for j in np.flatnonzero(row))
            slopes.append(_rhs(current + h * increment, u, kla))
        increment = sum(DOP853.B[j] * slopes[j] for j in np.flatnonzero(DOP853.B))
        return current + h * increment

    nxt = jax.lax.fori_loop(0, steps, substep, x)
    return jnp.concatenate((nxt, zu[10:11]))


def _with_value(zu, steps):
    nxt = _advance(zu, steps)
    return nxt, nxt


_predict = jax.jit(_advance)
_derivatives = jax.jit(jax.jacfwd(_with_value, has_aux=True))


@cache
def _cvodes():
    """Stiff fallback, also used as the independent native validation reference."""
    import casadi as ca
    from pcgym.model_classes import multistage_extraction

    z = ca.SX.sym("z", 11)
    u = ca.SX.sym("u", 2)
    model = multistage_extraction(Kla=ca.exp(z[10]), int_method="casadi")
    ode = ca.vertcat(*model(z[:10], u), 0.)
    flow = ca.integrator("extraction_flow", "cvodes", {"x": z, "p": u, "ode": ode},
                         0., DT, {"reltol": 1e-12, "abstol": 1e-14})
    nxt = flow(x0=z, p=u)["xf"]
    return (ca.Function("native_step", [z, u], [nxt]),
            ca.Function("native_tangent", [z, u], [ca.jacobian(nxt, z), ca.jacobian(nxt, u), nxt]))


def _inputs(m11, u2):
    m, u = np.asarray(m11, dtype=np.float64), np.asarray(u2, dtype=np.float64)
    if m.shape != (11,) or u.shape != (2,) or not np.isfinite(m).all() or not np.isfinite(u).all():
        raise ValueError("Expected 11 finite state coordinates and two finite flows")
    with np.errstate(over="ignore"):
        kla = np.exp(m[10])
    if not 0. < kla < np.inf:
        raise ValueError("log(Kla) must give a positive finite coefficient")
    # Subdivide according to advection and the magnitude of the local transfer
    # derivatives. Keep this integer fixed when differentiating the discrete flow.
    rate = np.max(np.abs(u)) / 5. + kla * (1. + 2. * max(1., np.max(np.abs(m[1:10:2]))))
    stiffness = DT * rate
    steps = max(1, int(np.ceil(stiffness / .4))) if stiffness <= 16. else None
    return m, u, steps


def step(m11, u2):
    """Advance the full 11-coordinate mean one interval."""
    m, u, steps = _inputs(m11, u2)
    if steps is None:
        nxt = np.asarray(_cvodes()[0](m, u)).ravel()
    else:
        with jax.enable_x64(True):
            nxt = np.asarray(_predict(np.r_[m, u], steps))
    if not np.isfinite(nxt).all():
        raise FloatingPointError("Nonfinite extraction prediction")
    return nxt


def linearize(m11, u2):
    """Return A, B, b with the tangent centered at (m11, u2) and f = A m + B u + b."""
    m, u, steps = _inputs(m11, u2)
    if steps is None:
        A, B, nxt = map(np.asarray, _cvodes()[1](m, u))
        nxt = nxt.ravel()
    else:
        with jax.enable_x64(True):
            jac, nxt = map(np.asarray, _derivatives(np.r_[m, u], steps))
        A, B = jac[:, :11], jac[:, 11:]
    b = nxt - A @ m - B @ u
    if not all(np.isfinite(x).all() for x in (A, B, b)):
        raise FloatingPointError("Nonfinite extraction linearization")
    return A, B, b
