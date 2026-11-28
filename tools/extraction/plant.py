"""Physical five-stage extraction column and scorer-owned paired experiment tapes.

The truth integrator uses PCGym's native right-hand side with a symbolic Kla.
The filter sees only y0, the action actually applied and each new outlet reading.
Time is in model hours, concentrations and both flows are in plant units.
"""

from functools import lru_cache
from importlib.metadata import distribution
import json

import numpy as np


DT = 0.05
OUTLETS = np.array([8, 1])  # X5, Y1
HIDDEN = np.array([0, 2, 3, 4, 5, 6, 7, 9])
NOMINAL_FLOW = np.array([75.0, 30.0])
FLOW_LOW = np.array([45.0, 18.0])
FLOW_HIGH = np.array([105.0, 45.0])
NOMINAL_KLA = 5.0
CHANGED_KLA = 4.5
SENSOR_SD = np.array([0.002, 0.003])
PROCESS_SD = 0.001
INITIAL_SD = 0.035
LOG_KLA_SD = 0.10
PCGYM_COMMIT = "d26cbc0690024efef488dc6a304d37654934e525"


def pcgym_revision():
    dist = distribution("pcgym")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id")
    if dist.version != "0.1.8" or revision != PCGYM_COMMIT:
        raise RuntimeError(f"Expected PCGym 0.1.8 at {PCGYM_COMMIT}, got {dist.version} at {revision}")
    return revision


def _native(kla):
    from pcgym.model_classes import multistage_extraction

    return multistage_extraction(Kla=kla, int_method="casadi")


@lru_cache(maxsize=1)
def _steady():
    from scipy.optimize import root

    pcgym_revision()
    model = _native(NOMINAL_KLA)
    sol = root(lambda x: model(x, NOMINAL_FLOW), np.tile([0.33, 0.45], 5), tol=1e-12)
    if not sol.success or max(abs(sol.fun)) > 1e-10:
        raise RuntimeError("Nominal extraction steady-state solve failed")
    return sol.x


def nominal_profile():
    return _steady().copy()


def observe(state, noise):
    """Return exactly the two noisy physical outlet readings in X5,Y1 order."""
    if np.shape(state) != (10,) or np.shape(noise) != (2,):
        raise ValueError("Expected ten physical concentrations and two additive errors")
    return np.asarray(state)[OUTLETS] + np.asarray(noise)


class Plant:
    """Native PCGym extraction ODE, integrated by CasADi CVODES in float64."""

    def __init__(self):
        import casadi as ca

        pcgym_revision()
        x = ca.SX.sym("x", 10)
        p = ca.SX.sym("p", 3)  # L, G, physical Kla
        dx = ca.vertcat(*_native(p[2])(x, p[:2]))
        self._step = ca.integrator("truth", "cvodes", {"x": x, "p": p, "ode": dx}, 0, DT,
                                   {"abstol": 1e-12, "reltol": 1e-10})

    def step(self, x10, u2, kla):
        """Advance one transition using the applied flows, before the stored process shock."""
        x, u = np.asarray(x10, dtype=float), np.asarray(u2, dtype=float)
        if x.shape != (10,) or not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError("Expected ten finite nonnegative physical concentrations")
        if u.shape != (2,) or not np.all((FLOW_LOW <= u) & (u <= FLOW_HIGH)):
            raise ValueError("Applied flows must lie inside the physical control envelope")
        if np.ndim(kla) or not np.isfinite(kla) or kla <= 0:
            raise ValueError("Expected a finite positive scalar Kla")
        out = self._step(x0=x, p=np.r_[u, kla])["xf"].full().ravel()
        if not np.isfinite(out).all() or np.any(out < 0):
            raise ArithmeticError("Invalid physical concentrations after integration")
        return out


def _probe(rng, n, channel):
    """Independent, sustained flow perturbations with random asynchronous holds."""
    lo, hi = FLOW_LOW[channel], FLOW_HIGH[channel]
    out = np.empty(n)
    start = 0
    while start < n:
        end = start + int(rng.integers(5, 18))
        out[start:end] = rng.uniform(lo, hi)
        start = end
    return out


def _schedule(n, condition, switch_rng, goal_rng):
    kla = np.full(n, NOMINAL_KLA)
    event = np.full(n, "nominal", dtype="<U12")
    target = np.empty(n)
    for offset in range(0, n, 600):
        k = np.arange(min(600, n - offset))
        # A 120-tick reference repeats at both abrupt visits. Its phase and levels
        # come from a stream independent of the plant, probes and switch times.
        shift = goal_rng.integers(-4, 5)
        lo, hi = np.array([0.30, 0.37]) + goal_rng.uniform(-0.005, 0.005, 2)
        target[offset:offset + len(k)] = np.where((k + shift) % 120 < 60, lo, hi)
        start = 80 + switch_rng.integers(-4, 5)
        dwell = 65 + switch_rng.integers(-3, 4)
        ramp = 450 + switch_rng.integers(-4, 5)
        if condition == "stationary":
            continue
        levels = kla[offset:offset + len(k)]
        events = event[offset:offset + len(k)]
        for visit, onset in enumerate((start, start + 240), start=1):
            changed = (k >= onset) & (k < onset + dwell)
            levels[changed] = CHANGED_KLA
            events[changed] = f"step{visit}"
            events[k >= onset + dwell] = f"return{visit}"
        # Twenty equal increments each way; the last up increment is nominal.
        age = k - ramp
        down, hold, up = ((age >= a) & (age < b) for a, b in [(0, 20), (20, 56), (56, 75)])
        delta = NOMINAL_KLA - CHANGED_KLA
        levels[down] = NOMINAL_KLA - delta * (age[down] + 1) / 20
        levels[hold] = CHANGED_KLA
        levels[up] = CHANGED_KLA + delta * (age[up] - 55) / 20
        events[down], events[hold], events[up] = "ramp_down", "ramp_hold", "ramp_up"
        events[age >= 75] = "return_ramp"
    return kla, event, target


def design(seed, n, condition):
    """Pre-generate scorer-owned exogenous streams; row t is transition t -> t+1.

    y0 is the reset reading. noise[t] is the physical additive error for the
    next reading, after adding process[t] to the deterministic transition.
    target[t] applies to that next physical X5. All streams are prefix-stable.
    """
    if condition not in ("stationary", "changing") or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("Expected a positive number of transitions and stationary or changing")
    init, reset, sensor, proc, liquid, gas, switches, goals = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(8))
    initial = nominal_profile() + init.normal(size=10) * INITIAL_SD
    if not np.all((initial > 0) & (initial < 1)):
        raise ValueError("Initial profile outside the physical experiment range")
    reset_noise = reset.standard_normal(2) * SENSOR_SD
    kla, event, target = _schedule(n, condition, switches, goals)
    return dict(initial=initial, reset_noise=reset_noise, y0=observe(initial, reset_noise),
                flows=np.column_stack((_probe(liquid, n, 0), _probe(gas, n, 1))),
                process=proc.standard_normal((n, 10)) * PROCESS_SD,
                noise=sensor.standard_normal((n, 2)) * SENSOR_SD,
                kla=kla, target=target, phase=(kla != NOMINAL_KLA).astype(int), event=event)
