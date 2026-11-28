"""Two-window control with one carried joint Gaussian belief."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import wraps
from hashlib import sha256
import json
from threading import RLock

import jax
import numpy as np

from cffg.bundle import load_bundle
from cffg.compiler import compile_template
from cffg.core import GaussianFilterSpec, GaussianPlanSpec, HybridFilterSpec, HybridPlanSpec
from cffg.ef import resolved_correlation
from cffg.engine import Engine
from cffg.neural_artifacts import array_digest
from cffg.params import set_params
from cffg.runtime_types import FailureDiagnostic, InferenceError, RunStatus, _replace


@dataclass(frozen=True)
class GaussianDecision:
    """A proposal or marked hold at one outstanding controller tick."""

    tick: int
    command: np.ndarray | int
    free_energy: float | None
    status: RunStatus = RunStatus.CONVERGED
    probabilities: np.ndarray | None = None
    held: bool = False
    failure: FailureDiagnostic | None = None


@dataclass(frozen=True)
class _Pending:
    preview: str
    decision: GaussianDecision
    feedback: str | None = None


def _array(value, shape, dtype, label, *, exact=False):
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu"
            or (exact and raw.dtype != dtype)):
        raise ValueError(f"invalid {label} shape or dtype")
    with np.errstate(over="ignore", invalid="ignore"):
        value = raw.astype(dtype)
    if not np.isfinite(value).all():
        raise ValueError(f"{label} must be finite at the execution precision")
    return value


def _moments(mean, cov, dim, dtype, *, exact=False):
    mean = _array(mean, (dim,), dtype, "Gaussian mean", exact=exact)
    cov = _array(cov, (dim, dim), dtype, "Gaussian covariance", exact=exact)
    try:
        with np.errstate(invalid="raise", divide="raise", over="raise"):
            resolved_correlation(cov)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise ValueError("Gaussian covariance is unresolved at the execution precision") from error
    return mean, cov


def _prior_digest(priors):
    return array_digest({
        json.dumps([name, field]): value
        for name, values in priors.items()
        for field, value in values.items()
    })


def _validate_learning(plan, filt):
    prior = filt.learnables.initial_params
    for name, values in plan.learnables.initial_params.items():
        if name not in prior or _prior_digest({name: values}) != _prior_digest({name: prior[name]}):
            raise ValueError(f"plan learnable {name!r} requires the same immutable filter prior")
    if any(not p.sources for p in filt.learnables.plans.values()):
        raise ValueError("each filter learnable requires an observation-source factor")


def _copy(value):
    return value.copy() if isinstance(value, np.ndarray) else value


def _synchronized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


class GaussianRollingAgent:
    """Carry a proper Gaussian across affine or known-selector hybrid windows."""

    def __init__(self, plan_path, filter_path, *, n_sweeps=100, tol=1e-7, select="map",
                 seed=0, on_failure="raise", initial_command=None):
        if type(n_sweeps) is not int or n_sweeps < 1 or not np.isfinite(tol) or tol <= 0:
            raise ValueError("positive iteration budget and finite positive tolerance required")
        if select not in ("map", "sample"):
            raise ValueError("select must be 'map' or 'sample'")
        if on_failure not in ("raise", "hold"):
            raise ValueError("on_failure must be 'raise' or 'hold'")
        if on_failure == "hold" and initial_command is None:
            raise ValueError("hold policy requires an initial command")
        if on_failure == "raise" and initial_command is not None:
            raise ValueError("initial_command requires the hold policy")
        plan, filt = load_bundle(plan_path), load_bundle(filter_path)
        self._p, self._f = plan.template.agent, filt.template.agent
        self._hybrid = isinstance(self._p, HybridPlanSpec)
        if not ((self._hybrid and isinstance(self._f, HybridFilterSpec))
                or (isinstance(self._p, GaussianPlanSpec) and isinstance(self._f, GaussianFilterSpec))):
            raise ValueError("expected matching Gaussian planning and filtering bundle declarations")
        if (self._p.coordinates != self._f.coordinates or self._p.commands != self._f.commands):
            raise ValueError("planning and filter coordinate names and order must agree")
        if select == "sample" and not self._hybrid:
            raise ValueError("sample selection requires a categorical command")
        for bundle in (plan, filt):
            factors = {"gaussian", "gaussian_linear", "equality"}
            if self._hybrid:
                factors |= {"categorical", "gaussian_mixture_linear"}
            if bundle.template.plates or bundle.template.constraints or any(
                factor.type not in factors for factor in bundle.template.factors.values()
            ):
                raise ValueError("Gaussian control requires flat affine or known-selector Gaussian windows")
        _validate_learning(plan, filt)
        self.plan = Engine(compile_template(plan.template, arrays=plan.arrays))
        self.filter = Engine(compile_template(filt.template, arrays=filt.arrays))
        self.learnables = filt.learnables
        # Check family, shapes and target ownership at open, before planning.
        self.learnables.write(self.plan, self.plan.init_state().params)

        self._x64 = bool(jax.config.jax_enable_x64)
        self._dtype = self.plan.init_state().g_eta.dtype
        self._dim = len(self._p.coordinates)
        self._nu = len(self._p.commands)
        self._budget = n_sweeps
        self._tol = float(tol)
        self._select = select
        self._on_failure = on_failure
        self._rng = np.random.default_rng(seed)
        self._initial_command = None if initial_command is None else self._command(
            initial_command, "initial held action"
        )
        self._legacy = not self._hybrid and not self.learnables.plans and on_failure == "raise"
        identity = {
            "structures": [b.manifest["structure_hash"] for b in (plan, filt)],
            "blobs": [b.manifest["blob_sha256"] for b in (plan, filt)],
            "dtype": self._dtype.str, "x64": self._x64, "n_sweeps": n_sweeps, "tol": self._tol,
        }
        if not self._legacy:
            identity.update(
                mode="hybrid" if self._hybrid else "continuous",
                prior=_prior_digest(self.learnables.initial_params), select=select, on_failure=on_failure,
                hold=None if initial_command is None else array_digest({"action": self._command_array(self._initial_command)}),
            )
        self._identity = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        prior = plan.template.factors[self._p.prior].param_blobs
        self._initial = _moments(plan.arrays[prior["mean"]], plan.arrays[prior["cov"]],
                                 self._dim, self._dtype)
        if not self._hybrid:
            offset_name = filt.template.factors[self._f.transition].param_blobs["offset"]
            self._offset = np.asarray(filt.arrays[offset_name], dtype=self._dtype)
            self._control = _array(filt.arrays[self._f.control_matrix],
                                   (len(self._offset), self._nu), self._dtype, "control matrix")
        self._lock = RLock()
        self.tick = 0
        self._pending = None
        self.reset_episode()

    @_synchronized
    def reset_episode(self):
        """Start from the declared initial belief and retain the learned posterior."""
        if self._pending is not None:
            raise ValueError("cannot reset with an outstanding decision")
        self._moments = self._initial
        self._applied = None
        self.episode_tick = 0
        return self.belief

    @property
    @_synchronized
    def belief(self):
        return tuple(x.copy() for x in self._moments)

    def _command(self, value, label, *, exact=False):
        if not self._hybrid:
            return _array(value, (self._nu,), self._dtype, label, exact=exact)
        command = np.asarray(value)
        if (command.shape != () or command.dtype.kind not in "iu"
                or (exact and command.dtype != np.dtype(np.int64))
                or not 0 <= int(command) < self._nu):
            raise ValueError(f"invalid {label} selector")
        return int(command)

    def _command_array(self, value):
        dtype = np.int64 if self._hybrid else self._dtype
        return np.empty(0, dtype) if value is None else np.asarray(value, dtype=dtype)

    def _run(self, engine, spec, data, updates):
        if bool(jax.config.jax_enable_x64) != self._x64:
            raise ValueError("execution precision changed since the agent was created")
        state = engine.init_state()
        mean, cov = self._moments
        params = set_params(engine, self.learnables.write(engine, state.params), {
            spec.prior: {"mean": mean, "cov": cov}, **updates,
        })
        state = engine.bind_data(_replace(state, params=params), data)
        result = engine.run(state, self._budget, tol=self._tol)
        result.require_converged()
        return result

    def _new_decision(self, data):
        result = self._run(self.plan, self._p, data, {})
        energy = result.free_energies[-1] if result.free_energies else np.nan
        if not np.isfinite(energy):
            diagnostic = FailureDiagnostic("nonfinite_objective", None, "ledger", result.iterations)
            raise InferenceError(replace(result, status=RunStatus.DIVERGED, failure=diagnostic))
        q = None
        if self._hybrid:
            q = np.asarray(self.plan.marginals(result.state, [self._p.action])[self._p.action],
                           dtype=np.float64)
            mass = q.sum()
            valid = q.shape == (self._nu,) and np.all(np.isfinite(q) & (q >= 0)) and np.isfinite(mass) and mass > 0
            if valid:
                q = q / mass
                command = int(np.argmax(q) if self._select == "map" else self._rng.choice(self._nu, p=q))
        else:
            mean, _ = self.plan.gaussian_marginal(result.state, self._p.action)
            command = mean[list(self._p.indices)]
            valid = np.isfinite(command).all()
        if not valid:
            diagnostic = FailureDiagnostic("invalid_action_readout", None, "readout", result.iterations)
            raise InferenceError(replace(result, status=RunStatus.DIVERGED, failure=diagnostic))
        return GaussianDecision(self.tick, command, float(energy), probabilities=q)

    @_synchronized
    def propose(self, preview):
        """Stage a decision before physical execution, with selection at most once per tick."""
        if set(preview) != set(self._p.preview_ports):
            raise ValueError("preview must name every declared planning DATA port")
        data = {}
        for port, value in preview.items():
            dim = self.plan.arena.gauss_data[port][1]
            data[port] = None if value is None else _array(value, (dim,), self._dtype, port)
        key = array_digest({p: np.empty(0, self._dtype) if v is None else v for p, v in data.items()})
        if self._pending is None:
            try:
                decision = self._new_decision(data)
            except InferenceError as error:
                if self._on_failure == "raise":
                    raise
                result = error.result
                command = self._initial_command if self._applied is None else self._applied
                diagnostic = result.failure or FailureDiagnostic(
                    result.status.value, None, "plan", result.iterations
                )
                decision = GaussianDecision(
                    self.tick, command, None, result.status, held=True, failure=diagnostic,
                )
            self._pending = _Pending(key, decision)
        if self._pending.preview != key:
            raise ValueError("preview changed for the outstanding decision")
        decision = self._pending.decision
        return replace(decision, command=_copy(decision.command), probabilities=_copy(decision.probabilities))

    @_synchronized
    def observe(self, applied, observation, *, tick):
        """Commit one applied command and next sensor reading, or predict for None."""
        if type(tick) is not int or tick != self.tick or self._pending is None:
            raise ValueError("feedback requires the outstanding decision tick")
        applied = self._command(applied, "applied action")
        port = self._f.observation_port
        if observation is not None:
            observation = _array(observation, (self.filter.arena.gauss_data[port][1],),
                                 self._dtype, port)
        feedback = array_digest({
            "action": self._command_array(applied),
            "observation": np.empty(0, self._dtype) if observation is None else observation,
        })
        if self._pending.feedback is not None and self._pending.feedback != feedback:
            raise ValueError("feedback changed for the outstanding decision")
        self._pending = replace(self._pending, feedback=feedback)
        data, updates = {port: observation}, {}
        if self._hybrid:
            data[self._f.selector_port] = applied
        else:
            offset = self._offset + self._control @ applied
            updates[self._f.transition] = {"offset": offset}
        result = self._run(self.filter, self._f, data, updates)
        moments = _moments(*self.filter.gaussian_marginal(result.state, self._f.posterior),
                           self._dim, self._dtype)
        if observation is not None and self.learnables.plans:
            # The learner commits all posterior owners together after validation.
            self.learnables.accumulate(self.filter, result)
        self._moments, self._applied, self._pending = moments, applied, None
        self.tick += 1
        self.episode_tick += 1
        return self.belief

    @_synchronized
    def checkpoint(self, path):
        """Atomically save a committed boundary or an outstanding decision."""
        from cffg._gaussian_agent_checkpoint import save

        save(self, path)

    @_synchronized
    def restore(self, path):
        """Validate the complete checkpoint before replacing live state."""
        from cffg._gaussian_agent_checkpoint import restore

        restore(self, path)
