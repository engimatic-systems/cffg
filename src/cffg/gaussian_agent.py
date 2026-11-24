"""Two-window control with a full Gaussian belief and known affine commands."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import jax
import numpy as np

from cffg.bundle import load_bundle
from cffg.compiler import compile_template
from cffg.core import GaussianFilterSpec, GaussianPlanSpec
from cffg.engine import Engine
from cffg.neural_artifacts import array_digest, write_artifact
from cffg.params import set_params
from cffg.readouts.gaussian import _resolved_correlation
from cffg.runtime_types import _replace, RunStatus


@dataclass(frozen=True)
class GaussianDecision:
    tick: int
    command: np.ndarray
    free_energy: float
    status: RunStatus = RunStatus.CONVERGED


def _array(value, shape, dtype, label, *, exact=False):
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu"
            or (exact and raw.dtype != dtype)):
        raise ValueError(f"invalid {label} shape or dtype")
    value = raw.astype(dtype)
    if not np.isfinite(value).all():
        raise ValueError(f"{label} must be finite at the execution precision")
    return value


def _snapshot_digest(metadata, arrays):
    encoded = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    return array_digest({**arrays, "metadata": encoded})


def _moments(mean, cov, dim, dtype, *, exact=False):
    mean = _array(mean, (dim,), dtype, "Gaussian mean", exact=exact)
    cov = _array(cov, (dim, dim), dtype, "Gaussian covariance", exact=exact)
    try:
        with np.errstate(invalid="raise", divide="raise", over="raise"):
            _resolved_correlation(cov)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise ValueError("Gaussian covariance is unresolved at the execution precision") from error
    return mean, cov


class GaussianRollingAgent:
    """Fixed affine models, deterministic mean actions and raise-on-failure planning."""

    def __init__(self, plan_path, filter_path, *, n_sweeps=100, tol=1e-7):
        if type(n_sweeps) is not int or n_sweeps < 1 or not np.isfinite(tol) or tol <= 0:
            raise ValueError("positive iteration budget and finite positive tolerance required")
        plan, filt = load_bundle(plan_path), load_bundle(filter_path)
        self._p, self._f = plan.template.agent, filt.template.agent
        if not isinstance(self._p, GaussianPlanSpec) or not isinstance(self._f, GaussianFilterSpec):
            raise ValueError("expected Gaussian planning and filtering bundle declarations")
        if (self._p.coordinates != self._f.coordinates or self._p.commands != self._f.commands):
            raise ValueError("planning and filter coordinate names and order must agree")
        for bundle in (plan, filt):
            if bundle.learnables.plans:
                raise ValueError("this Gaussian controller requires a fixed-parameter model")
            if bundle.template.plates or bundle.template.constraints or any(
                factor.type not in {"gaussian", "gaussian_linear", "equality"}
                for factor in bundle.template.factors.values()
            ):
                raise ValueError("this controller supports flat affine Gaussian windows")
        self.plan = Engine(compile_template(plan.template, arrays=plan.arrays))
        self.filter = Engine(compile_template(filt.template, arrays=filt.arrays))
        self._x64 = bool(jax.config.jax_enable_x64)
        self._dtype = self.plan.init_state().g_eta.dtype
        self._dim = len(self._p.coordinates)
        self._nu = len(self._p.commands)
        self._budget = n_sweeps
        self._tol = float(tol)
        identity = {
            "structures": [b.manifest["structure_hash"] for b in (plan, filt)],
            "blobs": [b.manifest["blob_sha256"] for b in (plan, filt)],
            "dtype": self._dtype.str, "x64": self._x64, "n_sweeps": n_sweeps, "tol": self._tol,
        }
        self._identity = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        prior = plan.template.factors[self._p.prior].param_blobs
        self._initial = _moments(plan.arrays[prior["mean"]], plan.arrays[prior["cov"]],
                                 self._dim, self._dtype)
        offset_name = filt.template.factors[self._f.transition].param_blobs["offset"]
        self._offset = np.asarray(filt.arrays[offset_name], dtype=self._dtype)
        self._control = _array(filt.arrays[self._f.control_matrix],
                               (len(self._offset), self._nu), self._dtype, "control matrix")
        self.tick = 0
        self._pending = None
        self.reset_episode()

    def reset_episode(self):
        """Start from the declared planning prior. Pending physical interactions must finish."""
        if self._pending is not None:
            raise ValueError("cannot reset with an outstanding decision")
        self._moments = self._initial
        self._applied = np.empty(0, dtype=self._dtype)
        self._pending = None
        self.episode_tick = 0
        return self.belief

    @property
    def belief(self):
        return tuple(x.copy() for x in self._moments)

    def _run(self, engine, spec, data, updates):
        if bool(jax.config.jax_enable_x64) != self._x64:
            raise ValueError("execution precision changed since the agent was created")
        state = engine.init_state()
        mean, cov = self._moments
        params = set_params(engine, state.params, {
            spec.prior: {"mean": mean, "cov": cov}, **updates,
        })
        state = engine.bind_data(_replace(state, params=params), data)
        result = engine.run(state, self._budget, tol=self._tol)
        result.require_converged()
        return result

    def propose(self, preview):
        """Return this tick's raw vector mean. Repeated calls return an independent copy."""
        if set(preview) != set(self._p.preview_ports):
            raise ValueError("preview must name every declared planning DATA port")
        data = {}
        for port, value in preview.items():
            dim = self.plan.arena.gauss_data[port][1]
            data[port] = None if value is None else _array(value, (dim,), self._dtype, port)
        key = array_digest({p: np.empty(0, self._dtype) if v is None else v for p, v in data.items()})
        if self._pending is None:
            result = self._run(self.plan, self._p, data, {})
            mean, _ = self.plan.gaussian_marginal(result.state, self._p.action)
            command = _array(mean[list(self._p.indices)], (self._nu,), self._dtype, "proposal")
            self._pending = key, command, float(result.free_energies[-1])
        saved_key, command, energy = self._pending
        if saved_key != key:
            raise ValueError("preview changed for the outstanding decision")
        return GaussianDecision(self.tick, command.copy(), energy)

    def observe(self, applied, observation, *, tick):
        """Commit one actual command and its next sensor reading, or None for missing."""
        if type(tick) is not int or tick != self.tick or self._pending is None:
            raise ValueError("feedback requires the outstanding decision tick")
        applied = _array(applied, (self._nu,), self._dtype, "applied action")
        port = self._f.observation_port
        if observation is not None:
            observation = _array(observation, (self.filter.arena.gauss_data[port][1],),
                                 self._dtype, port)
        offset = self._offset + self._control @ applied
        result = self._run(self.filter, self._f, {port: observation},
                           {self._f.transition: {"offset": offset}})
        moments = self.filter.gaussian_marginal(result.state, self._f.posterior)
        self._moments = _moments(*moments, self._dim, self._dtype)
        self._applied = applied
        self._pending = None
        self.tick += 1
        self.episode_tick += 1
        return self.belief

    def checkpoint(self, path):
        """Atomically save a committed boundary or an outstanding proposal."""
        pending = self._pending
        arrays = dict(mean=self._moments[0], cov=self._moments[1], applied=self._applied,
                      proposal=np.empty(0, self._dtype) if pending is None else pending[1])
        metadata = dict(version=2, identity=self._identity, tick=self.tick, episode_tick=self.episode_tick,
                        preview=None if pending is None else pending[0],
                        energy=None if pending is None else pending[2])
        metadata["digest"] = _snapshot_digest(metadata, arrays)
        write_artifact(path, metadata, arrays)

    def restore(self, path):
        """Validate the complete checkpoint against the two loaded models before committing."""
        with np.load(path, allow_pickle=False) as data:
            fields = {"metadata", "mean", "cov", "applied", "proposal"}
            if len(data.files) != len(fields) or set(data.files) != fields:
                raise ValueError("invalid Gaussian agent checkpoint inventory")
            raw = data["metadata"]
            if raw.shape != () or raw.dtype.kind != "U":
                raise ValueError("invalid Gaussian agent checkpoint metadata")
            meta = json.loads(str(raw))
            arrays = {name: data[name] for name in fields - {"metadata"}}
        if (not isinstance(meta, dict)
                or set(meta) != {"version", "identity", "tick", "episode_tick", "preview", "energy", "digest"}
                or type(meta["version"]) is not int or meta["version"] != 2
                or meta["identity"] != self._identity):
            raise ValueError("incompatible Gaussian agent checkpoint")
        if meta.pop("digest") != _snapshot_digest(meta, arrays):
            raise ValueError("incompatible Gaussian agent checkpoint")
        tick, episode = meta["tick"], meta["episode_tick"]
        if type(tick) is not int or type(episode) is not int or not 0 <= episode <= tick:
            raise ValueError("invalid Gaussian agent cadence")
        moments = _moments(arrays["mean"], arrays["cov"], self._dim, self._dtype, exact=True)
        applied = _array(arrays["applied"], (self._nu if episode else 0,), self._dtype,
                         "previous action", exact=True)
        key, energy = meta["preview"], meta["energy"]
        command = _array(arrays["proposal"], (0 if key is None else self._nu,), self._dtype,
                         "proposal", exact=True)
        if key is None and energy is None:
            pending = None
        elif (isinstance(key, str) and len(key) == 64 and all(c in "0123456789abcdef" for c in key)
              and type(energy) is float and np.isfinite(energy)):
            pending = key, command, energy
        else:
            raise ValueError("invalid pending Gaussian decision")
        self._moments, self._applied, self._pending = moments, applied, pending
        self.tick, self.episode_tick = tick, episode
