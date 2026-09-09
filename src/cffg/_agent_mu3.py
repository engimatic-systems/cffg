"""Functional-mu3 planning state owned by :class:`RollingAgent`."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from cffg.engine import Engine, RunResult, RunStatus, State, _replace
from cffg.outer import OuterTerminationReason


_STATE_FIELDS = tuple(field.name for field in fields(State))
_RESULT_FIELDS = (
    "iterations",
    "message_deltas",
    "free_energies",
    "residual",
    "inner_residual",
    "mu3_ess",
    "mu3_log_normalizers",
    "outer_residual",
    "outer_merit",
    "outer_gradient_norm",
    "trust_radius",
    "accepted_steps",
    "rejected_steps",
    "outer_map_evaluations",
    "trial_evaluations",
    "linear_iterations",
    "solver_compilations",
)


@dataclass
class PlanningRuntime:
    """One agent's fixed-particle planning epoch and next-key cursor."""

    key_cursor: jax.Array | None
    refresh_policy: str = "trial"
    epoch: int = 0
    result: RunResult | None = None
    run_required: bool = True
    refresh_required: bool = True
    cache_config: dict | None = None

    def __post_init__(self) -> None:
        if self.refresh_policy not in {"trial", "posterior"}:
            raise ValueError("mu3_refresh must be 'trial' or 'posterior'")

    def invalidate(self) -> None:
        self.run_required = True
        if self.refresh_policy == "trial":
            self.refresh_required = True

    def refresh(self) -> None:
        self.run_required = True
        self.refresh_required = True

    def run(
        self,
        engine: Engine,
        initial: State,
        n_iters: int,
        *,
        tol: float,
        run_options: dict,
    ) -> RunResult:
        if not engine.arena.mu3_groups:
            self.result = engine.run(initial, n_iters, tol=tol, **run_options)
            self.cache_config = _run_config(engine, n_iters, tol, run_options)
            self.run_required = False
            self.refresh_required = False
            return self.result

        current_config = _run_config(engine, n_iters, tol, run_options)
        if self.result is not None:
            if self.cache_config != current_config:
                self.run_required = True
                if (
                    self.cache_config is not None
                    and self.cache_config.get("samples") != current_config["samples"]
                ):
                    self.refresh_required = True
            if not _same_base(engine, initial, self.result.state):
                self.run_required = True
                self.refresh_required = True
            elif not prepared_inputs_match(engine, initial, self.result.state):
                self.run_required = True
                if self.refresh_policy == "trial":
                    self.refresh_required = True
        if self.result is not None and not self.run_required:
            return self.result

        if self.refresh_required or self.result is None:
            if self.key_cursor is None:
                raise ValueError(
                    "functional mu3 planning requires an explicit JAX key "
                    "before the first planning run"
                )
            initial, self.key_cursor = engine.prepare_mu3(initial, self.key_cursor)
            self.epoch += 1
        else:
            initial = _reuse_particles(engine, initial, self.result.state)

        self.result = engine.run(initial, n_iters, tol=tol, **run_options)
        self.cache_config = current_config
        self.run_required = False
        self.refresh_required = False
        return self.result


def functional(engine: Engine) -> bool:
    """The compiled evaluator declaration, not an agent flag, is authoritative."""
    return bool(engine.arena.mu3_groups)


def prepared_inputs_match(engine: Engine, initial: State, solved: State) -> bool:
    """Compare immutable planning inputs while excluding mu3-derived values."""
    derived = {
        key
        for plan in engine.arena.mu3_groups
        for key in solved.params
        if key.startswith(plan.name + ":")
    }
    for plan in engine.arena.mu3_groups:
        derived.add(plan.gfe_name + ":Abar")
        derived.add(plan.gfe_name + ":hbar")
    return all(
        key in solved.params
        and np.array_equal(np.asarray(value), np.asarray(solved.params[key]))
        for key, value in initial.params.items()
        if key not in derived
    )


def checkpoint_runtime(
    data: dict[str, np.ndarray],
    engine: Engine,
    runtime: PlanningRuntime,
    *,
    n_iters: int,
    tol: float,
    run_options: dict,
) -> None:
    """Append a versioned stochastic planning epoch to an agent checkpoint."""
    current_config = _run_config(engine, n_iters, tol, run_options)
    stale_config = runtime.result is not None and runtime.cache_config != current_config
    stale_shape = bool(
        stale_config
        and runtime.cache_config is not None
        and runtime.cache_config.get("samples") != current_config["samples"]
    )
    stored_result = None if stale_shape else runtime.result
    metadata = {
        "epoch": runtime.epoch,
        "run_required": runtime.run_required or stale_config,
        "refresh_required": runtime.refresh_required or stale_shape,
        "refresh_policy": runtime.refresh_policy,
        "samples": engine.mu3_samples,
        "min_ess_fraction": engine.mu3_min_ess_fraction,
        "schedule_fingerprint": engine.schedule.fingerprint,
        "evaluators": _evaluator_identity(engine),
        "has_result": stored_result is not None,
        "cache_config": None if stale_shape else runtime.cache_config,
    }
    if runtime.key_cursor is None:
        metadata["has_key"] = False
    else:
        metadata |= {
            "has_key": True,
            "key_impl": str(jax.random.key_impl(runtime.key_cursor)),
            "typed_key": str(runtime.key_cursor.dtype).startswith("key<"),
        }
        data["mu3:key"] = np.asarray(jax.random.key_data(runtime.key_cursor))
    if stored_result is not None:
        metadata["result"] = _result_metadata(stored_result)
        _store_state(data, stored_result.state)
    data["mu3:metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))


def restore_runtime(
    data: Mapping[str, np.ndarray], engine: Engine
) -> PlanningRuntime:
    """Validate and reconstruct a stochastic epoch without drawing particles."""
    if "mu3:metadata" not in data:
        raise ValueError(
            "legacy checkpoint cannot restore a functional mu3 agent: "
            "the checkpoint has no key or prepared planning epoch"
        )
    metadata = json.loads(str(data["mu3:metadata"]))
    expected = {
        "samples": engine.mu3_samples,
        "min_ess_fraction": engine.mu3_min_ess_fraction,
        "schedule_fingerprint": engine.schedule.fingerprint,
        "evaluators": _evaluator_identity(engine),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"functional checkpoint {name} does not match this agent: "
                f"{metadata.get(name)!r} != {value!r}"
            )

    key = None
    if metadata.get("has_key"):
        if "mu3:key" not in data:
            raise KeyError("functional checkpoint is missing its JAX key cursor")
        key_data = jnp.asarray(data["mu3:key"])
        wrapped = jax.random.wrap_key_data(key_data, impl=metadata["key_impl"])
        key = wrapped if metadata.get("typed_key") else jax.random.key_data(wrapped)

    result = None
    if metadata.get("has_result"):
        state = _load_state(data, engine)
        result = _load_result(state, metadata.get("result", {}))
        _validate_prepared_state(engine, result)
    runtime = PlanningRuntime(
        key_cursor=key,
        refresh_policy=metadata["refresh_policy"],
        epoch=int(metadata["epoch"]),
        result=result,
        run_required=bool(metadata["run_required"]),
        refresh_required=bool(metadata["refresh_required"]),
        cache_config=metadata.get("cache_config"),
    )
    if result is None and not runtime.run_required:
        raise ValueError("functional checkpoint has no result but marks planning reusable")
    if result is not None and runtime.cache_config is None:
        raise ValueError("functional checkpoint result has no runtime configuration")
    return runtime


def _same_base(engine: Engine, initial: State, prepared: State) -> bool:
    for plan in engine.arena.mu3_groups:
        counts = np.asarray(initial.params[plan.count_key])[plan.rows]
        base = np.asarray(prepared.params[plan.name + ":base_alpha"])
        if not all(np.array_equal(row, base) for row in counts):
            return False
    return True


def _reuse_particles(engine: Engine, initial: State, prepared: State) -> State:
    if not _same_base(engine, initial, prepared):
        raise ValueError("cannot reuse mu3 particles after the base posterior changed")
    params = dict(initial.params)
    for plan in engine.arena.mu3_groups:
        prefix = plan.name
        params[prefix + ":samples"] = prepared.params[prefix + ":samples"]
        params[prefix + ":base_alpha"] = prepared.params[prefix + ":base_alpha"]
        params[prefix + ":ready"] = jnp.asarray(True)
    return _replace(initial, params=params)


def _evaluator_identity(engine: Engine) -> list[dict]:
    return [
        {
            "name": plan.name,
            "evaluator": plan.evaluator,
            "parameter": plan.param_blob,
            "members": list(plan.members),
            "rows": np.asarray(plan.rows).tolist(),
        }
        for plan in engine.arena.mu3_groups
    ]


def _run_config(
    engine: Engine, n_iters: int, tol: float, run_options: dict
) -> dict:
    return {
        "n_iters": int(n_iters),
        "tol": float(tol),
        "run_options": dict(sorted(run_options.items())),
        "samples": engine.mu3_samples,
        "min_ess_fraction": engine.mu3_min_ess_fraction,
        "schedule_fingerprint": engine.schedule.fingerprint,
    }


def _store_state(data: dict[str, np.ndarray], state: State) -> None:
    metadata = {"params": sorted(state.params)}
    for name in _STATE_FIELDS:
        value = getattr(state, name)
        if name == "params":
            for key, array in value.items():
                data[f"mu3:state:param:{key}"] = np.asarray(array)
        else:
            data[f"mu3:state:{name}"] = np.asarray(value)
    data["mu3:state:metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))


def _load_state(data: Mapping[str, np.ndarray], engine: Engine) -> State:
    if "mu3:state:metadata" not in data:
        raise KeyError("functional checkpoint is missing planning-state metadata")
    metadata = json.loads(str(data["mu3:state:metadata"]))
    expected = engine.init_state()
    if set(metadata.get("params", ())) != set(expected.params):
        raise ValueError("functional checkpoint parameter layout does not match this plan")
    restored: dict[str, object] = {}
    for name in _STATE_FIELDS:
        if name == "params":
            params = {}
            for key, expected_array in expected.params.items():
                storage = f"mu3:state:param:{key}"
                if storage not in data:
                    raise KeyError(f"functional checkpoint is missing state parameter {key!r}")
                params[key] = _checked_array(data[storage], expected_array, storage)
            restored[name] = params
        else:
            storage = f"mu3:state:{name}"
            if storage not in data:
                raise KeyError(f"functional checkpoint is missing state field {name!r}")
            restored[name] = _checked_array(data[storage], getattr(expected, name), storage)
    return State(**restored)


def _checked_array(value: np.ndarray, expected: jax.Array, label: str) -> jax.Array:
    array = np.asarray(value)
    shape, dtype = np.shape(expected), np.asarray(expected).dtype
    if array.shape != shape:
        raise ValueError(f"{label} shape {array.shape} != {shape}")
    if array.dtype != dtype:
        raise ValueError(
            f"{label} dtype {array.dtype} != {dtype}; check the JAX x64 configuration"
        )
    return jnp.asarray(array)


def _result_metadata(result: RunResult) -> dict:
    metadata = {name: getattr(result, name) for name in _RESULT_FIELDS}
    for name in ("message_deltas", "free_energies", "mu3_ess", "mu3_log_normalizers"):
        metadata[name] = list(metadata[name])
    metadata |= {
        "status": result.status.value,
        "inner_status": result.inner_status.value,
        "mu3_statuses": [status.value for status in result.mu3_statuses],
        "termination_reason": result.termination_reason.value,
    }
    return metadata


def _load_result(state: State, metadata: dict) -> RunResult:
    missing = sorted(set(_RESULT_FIELDS) - set(metadata))
    if missing:
        raise KeyError(f"functional checkpoint result is missing {missing}")
    values = {name: metadata[name] for name in _RESULT_FIELDS}
    for name in ("message_deltas", "free_energies", "mu3_ess", "mu3_log_normalizers"):
        values[name] = tuple(values[name])
    return RunResult(
        state=state,
        status=RunStatus(metadata["status"]),
        inner_status=RunStatus(metadata["inner_status"]),
        mu3_statuses=tuple(RunStatus(value) for value in metadata["mu3_statuses"]),
        termination_reason=OuterTerminationReason(metadata["termination_reason"]),
        **values,
    )


def _validate_prepared_state(engine: Engine, result: RunResult) -> None:
    expected_ess, expected_logz, expected_statuses = engine._mu3_result_diagnostics(
        result.state
    )
    if (
        not np.array_equal(expected_ess, result.mu3_ess, equal_nan=True)
        or not np.array_equal(
            expected_logz, result.mu3_log_normalizers, equal_nan=True
        )
        or expected_statuses != result.mu3_statuses
    ):
        raise ValueError("functional checkpoint result diagnostics disagree with its state")
    if (
        result.status is RunStatus.CONVERGED
        and any(status is not RunStatus.CONVERGED for status in result.mu3_statuses)
    ):
        raise ValueError("functional checkpoint marks failed mu3 evaluation converged")
    for plan in engine.arena.mu3_groups:
        prefix = plan.name
        if not bool(np.asarray(result.state.params[prefix + ":ready"])):
            raise ValueError(f"functional checkpoint evaluator {plan.name!r} is unprepared")
        counts = np.asarray(result.state.params[plan.count_key])[plan.rows]
        base = np.asarray(result.state.params[prefix + ":base_alpha"])
        if not all(np.array_equal(row, base) for row in counts):
            raise ValueError(f"functional checkpoint evaluator {plan.name!r} has a stale base")
