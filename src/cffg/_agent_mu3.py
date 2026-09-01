"""Functional-mu3 planning state owned by :class:`RollingAgent`."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from cffg import efrules
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
_RESULT_METADATA_FIELDS = set(_RESULT_FIELDS) | {
    "status",
    "inner_status",
    "mu3_statuses",
    "termination_reason",
}
_RUNTIME_METADATA_FIELDS = {
    "epoch",
    "run_required",
    "refresh_required",
    "refresh_policy",
    "samples",
    "min_ess_fraction",
    "schedule_fingerprint",
    "evaluators",
    "has_result",
    "cache_config",
    "has_key",
}


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
    data: Mapping[str, np.ndarray],
    engine: Engine,
    *,
    n_iters: int,
    tol: float,
    run_options: dict,
    refresh_policy: str,
) -> PlanningRuntime:
    """Validate and reconstruct a stochastic epoch without drawing particles."""
    if "mu3:metadata" not in data:
        raise ValueError(
            "legacy checkpoint cannot restore a functional mu3 agent: "
            "the checkpoint has no key or prepared planning epoch"
        )
    metadata = _json_object(data["mu3:metadata"], "mu3:metadata")
    metadata_fields = set(_RUNTIME_METADATA_FIELDS)
    if metadata.get("has_key") is True:
        metadata_fields |= {"key_impl", "typed_key"}
    if metadata.get("has_result") is True:
        metadata_fields.add("result")
    if set(metadata) != metadata_fields:
        raise KeyError("functional checkpoint runtime metadata has unknown or missing fields")
    expected = {
        "samples": engine.mu3_samples,
        "min_ess_fraction": engine.mu3_min_ess_fraction,
        "schedule_fingerprint": engine.schedule.fingerprint,
        "evaluators": _evaluator_identity(engine),
        "refresh_policy": refresh_policy,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"functional checkpoint {name} does not match this agent: "
                f"{metadata.get(name)!r} != {value!r}"
            )

    for name in ("has_key", "has_result", "run_required", "refresh_required"):
        if not isinstance(metadata[name], bool):
            raise ValueError(f"functional checkpoint {name} must be boolean")
    if (
        isinstance(metadata["epoch"], bool)
        or not isinstance(metadata["epoch"], int)
        or metadata["epoch"] < 0
    ):
        raise ValueError("functional checkpoint epoch must be a non-negative integer")
    if metadata["refresh_required"] and not metadata["run_required"]:
        raise ValueError("functional checkpoint cannot require refresh without a run")
    if metadata["has_result"] and not metadata["has_key"]:
        raise ValueError("functional checkpoint result has no next JAX key cursor")
    if metadata["has_result"] and metadata["epoch"] == 0:
        raise ValueError("functional checkpoint result has no sampling epoch")
    if metadata["epoch"] > 0 and not metadata["has_key"]:
        raise ValueError("functional checkpoint sampling epoch has no JAX key cursor")

    expected_mu3_fields = {"mu3:metadata"}
    key = None
    if metadata["has_key"]:
        if "mu3:key" not in data:
            raise KeyError("functional checkpoint is missing its JAX key cursor")
        if not isinstance(metadata["key_impl"], str) or not isinstance(
            metadata["typed_key"], bool
        ):
            raise ValueError("functional checkpoint JAX key metadata is invalid")
        expected_mu3_fields.add("mu3:key")
        key_data = jnp.asarray(data["mu3:key"])
        try:
            wrapped = jax.random.wrap_key_data(key_data, impl=metadata["key_impl"])
        except (TypeError, ValueError) as error:
            raise ValueError("functional checkpoint JAX key is invalid") from error
        key = wrapped if metadata.get("typed_key") else jax.random.key_data(wrapped)

    result = None
    if metadata["has_result"]:
        expected_state = engine.init_state()
        expected_mu3_fields |= {
            "mu3:state:metadata",
            *(f"mu3:state:{name}" for name in _STATE_FIELDS if name != "params"),
            *(f"mu3:state:param:{key}" for key in expected_state.params),
        }
        state = _load_state(data, engine)
        result = _load_result(state, metadata.get("result", {}))
    actual_mu3_fields = {key for key in data if key.startswith("mu3:")}
    if actual_mu3_fields != expected_mu3_fields:
        raise KeyError(
            "functional checkpoint field inventory does not match its metadata"
        )

    cache_config = metadata["cache_config"]
    if result is None and cache_config is not None:
        raise ValueError("functional checkpoint without a result has cached run state")
    if result is not None and cache_config is None:
        raise ValueError("functional checkpoint result has no runtime configuration")
    if cache_config is not None:
        _validate_cache_config(cache_config)
        if (
            cache_config["samples"] != engine.mu3_samples
            or cache_config["schedule_fingerprint"] != engine.schedule.fingerprint
        ):
            raise ValueError(
                "functional checkpoint cached state has incompatible structure"
            )
    if result is not None:
        _validate_prepared_state(engine, result)
        threshold = cache_config["min_ess_fraction"] * cache_config["samples"]
        if any(
            status is RunStatus.CONVERGED and ess < threshold
            for status, ess in zip(result.mu3_statuses, result.mu3_ess, strict=True)
        ):
            raise ValueError(
                "functional checkpoint marks insufficient effective sample size "
                "converged"
            )
    current_config = _run_config(engine, n_iters, tol, run_options)
    if result is not None and not metadata["run_required"]:
        if cache_config != current_config:
            raise ValueError(
                "functional checkpoint reusable result has stale runtime configuration"
            )
    runtime = PlanningRuntime(
        key_cursor=key,
        refresh_policy=metadata["refresh_policy"],
        epoch=int(metadata["epoch"]),
        result=result,
        run_required=bool(metadata["run_required"]),
        refresh_required=bool(metadata["refresh_required"]),
        cache_config=cache_config,
    )
    if result is None and not runtime.run_required:
        raise ValueError("functional checkpoint has no result but marks planning reusable")
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
    metadata = _json_object(
        data["mu3:state:metadata"], "mu3:state:metadata"
    )
    expected = engine.init_state()
    if set(metadata) != {"params"} or metadata["params"] != sorted(expected.params):
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
    if not isinstance(metadata, dict):
        raise ValueError("functional checkpoint result metadata must be an object")
    if set(metadata) != _RESULT_METADATA_FIELDS:
        raise KeyError(
            "functional checkpoint result metadata has unknown or missing fields"
        )
    counter_fields = {
        "iterations",
        "accepted_steps",
        "rejected_steps",
        "outer_map_evaluations",
        "trial_evaluations",
        "linear_iterations",
        "solver_compilations",
    }
    if any(
        isinstance(metadata[name], bool)
        or not isinstance(metadata[name], int)
        or metadata[name] < 0
        for name in counter_fields
    ):
        raise ValueError(
            "functional checkpoint result counters must be non-negative integers"
        )
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


def _validate_cache_config(config: object) -> None:
    expected_fields = {
        "n_iters",
        "tol",
        "run_options",
        "samples",
        "min_ess_fraction",
        "schedule_fingerprint",
    }
    if not isinstance(config, dict) or set(config) != expected_fields:
        raise ValueError(
            "functional checkpoint cached runtime configuration is invalid"
        )
    integer_fields = ("n_iters", "samples")
    if any(
        isinstance(config[name], bool) or not isinstance(config[name], int)
        for name in integer_fields
    ):
        raise ValueError(
            "functional checkpoint cached runtime configuration is invalid"
        )
    if (
        isinstance(config["tol"], bool)
        or not isinstance(config["tol"], (int, float))
        or isinstance(config["min_ess_fraction"], bool)
        or not isinstance(config["min_ess_fraction"], (int, float))
        or not isinstance(config["run_options"], dict)
        or not isinstance(config["schedule_fingerprint"], str)
    ):
        raise ValueError(
            "functional checkpoint cached runtime configuration is invalid"
        )
    if (
        config["n_iters"] < 0
        or config["samples"] <= 0
        or not np.isfinite(config["tol"])
        or config["tol"] <= 0
        or not np.isfinite(config["min_ess_fraction"])
        or not 0 < config["min_ess_fraction"] <= 1
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            for value in config["run_options"].values()
        )
    ):
        raise ValueError(
            "functional checkpoint cached runtime configuration is invalid"
        )


def _json_object(value: object, label: str) -> dict:
    scalar = np.asarray(value)
    if scalar.shape != () or not isinstance(scalar.item(), (str, np.str_)):
        raise ValueError(f"functional checkpoint {label} must be a JSON string")
    try:
        decoded = json.loads(str(scalar.item()))
    except json.JSONDecodeError as error:
        raise ValueError(f"functional checkpoint {label} is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"functional checkpoint {label} must decode to an object")
    return decoded


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
    if result.status is RunStatus.CONVERGED and (
        result.inner_status is not RunStatus.CONVERGED
        or not np.isfinite(result.residual)
        or result.residual < 0
        or not np.isfinite(result.inner_residual)
        or result.inner_residual < 0
        or not all(np.isfinite(delta) and delta >= 0 for delta in result.message_deltas)
        or not all(np.isfinite(result.free_energies))
        or not np.isfinite(result.outer_residual)
        or result.outer_residual < 0
        or not np.isfinite(result.outer_merit)
        or result.outer_merit < 0
        or not np.isfinite(result.outer_gradient_norm)
        or result.outer_gradient_norm < 0
        or not np.isfinite(result.trust_radius)
        or result.trust_radius <= 0
        or result.termination_reason is not OuterTerminationReason.ROOT_CONVERGED
    ):
        raise ValueError(
            "functional checkpoint converged result disagrees with its solver diagnostics"
        )
    if not all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(result.state)
    ):
        raise ValueError(
            "functional checkpoint result contains non-finite state"
        )
    for plan, status in zip(
        engine.arena.mu3_groups, result.mu3_statuses, strict=True
    ):
        prefix = plan.name
        if not bool(np.asarray(result.state.params[prefix + ":ready"])):
            raise ValueError(f"functional checkpoint evaluator {plan.name!r} is unprepared")
        counts = np.asarray(result.state.params[plan.count_key])[plan.rows]
        base = np.asarray(result.state.params[prefix + ":base_alpha"])
        if not all(np.array_equal(row, base) for row in counts):
            raise ValueError(f"functional checkpoint evaluator {plan.name!r} has a stale base")
        _validate_evaluator_state(result.state, plan, status)


def _validate_evaluator_state(state: State, plan, status: RunStatus) -> None:
    """Validate committed updates and retained-state failure transactions."""
    prefix = plan.name
    status_code = int(np.asarray(state.params[prefix + ":status"]))
    expected_status = {
        0: RunStatus.CONVERGED,
        1: RunStatus.BUDGET_EXHAUSTED,
        2: RunStatus.DIVERGED,
    }.get(status_code)
    if expected_status is not status:
        raise ValueError(
            f"functional checkpoint evaluator {plan.name!r} has invalid status"
        )

    samples = np.asarray(state.params[prefix + ":samples"])
    weights = np.asarray(state.params[prefix + ":weights"])
    dtype = samples.dtype
    tolerance = 256.0 * np.finfo(dtype).eps * max(1.0, np.sqrt(samples.shape[0]))
    if (
        np.any(samples < 0)
        or not np.allclose(
            samples.sum(axis=1), 1.0, rtol=tolerance, atol=tolerance
        )
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, rtol=tolerance, atol=tolerance)
    ):
        raise ValueError(
            f"functional checkpoint evaluator {plan.name!r} has invalid particles or weights"
        )

    ess = float(np.asarray(state.params[prefix + ":ess"]))
    expected_ess = 1.0 / float(np.sum(np.square(weights)))
    # Weight collapse commits its failure marker but retains coherent weights.
    collapsed = status is RunStatus.DIVERGED and ess == 0.0
    if not collapsed and (
        not 1.0 <= ess <= samples.shape[0]
        or not np.isclose(ess, expected_ess, rtol=tolerance, atol=tolerance)
    ):
        raise ValueError(
            f"functional checkpoint evaluator {plan.name!r} has inconsistent ESS"
        )

    expected_abar = np.einsum("s,soi->oi", weights, samples)
    log_samples = np.where(samples > 0, np.log(samples), 0.0)
    particle_h = -np.sum(
        np.where(samples > 0, samples * log_samples, 0.0), axis=1
    )
    expected_hbar = np.einsum("s,si->i", weights, particle_h)
    abar = np.asarray(state.params[plan.gfe_name + ":Abar"])[plan.rows]
    hbar = np.asarray(state.params[plan.gfe_name + ":hbar"])[plan.rows]
    weighted_moments = np.allclose(
        abar, expected_abar, rtol=tolerance, atol=tolerance
    ) and np.allclose(hbar, expected_hbar, rtol=tolerance, atol=tolerance)
    base = state.params[prefix + ":base_alpha"]
    base_abar = np.asarray(
        efrules.DIRICHLET.mean_from_concentration(base, axis=0)
    )
    base_hbar = -np.asarray(
        efrules.DIRICHLET.expected_probability_log_probability(base, axis=0)
    ).sum(axis=0)
    prepared_moments = np.allclose(
        abar, base_abar, rtol=tolerance, atol=tolerance
    ) and np.allclose(hbar, base_hbar, rtol=tolerance, atol=tolerance)
    # A failed solve retains either preparation or the last successful update.
    if not weighted_moments and not (
        status is not RunStatus.CONVERGED and prepared_moments
    ):
        raise ValueError(
            f"functional checkpoint evaluator {plan.name!r} has inconsistent moments"
        )
    log_normalizer = float(state.params[prefix + ":log_normalizer"])
    expected_phi = float(state.params[prefix + ":expected_phi"])
    kl = float(state.params[prefix + ":kl"])
    if not np.isclose(
        kl, expected_phi - log_normalizer, rtol=tolerance, atol=tolerance
    ):
        raise ValueError(
            f"functional checkpoint evaluator {plan.name!r} has inconsistent ledger"
        )
