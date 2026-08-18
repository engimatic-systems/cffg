"""Versioned, atomic persistence for :class:`cffg.agent.RollingAgent`."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from cffg._agent_mu3 import (
    PlanningRuntime,
    checkpoint_runtime,
    prepared_inputs_match,
    restore_runtime,
)
from cffg.compiler import _log_of, _resolve_arrays
from cffg.core.ir import AgentSpec, Template
from cffg.engine import _replace
from cffg.learn import Learnables

if TYPE_CHECKING:
    from cffg.agent import RollingAgent


_CHECKPOINT_VERSION = 1
_LEGACY_KEYS = {"counts": "A", "b_counts": "B"}
_CORE_FIELDS = {
    "checkpoint_version",
    "checkpoint_fields",
    "agent:mode",
    "plan_structure_hash",
    "filter_structure_hash",
    "plan_parameter_hash",
    "filter_parameter_hash",
    "horizon",
    "damp",
    "n_sweeps",
    "select",
    "jax_enable_x64",
    "tick",
    "episode_tick",
    "rng_state",
}
_MU3_CONFIG_FIELDS = {
    "mu3_samples",
    "mu3_min_ess_fraction",
    "mu3_refresh",
}


def checkpoint_agent(agent: RollingAgent, path: str | Path) -> None:
    """Write one exact physical-field manifest and all continuation state."""
    mode = _agent_mode(agent)
    data = {
        "checkpoint_version": np.asarray(_CHECKPOINT_VERSION),
        "agent:mode": np.asarray(mode),
        "plan_structure_hash": np.asarray(agent._plan_structure_hash),
        "filter_structure_hash": np.asarray(agent._filter_structure_hash),
        "plan_parameter_hash": np.asarray(agent._plan_parameter_hash),
        "filter_parameter_hash": np.asarray(agent._filter_parameter_hash),
        "horizon": np.asarray(int(agent.horizon)),
        "damp": np.asarray(float(agent.damp)),
        "n_sweeps": np.asarray(int(agent.n_sweeps)),
        "select": np.asarray(str(agent.select)),
        "jax_enable_x64": np.asarray(agent._jax_enable_x64),
        "tick": np.asarray(agent.tick),
        "episode_tick": np.asarray(agent.episode_tick),
        "rng_state": np.asarray(json.dumps(agent.rng.bit_generator.state)),
    }
    if agent._has_legacy_belief:
        data["belief"] = agent.belief
    else:
        for name, belief in agent.beliefs.items():
            data[f"belief:{name}"] = belief
    for name, alpha in agent.learnables.state_dict().items():
        data[f"learn:{name}"] = alpha
    for name, alpha in agent.goal_priors.items():
        data[f"goal:prior:{name}"] = alpha
    for name, alpha in (agent._pending_goal_priors or {}).items():
        data[f"goal:pending:{name}"] = alpha
    if mode == "functional_mu3":
        data.update(
            {
                "mu3_samples": np.asarray(int(agent.plan.mu3_samples)),
                "mu3_min_ess_fraction": np.asarray(
                    float(agent.plan.mu3_min_ess_fraction)
                ),
                "mu3_refresh": np.asarray(agent._planning.refresh_policy),
            }
        )
        checkpoint_runtime(
            data,
            agent.plan,
            agent._planning,
            n_iters=agent.n_sweeps,
            tol=1e-10,
            run_options=_planning_run_options(agent),
        )
    data["checkpoint_fields"] = np.asarray(
        json.dumps(sorted((*data, "checkpoint_fields")))
    )
    np.savez(path, **data)


def restore_agent(agent: RollingAgent, path: str | Path) -> None:
    """Stage and validate every component before mutating ``agent``."""
    with np.load(path, allow_pickle=False) as archive:
        archive_fields = list(archive.files)
        if len(archive_fields) != len(set(archive_fields)):
            raise KeyError("checkpoint contains duplicate array names")
        data = {key: archive[key].copy() for key in archive_fields}
    fields = set(data)
    version = (
        None
        if "checkpoint_version" not in data
        else _checkpoint_int(data["checkpoint_version"], "checkpoint_version")
    )

    if version is None:
        if _agent_mode(agent) == "functional_mu3":
            restore_runtime(
                data,
                agent.plan,
                n_iters=agent.n_sweeps,
                tol=1e-10,
                run_options=_planning_run_options(agent),
                refresh_policy=agent._planning.refresh_policy,
            )
        _validate_legacy_fields(fields)
        staged_runtime = PlanningRuntime(None, agent._planning.refresh_policy)
    elif version == _CHECKPOINT_VERSION:
        staged_runtime = _validate_versioned_envelope(agent, data, fields)
    else:
        raise ValueError(
            f"checkpoint version {version} is not supported; "
            f"expected {_CHECKPOINT_VERSION}"
        )

    staged_beliefs = _restore_beliefs(agent, data, fields)
    staged_tick = _checkpoint_int(data["tick"], "tick")
    staged_episode_tick = (
        _checkpoint_int(data["episode_tick"], "episode_tick")
        if "episode_tick" in fields
        else staged_tick
    )
    if staged_tick < 0 or not 0 <= staged_episode_tick <= staged_tick:
        raise ValueError(
            f"checkpoint cadence is invalid: tick={staged_tick}, "
            f"episode_tick={staged_episode_tick}"
        )

    staged_rng = agent.rng.bit_generator.state
    if "rng_state" in fields:
        rng_state = _checkpoint_json(data["rng_state"], "rng_state")
        candidate_rng = np.random.default_rng()
        try:
            candidate_rng.bit_generator.state = rng_state
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                "checkpoint field 'rng_state' is not a valid bit-generator state"
            ) from error
        staged_rng = candidate_rng.bit_generator.state
    elif agent.select != "map":
        raise KeyError("legacy checkpoint has no RNG state for stochastic selection")

    learnable_state = _restore_learnable_fields(agent, data, fields)
    probe_learnables = Learnables(
        agent.plan,
        priors={
            name: {key: value.copy() for key, value in params.items()}
            for name, params in agent.learnables.params.items()
        },
    )
    probe_learnables.load_state_dict(learnable_state, exact=True)
    restored_goal, pending = _restore_goals(agent, data, fields)

    if (
        _agent_mode(agent) == "functional_mu3"
        and staged_runtime.result is not None
        and not staged_runtime.run_required
    ):
        expected = _planning_inputs(
            agent, probe_learnables, staged_beliefs, restored_goal
        )
        if not prepared_inputs_match(
            agent.plan, expected, staged_runtime.result.state
        ):
            raise ValueError(
                "functional checkpoint planning state is stale for its "
                "beliefs or learnable posterior"
            )

    agent.beliefs = staged_beliefs
    agent.tick = staged_tick
    agent.episode_tick = staged_episode_tick
    agent.rng.bit_generator.state = staged_rng
    agent.learnables.params = {
        name: {key: value.copy() for key, value in params.items()}
        for name, params in probe_learnables.params.items()
    }
    agent.goal_priors = restored_goal
    agent._pending_goal_priors = pending or None
    agent._planning = staged_runtime
    agent._last_filter_result = None


def static_parameter_hash(
    template: Template,
    arrays: dict[str, np.ndarray],
    spec: AgentSpec,
    *,
    goal_priors: set[str],
) -> str:
    """Hash grounded parameters that checkpoint state does not replace."""
    resolved = _resolve_arrays(template, arrays, None)
    fully_dynamic_blobs = {
        blob
        for learnable in template.learnables.values()
        for blob in learnable.blobs
    }
    dynamic_factors = {binding.prior for binding in spec.beliefs} | goal_priors
    readers = {
        blob: {
            factor.name
            for factor in template.factors.values()
            if blob in factor.param_blobs.values()
        }
        for blob in resolved
    }

    digest = hashlib.sha256()
    for name in sorted(resolved):
        if name in fully_dynamic_blobs or readers[name] <= dynamic_factors:
            continue
        value = np.ascontiguousarray(resolved[name])
        if value.dtype.hasobject:
            raise TypeError(f"parameter blob {name!r} cannot have object dtype")
        metadata = json.dumps(
            [name, value.dtype.str, list(value.shape)], separators=(",", ":")
        ).encode()
        raw = value.tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def check_belief(
    value: np.ndarray,
    label: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
    normalized: bool = False,
) -> np.ndarray:
    """Validate categorical mass while preserving checkpoint bits exactly."""
    value = np.asarray(value)
    if value.ndim != 1:
        raise ValueError(f"{label} must be a 1-D categorical belief, got {value.shape}")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(f"{label} shape {value.shape} != {expected_shape}")
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{label} must be finite and non-negative")
    total = value.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{label} must have positive finite mass")
    if normalized:
        real_dtype = value.dtype if np.issubdtype(value.dtype, np.floating) else float
        tolerance = np.finfo(real_dtype).eps * max(8, value.size * 4)
        if abs(float(total) - 1.0) > tolerance:
            raise ValueError(f"{label} must sum to one within dtype precision")
    return value.copy()


def _validate_versioned_envelope(
    agent: RollingAgent,
    data: dict[str, np.ndarray],
    fields: set[str],
) -> PlanningRuntime:
    manifest = _checkpoint_json(data.get("checkpoint_fields"), "checkpoint_fields")
    if (
        not isinstance(manifest, list)
        or any(not isinstance(field, str) for field in manifest)
        or len(manifest) != len(set(manifest))
        or set(manifest) != fields
    ):
        raise KeyError("checkpoint field manifest does not match its arrays")

    mode = _checkpoint_string(data.get("agent:mode"), "agent:mode")
    expected_mode = _agent_mode(agent)
    if mode != expected_mode:
        raise ValueError(f"{mode} checkpoint cannot restore a {expected_mode} agent")

    belief_fields = (
        {"belief"}
        if agent._has_legacy_belief
        else {f"belief:{name}" for name in agent.beliefs}
    )
    learn_fields = {f"learn:{name}" for name in agent.learnables.state_dict()}
    goal_fields = {f"goal:prior:{name}" for name in agent.goal_priors}
    pending_fields = {f"goal:pending:{name}" for name in agent.goal_priors}
    required = _CORE_FIELDS | belief_fields | learn_fields | goal_fields
    if mode == "functional_mu3":
        required |= _MU3_CONFIG_FIELDS | {"mu3:metadata"}
    missing = sorted(required - fields)
    unknown = sorted(
        field
        for field in fields - required - pending_fields
        if mode != "functional_mu3" or not field.startswith("mu3:")
    )
    if missing:
        raise KeyError(f"checkpoint is missing required fields {missing}")
    if unknown:
        raise KeyError(f"unknown checkpoint fields {unknown}")

    for label, expected in (
        ("plan", agent._plan_structure_hash),
        ("filter", agent._filter_structure_hash),
    ):
        actual = _checkpoint_string(
            data[f"{label}_structure_hash"], f"{label}_structure_hash"
        )
        if actual != expected:
            raise ValueError(
                f"checkpoint {label} structure mismatch: {actual!r} != {expected!r}"
            )
    for label, expected in (
        ("plan", agent._plan_parameter_hash),
        ("filter", agent._filter_parameter_hash),
    ):
        actual = _checkpoint_string(
            data[f"{label}_parameter_hash"], f"{label}_parameter_hash"
        )
        if actual != expected:
            raise ValueError(
                f"checkpoint {label} parameter mismatch: {actual!r} != {expected!r}"
            )

    configuration = {
        "horizon": _checkpoint_int(data["horizon"], "horizon"),
        "damp": _checkpoint_float(data["damp"], "damp"),
        "n_sweeps": _checkpoint_int(data["n_sweeps"], "n_sweeps"),
        "select": _checkpoint_string(data["select"], "select"),
        "jax_enable_x64": _checkpoint_bool(data["jax_enable_x64"], "jax_enable_x64"),
    }
    expected_configuration = {
        "horizon": agent.horizon,
        "damp": agent.damp,
        "n_sweeps": agent.n_sweeps,
        "select": agent.select,
        "jax_enable_x64": agent._jax_enable_x64,
    }
    if mode == "functional_mu3":
        configuration.update(
            {
                "mu3_samples": _checkpoint_int(data["mu3_samples"], "mu3_samples"),
                "mu3_min_ess_fraction": _checkpoint_float(
                    data["mu3_min_ess_fraction"], "mu3_min_ess_fraction"
                ),
                "mu3_refresh": _checkpoint_string(data["mu3_refresh"], "mu3_refresh"),
            }
        )
        expected_configuration.update(
            {
                "mu3_samples": agent.plan.mu3_samples,
                "mu3_min_ess_fraction": agent.plan.mu3_min_ess_fraction,
                "mu3_refresh": agent._planning.refresh_policy,
            }
        )
    mismatches = {
        name: (configuration[name], expected_configuration[name])
        for name in configuration
        if configuration[name] != expected_configuration[name]
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"checkpoint runtime configuration mismatch: {details}")

    if mode != "functional_mu3":
        return PlanningRuntime(None, agent._planning.refresh_policy)
    runtime = restore_runtime(
        data,
        agent.plan,
        n_iters=agent.n_sweeps,
        tol=1e-10,
        run_options=_planning_run_options(agent),
        refresh_policy=agent._planning.refresh_policy,
    )
    expected_mu3: dict[str, np.ndarray] = {}
    checkpoint_runtime(
        expected_mu3,
        agent.plan,
        runtime,
        n_iters=agent.n_sweeps,
        tol=1e-10,
        run_options=_planning_run_options(agent),
    )
    actual_mu3_fields = {field for field in fields if field.startswith("mu3:")}
    if actual_mu3_fields != set(expected_mu3):
        raise KeyError(
            "functional checkpoint field inventory does not match its metadata"
        )
    return runtime


def _validate_legacy_fields(fields: set[str]) -> None:
    unknown = sorted(
        field
        for field in fields
        if field not in {"tick", "episode_tick", "rng_state", "belief", *_LEGACY_KEYS}
        and not field.startswith(("belief:", "learn:", "goal:prior:", "goal:pending:"))
    )
    if unknown:
        raise KeyError(f"unknown legacy checkpoint fields {unknown}")


def _restore_beliefs(
    agent: RollingAgent, data: dict[str, np.ndarray], fields: set[str]
) -> dict[str, np.ndarray]:
    named = {field for field in fields if field.startswith("belief:")}
    if "belief" in fields and named:
        raise KeyError("checkpoint mixes legacy 'belief' with named belief fields")
    if "belief" in fields:
        if not agent._has_legacy_belief:
            raise KeyError(
                "legacy single-belief checkpoint cannot restore a multi-block agent"
            )
        only = next(iter(agent.beliefs))
        return {
            only: check_belief(
                data["belief"],
                f"checkpoint belief {only!r}",
                expected_shape=agent.beliefs[only].shape,
                normalized=True,
            )
        }

    restored = {
        key[len("belief:") :]: data[key]
        for key in fields
        if key.startswith("belief:")
    }
    if set(restored) != set(agent.beliefs):
        raise KeyError(
            "checkpoint boundary beliefs do not match this agent: "
            f"have {sorted(restored)}, expected {sorted(agent.beliefs)}"
        )
    return {
        name: check_belief(
            value,
            f"checkpoint belief {name!r}",
            expected_shape=agent.beliefs[name].shape,
            normalized=True,
        )
        for name, value in restored.items()
    }


def _restore_learnable_fields(
    agent: RollingAgent, data: dict[str, np.ndarray], fields: set[str]
) -> dict[str, np.ndarray]:
    state = {
        key[len("learn:") :]: data[key]
        for key in fields
        if key.startswith("learn:")
    }
    for old, name in _LEGACY_KEYS.items():
        if old not in fields:
            continue
        if name not in agent.learnables.params:
            raise KeyError(
                f"checkpoint holds legacy {old!r} counts but this agent does "
                f"not learn {name!r}"
            )
        if name in state:
            raise KeyError(
                f"checkpoint names learnable {name!r} through both "
                f"'learn:{name}' and legacy {old!r}"
            )
        state[name] = data[old]
    return state


def _restore_goals(
    agent: RollingAgent, data: dict[str, np.ndarray], fields: set[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    restored = {
        key[len("goal:prior:") :]: data[key]
        for key in fields
        if key.startswith("goal:prior:")
    }
    pending = {
        key[len("goal:pending:") :]: data[key]
        for key in fields
        if key.startswith("goal:pending:")
    }
    if not agent.goal_priors:
        if restored or pending:
            raise KeyError(
                "checkpoint contains learned-goal state but this agent has "
                "learn_goal=False"
            )
        return {}, {}
    if set(restored) != set(agent.goal_priors):
        raise KeyError(
            "checkpoint learned-goal priors do not match this agent: "
            f"have {sorted(restored)}, expected {sorted(agent.goal_priors)}"
        )
    unknown_pending = sorted(set(pending) - set(agent.goal_priors))
    if unknown_pending:
        raise KeyError(f"checkpoint has unknown pending goal priors {unknown_pending}")
    return (
        {
            name: _check_goal_counts(
                value,
                f"checkpoint goal prior {name!r}",
                agent.goal_priors[name].shape,
            )
            for name, value in restored.items()
        },
        {
            name: _check_goal_counts(
                value,
                f"checkpoint pending goal {name!r}",
                agent.goal_priors[name].shape,
            )
            for name, value in pending.items()
        },
    )


def _planning_inputs(
    agent: RollingAgent,
    learnables: Learnables,
    beliefs: dict[str, np.ndarray],
    goals: dict[str, np.ndarray],
):
    state = agent.plan.init_state()
    params = learnables.write(agent.plan, dict(state.params))
    for name, binding in agent._plan_bindings.items():
        address = agent.plan.arena.member_of[binding.prior].parameter("statistic")
        params[address.key] = params[address.key].at[address.index].set(
            jnp.asarray(_log_of(beliefs[name]))
        )
    state = _replace(state, params=params)
    return agent.plan.with_goal_priors(state, goals) if goals else state


def _planning_run_options(agent: RollingAgent) -> dict:
    if agent.plan.arena.gfe_groups or agent.plan.arena.gfe_tm_groups:
        return {}
    return {"damp": agent.damp}


def _agent_mode(agent: RollingAgent) -> str:
    return "functional_mu3" if agent.functional_mu3 else "analytic"


def _checkpoint_string(value: object, label: str) -> str:
    if value is None:
        raise KeyError(f"checkpoint is missing required field {label!r}")
    scalar = np.asarray(value)
    if scalar.shape != () or not isinstance(scalar.item(), (str, np.str_)):
        raise ValueError(f"checkpoint field {label!r} must be a string scalar")
    return str(scalar.item())


def _checkpoint_json(value: object, label: str) -> object:
    raw = _checkpoint_string(value, label)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"checkpoint field {label!r} is not valid JSON") from error


def _checkpoint_int(value: object, label: str) -> int:
    scalar = np.asarray(value)
    item = scalar.item() if scalar.shape == () else None
    if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
        raise ValueError(f"checkpoint field {label!r} must be an integer scalar")
    return int(item)


def _checkpoint_bool(value: object, label: str) -> bool:
    scalar = np.asarray(value)
    item = scalar.item() if scalar.shape == () else None
    if not isinstance(item, (bool, np.bool_)):
        raise ValueError(f"checkpoint field {label!r} must be a boolean scalar")
    return bool(item)


def _checkpoint_float(value: object, label: str) -> float:
    scalar = np.asarray(value)
    item = scalar.item() if scalar.shape == () else None
    if isinstance(item, (bool, np.bool_)) or not isinstance(item, (float, np.floating)):
        raise ValueError(f"checkpoint field {label!r} must be a float scalar")
    if not np.isfinite(item):
        raise ValueError(f"checkpoint field {label!r} must be finite")
    return float(item)


def _check_goal_counts(
    value: np.ndarray, label: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != expected_shape:
        raise ValueError(f"{label} shape {value.shape} != {expected_shape}")
    if not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"{label} must contain finite positive counts")
    return value.copy()
