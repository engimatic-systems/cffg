"""Persistent state storage and compiler-planned costs for named schedule units."""

from __future__ import annotations

import operator
import numpy as np
from math import isfinite
from time import perf_counter

import jax


STATE_ARRAYS = (
    "messages",
    "g_eta",
    "g_lam",
    "g_obs",
    "g_mask",
    "ef_eta",
    "ef_obs",
    "ef_mask",
)


def state_storage(state, batch_size=None):
    actual_batch = state.messages.shape[0] if state.messages.ndim > 1 else 1
    if actual_batch < 1:
        raise ValueError("An initialized batch must contain at least one state")
    arrays = {name: int(getattr(state, name).nbytes) for name in STATE_ARRAYS}
    parameters = {
        jax.tree_util.keystr(path): int(leaf.nbytes)
        for path, leaf in jax.tree_util.tree_flatten_with_path(state.params)[0]
    }
    if batch_size is not None:
        batch_size = operator.index(batch_size)
        if batch_size < 1 or state.messages.ndim != 1:
            raise ValueError("Supply a positive batch size with one unbatched state")
        for group in (arrays, parameters):
            for category, nbytes in group.items():
                total = nbytes * batch_size
                if total > np.iinfo(np.int64).max:
                    raise OverflowError(
                        f"Batch size {batch_size} overflows byte count for {category}"
                    )
                group[category] = total
        if sum(arrays.values()) + sum(parameters.values()) > np.iinfo(np.int64).max:
            raise OverflowError(
                f"Batch size {batch_size} overflows total persistent bytes"
            )
    return {
        "batch_size": batch_size if batch_size is not None else actual_batch,
        "arenas": arrays,
        "parameter_leaves": parameters,
        "parameter_bytes": sum(parameters.values()),
        "state_bytes": sum(arrays.values()) + sum(parameters.values()),
    }


def cost_values(raw):
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError("Expected one backend cost dictionary")
    result = {}
    for key in ("flops", "bytes accessed"):
        value = raw.get(key)
        if not isinstance(value, (float, int)) or not isfinite(value) or value < 0:
            raise ValueError(f"Invalid backend cost field {key!r}: {value!r}")
        result[key] = value
    return result


def memory_values(stats):
    if stats is None:
        raise NotImplementedError("Backend has no executable memory accounting")
    result = {}
    for domain, prefix in (("device", ""), ("host", "host_")):
        fields = {
            key: getattr(stats, prefix + key + "_size_in_bytes", None)
            for key in ("argument", "output", "alias", "temp", "generated_code")
        }
        if all(
            fields[key] is not None for key in ("argument", "output", "alias", "temp")
        ):
            fields["planned_peak"] = (
                fields["argument"] + fields["output"] - fields["alias"] + fields["temp"]
            )
        result[domain] = fields
    return result


def _backend_result(method, parse):
    try:
        raw = method()
    except NotImplementedError as error:
        return None, {
            "kind": "unavailable",
            "type": type(error).__name__,
            "message": str(error),
        }
    except RuntimeError as error:
        if not any(
            word in str(error).lower()
            for word in (
                "unimplemented",
                "unsupported",
                "not supported",
                "not implemented",
            )
        ):
            raise
        return None, {
            "kind": "unavailable",
            "type": type(error).__name__,
            "message": str(error),
        }
    try:
        return parse(raw), None
    except (ValueError, NotImplementedError) as error:
        return None, {
            "kind": "unavailable"
            if isinstance(error, NotImplementedError)
            else "malformed",
            "type": type(error).__name__,
            "message": str(error),
        }


def executable_reports(
    engine, state, *, damp=0.0, harden=False, batch_size=None, time_units=False
):
    sweep_args = tuple(getattr(state, k) for k in STATE_ARRAYS) + (state.params,)
    priming_args = (
        state.messages,
        state.g_eta,
        state.g_lam,
        state.ef_eta,
        state.g_obs,
        state.g_mask,
        state.ef_obs,
        state.ef_mask,
        state.params,
    )
    units = [
        ("flooding_step", lambda *args: engine._step(*args, damp, harden), sweep_args)
    ]
    if engine._exact_unit is not None:
        units.append(("two_pass_exact", engine._exact_unit, sweep_args))
    if engine.schedule.kind in ("mixed", "colored"):
        units.append(
            (
                engine.schedule.kind + "_step",
                lambda *args: engine._iter_step(*args, damp, harden),
                sweep_args,
            )
        )
    if engine._prime is not None:
        units.append((engine.schedule.kind + "_prime", engine._prime, priming_args))
    batch = (
        batch_size
        if batch_size is not None
        else state.messages.shape[0]
        if state.messages.ndim > 1
        else None
    )
    results = {}
    for unit, fn, args in units:
        signatures = args
        if batch_size is not None:
            signatures = jax.tree.map(
                lambda x: jax.ShapeDtypeStruct((batch_size, *x.shape), x.dtype), args
            )
        start = perf_counter()
        lowered = jax.jit(jax.vmap(fn) if batch is not None else fn).lower(*signatures)
        lowering = perf_counter() - start
        start = perf_counter()
        executable = lowered.compile()
        compilation = perf_counter() - start
        costs, cost_error = _backend_result(executable.cost_analysis, cost_values)
        memory, memory_error = _backend_result(
            executable.memory_analysis, memory_values
        )
        for diagnostic in (cost_error, memory_error):
            if diagnostic:
                diagnostic["backend"] = jax.default_backend()
        results[unit] = {
            "unit": unit,
            "backend": jax.default_backend(),
            "batch_size": batch or 1,
            "cost": costs,
            "memory_bytes": memory,
            "cost_diagnostic": cost_error,
            "memory_diagnostic": memory_error,
            "lowering_seconds": lowering,
            "compilation_seconds": compilation,
        }
        if time_units:
            inputs = (
                args
                if batch_size is None
                else jax.tree.map(
                    lambda x: jax.numpy.broadcast_to(x, (batch_size, *x.shape)), args
                )
            )
            jax.block_until_ready(inputs)
            start = perf_counter()
            jax.block_until_ready(executable(*inputs))
            results[unit]["first_execution_seconds"] = perf_counter() - start
    return results
