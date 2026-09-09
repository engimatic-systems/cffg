"""Validated named writes into compiled parameter arenas."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cffg.compiled import MemberAddress, ParameterAddress
    from cffg.engine import Engine


_PROBABILITY_AXES = {
    ("categorical", "statistic"): 0,
    ("matrix_cat", "matrix"): 0,
    ("transition_mixture", "matrices"): 1,
}


@dataclass(frozen=True)
class _RuntimeWrite:
    key: str
    index: int
    value: np.ndarray


def _numeric(name: str, value: object) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be a real numeric array")
    return array


def _working_float(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        return array
    return array.astype(np.float64)


def _symmetry_ok(matrices: np.ndarray) -> bool:
    dtype = matrices.dtype
    scale = np.max(np.abs(matrices), axis=(-2, -1))
    scale = np.maximum(scale, np.finfo(dtype).tiny)
    error = np.max(
        np.abs(matrices - matrices.swapaxes(-1, -2)), axis=(-2, -1)
    )
    # Admit only the roundoff that a dtype-sized matrix comparison accumulates.
    tolerance = 16.0 * matrices.shape[-1] * np.finfo(dtype).eps * scale
    return bool(np.all(error <= tolerance))


def _validate_probability(
    factor_type: str, role: str, value: np.ndarray, name: str, axis=None
) -> None:
    if axis is None:
        axis = _PROBABILITY_AXES.get((factor_type, role))
    if axis is None:
        raise ValueError(
            f"{name}: role {role!r} of {factor_type!r} is not a probability parameter"
        )
    working = _working_float(value)
    if not np.all(np.isfinite(working)) or np.any(working < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    totals = working.sum(axis=axis)
    # Cover summation roundoff without relaxing the serialized simplex.
    tolerance = (
        16.0
        * working.shape[axis]
        * np.finfo(working.dtype).eps
        * np.maximum(1.0, np.abs(totals))
    )
    if not np.all(np.abs(totals - 1.0) <= tolerance):
        raise ValueError(
            f"{name} must sum to one along serialized axis {axis}"
        )


def _validate_covariance(value: np.ndarray, name: str) -> None:
    working = _working_float(value)
    if (
        working.ndim < 2
        or working.shape[-1] != working.shape[-2]
        or not np.all(np.isfinite(working))
    ):
        raise ValueError(f"{name} must contain finite square covariance matrices")
    if not _symmetry_ok(working):
        raise ValueError(f"{name} must be symmetric")
    try:
        # Preserve source-dtype symmetry tolerance; host linalg also accepts f16.
        np.linalg.cholesky(np.asarray(working, dtype=np.float64))
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} must be positive definite") from None


def _serialized_shape(
    params: Mapping[str, object],
    member: "MemberAddress",
    address: "ParameterAddress",
) -> tuple[int, ...]:
    if address.key not in params:
        raise KeyError(f"parameter mapping has no runtime array {address.key!r}")
    runtime_shape = tuple(np.shape(params[address.key]))
    if not runtime_shape or address.index >= runtime_shape[0]:
        raise ValueError(
            f"runtime address {address.key!r}[{address.index}] is outside shape "
            f"{runtime_shape}"
        )
    row_shape = runtime_shape[1:]
    if address.permutation:
        return tuple(row_shape[i] for i in np.argsort(address.permutation))
    return row_shape


def _runtime_layout(
    factor_type: str, address: "ParameterAddress", value: np.ndarray
) -> np.ndarray:
    if address.permutation:
        return value.transpose(address.permutation)
    return value


def _transform(
    factor_type: str, address: "ParameterAddress", value: np.ndarray
) -> np.ndarray:
    from cffg.compiled import NEG

    if address.transform == "identity":
        transformed = value
    elif address.transform == "probability_to_log":
        working = _working_float(value)
        dtype = working.dtype
        transformed = np.where(
            working > 0.0,
            np.log(np.maximum(working, np.finfo(dtype).tiny)),
            np.asarray(NEG, dtype=dtype),
        )
    else:
        transformed = np.linalg.inv(_working_float(value))
    return _runtime_layout(factor_type, address, np.asarray(transformed))


def _apply_runtime_writes(
    params: Mapping[str, object], writes: list[_RuntimeWrite]
) -> dict[str, object]:
    """Validate all row writes, then update each stacked array once."""
    pending: dict[tuple[str, int], np.ndarray] = {}
    for write in writes:
        if write.key not in params:
            raise KeyError(f"parameter mapping has no runtime array {write.key!r}")
        destination = np.asarray(params[write.key])
        if destination.ndim == 0 or not 0 <= write.index < destination.shape[0]:
            raise ValueError(
                f"runtime address {write.key!r}[{write.index}] is outside "
                f"shape {destination.shape}"
            )
        if write.value.shape != destination.shape[1:]:
            raise ValueError(
                f"runtime write {write.key!r}[{write.index}] shape "
                f"{write.value.shape} != {destination.shape[1:]}"
            )
        row = (write.key, write.index)
        if row in pending and not np.array_equal(pending[row], write.value):
            raise ValueError(f"conflicting values reach runtime row {row}")
        pending[row] = write.value

    by_key: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (key, index), value in pending.items():
        by_key[key].append((index, value))

    out = dict(params)
    for key, rows in by_key.items():
        rows.sort(key=lambda item: item[0])
        indices = jnp.asarray([index for index, _ in rows], dtype=jnp.int32)
        values = jnp.asarray(
            np.stack([value for _, value in rows]), dtype=out[key].dtype
        )
        out[key] = out[key].at[indices].set(values)
    return out


def set_params(
    engine: "Engine",
    params: Mapping[str, object],
    updates: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Return a new runtime parameter mapping with validated named updates.

    Factor names resolve through the compiler's role-bearing member map.
    Updating a serialized blob and role fans out to every runtime member that
    reads the same pair, preserving template-level parameter tying.
    """
    if not isinstance(updates, Mapping):
        raise TypeError("updates must map factor names to role mappings")

    requested: dict[tuple[str, str], np.ndarray] = {}
    for factor_name, role_updates in updates.items():
        if factor_name not in engine.arena.member_of:
            raise KeyError(f"unknown or parameterless factor {factor_name!r}")
        if factor_name not in engine.arena.writable_members:
            raise ValueError(
                f"factor {factor_name!r} has only an internal composite address"
            )
        if not isinstance(role_updates, Mapping):
            raise TypeError(f"updates for factor {factor_name!r} must map roles to arrays")
        member = engine.arena.member_of[factor_name]
        known_roles = {address.role for address in member.parameters}
        unknown_roles = sorted(set(role_updates) - known_roles)
        if unknown_roles:
            raise KeyError(
                f"factor {factor_name!r} has roles {sorted(known_roles)}, not "
                f"{unknown_roles}"
            )
        for role, raw in role_updates.items():
            address = member.parameter(role)
            name = f"factor {factor_name!r} role {role!r}"
            value = _numeric(name, raw)
            expected_shape = _serialized_shape(params, member, address)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {value.shape} != serialized blob shape "
                    f"{expected_shape}"
                )
            if address.transform == "probability_to_log":
                _validate_probability(member.factor_type, role, value, name, address.probability_axis)
            elif address.transform == "covariance_to_precision":
                _validate_covariance(value, name)
            elif not np.all(np.isfinite(_working_float(value))):
                raise ValueError(f"{name} must be finite")
            blob_role = (address.blob, role)
            if blob_role in requested and not np.array_equal(
                requested[blob_role], value
            ):
                raise ValueError(
                    f"tied blob {address.blob!r} role {role!r} received unequal values"
                )
            requested[blob_role] = value.copy()

    writes: list[_RuntimeWrite] = []
    for member in engine.arena.member_of.values():
        for address in member.parameters:
            blob_role = (address.blob, address.role)
            if blob_role not in requested:
                continue
            value = requested[blob_role]
            writes.append(
                _RuntimeWrite(
                    address.key,
                    address.index,
                    _transform(member.factor_type, address, value),
                )
            )
    return _apply_runtime_writes(params, writes)
