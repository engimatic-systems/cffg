"""Shape/dtype inventory of published values, without runtime construction.

Stored arrays are counted once per published object, with aliases listed.
Logical bytes exclude Python/container overhead and imply neither device
allocation nor physical or compiled peak memory. Future State and scratch
requirements are separate declarations, not storage owned by a compiled plan.
Only the eight ordinary nonparameter State buffers are counted; runtime
parameters and complete State storage remain unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from math import prod

import numpy as np

from cffg._immutable import FrozenArray, ImmutableRecord
from cffg.compiled import Arena
from cffg.hybrid_plans import HybridComponentPlan


@dataclass(frozen=True)
class ArrayInventory(ImmutableRecord):
    path: str
    shape: tuple[int, ...]
    dtype: str
    aliases: tuple[str, ...] = ()

    @property
    def elements(self) -> int:
        return prod(self.shape)

    @property
    def logical_bytes(self) -> int:
        return self.elements * np.dtype(self.dtype).itemsize


@dataclass(frozen=True)
class HybridStorageInventory(ImmutableRecord):
    component: str
    approximation: str
    coordinate: tuple[ArrayInventory, ...] | None
    scratch: tuple[ArrayInventory, ...] | None

    @property
    def coordinate_bytes(self) -> int | None:
        return _bytes(self.coordinate)

    @property
    def scratch_bytes(self) -> int | None:
        return _bytes(self.scratch)


@dataclass(frozen=True)
class AnalyticInventory(ImmutableRecord):
    """Stored defaults and partial future buffers, never a complete State or peak.

    The ordinary nonparameter subset contains messages, g_eta, g_lam, g_obs,
    g_mask, ef_eta, ef_obs and ef_mask. Stored defaults do not specify runtime
    State.params storage.
    """

    dtype: str
    stored_plan_arrays: tuple[ArrayInventory, ...]
    stored_parameter_arrays: tuple[ArrayInventory, ...] | None
    ordinary_nonparameter_state_requirements: tuple[ArrayInventory, ...] | None
    hybrid_requirements: tuple[HybridStorageInventory, ...]

    @property
    def stored_plan_bytes(self) -> int:
        return _bytes(self.stored_plan_arrays)

    @property
    def stored_parameter_bytes(self) -> int | None:
        return _bytes(self.stored_parameter_arrays)

    @property
    def runtime_parameter_bytes(self) -> None:
        return None

    @property
    def complete_state_bytes(self) -> None:
        return None

    @property
    def physical_peak_bytes(self) -> None:
        return None

    @property
    def compiled_peak_bytes(self) -> None:
        return None


def _bytes(arrays):
    return None if arrays is None else sum(a.logical_bytes for a in arrays)


def analytic_inventory(value: Arena | HybridComponentPlan, *, dtype) -> AnalyticInventory:
    """Inspect Arena or plan metadata using an explicit future floating dtype.

    Stored arrays keep their actual dtypes. The requested float32/float64 dtype
    applies only to declared future floating buffers, independent of JAX's
    precision setting. A standalone plan has no available parameter inventory.
    The ordinary State requirements cover its eight nonparameter buffers only;
    runtime parameter and complete State byte counts remain unavailable.
    Unknown coordinate/scratch layouts remain None. No solver-layout builder,
    engine, state, numerical evaluation, differentiation or compilation runs.
    """
    if dtype is None:
        raise ValueError("analytic inventory requires explicit float32 or float64 dtype")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("analytic inventory requires explicit float32 or float64 dtype")
    if not isinstance(value, (Arena, HybridComponentPlan)):
        raise TypeError("analytic inventory requires an Arena or HybridComponentPlan")

    arrays = {}

    def visit(item, path, category):
        if isinstance(item, FrozenArray):
            if id(item) not in arrays:
                arrays[id(item)] = (category, item, [path])
            else:
                arrays[id(item)][2].append(path)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}[{key!r}]", category)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name), f"{path}.{field.name}", category)
        elif isinstance(item, tuple):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", category)

    ordinary_nonparameter = None
    if isinstance(value, Arena):
        visit(value.log_params, "arena.log_params", "parameters")
        for field in fields(value):
            if field.name != "log_params":
                visit(getattr(value, field.name), f"arena.{field.name}", "plan")
        plans = value.hybrid_components
        counts = {
            "messages": (value.n_message_values,),
            "g_eta": (value.n_g_eta_values,),
            "g_lam": (value.n_g_lam_values,),
            "g_obs": (value.n_g_obs_values,),
            "g_mask": (value.n_obs_rows,),
            "ef_eta": (value.n_efslots, 2),
            "ef_obs": (value.n_ef_obs,),
            "ef_mask": (value.n_ef_obs,),
        }
        ordinary_nonparameter = tuple(
            ArrayInventory(f"state.{name}", shape, dtype.name)
            for name, shape in counts.items()
        )
    else:
        visit(value, "plan", "plan")
        plans = (value,)

    def stored(category):
        return tuple(ArrayInventory(paths[0], tuple(int(d) for d in array.shape),
                                    array.dtype.str, tuple(paths[1:]))
                     for owner, array, paths in arrays.values() if owner == category)

    def requirements(plan, kind):
        declared = getattr(plan, f"{kind}_requirements")
        if declared is None:
            return None
        return tuple(ArrayInventory(f"{plan.name}.{kind}.{r.name}",
                                    tuple(int(d) for d in r.shape), r.dtype or dtype.name)
                     for r in declared)

    return AnalyticInventory(
        dtype.name, stored("plan"),
        stored("parameters") if isinstance(value, Arena) else None,
        ordinary_nonparameter,
        tuple(HybridStorageInventory(plan.name, plan.approximation,
                                     requirements(plan, "coordinate"),
                                     requirements(plan, "scratch")) for plan in plans),
    )
