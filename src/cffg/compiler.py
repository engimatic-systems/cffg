"""Template → immutable executable arena.

Topology records wiring; specialized and registered owners interpret factor
declarations. Resolution owns source loading, numeric validation, observations
and learning addresses. This facade retains public compiled-value exports and
compatibility helpers; final ledger tables and construction belong to assembly.

Each Forney edge owns two directional message slots. Persistent arrays use
exact-sized segments; local factor batches use static shape signatures.
Parameters remain per-instance inputs, not baked compiled constants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cffg.assembly import assemble_arena
from cffg.compiled import (
    NEG,
    PARAMETER_TRANSFORMS,
    Arena,
    ConstraintPlan,
    DirichletGroupPlan,
    EdgeInfo,
    EfGroupPlan,
    GaussGroupPlan,
    GfePlan,
    GfeTmPlan,
    GoalParameterPlan,
    GroupPlan,
    HybridGfePlan,
    LearnPlan,
    LearnTarget,
    MemberAddress,
    MixGroupPlan,
    Mu3Plan,
    ParameterAddress,
    PrecGroupPlan,
    RecurrenceComponentPlan,
    RecurrencePlan,
)
from cffg.compiler_errors import CompileError
from cffg.core.ir import Template
from cffg.core.plates import expand_plates
from cffg.core.validate import validate
from cffg.registered import packing as _registered_packing
from cffg.registered.capabilities import snapshot_registry
from cffg.registered.lowering import lower_registered
from cffg.resolution import sources as _source_helpers
from cffg.resolution.resolve import resolve_compile_data
from cffg.specialized import (
    OrdinaryMembership,
    complete_specialized,
    discover_specialized,
    validate_admission,
)
from cffg.specialized import packing as _specialized_packing
from cffg.specialized.discovery import dirichlet_shape
from cffg.topology import build_topology

__all__ = [
    "NEG",
    "PARAMETER_TRANSFORMS",
    "Arena",
    "CompileError",
    "ConstraintPlan",
    "DirichletGroupPlan",
    "EdgeInfo",
    "EfGroupPlan",
    "GaussGroupPlan",
    "GfePlan",
    "GfeTmPlan",
    "GoalParameterPlan",
    "GroupPlan",
    "HybridGfePlan",
    "LearnPlan",
    "LearnTarget",
    "MemberAddress",
    "MixGroupPlan",
    "Mu3Plan",
    "ParameterAddress",
    "PrecGroupPlan",
    "RecurrenceComponentPlan",
    "RecurrencePlan",
    "compile_template",
]

_resolve_arrays = _source_helpers.resolve_arrays
_blob_shape = _source_helpers._blob_shape
_load_blob = _source_helpers._load_blob
_stack_params = _registered_packing._stack_params
_stack_gauss_params = _registered_packing._stack_gauss_params
_stack_mix_params = _specialized_packing._stack_mix_params
_stack_prec_means = _specialized_packing._stack_prec_means
_stack_recurrence_params = _specialized_packing._stack_recurrence_params
_log_of = _specialized_packing._log_of
_expect = _specialized_packing._expect
_stochastic = _specialized_packing._stochastic


def compile_template(
    t: Template,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    root: str | Path | None = None,
    semiring: str = "sum",
) -> Arena:
    """Ground `t` into an Arena. Blob data comes from `arrays` (name -> array,
    wins) or is loaded from `root`/blob.path.

    `semiring` selects the inference algebra (cff-e8zk): "sum" is sum-product
    (the BFE ledger; -log Z on trees), "max" is min-sum / max-product (MAP
    decoding; Engine.objective, not Engine.bfe). MAP semantics are defined in v1
    for discrete BP groups and hard constraints only — "max" REJECTS vmp/mean-
    field beads, P-substitution/GFE composites, and Gaussian/EF/hybrid groups."""
    if semiring not in ("sum", "max"):
        raise CompileError(f"semiring must be 'sum' or 'max', got {semiring!r}")
    t = expand_plates(t)
    validate(t)
    topology = build_topology(t)
    try:
        registry = snapshot_registry(t)
    except (ValueError, TypeError, KeyError) as error:
        raise CompileError(str(error)) from error
    claims = discover_specialized(t, topology, semiring, registry)
    validate_admission(t.name, claims)
    registered = lower_registered(t, topology, claims, semiring, registry)
    drafts = complete_specialized(claims, OrdinaryMembership(registered.bp_members))
    data = resolve_compile_data(t, arrays, root, topology, registered, drafts)

    return assemble_arena(topology, registered, drafts, data, semiring)


def _dirichlet_shape(t: Template, likelihood_name: str) -> tuple[tuple[int, ...], int]:
    """Compatibility entry for the specialized concentration layout owner."""
    try:
        return dirichlet_shape(t, likelihood_name)
    except ValueError as error:
        raise CompileError(str(error)) from error
