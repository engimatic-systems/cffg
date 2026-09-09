"""Hard-constraint drafts and their family-owned payload validation."""

import numpy as np

from cffg.compiled import ConstraintPlan

from .contracts import ArrayRecipe, ConstraintDraft, SpecializedClaims
from .requirements import array_requirement


class ConstraintValidationError(ValueError):
    """An explicit family diagnostic, distinct from incidental array errors."""


def lower_constraints(claims: SpecializedClaims) -> tuple[ConstraintDraft, ...]:
    topology = claims.topology
    drafts = []
    for name in claims.constraints:
        factor = topology.factors[name]
        roles = tuple(f"x{i}" for i in range(len(factor.edges)))
        cards = tuple(
            topology.variables[factor.edges[role]].cardinality for role in roles
        )
        gather = np.array(
            [topology.opposite[name, role] for role in roles], dtype=np.int64
        )
        scatter = np.array(
            [topology.slot_of[name, role] for role in roles], dtype=np.int64
        )
        payload = (
            None
            if factor.type == "alldifferent"
            else "k"
            if factor.type in ("exactly_k", "at_most_k")
            else "table"
        )
        requirement = array_requirement(
            claims, name, ArrayRecipe.CONSTRAINT, (name,), payload=payload
        )
        drafts.append(
            ConstraintDraft(
                name, factor.type, roles, cards, gather, scatter, requirement
            )
        )
    return tuple(drafts)


def resolve_constraint(
    draft: ConstraintDraft, resolved: dict[str, np.ndarray]
) -> ConstraintPlan:
    """Run this family's existing checks at the late constraint checkpoint."""
    fname, cards = draft.name, draft.cards
    n = len(cards)
    k, table_arr = 0, None
    if draft.type == "alldifferent":
        K = cards[0]
        if any(c != K for c in cards):
            raise ConstraintValidationError(
                f"factor {fname!r}: alldifferent needs one shared cardinality, got {cards}"
            )
        if K < n:
            raise ConstraintValidationError(
                f"factor {fname!r}: alldifferent over {n} edges needs cardinality "
                f"K >= {n}, got K={K} (pigeonhole: unsatisfiable)"
            )
        if n > 10 or K > 10:
            raise ConstraintValidationError(
                f"factor {fname!r}: exact alldifferent DP is capped at n, K <= 10 "
                f"(got n={n}, K={K}); larger n is the permanent-approximation regime "
                "(Huang & Jebara, arXiv:0908.1769)"
            )
    elif draft.type in ("exactly_k", "at_most_k"):
        if any(c != 2 for c in cards):
            raise ConstraintValidationError(
                f"factor {fname!r}: {draft.type} operates on binary edges (cardinality 2), "
                f"got {cards}"
            )
        karr = resolved[draft.arrays.inputs[0].blob]
        if karr.size != 1 or int(karr.reshape(())) != karr.reshape(()):
            raise ConstraintValidationError(
                f"factor {fname!r}: parameter 'k' must be a scalar integer"
            )
        k = int(karr.reshape(()))
        if not 0 <= k <= n:
            raise ConstraintValidationError(
                f"factor {fname!r}: k={k} out of range [0, {n}] for {n} binary edges"
            )
    else:  # table
        table_arr = np.asarray(resolved[draft.arrays.inputs[0].blob])
        if table_arr.ndim != 2 or table_arr.shape[1] != n:
            raise ConstraintValidationError(
                f"factor {fname!r}: table must be [T, {n}] (T tuples over {n} edges), "
                f"got shape {table_arr.shape}"
            )
        if table_arr.shape[0] >= 100000:
            raise ConstraintValidationError(
                f"factor {fname!r}: table has {table_arr.shape[0]} tuples (>= 1e5); "
                "large tables leave the explicit-enumeration regime"
            )
        if not np.issubdtype(table_arr.dtype, np.integer) and not np.allclose(
            table_arr, np.round(table_arr)
        ):
            raise ConstraintValidationError(
                f"factor {fname!r}: table entries must be integers"
            )
        table_arr = np.round(table_arr).astype(np.int64)
        for j in range(n):
            if table_arr[:, j].min() < 0 or table_arr[:, j].max() >= cards[j]:
                raise ConstraintValidationError(
                    f"factor {fname!r}: table column {j} has values outside [0, {cards[j]})"
                )

    return ConstraintPlan(
        name=fname,
        type=draft.type,
        cards=cards,
        members=(fname,),
        gather=draft.gather,
        scatter=draft.scatter,
        k=k,
        table=table_arr,
    )
