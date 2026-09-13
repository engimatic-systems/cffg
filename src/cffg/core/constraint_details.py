"""Detect contradictory P-substitution metadata before dictionary conversion."""

import json
from collections.abc import Iterable

from cffg.core.ir import ConstraintKind, ConstraintSpec

_MISSING = object()


def _display(value: object) -> str:
    if value is _MISSING:
        return "<missing>"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def p_substitution_conflicts(constraints: Iterable[ConstraintSpec]) -> list[str]:
    """Compare complete declarations without discarding repeated raw keys.

    Identical repetitions are valid and remain in the template. Comparisons
    use canonical JSON values, so mapping-key order does not change meaning.
    Numeric representations remain distinct, as in the serialized structure:
    1 and 1.0 are not identical declarations, nor are 0.0 and -0.0.
    A missing field is distinct from every explicit value. This checks only
    conflicts, not factor existence, supported roles, or evaluator admission.
    """
    targets: dict[str, list[dict[str, list[object]]]] = {}
    for constraint in constraints:
        if constraint.kind is not ConstraintKind.P_SUBSTITUTION:
            continue
        detail: dict[str, list[object]] = {}
        for key, value in constraint.detail:
            detail.setdefault(key, []).append(value)
        targets.setdefault(constraint.target, []).append(detail)

    problems = []
    for target, declarations in sorted(targets.items()):
        keys = {key for detail in declarations for key in detail}
        conflicts = []
        for key in sorted(keys):
            values = [
                value
                for detail in declarations
                for value in detail.get(key, [_MISSING])
            ]
            if len(values) < 2:
                continue
            alternatives = sorted({_display(value) for value in values})
            if len(alternatives) > 1:
                conflicts.append(f"{key!r}: " + " vs ".join(alternatives))
        if conflicts:
            problems.append(
                f"p_substitution on {target!r}: conflicting declarations for "
                + "; ".join(conflicts)
            )
    return problems
