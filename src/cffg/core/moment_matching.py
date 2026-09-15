"""Explicit endpoint and sufficient-statistic declarations (Part I, §5.4).

Selecting one endpoint relaxes its marginal-consistency requirement to the
declared moments. The opposite endpoint is an independent declaration.
Legacy factor-only payloads retain their meaning until explicitly migrated.
"""

from dataclasses import dataclass
from enum import Enum

from cffg.core.ir import ConstraintKind, ConstraintSpec, Template


class MomentProjection(str, Enum):
    GAUSSIAN = "gaussian_first_second"


@dataclass(frozen=True)
class MomentMatchingEndpoint:
    role: str
    projection: MomentProjection


def moment_matching(
    factor: str, role: str, projection: MomentProjection
) -> ConstraintSpec:
    """Select E[X] and E[XXᵀ] at a named Gaussian factor endpoint."""
    if not isinstance(role, str) or not role:
        raise ValueError("moment matching requires a nonempty structural role")
    return ConstraintSpec(
        ConstraintKind.MOMENT_MATCHING,
        factor,
        (("projection", MomentProjection(projection).value), ("role", role)),
    )


def moment_matching_endpoint(
    constraint: ConstraintSpec, template: Template
) -> MomentMatchingEndpoint:
    """Validate a complete declaration without guessing absent information."""
    prefix = f"factor {constraint.target!r}: "
    if constraint.kind is not ConstraintKind.MOMENT_MATCHING:
        raise ValueError(prefix + "expected a moment-matching constraint")
    if not constraint.detail:
        raise ValueError(
            prefix + "legacy moment-matching syntax does not identify "
            "an incident edge endpoint or projection"
        )
    keys = [key for key, _ in constraint.detail]
    if len(keys) != len(set(keys)):
        raise ValueError(prefix + "duplicate moment-matching detail keys")
    if set(keys) != {"role", "projection"}:
        raise ValueError(prefix + "moment matching requires exactly 'role' and 'projection'")
    detail = constraint.detail_dict()
    role = detail["role"]
    factor = template.factors.get(constraint.target)
    if factor is None:
        raise ValueError(prefix + "unknown moment-matching factor")
    if not isinstance(role, str) or role not in factor.edges:
        raise ValueError(prefix + f"moment-matching role {role!r} is not a structural edge")
    try:
        projection = MomentProjection(detail["projection"])
    except (ValueError, TypeError) as exc:
        raise ValueError(prefix + f"unknown moment projection {detail['projection']!r}") from exc
    variable = template.variables.get(factor.edges[role])
    if variable is None or not variable.dim:
        raise ValueError(prefix + "Gaussian moment projection requires a Gaussian edge")
    return MomentMatchingEndpoint(role, projection)


def migrate_moment_matching(
    constraint: ConstraintSpec,
    template: Template,
    *,
    role: str,
    projection: MomentProjection,
) -> ConstraintSpec:
    """Replace a legacy placeholder using caller-supplied endpoint semantics."""
    if constraint.kind is not ConstraintKind.MOMENT_MATCHING or constraint.detail:
        raise ValueError("explicit migration requires a legacy factor-only moment constraint")
    result = moment_matching(constraint.target, role, projection)
    moment_matching_endpoint(result, template)
    return result


def moment_matching_problems(template: Template) -> list[str]:
    """Preserve legacy placeholders; reject incomplete or invalid new details."""
    problems = []
    for constraint in template.constraints:
        if constraint.kind is ConstraintKind.MOMENT_MATCHING and constraint.detail:
            try:
                moment_matching_endpoint(constraint, template)
            except ValueError as exc:
                problems.append(str(exc))
    return problems
