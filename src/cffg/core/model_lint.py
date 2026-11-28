"""Optional structural diagnostics for valid model declarations."""

from dataclasses import dataclass

from cffg.core.ir import ConstraintKind, PortRole, Template
from cffg.core.plates import expand_plates
from cffg.core.validate import validate


@dataclass(frozen=True)
class LintFinding:
    code: str
    owner: str
    message: str


def lint(template: Template) -> tuple[LintFinding, ...]:
    """Inspect the expanded, validated template without loading parameter arrays."""
    expanded = expand_plates(template)
    validate(expanded)
    return _lint_validated(expanded)


def _lint_validated(template: Template) -> tuple[LintFinding, ...]:
    data = {port.variable for port in template.ports.values() if port.role is PortRole.DATA}
    goals = {f.edges["out"]: f.name for f in template.factors.values() if f.type == "categorical"}
    substituted = {c.target for c in template.constraints if c.kind is ConstraintKind.P_SUBSTITUTION
                   and c.detail_dict().get("substituted_role") == "out"}
    mean_field = {c.target for c in template.constraints if c.kind is ConstraintKind.MEAN_FIELD}
    findings = []
    for name in sorted(template.factors):
        factor = template.factors[name]
        if factor.type == "matrix_cat" and name not in substituted:
            output = factor.edges["out"]
            if output in goals and output not in data:
                findings.append(LintFinding(
                    "missing_p_substitution", name,
                    f"Likelihood {name!r} faces categorical goal {goals[output]!r} on {output!r}. "
                    "This pair contributes no P-substitution epistemic term. Ordinary KL-control "
                    "remains valid.",
                ))
        if name in mean_field and len(factor.edges) == 1 and not factor.param_edges:
            findings.append(LintFinding(
                "unary_mean_field", name,
                f"Factor {name!r} has one random-variable edge. Its mean-field split equals "
                "the unsplit belief.",
            ))
    return tuple(findings)
