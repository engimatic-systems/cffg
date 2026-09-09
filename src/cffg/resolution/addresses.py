"""Materialize declared member addresses and bind learning source/target pairs."""

from dataclasses import dataclass, field

import numpy as np

from cffg.array_requirements import ParameterRuntime
from cffg.compiled import LearnPlan, LearnTarget, MemberAddress, ParameterAddress
from cffg.compiler_errors import CompileError
from cffg.core.ir import Template


@dataclass
class AddressBuilder:
    """Private mutable bookkeeping; final resolved data owns publication."""

    member_of: dict[str, MemberAddress] = field(default_factory=dict)
    writable_members: set[str] = field(default_factory=set)
    learning_parameters: dict[str, ParameterRuntime] = field(default_factory=dict)
    statistic_sources: set[str] = field(default_factory=set)

    def add(self, template, requirement):
        for intent in requirement.members:
            factor = template.factors[intent.member]
            if factor.param_blobs:
                missing = sorted(set(factor.param_blobs) - set(intent.runtime))
                if missing:
                    raise CompileError(
                        f"factor {intent.member!r}: no runtime parameter address for roles {missing}"
                    )
                parameters = []
                for role, blob in sorted(factor.param_blobs.items()):
                    runtime = intent.runtime[role]
                    parameters.append(
                        ParameterAddress(
                            role,
                            blob,
                            runtime.key,
                            intent.index,
                            runtime.transform,
                            runtime.probability_axis,
                            runtime.permutation,
                        )
                    )
                    self.learning_parameters[runtime.key] = runtime
                self.member_of[intent.member] = MemberAddress(
                    factor.type, tuple(parameters)
                )
                if intent.statistic_source:
                    self.statistic_sources.add(intent.member)
            if intent.writable:
                self.writable_members.add(intent.member)


def resolve_learnables(
    t: Template,
    resolved: dict[str, np.ndarray],
    addresses: AddressBuilder,
) -> list[LearnPlan]:
    """Bind each declared learnable to its stat sources and write targets
    after every earlier compile gate has passed.

    Every factor reading one of the learnable's blobs becomes a target, so a
    posterior update reaches every instance that shares the parameter — the
    tying the IR already expresses by blob name. Sources are the subset that
    live in a plain discrete group, because those are the nodes whose beliefs
    Engine.node_beliefs reports; GFE composites are write-only here."""
    from cffg.learn import FAMILIES

    plans: list[LearnPlan] = []
    member_of = addresses.member_of
    for name in sorted(t.learnables):
        spec = t.learnables[name]
        if spec.family not in FAMILIES:
            raise CompileError(
                f"learnable {name!r}: unknown conjugate family {spec.family!r} "
                f"(known: {sorted(FAMILIES)})"
            )
        blobs = set(spec.blobs)
        readers = sorted(
            f.name for f in t.factors.values() if blobs & set(f.param_blobs.values())
        )
        if not readers:
            raise CompileError(
                f"learnable {name!r}: no factor reads blobs {sorted(blobs)}"
            )
        ftypes = {t.factors[r].type for r in readers}
        if len(ftypes) > 1:
            raise CompileError(
                f"learnable {name!r}: blobs {sorted(blobs)} are read by mixed factor "
                f"types {sorted(ftypes)}; one posterior needs one sufficient statistic"
            )
        ftype = ftypes.pop()
        family = FAMILIES[spec.family]
        # Only the blobs this template actually reads: one learnable spans a
        # probability blob and a counts blob (build_rollout's A/Ac, Bs/Bc), and
        # a perception graph resolves the first while a planning graph with
        # parameter uncertainty resolves the second.
        shapes = {
            b: tuple(np.shape(resolved[b])) for b in sorted(blobs) if b in resolved
        }
        # Which parameter ROLE each blob fills. A multi-parameter family
        # writes its outputs by role, so a gaussian_linear's matrix, offset
        # and cov blobs each receive their own part of one MNIW rollover.
        roles = {
            b: role
            for r in readers
            for role, b in t.factors[r].param_blobs.items()
            if b in blobs
        }
        try:
            family.check(name, ftype, shapes, roles)
        except ValueError as err:
            raise CompileError(str(err)) from None

        targets: list[LearnTarget] = []
        sources: list[str] = []
        for r in readers:
            if r not in member_of:
                raise CompileError(
                    f"learnable {name!r}: factor {r!r} ({t.factors[r].type}) has no "
                    "addressable parameter slot"
                )
            member = member_of[r]
            read = sorted(blobs & set(t.factors[r].param_blobs.values()))
            for blob in read:
                role = roles[blob]
                parameter = member.parameter(role)
                if parameter.blob != blob:
                    raise CompileError(
                        f"learnable {name!r}: factor {r!r} role {role!r} reads "
                        f"{parameter.blob!r}, not {blob!r}"
                    )
                interpretation = addresses.learning_parameters[parameter.key]
                targets.append(
                    LearnTarget(
                        parameter.key,
                        parameter.index,
                        interpretation.learning_form,
                        interpretation.learning_layout,
                        blob,
                        role,
                        parameter.transform,
                    )
                )
            if r in addresses.statistic_sources and (
                spec.sources is None or r in spec.sources
            ):
                sources.append(r)
        # A graph with no sources is normal, not an error: the planning
        # template reads A only through GFE composites, and learns nothing.
        # Statistics come from the perception template, parameters flow back.
        plans.append(
            LearnPlan(name, spec.family, ftype, shapes, tuple(sources), tuple(targets))
        )
    return plans
