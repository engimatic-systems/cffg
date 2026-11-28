"""Plan ternary equalities connecting one hub to an ordered set of leaves."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from cffg.core.compose import ComposeError
from cffg.core.ir import FactorSpec, VariableSpec


@dataclass(frozen=True)
class EqualityTree:
    """A flat wiring plan independent of the domain and fragment representation."""

    hub: str
    branches: tuple[str, ...]
    links: tuple[str, ...]
    factors: tuple[FactorSpec, ...]

    def install(self, variables, factors, prototype: VariableSpec) -> None:
        """Add the intermediate variables and equalities after checking all names."""
        for name in self.links:
            if name in variables:
                raise ComposeError(f"equality tree for {self.hub!r}: variable {name!r} already exists")
        for factor in self.factors:
            if factor.name in factors:
                raise ComposeError(f"equality tree for {self.hub!r}: factor {factor.name!r} already exists")
        variables.update((name, replace(prototype, name=name)) for name in self.links)
        factors.update((factor.name, factor) for factor in self.factors)


def plan_equality_tree(hub: str, branches: Sequence[str], prefix: str) -> EqualityTree:
    """Split contiguous leaf intervals at their midpoint, preserving input order.

    A singleton aliases the hub. For N >= 2 the plan has N-1 equality factors,
    N-2 links and at most ceil(log2(N)) factors from the hub to any branch.
    """
    branches, links, junctions = equality_wiring(hub, branches, prefix)
    factors = tuple(
        FactorSpec(name, "equality", {"a": inlet, "b": left, "c": right})
        for name, inlet, left, right in junctions
    )
    return EqualityTree(hub, branches, links, factors)


def equality_wiring(hub: str, branches: Sequence[str], prefix: str):
    """Return leaf names, internal links and ordered (factor, inlet, left, right) addresses."""
    branches = tuple(branches)
    if not branches or len(set((hub, *branches))) != len(branches) + 1:
        raise ComposeError(f"equality tree for {hub!r} requires distinct endpoints and at least one branch")

    factors = []
    links = []

    def subtree(first: int, stop: int, inlet: str) -> None:
        middle = (first + stop) // 2
        children = []
        intervals = []
        for lo, hi in ((first, middle), (middle, stop)):
            if hi - lo == 1:
                children.append(branches[lo])
            else:
                link = f"{prefix}/link{len(links)}"
                links.append(link)
                children.append(link)
                intervals.append((lo, hi, link))
        factor = f"{prefix}/eq{len(factors)}"
        factors.append((factor, inlet, *children))
        for lo, hi, link in intervals:
            subtree(lo, hi, link)

    if len(branches) > 1:
        subtree(0, len(branches), hub)
    clashes = set(links).intersection((hub, *branches))
    if clashes:
        raise ComposeError(
            f"equality tree for {hub!r}: generated link collides with endpoint {min(clashes)!r}"
        )
    return branches, tuple(links), tuple(factors)
