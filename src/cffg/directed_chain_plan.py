"""Recognize the directed Taylor envelope of DERIVATIONS D13.6.

A single Gaussian prior feeds a constant-dimensional chain with at least one
transition. Every nonterminal state has a three-way equality joining its
prediction, one conditioned linear emission and the input of the next
transition. The final prediction feeds its emission directly. Cutting any
transition thus leaves only the prefix and its linear data on the input side.
One compiled Taylor group supplies the transition function. Its parameter and
noise rows can differ. Emissions can have different output dimensions.

Noise covariances must be positive definite. Each DATA row can be present or
missing, and a linear goal may supply rank-deficient state information. Fixed
parameter blobs may be shared. Learnable parameters, other factor families,
extra components, nonlinear sensors and sigma-point approximations fall outside
this envelope. Recognition reads only compiled structure. The executor must
check live parameters, masks, observations, cavity propriety, Taylor values and
derivatives, and the resulting messages and free energy.
"""

from dataclasses import dataclass

from cffg.compiled import Arena, GaussGroupPlan


@dataclass(frozen=True)
class GaussianMember:
    """One compiled factor and its logical Gaussian slots in group role order."""

    group: GaussGroupPlan
    row: int
    gather: tuple[int, ...]
    scatter: tuple[int, ...]


@dataclass(frozen=True)
class DirectedChainSite:
    emission: GaussianMember
    obs_row: int
    transition: GaussianMember | None
    equality: GaussianMember | None
    # Logical slots written toward the incoming prediction, emission, transition.
    equality_scatter: tuple[int, int, int] | None


@dataclass(frozen=True)
class DirectedChainPlan:
    prior: GaussianMember
    sites: tuple[DirectedChainSite, ...]
    dim: int


def _supported_group(group: GaussGroupPlan) -> bool:
    if group.type == "equality":
        return group.kernel is None and group.roles == ("a", "b", "c")
    kernel = group.kernel
    if kernel is None or kernel.version != "1.0.0":
        return False
    if group.type == "gaussian":
        return (kernel.impl_id == "cffg.gaussian.gaussian" and group.roles == ("out",)
                and group.observed_role is None and kernel.binding.objective == "analytic")
    if group.roles != ("out", "in"):
        return False
    if group.type == "gaussian_linear_obs":
        return (kernel.impl_id == "cffg.gaussian.gaussian_linear" and group.observed_role == 0
                and kernel.binding.objective == "analytic")
    return (group.type == "gaussian_nonlinear" and group.observed_role is None
            and group.dims[0] == group.dims[1] and kernel.impl_id == "cffg.gaussian.nonlinear"
            and kernel.binding.objective == "linearization_frozen_gaussian_bethe"
            and kernel.binding.taylor_linearize is not None)


def _split(equality, pred_role, readers):
    """Identify the two other equality legs by the factor roles that read them."""
    legs = {}
    for role, slot in enumerate(equality.scatter):
        if role == pred_role:
            continue
        reader = readers.get(slot)
        if reader is None:
            return None
        member, input_role = reader
        if input_role != 1:
            return None
        legs[member.group.type] = (member, slot)
    if "gaussian_linear_obs" not in legs or "gaussian_nonlinear" not in legs:
        return None
    emission, emission_slot = legs["gaussian_linear_obs"]
    transition, transition_slot = legs["gaussian_nonlinear"]
    return DirectedChainSite(
        emission, int(emission.group.obs_rows[emission.row]), transition, equality,
        (equality.scatter[pred_role], emission_slot, transition_slot),
    )


def recognize_directed_chain(arena: Arena) -> DirectedChainPlan | None:
    """Return the complete ordered structure, or None for the ordinary solver."""
    if (arena.semiring != "sum" or arena.n_slots or arena.n_efslots or arena.learn_plans
            or arena.groups or arena.ef_groups or arena.dirichlet_groups or arena.constraint_groups
            or arena.mix_groups or arena.prec_groups or arena.recurrence_groups or arena.gfe_groups
            or arena.gfe_tm_groups or arena.hybrid_gfe_groups or arena.mu3_groups):
        return None

    readers = {}
    prior = None
    transition_count = 0
    members = 0
    for group in arena.gauss_groups:
        if not _supported_group(group):
            return None
        if group.type == "gaussian_nonlinear":
            if transition_count:
                return None
            transition_count = len(group.members)
        for row, (gather, scatter) in enumerate(zip(group.gather, group.scatter, strict=True)):
            member = GaussianMember(group, row, tuple(map(int, gather)), tuple(map(int, scatter)))
            if group.type == "gaussian":
                if prior is not None:
                    return None
                prior = member
            for role, slot in enumerate(member.gather):
                readers[slot] = member, role
            members += 1
    if prior is None or not transition_count:
        return None

    sites = []
    prediction = prior
    for _ in range(transition_count):
        reader = readers.get(prediction.scatter[0])
        if reader is None:
            return None
        equality, pred_role = reader
        if equality.group.type != "equality":
            return None
        site = _split(equality, pred_role, readers)
        if site is None:
            return None
        sites.append(site)
        prediction = site.transition

    terminal = readers.get(prediction.scatter[0])
    if terminal is None:
        return None
    emission, role = terminal
    if emission.group.type != "gaussian_linear_obs" or role != 1:
        return None
    # Each directed leg has exactly one peer. A walk reaching this terminal
    # cannot have revisited a state. Account for every row, including islands.
    if members != 1 + 3 * len(sites) + 1:
        return None
    sites.append(DirectedChainSite(emission, int(emission.group.obs_rows[emission.row]),
                                   None, None, None))
    return DirectedChainPlan(prior, tuple(sites), prior.group.dims[0])
