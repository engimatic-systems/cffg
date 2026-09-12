"""Internal immutable plans and addresses shared by compiler consumers.

Constructors snapshot mutable input data. Stored arrays are FrozenArray values;
NumPy conversion produces independent working storage. Inference updates belong
to runtime State, never to these shared plans or their initial parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from cffg import kernels
from cffg._immutable import FrozenArray, ImmutableRecord

NEG = -1.0e30  # Reserved log-space zero sentinel.


@dataclass(frozen=True)
class EdgeInfo(ImmutableRecord):
    name: str
    card: int
    slot0: int  # side-0 message: into the edge from its first endpoint
    slot1: int  # side-1 message: second endpoint, or the external side
    interior: bool  # both sides are factors (degree 2)
    dim: int = 0  # Gaussian dimension; 0 = discrete. Gaussian slots index the
    # (eta, Lambda) arena, discrete slots the log-probability arena — the two
    # numberings are independent (cff-4e5z).
    ef: str = ""  # EF family name; slots then index the natural-parameter
    # arena ef_eta [n_efslots, 2], a third independent numbering (cff-8s6a).
    param_shape: tuple[int, ...] = ()  # Dirichlet parameter edge, packed into
    # the flat message arena as one concentration/count value per table cell.
    # Empty means an ordinary discrete edge.  Parameter slots are *not*
    # categorical log messages and are excluded from discrete normalization,
    # exact-support checks, decoding and the ordinary edge-entropy ledger.
    param_axis: int = 0  # output/concentration axis inside param_shape


@dataclass(frozen=True)
class GroupPlan(ImmutableRecord):
    """All factors of one (type, cardinality-signature, mode), run together."""

    name: str
    type: str
    roles: tuple[str, ...]
    cards: tuple[int, ...]  # cardinality per role
    members: tuple[str, ...]  # factor names, defines the stacking order
    gather: FrozenArray  # [n_members, n_roles] slot to read (opposite side)
    scatter: FrozenArray  # [n_members, n_roles] slot to write (own side)
    mode: str = "bp"  # "bp" (sum-product) | "vmp" (mean-field bead)
    gather_both: FrozenArray | None = None  # [M, R, 2] both slots (vmp: marginals)
    kernel: kernels.FactorKernel | None = None
    parameter_paths: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DirichletGroupPlan(ImmutableRecord):
    """Mean-field categorical likelihoods with an in-graph Dirichlet table.

    One member consumes a structural likelihood factor and its facing
    ``dirichlet`` prior.  Structural edges remain categorical log messages;
    the parameter edge's two packed slots carry the prior concentrations
    ``alpha0`` and the likelihood's fractional counts.  Their sum is the
    proper posterior concentration.  Packing concentrations rather than
    ``alpha0 - 1`` avoids erasing sparse positive float32 priors by
    cancellation at one.
    """

    name: str
    type: str  # categorical | matrix_cat | transition_mixture
    roles: tuple[str, ...]
    cards: tuple[int, ...]
    members: tuple[str, ...]  # likelihood factor names
    priors: tuple[str, ...]  # facing dirichlet factor names
    gather_both: FrozenArray  # [M, R, 2] both structural slots
    scatter: FrozenArray  # [M, R] likelihood-owned structural slots
    prior_scatter: FrozenArray  # [M] prior-owned parameter slots
    count_scatter: FrozenArray  # [M] likelihood-owned parameter slots
    param_shape: tuple[int, ...]  # canonical serialized layout
    output_axis: int  # Dirichlet-normalized axis in param_shape

    @property
    def param_size(self) -> int:
        return int(np.prod(self.param_shape))


@dataclass(frozen=True)
class ConstraintPlan(ImmutableRecord):
    """One hard-constraint (indicator) factor, run on its own (cff-e8zk).

    Carries structural metadata only — NO dense potential is stacked. Edges are
    the variadic roles x0..x{n-1} in order; `gather`/`scatter` are per-edge
    slots (read opposite, write own). `cards` is the per-edge cardinality. The
    static compile-time detail (`k` for exactly/at_most, `table` [T, n] for
    table) is baked into the plan. The same plan drives sum-product (BFE) and
    max-product (MAP): the engine passes the semiring's reduction op."""

    name: str
    type: str
    cards: tuple[int, ...]  # per-edge cardinality
    members: tuple[str, ...]  # single factor name (kept a tuple for symmetry)
    gather: FrozenArray  # [n] slot to read (opposite side)
    scatter: FrozenArray  # [n] slot to write (own side)
    k: int = 0  # exactly_k / at_most_k threshold
    table: FrozenArray | None = None  # [T, n] allowed tuples (table)


@dataclass(frozen=True)
class GaussGroupPlan(ImmutableRecord):
    """All Gaussian-family factors of one (type, dim-signature), run together.

    Types: "gaussian" (prior), "gaussian_linear", "gaussian_linear_obs" (a
    gaussian_linear whose out edge is a DATA port — compiled to *conditioning*
    on the runtime observation, never a huge-precision clamp, so the
    differential-entropy ledger stays finite; see cffg.gauss), and "equality"
    (family-polymorphic, routed here when its edges are Gaussian)."""

    name: str
    type: str
    roles: tuple[str, ...]
    dims: tuple[int, ...]  # dimension per role
    members: tuple[str, ...]
    gather: FrozenArray  # [M, R] gaussian slot to read (opposite side)
    scatter: FrozenArray  # [M, R] gaussian slot to write (own side)
    obs_rows: FrozenArray | None = None  # [M] rows into the g_obs array
    kernel: kernels.FactorKernel | None = None
    parameter_paths: tuple[tuple[str, str], ...] = ()
    observed_role: int | None = None

    @property
    def observed_dim(self) -> int:
        return self.dims[self.observed_role] if self.observed_role is not None else 0


@dataclass(frozen=True)
class MixGroupPlan(ImmutableRecord):
    """All hybrid Gaussian-mixture factors of one (type, dim-signature,
    selector cardinality), run together (cff-pksr). A hybrid node straddles
    two arenas: its out(/in) edges live in the Gaussian (eta, Lambda) arena
    (g_gather/g_scatter, read-opposite/write-own like GaussGroupPlan), its
    selector edge in the discrete log-probability arena (s_gather/s_scatter).

    Types: "gaussian_mixture" (roles out, selector; dims (do,)) and
    "gaussian_mixture_linear" (roles out, in, selector; dims (do, di)).
    Parameters are per-regime stacks keyed under f"{name}:W" [M, K, do, do]
    plus f"{name}:m" (mixture) or f"{name}:A"/f"{name}:b" (linear)."""

    name: str
    type: str
    dims: tuple[int, ...]  # (do,) or (do, di)
    k_sel: int  # selector cardinality K (the per-regime stack axis)
    members: tuple[str, ...]
    g_gather: FrozenArray  # [M, R_g] gaussian slot to read (opposite side)
    g_scatter: FrozenArray  # [M, R_g] gaussian slot to write (own side)
    s_gather: FrozenArray  # [M] selector discrete slot to read (opposite side)
    s_scatter: FrozenArray  # [M] selector discrete slot to write (own side)


@dataclass(frozen=True)
class RecurrencePlan(ImmutableRecord):
    """Source-specific stick-breaking transitions across two arenas.

    The two discrete roles read their facing cavities for structured
    sum-product. The Gaussian drive reads its complete edge marginal for the
    coordinate-optimal PG/JJ bound. The factor writes two categorical
    messages and one possibly singular Gaussian likelihood message.
    """

    name: str
    cards: tuple[int, int]  # (K_out, K_in), equal in the rSLDS v1 family
    dim: int
    members: tuple[str, ...]
    d_gather_both: FrozenArray  # [M, 2 roles, 2 edge slots]
    d_scatter: FrozenArray  # [M, 2 roles]
    drive_gather_both: FrozenArray  # [M, 2 edge slots]
    drive_scatter: FrozenArray  # [M]
    component_ids: FrozenArray  # [M] rows into Arena.recurrence_components


@dataclass(frozen=True)
class RecurrenceComponentPlan(ImmutableRecord):
    """Derived categorical component containing recurrence-link factors.

    ``acyclic`` certifies exact sum-product marginals for the component's
    bounded factor potentials. It does not make the JJ Gaussian expectation
    exact and it does not claim exactness for a component with a loop.
    """

    members: tuple[str, ...]
    categorical_factors: tuple[str, ...]
    acyclic: bool


@dataclass(frozen=True)
class PrecGroupPlan(ImmutableRecord):
    """All gaussian_prec factors, run together (cff-7c5o). A scalar Gaussian
    node f(x; m, τ) = N(x; m, τ⁻¹) whose precision lives on an ef:gamma edge —
    the first factor straddling the Gaussian (eta, Lambda) arena (its `out`
    edge) and the EF natural-parameter arena (its `prec` edge). Both VMP
    messages read *marginals* (both slots of each edge, like the discrete vmp
    groups), so the gather tables carry both slots. Parameters: the scalar
    means stacked [M, 1] under f"{name}:m"."""

    name: str
    members: tuple[str, ...]
    x_gather_both: FrozenArray  # [M, 2] both Gaussian slots of the out edge
    tau_gather_both: FrozenArray  # [M, 2] both EF slots of the prec edge
    x_scatter: FrozenArray  # [M] own Gaussian slot on the out edge
    tau_scatter: FrozenArray  # [M] own EF slot on the prec edge


@dataclass(frozen=True)
class EfGroupPlan(ImmutableRecord):
    """All EF-family factors of one (type, family), run together (cff-8s6a).

    Every non-equality type dispatches through its registered FactorKernel
    (cffg.kernels, cff-ri8m): prior kernels stack constant natural parameters
    into params["<name>:<key>"], conditioned observation kernels
    (observed_out) take the "out" edge as a DATA port with the message a
    constant conjugate increment to the parameter edge (missing observation
    drops the node — the Gaussian conditioning convention). "equality" stays
    structural: family-polymorphic sum-minus-self on natural parameters."""

    name: str
    type: str
    family: str  # message family of the edges it reads/writes
    roles: tuple[str, ...]  # message-carrying roles only (obs "out" is data)
    members: tuple[str, ...]
    gather: FrozenArray  # [M, R] ef slot to read (opposite side)
    scatter: FrozenArray  # [M, R] ef slot to write (own side)
    obs_rows: FrozenArray | None = None  # [M] rows into the ef_obs array


@dataclass(frozen=True)
class GfeTmPlan(ImmutableRecord):
    """A P-substituted transition_mixture node (cff-fk2u): the dynamics-
    curiosity composite. Solves the Corollary-1 fixed point over the joint
    (z, u) simplex with the downstream state message in the goal prior's
    role; `learned=True` (detail parameter_uncertainty=dirichlet) reads
    per-action Dirichlet counts α [U, K', K] and adds the count-novelty
    channel. FE contribution is a GFE term with optimistic-bound semantics
    (LIMITATIONS S1c)."""

    name: str
    cards: tuple[int, int, int]  # (K_out, K_in, U)
    members: tuple[str, ...]
    gather: FrozenArray  # [M, 3] incoming slots: (out=μ_down, in=μ_in, sel=μ_e)
    scatter: FrozenArray  # [M, 3] outgoing slots, same role order
    learned: bool = False


@dataclass(frozen=True)
class HybridGfePlan(ImmutableRecord):
    """A D5 P-substituted Gaussian-mixture node and absorbed Gaussian goal.

    The hybrid factor retains its selector and Gaussian state slots, while
    the substituted ``out`` edge stores the collapsed predictive only. The
    facing Gaussian prior contributes goal parameters but no ordinary BP
    message or separate ledger term.
    """

    name: str
    type: str
    dims: tuple[int, ...]
    k_sel: int
    members: tuple[str, ...]
    goal_members: tuple[str, ...]
    g_gather: FrozenArray
    g_scatter: FrozenArray
    s_gather: FrozenArray
    s_scatter: FrozenArray


PARAMETER_TRANSFORMS = frozenset(
    {"identity", "probability_to_log", "covariance_to_precision"}
)


@dataclass(frozen=True)
class ParameterAddress(ImmutableRecord):
    """One serialized parameter role's row in the runtime arena."""

    role: str
    blob: str
    key: str
    index: int
    transform: str
    probability_axis: int | None = None
    permutation: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.transform not in PARAMETER_TRANSFORMS:
            raise ValueError(f"unknown parameter transform {self.transform!r}")


@dataclass(frozen=True)
class MemberAddress(ImmutableRecord):
    """All runtime parameter rows owned by one serialized factor."""

    factor_type: str
    parameters: tuple[ParameterAddress, ...]

    def parameter(self, role: str) -> ParameterAddress:
        matches = tuple(address for address in self.parameters if address.role == role)
        if len(matches) != 1:
            raise KeyError(
                f"factor type {self.factor_type!r} has parameter roles "
                f"{sorted(address.role for address in self.parameters)}, not {role!r}"
            )
        return matches[0]


@dataclass(frozen=True)
class LearnTarget(ImmutableRecord):
    """One place a learnable posterior writes back into stacked params.

    `form` is how that consumer reads the parameter: "expected" wants the
    family's expected natural parameter (Dirichlet: E[log θ] = ψ(α) − ψ(α₀)),
    "counts" wants the raw hyperparameters α, which is what the GFE
    composites differentiate for their parameter-novelty term.

    `layout` names the blob → stacked-params transform. Discrete BP groups
    hold transition_mixture as [K', K, U] while the blob (and every other
    consumer) is [U, K', K]; "as_is" everywhere else.

    `blob` and `role` are what route a multi-parameter family. An MNIW
    posterior produces a matrix, an offset and a covariance; each of the
    gaussian_linear's three blobs fills one parameter role, and the target
    says which. For a one-role family every target names the same role."""

    key: str  # params key of the stacked group
    index: int  # member row within that group
    form: str  # "expected" | "counts"
    layout: str  # "as_is" | "tm"
    blob: str = ""  # the blob this target's factor reads
    role: str = "statistic"  # the parameter role that blob fills
    transform: str = "identity"  # serialized value -> runtime row


@dataclass(frozen=True)
class LearnPlan(ImmutableRecord):
    """A declared learnable, resolved against the compiled arena (cff-x0sv).

    `sources` are the factor instances whose node beliefs supply sufficient
    statistics — every instance reading a blob of this learnable, which is how
    tying accumulates: build_rollout's slices and a plate's copies all name
    one blob, so they all pay into one posterior.

    GFE composites are targets but never sources: the composite owns its own
    update path. Ordinary discrete and Gaussian factors are sources through
    the readout selected by their conjugate family."""

    name: str
    family: str
    semantic_identity: str
    ftype: str  # factor type of the sources — fixes the sufficient statistic
    shapes: Mapping[str, tuple[int, ...]]  # blob -> array shape
    sources: tuple[str, ...]  # factor names, sorted
    targets: tuple[LearnTarget, ...]


@dataclass(frozen=True)
class GfePlan(ImmutableRecord):
    """A P-substituted goal-observation composite (Part II §4): matrix_cat
    observation node facing a categorical goal node across the x edge. The
    composite owns both factors' updates and their joint FE term.

    `learned=True` (P-bead detail parameter_uncertainty=dirichlet): the matrix
    parameter is a Dirichlet *count* array α, not probabilities. Messages then
    use Ā = E[A] for probability terms and the closed-form coupled term
    E[A log A] for the pseudo-entropy — the parameter-novelty channel."""

    name: str
    cards: tuple[int, int]  # (M_obs, K_state)
    members: tuple[str, ...]  # obs-factor names
    gather_z: FrozenArray  # [M] incoming state-message slot (μ_D)
    scatter_z: FrozenArray  # [M] outgoing backward-message slot (μ 2)
    scatter_x: FrozenArray  # [M] obs-side x slot (posterior predictive q_x)
    learned: bool = False
    mu3: bool = False
    param_blobs: tuple[str, ...] = ()
    goal_learned: bool = False
    goal_members: tuple[str, ...] = ()
    goal_param_edges: tuple[str, ...] = ()
    goal_priors: tuple[str, ...] = ()
    goal_prior_blobs: tuple[str, ...] = ()
    goal_tie_groups: tuple[tuple[int, ...], ...] = ()
    goal_prior_scatter: FrozenArray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    goal_count_scatter: FrozenArray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )


@dataclass(frozen=True)
class GoalParameterPlan(ImmutableRecord):
    """One learned goal shared by blob identity, independent of shape batches.

    Group rows select each composite's prediction exactly once. The first
    prior/count slot is canonical; all slots on that side carry the same value.
    """

    prior_blob: str
    cardinality: int
    group_rows: tuple[tuple[str, tuple[int, ...]], ...]
    parameter_edges: tuple[str, ...]
    prior_scatter: FrozenArray
    count_scatter: FrozenArray


@dataclass(frozen=True)
class Mu3Plan(ImmutableRecord):
    """One serialized functional parameter-message evaluator.

    Every ``row`` is a future GFE slice reading the same Dirichlet parameter
    blob.  Their nonconjugate log messages add before normalization against
    the proper base posterior held in ``count_key``.  Particles and weights
    are deliberately absent: they are derived runtime state, reconstructed
    from this evaluator identity plus an explicit PRNG key.
    """

    name: str
    evaluator: str
    gfe_name: str
    rows: FrozenArray
    members: tuple[str, ...]
    param_blob: str
    count_key: str
    cards: tuple[int, int]


@dataclass(frozen=True)
class Arena(ImmutableRecord):
    template_name: str
    n_slots: int
    message_starts: FrozenArray  # [n_slots + 1] prefix offsets into flat messages
    edges: Mapping[str, EdgeInfo]
    groups: tuple[GroupPlan, ...]
    data_slots: Mapping[str, int]  # DATA port name -> external slot
    semiring: str = "sum"  # "sum" (sum-product / BFE) | "max" (min-sum / MAP)
    dirichlet_groups: tuple[DirichletGroupPlan, ...] = field(default_factory=tuple)
    constraint_groups: tuple[ConstraintPlan, ...] = field(default_factory=tuple)
    log_params: Mapping[str, FrozenArray] = field(default_factory=dict)  # group -> [M, ...]
    # Gaussian arena (cff-4e5z): eta and Lambda are separate flat arrays.
    # Prefix offsets make each logical slot exactly d and d*d values;
    # vacuous = zeros, so init and half-edges remain free.
    n_gslots: int = 0
    g_eta_starts: FrozenArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.int64)
    )
    g_lam_starts: FrozenArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.int64)
    )
    gauss_groups: tuple[GaussGroupPlan, ...] = field(default_factory=tuple)
    gauss_data: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    # Gaussian DATA port -> (row in g_obs, dim of the observed edge)
    n_obs_rows: int = 0
    g_obs_starts: FrozenArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.int64)
    )
    mix_groups: tuple[MixGroupPlan, ...] = field(default_factory=tuple)
    # Hybrid Gaussian-mixture groups (cff-pksr): each writes both the Gaussian
    # arena (out/in edges) and the discrete arena (selector edge).
    recurrence_groups: tuple[RecurrencePlan, ...] = field(default_factory=tuple)
    recurrence_components: tuple[RecurrenceComponentPlan, ...] = ()
    prec_groups: tuple[PrecGroupPlan, ...] = field(default_factory=tuple)
    # gaussian_prec groups (cff-7c5o): each writes both the Gaussian arena
    # (out edge) and the EF arena (prec edge).
    gauss_interior: tuple[tuple[int, FrozenArray], ...] = field(default_factory=tuple)
    # [(dim, [E, 2] slot pairs)] interior Gaussian edges for the entropy ledger
    # EF arena (cff-8s6a): messages are natural parameters ef_eta
    # [n_efslots, 2], zero-padded past each family's n_params; vacuous =
    # zeros, so init and half-edges are free (η = 0 is the empty product).
    n_efslots: int = 0
    ef_groups: tuple[EfGroupPlan, ...] = field(default_factory=tuple)
    ef_data: Mapping[str, tuple[int, str]] = field(default_factory=dict)
    # EF DATA port -> (row in the ef_obs/ef_mask arrays, observation family)
    n_ef_obs: int = 0
    ef_interior: tuple[tuple[str, FrozenArray], ...] = field(default_factory=tuple)
    # [(family, [E, 2] slot pairs)] interior EF edges for the entropy ledger
    gfe_groups: tuple[GfePlan, ...] = field(default_factory=tuple)
    goal_parameter_plans: tuple[GoalParameterPlan, ...] = ()
    mu3_groups: tuple[Mu3Plan, ...] = field(default_factory=tuple)
    gfe_tm_groups: tuple[GfeTmPlan, ...] = field(default_factory=tuple)
    hybrid_gfe_groups: tuple[HybridGfePlan, ...] = field(default_factory=tuple)
    gfe_x_edges: frozenset[str] = frozenset()  # entropy absorbed into composite term
    delta_pairs: FrozenArray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )  # [n_delta_edges, 2] slot pairs projected to a pointmass each sweep
    member_of: Mapping[str, MemberAddress] = field(default_factory=dict)
    # Factor name -> role-bearing rows in the stacked runtime parameters.
    # One factor can span several arrays (for example A, b and W).
    writable_members: frozenset[str] = frozenset()
    # Synthetic GFE members retain internal learning addresses but are not
    # direct named-write entry points.
    learn_plans: tuple[LearnPlan, ...] = field(default_factory=tuple)
    # Declared learnables resolved to stat sources and write targets (cff-x0sv).

    @property
    def interior_slot_pairs(self) -> np.ndarray:
        return np.array(
            [
                [e.slot0, e.slot1]
                for e in self.edges.values()
                if (
                    e.interior
                    and not e.dim
                    and not e.ef
                    and not e.param_shape
                    and e.name not in self.gfe_x_edges
                )
            ],
            dtype=np.int64,
        ).reshape(-1, 2)

    @property
    def interior_slot_groups(self) -> list[tuple[int, np.ndarray]]:
        """Interior discrete slot pairs grouped by cardinality.

        The logical slot table stays useful for graph inspection, while the
        grouped form lets the VM materialize only shape-local dense blocks.
        """
        by_card: dict[int, list[list[int]]] = {}
        for e in self.edges.values():
            if (
                e.interior
                and not e.dim
                and not e.ef
                and not e.param_shape
                and e.name not in self.gfe_x_edges
            ):
                by_card.setdefault(e.card, []).append([e.slot0, e.slot1])
        return [
            (card, np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
            for card, pairs in sorted(by_card.items())
        ]

    @property
    def delta_slot_groups(self) -> list[tuple[int, np.ndarray]]:
        """Delta-constrained logical slot pairs grouped by cardinality."""
        if not len(self.delta_pairs):
            return []
        card_of = {
            slot: e.card
            for e in self.edges.values()
            if not e.dim and not e.ef and not e.param_shape
            for slot in (e.slot0, e.slot1)
        }
        by_card: dict[int, list[np.ndarray]] = {}
        for pair in self.delta_pairs:
            by_card.setdefault(card_of[int(pair[0])], []).append(pair)
        return [
            (card, np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
            for card, pairs in sorted(by_card.items())
        ]

    @property
    def n_message_values(self) -> int:
        return int(self.message_starts[-1])

    @property
    def n_g_eta_values(self) -> int:
        return int(self.g_eta_starts[-1])

    @property
    def n_g_lam_values(self) -> int:
        return int(self.g_lam_starts[-1])

    @property
    def n_g_obs_values(self) -> int:
        return int(self.g_obs_starts[-1])

    @staticmethod
    def _segment_indices(starts: FrozenArray, slots: np.ndarray | int, width: int) -> np.ndarray:
        base = starts[np.asarray(slots, dtype=np.int64)]
        return base[..., None] + np.arange(width, dtype=np.int64)

    def message_indices(self, slots: np.ndarray | int, card: int) -> np.ndarray:
        """Flat value indices for discrete logical ``slots`` of one card."""
        return self._segment_indices(self.message_starts, slots, card)

    def g_eta_indices(self, slots: np.ndarray | int, dim: int) -> np.ndarray:
        """Flat eta indices for Gaussian logical ``slots`` of one dimension."""
        return self._segment_indices(self.g_eta_starts, slots, dim)

    def g_lam_indices(self, slots: np.ndarray | int, dim: int) -> np.ndarray:
        """Flat Lambda indices, shaped ``slots + (dim, dim)``."""
        base = self.g_lam_starts[np.asarray(slots, dtype=np.int64)]
        block = np.arange(dim * dim, dtype=np.int64).reshape(dim, dim)
        return base[..., None, None] + block

    def g_obs_indices(self, rows: np.ndarray | int, dim: int) -> np.ndarray:
        """Flat conditioned-observation indices for equal-dimension rows."""
        return self._segment_indices(self.g_obs_starts, rows, dim)
