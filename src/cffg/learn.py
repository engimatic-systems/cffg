"""Declared learnables: the generic conjugate-update API (cff-x0sv).

Continual learning used to be two hardcoded paths in RollingAgent — `learn_a`
walked the matrix_cat and GFE groups by name, `learn_b` the transition_mixture
groups, and both assumed one flat parameter array bound to the factor names
build_rollout happens to emit. This module replaces that with a declaration.

Three pieces:

- **The declaration** (`Template.learnables`, cffg.core.ir.LearnableSpec):
  a named conjugate posterior over a set of blobs. Blobs are the unit because
  blobs are how the IR ties parameters — build_rollout names one `Bs` on every
  slice's transition, plates name one blob across every copy — so counts from
  every instance sharing a blob accumulate into one posterior. Parameter tying
  is what makes learning transfer across states.

- **The family** (`FAMILIES`): the conjugate update itself, plus the map from
  a node belief to that family's sufficient statistic, keyed by the factor
  type it is attached to. `Dirichlet` covers the discrete vocabulary; the
  Gaussian half of the contract is `cffg.conj` (NIW/MNIW), which reads
  `Engine.gaussian_node_stats` instead of `Engine.node_beliefs` and owns
  several blobs at different shapes — hence `shapes` being a per-blob map.

- **The runtime** (`Learnables`): holds the hyperparameters, accumulates
  statistics from a converged state, and writes parameters back. It has no
  opinion about *when* — the caller decides that a transition posterior
  updates every tick while an initial-state posterior updates once per
  episode, because that is a modeling choice, not a mechanical one.

Exactness (per family, documented because the FE ledger depends on it):

- Dirichlet on a BP group is exact conjugacy. The node belief at a discrete
  factor on a tree is the exact posterior over that factor's variables, and
  its expectation is the exact sufficient statistic of the Dirichlet update.
- Dirichlet on a VMP (mean-field) group is approximate: the node belief is
  the product of edge marginals, so the update integrates against a
  factorized measure. This is the standard mean-field conjugate update, not
  an exact one.
- The rollover writes E[log θ] = ψ(α) − ψ(α₀), the mean-field-consistent
  point (Part II §5), never log E[θ]. GFE composites instead read raw α: the
  parameter-novelty term differentiates the counts directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cffg.compiler import LearnPlan
    from cffg.engine import Engine, State


class Family(Protocol):
    """A conjugate family: what shape the hyperparameters take, how a node
    belief turns into sufficient statistics, and how the posterior rolls over
    into the parameters the message rules consume."""

    name: str

    def check(self, learnable: str, ftype: str, shapes: dict[str, tuple[int, ...]]) -> None:
        """Reject a declaration this family cannot serve, at compile time."""

    def stats(self, ftype: str, belief: np.ndarray) -> np.ndarray:
        """Node belief -> a sufficient-statistic increment in blob layout."""

    def expected(self, alpha: np.ndarray, ftype: str) -> np.ndarray:
        """Hyperparameters -> the expected natural parameter, blob layout."""

    def axis(self, ftype: str) -> int:
        """The axis the conditional normalizes over, in blob layout."""

    def validate_prior(self, learnable: str, alpha: np.ndarray) -> None:
        """Reject hyperparameters outside the family's admissible set."""


def _elog(counts: np.ndarray, axis: int) -> np.ndarray:
    """E[log θ] = ψ(α) − ψ(α₀) over Dirichlet columns laid out along `axis`."""
    from jax.scipy.special import digamma

    counts = np.asarray(counts)
    total = counts.sum(axis=axis, keepdims=True)
    return np.asarray(digamma(counts)) - np.asarray(digamma(total))


@dataclass(frozen=True)
class Dirichlet:
    """Conjugate counts for the discrete factor vocabulary.

    The count axis is the factor's *output* role, because that is the axis a
    discrete conditional normalizes over:

        categorical         blob [K]           columns: the whole vector
        matrix_cat          blob [K_out, K_in] columns: one per input state
        transition_mixture  blob [U, K', K]    columns: one per (action, state)

    The sufficient statistic is the node belief itself, in blob layout. For a
    matrix_cat that is q(x, z) — with x clamped to an observation this is the
    familiar `counts[x] += q(z)`, but the general form also folds partial
    observations, where several outcomes retain mass. For a
    transition_mixture, node_beliefs reports q(z', z, u) as [K', K, U]; the
    blob is [U, K', K], so the statistic transposes. Summing over u rather
    than slicing at the executed action is the same number when the control is
    clamped, and the right number when it is not.
    """

    name: str = "dirichlet"

    _AXIS = {"categorical": 0, "matrix_cat": 0, "transition_mixture": 1}

    def check(self, learnable: str, ftype: str, shapes: dict[str, tuple[int, ...]]) -> None:
        if ftype not in self._AXIS:
            raise ValueError(
                f"learnable {learnable!r}: no Dirichlet sufficient statistic for "
                f"factor type {ftype!r} (have: {sorted(self._AXIS)})"
            )
        distinct = set(shapes.values())
        if len(distinct) > 1:
            raise ValueError(
                f"learnable {learnable!r}: blobs disagree on shape {shapes}; a "
                "Dirichlet posterior has one hyperparameter array"
            )

    def axis(self, ftype: str) -> int:
        return self._AXIS[ftype]

    def stats(self, ftype: str, belief: np.ndarray) -> np.ndarray:
        belief = np.asarray(belief, dtype=float)
        if ftype == "transition_mixture":
            return belief.transpose(2, 0, 1)  # [K', K, U] -> [U, K', K]
        return belief

    def expected(self, alpha: np.ndarray, ftype: str) -> np.ndarray:
        return _elog(alpha, axis=self._AXIS[ftype])

    def validate_prior(self, learnable: str, alpha: np.ndarray) -> None:
        if not np.all(alpha > 0):
            raise ValueError(f"learnable {learnable!r}: Dirichlet counts must be > 0")


FAMILIES: dict[str, Family] = {"dirichlet": Dirichlet()}


class Learnables:
    """Hyperparameters for every learnable an engine declares, plus the two
    operations that move them: `accumulate` folds a converged state's
    sufficient statistics in, `write` pushes the resulting parameters into a
    params dict for the next run.

    Hyperparameters live here, not in an engine, because one posterior serves
    several compiled graphs: the receding-horizon agent learns from a
    perception graph and spends the result in a planning graph, and the two
    address the same parameter through different groups. `accumulate` and
    `write` therefore take the engine to act on, and resolve targets against
    *that* graph's plans.

    Prior hyperparameters come from `priors`, keyed by learnable name. A
    learnable whose graph already stores raw counts (a GFE composite's
    Dirichlet parameter) can recover its prior from the compiled arena; one
    stored only as expected parameters cannot — the log of a probability
    table is not a hyperparameter — and must be given a prior.
    """

    def __init__(
        self,
        engine: "Engine",
        priors: dict[str, np.ndarray] | None = None,
    ):
        self.plans = {p.name: p for p in engine.arena.learn_plans}
        self.params: dict[str, np.ndarray] = {}
        priors = priors or {}
        for name, plan in self.plans.items():
            alpha = (
                np.asarray(priors[name], dtype=float).copy()
                if name in priors
                else self._from_compiled(engine, plan)
            )
            shape = next(iter(plan.shapes.values()))
            if alpha.shape != shape:
                raise ValueError(
                    f"learnable {name!r}: prior shape {alpha.shape} != blob shape {shape}"
                )
            FAMILIES[plan.family].validate_prior(name, alpha)
            self.params[name] = alpha
        for name in priors:
            if name not in self.plans:
                raise KeyError(f"prior for {name!r}, which this graph does not declare")

    @staticmethod
    def _from_compiled(engine: "Engine", plan: "LearnPlan") -> np.ndarray:
        for tgt in plan.targets:
            if tgt.form == "counts":
                return np.asarray(
                    engine.arena.log_params[tgt.key][tgt.index], dtype=float
                ).copy()
        raise ValueError(
            f"learnable {plan.name!r}: this graph stores only expected parameters, "
            "so its prior counts cannot be recovered — pass priors={...}"
        )

    # -- the two operations ------------------------------------------------

    def accumulate(
        self, engine: "Engine", state: "State", only: tuple[str, ...] | None = None
    ) -> None:
        """Fold the sufficient statistics of every source instance into its
        posterior. `only` restricts to named learnables — an initial-state
        prior updates once per episode while a transition updates every tick,
        and that cadence belongs to the caller.

        The state must be converged: the node belief of an unconverged graph
        is not a posterior, and folding it in is not a conjugate update.
        """
        beliefs = None
        for plan in engine.arena.learn_plans:
            name = plan.name
            if (only is not None and name not in only) or not plan.sources:
                continue
            if name not in self.params:
                raise KeyError(f"engine declares {name!r}, which this Learnables lacks")
            family = FAMILIES[plan.family]
            if beliefs is None:
                beliefs = engine.node_beliefs(state)
            for src in plan.sources:
                self.params[name] = self.params[name] + family.stats(
                    plan.ftype, beliefs[src]
                )

    def write(self, engine: "Engine", params: dict) -> dict:
        """Return `params` with every learnable's targets in this engine
        refreshed. Expected targets get E[log θ] in the consumer's layout;
        counts targets get the raw hyperparameters."""
        import jax.numpy as jnp

        out = dict(params)
        for plan in engine.arena.learn_plans:
            name = plan.name
            if name not in self.params:
                raise KeyError(f"engine declares {name!r}, which this Learnables lacks")
            family = FAMILIES[plan.family]
            alpha = self.params[name]
            expected = family.expected(alpha, plan.ftype)
            for tgt in plan.targets:
                value = alpha if tgt.form == "counts" else expected
                if tgt.layout == "tm":
                    value = value.transpose(1, 2, 0)  # [U, K', K] -> [K', K, U]
                out[tgt.key] = out[tgt.key].at[tgt.index].set(jnp.asarray(value))
        return out

    # -- readouts and persistence -----------------------------------------

    def expected(self, name: str) -> np.ndarray:
        """The posterior mean parameter α/α₀ — for reporting and recovery
        tests. Not what the rules consume (see the module docstring)."""
        plan = self.plans[name]
        alpha = self.params[name]
        axis = FAMILIES[plan.family].axis(plan.ftype)
        return alpha / alpha.sum(axis=axis, keepdims=True)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: alpha.copy() for name, alpha in self.params.items()}

    def load_state_dict(self, data: dict[str, np.ndarray]) -> None:
        for name, alpha in data.items():
            if name not in self.params:
                raise KeyError(f"no learnable {name!r} in this graph")
            if np.shape(alpha) != self.params[name].shape:
                raise ValueError(
                    f"learnable {name!r}: checkpoint shape {np.shape(alpha)} != "
                    f"{self.params[name].shape}"
                )
            self.params[name] = np.asarray(alpha, dtype=float).copy()
