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
  an engine readout to that family's sufficient statistic, keyed by the
  factor type it is attached to. `Dirichlet` covers the discrete vocabulary.
  Hyperparameters, statistics and rollover outputs are all keyed dicts rather
  than bare arrays, and a family names which engine readout it draws from,
  because a conjugate posterior is not always one array read off one
  readout: an MNIW over a `gaussian_linear` carries four hyperparameters,
  accumulates three statistics from `Engine.gaussian_node_stats`, and rolls
  over into three blobs at different shapes. Dirichlet is the degenerate case
  of that contract, not the shape of it.

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
from urllib.parse import quote, unquote

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cffg.compiler import LearnPlan
    from cffg.engine import Engine, State


#: Engine readouts a family can draw sufficient statistics from. A family
#: names one; `Learnables.accumulate` calls it once per state and hands each
#: source's entry to `Family.stats`. Discrete conjugacy integrates against
#: node beliefs; the Gaussian families integrate against the expected
#: sufficient statistics of the same nodes, which is a different readout of
#: the same object, not a different object.
READOUTS = {
    "node_beliefs": lambda engine, state: engine.node_beliefs(state),
    "gaussian_node_stats": lambda engine, state: engine.gaussian_node_stats(state),
}


class Family(Protocol):
    """A conjugate family: where its statistics come from, what its
    hyperparameters look like, and how the posterior rolls over into the
    parameters the message rules consume.

    Hyperparameters and statistics are both keyed dicts, not bare arrays,
    because a conjugate posterior is not always one array. An MNIW over a
    `gaussian_linear` carries `m`, `lam`, `psi` and `nu`, accumulates `Syy`,
    `Syx` and `Sxx`, and rolls over into three separate blobs — matrix,
    offset and covariance. `rollover` therefore returns values keyed by the
    parameter ROLE the target blob fills, which is how a multi-blob write
    routes. Dirichlet uses a single role and is the degenerate case, not the
    shape of the contract.
    """

    name: str
    #: key into READOUTS: which engine readout supplies this family's stats
    readout: str

    def check(
        self, learnable: str, ftype: str, shapes: dict[str, tuple[int, ...]], roles: dict[str, str]
    ) -> None:
        """Reject a declaration this family cannot serve, at compile time.
        `roles` maps blob name -> the parameter role it fills."""

    def stats(self, ftype: str, readout: object) -> dict[str, np.ndarray]:
        """One node's readout entry -> a sufficient-statistic increment."""

    def fold(
        self, params: dict[str, np.ndarray], stats: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Hyperparameters + statistics -> updated hyperparameters."""

    def rollover(self, params: dict[str, np.ndarray], ftype: str) -> dict[str, np.ndarray]:
        """Hyperparameters -> the values each parameter ROLE should hold,
        as the message rules consume them (expected natural parameters)."""

    def raw(self, params: dict[str, np.ndarray], ftype: str) -> dict[str, np.ndarray]:
        """Hyperparameters -> the values a `counts`-form target wants, keyed
        by role. GFE composites differentiate the hyperparameters directly
        rather than an expectation of them."""

    def mean(self, params: dict[str, np.ndarray], ftype: str) -> dict[str, np.ndarray]:
        """Hyperparameters -> the posterior MEAN parameter, keyed by role.
        For reporting and recovery tests, not for rollover."""

    def as_params(self, prior: object) -> dict[str, np.ndarray]:
        """Normalize a caller-supplied prior into keyed hyperparameters. A
        one-parameter family may accept a bare array here."""

    def validate_prior(
        self, learnable: str, params: dict[str, np.ndarray], shapes: dict[str, tuple[int, ...]]
    ) -> None:
        """Reject hyperparameters outside the family's admissible set, or
        inconsistent with the blob shapes they have to write."""


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

    Dirichlet is the degenerate case of the keyed contract: one
    hyperparameter array `alpha`, one statistic `n`, and one output role —
    whichever single parameter role the factor type takes.
    """

    name: str = "dirichlet"
    readout: str = "node_beliefs"

    _AXIS = {"categorical": 0, "matrix_cat": 0, "transition_mixture": 1}
    _ROLE = {
        "categorical": "statistic",
        "matrix_cat": "matrix",
        "transition_mixture": "matrices",
    }

    def check(
        self,
        learnable: str,
        ftype: str,
        shapes: dict[str, tuple[int, ...]],
        roles: dict[str, str],
    ) -> None:
        if ftype not in self._AXIS:
            raise ValueError(
                f"learnable {learnable!r}: no Dirichlet sufficient statistic for "
                f"factor type {ftype!r} (have: {sorted(self._AXIS)})"
            )
        if len(set(shapes.values())) > 1:
            raise ValueError(
                f"learnable {learnable!r}: blobs disagree on shape {shapes}; a "
                "Dirichlet posterior has one hyperparameter array"
            )
        wrong = {b: r for b, r in roles.items() if r != self._ROLE[ftype]}
        if wrong:
            raise ValueError(
                f"learnable {learnable!r}: a Dirichlet over {ftype!r} governs its "
                f"{self._ROLE[ftype]!r} role only, but {wrong} fills another"
            )

    def axis(self, ftype: str) -> int:
        return self._AXIS[ftype]

    def stats(self, ftype: str, readout: np.ndarray) -> dict[str, np.ndarray]:
        belief = np.asarray(readout, dtype=float)
        if ftype == "transition_mixture":
            belief = belief.transpose(2, 0, 1)  # [K', K, U] -> [U, K', K]
        return {"n": belief}

    def fold(
        self, params: dict[str, np.ndarray], stats: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {"alpha": params["alpha"] + stats["n"]}

    def rollover(self, params: dict[str, np.ndarray], ftype: str) -> dict[str, np.ndarray]:
        return {self._ROLE[ftype]: _elog(params["alpha"], axis=self._AXIS[ftype])}

    def raw(self, params: dict[str, np.ndarray], ftype: str) -> dict[str, np.ndarray]:
        return {self._ROLE[ftype]: params["alpha"]}

    def mean(self, params: dict[str, np.ndarray], ftype: str) -> dict[str, np.ndarray]:
        alpha = params["alpha"]
        return {self._ROLE[ftype]: alpha / alpha.sum(axis=self._AXIS[ftype], keepdims=True)}

    def validate_prior(
        self,
        learnable: str,
        params: dict[str, np.ndarray],
        shapes: dict[str, tuple[int, ...]],
    ) -> None:
        if set(params) != {"alpha"}:
            raise ValueError(
                f"learnable {learnable!r}: a Dirichlet posterior is {{'alpha'}}, "
                f"got {sorted(params)}"
            )
        alpha = params["alpha"]
        want = next(iter(shapes.values()), alpha.shape)
        if alpha.shape != want:
            raise ValueError(
                f"learnable {learnable!r}: prior shape {alpha.shape} != blob shape {want}"
            )
        if not np.all(alpha > 0):
            raise ValueError(f"learnable {learnable!r}: Dirichlet counts must be > 0")

    # A bare array is the natural spelling for a one-parameter family, so the
    # runtime accepts it and normalizes here rather than making every caller
    # write {"alpha": ...}.
    def as_params(self, prior: object) -> dict[str, np.ndarray]:
        if isinstance(prior, dict):
            return {k: np.asarray(v, dtype=float).copy() for k, v in prior.items()}
        return {"alpha": np.asarray(prior, dtype=float).copy()}


FAMILIES: dict[str, Family] = {"dirichlet": Dirichlet()}

_STATE_KEY_PREFIX = "v2:"


def _encode_state_key(name: str, key: str) -> str:
    """Encode a learnable/hyperparameter pair as one unambiguous npz key.

    Both components are percent-encoded before the colon separator is added,
    so names produced by nested composition (``outer/inner/B``) and arbitrary
    family-owned hyperparameter keys cannot collide.  The version prefix lets
    the loader retain the two historical spellings without guessing how a new
    checkpoint was written.
    """
    return (
        f"{_STATE_KEY_PREFIX}{quote(name, safe='')}:{quote(key, safe='')}"
    )


def _decode_state_key(flat: str) -> tuple[str, str]:
    payload = flat.removeprefix(_STATE_KEY_PREFIX)
    encoded_name, sep, encoded_key = payload.partition(":")
    if (
        not sep
        or quote(unquote(encoded_name), safe="") != encoded_name
        or quote(unquote(encoded_key), safe="") != encoded_key
    ):
        raise KeyError(f"invalid versioned learnable checkpoint key {flat!r}")
    return unquote(encoded_name), unquote(encoded_key)


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

    Prior hyperparameters come from `priors`, keyed by learnable name. Each
    is a dict keyed the way its family keys hyperparameters; a one-parameter
    family also accepts the bare array. A learnable whose graph already
    stores raw counts (a GFE composite's Dirichlet parameter) can recover its
    prior from the compiled arena; one stored only as expected parameters
    cannot — the log of a probability table is not a hyperparameter — and
    must be given a prior.
    """

    def __init__(
        self,
        engine: "Engine",
        priors: dict[str, object] | None = None,
    ):
        self.plans = {p.name: p for p in engine.arena.learn_plans}
        self.params: dict[str, dict[str, np.ndarray]] = {}
        priors = priors or {}
        for name in priors:
            if name not in self.plans:
                raise KeyError(f"prior for {name!r}, which this graph does not declare")
        for name, plan in self.plans.items():
            family = FAMILIES[plan.family]
            given = (
                priors[name] if name in priors else self._from_compiled(engine, plan)
            )
            params = family.as_params(given)
            family.validate_prior(name, params, plan.shapes)
            self.params[name] = params

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
        readouts: dict[str, dict] = {}
        for plan in engine.arena.learn_plans:
            name = plan.name
            if (only is not None and name not in only) or not plan.sources:
                continue
            if name not in self.params:
                raise KeyError(f"engine declares {name!r}, which this Learnables lacks")
            family = FAMILIES[plan.family]
            if family.readout not in readouts:
                readouts[family.readout] = READOUTS[family.readout](engine, state)
            per_node = readouts[family.readout]
            for src in plan.sources:
                self.params[name] = family.fold(
                    self.params[name], family.stats(plan.ftype, per_node[src])
                )

    def write(self, engine: "Engine", params: dict) -> dict:
        """Return `params` with every learnable's targets in this engine
        refreshed. Expected targets get the rolled-over parameters the rules
        consume; counts targets get the raw hyperparameters. A target names
        the parameter role its blob fills, so a family with several
        parameters writes each one where it belongs."""
        import jax.numpy as jnp

        from cffg.compiler import NEG

        out = dict(params)
        for plan in engine.arena.learn_plans:
            name = plan.name
            if name not in self.params:
                raise KeyError(f"engine declares {name!r}, which this Learnables lacks")
            family = FAMILIES[plan.family]
            by_form = {
                "expected": family.rollover(self.params[name], plan.ftype),
                "counts": family.raw(self.params[name], plan.ftype),
            }
            for tgt in plan.targets:
                value = by_form[tgt.form][tgt.role]
                if tgt.layout == "tm":
                    value = value.transpose(1, 2, 0)  # [U, K', K] -> [K', K, U]
                value = jnp.asarray(value, dtype=out[tgt.key].dtype)
                if plan.family == "dirichlet" and tgt.form == "expected":
                    # NEG's bit pattern is reserved for exact log-zero. A
                    # proper positive Dirichlet can produce any finite
                    # E[log theta], including (after a dtype cast) exactly
                    # that value. Move only this representational collision
                    # one dtype ULP toward zero; no magnitude floor or
                    # probability support is invented.
                    neg = jnp.asarray(NEG, dtype=value.dtype)
                    value = jnp.where(
                        value == neg,
                        jnp.nextafter(neg, jnp.asarray(0.0, dtype=value.dtype)),
                        value,
                    )
                out[tgt.key] = out[tgt.key].at[tgt.index].set(value)
        return out

    # -- readouts and persistence -----------------------------------------

    def expected(self, name: str, role: str | None = None) -> np.ndarray:
        """The posterior MEAN parameter — for reporting and recovery tests,
        not what the rules consume (see the module docstring). `role` picks
        one of a multi-parameter family's outputs; a one-role family needs no
        argument."""
        plan = self.plans[name]
        out = FAMILIES[plan.family].mean(self.params[name], plan.ftype)
        if role is not None:
            return out[role]
        if len(out) != 1:
            raise ValueError(
                f"learnable {name!r} has roles {sorted(out)}; name one"
            )
        return next(iter(out.values()))

    def state_dict(self) -> dict[str, np.ndarray]:
        """Versioned encoded ``(name, key)`` -> array for flat npz storage.

        A delimiter alone is insufficient: scoped learnable names contain
        ``/``, and one valid learnable name can equal another learnable's
        apparent ``name/key`` spelling. Encoding the two components
        independently makes the boundary structural rather than inferred.
        """
        return {
            _encode_state_key(name, key): arr.copy()
            for name, params in self.params.items()
            for key, arr in params.items()
        }

    def load_state_dict(self, data: dict[str, np.ndarray]) -> None:
        for flat, arr in data.items():
            if flat.startswith(_STATE_KEY_PREFIX):
                name, key = _decode_state_key(flat)
            else:
                # Backward compatibility for both pre-v2 spellings:
                #   "A"       — the original one-array family checkpoint
                #   "scope/A/alpha" — the briefly shipped slash spelling.
                # A pair of valid interpretations is refused rather than
                # silently restoring one posterior into another.
                candidates: list[tuple[str, str]] = []
                if flat in self.params and len(self.params[flat]) == 1:
                    candidates.append((flat, next(iter(self.params[flat]))))
                legacy_name, sep, legacy_key = flat.rpartition("/")
                if (
                    sep
                    and legacy_name in self.params
                    and legacy_key in self.params[legacy_name]
                ):
                    candidates.append((legacy_name, legacy_key))
                if len(candidates) > 1:
                    raise KeyError(
                        f"ambiguous legacy learnable checkpoint key {flat!r}; "
                        "rewrite it with a v2 checkpoint"
                    )
                if not candidates:
                    raise KeyError(
                        f"no learnable for checkpoint key {flat!r} in this graph"
                    )
                name, key = candidates[0]
            if name not in self.params:
                raise KeyError(f"no learnable {name!r} in this graph")
            if key not in self.params[name]:
                raise KeyError(f"learnable {name!r} has no hyperparameter {key!r}")
            if np.shape(arr) != self.params[name][key].shape:
                raise ValueError(
                    f"learnable {name!r}: checkpoint shape {np.shape(arr)} != "
                    f"{self.params[name][key].shape}"
                )
            self.params[name][key] = np.asarray(arr, dtype=float).copy()
