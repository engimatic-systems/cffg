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
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol
from urllib.parse import quote, unquote

import numpy as np

from cffg import ef, efrules
from cffg.core.learning_identity import LEARNING_IDENTITIES
from cffg.learning_schema import (
    PERSISTENT_MANIFEST_KEY,
    PersistentParameter,
    encode_persistent_manifest,
    resolve_persistent_state,
    restore_persistent_array,
    validate_persistent_manifest,
)
from cffg.learning_checkpoint import (
    CHECKPOINT_MANIFEST_KEY,
    checkpoint_context,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cffg.compiled import LearnPlan
    from cffg.engine import Engine, RunResult


#: Engine readouts a family can draw sufficient statistics from. A family
#: names one; `Learnables.accumulate` calls it once per state and hands each
#: source's entry to `Family.stats`. Discrete conjugacy integrates against
#: node beliefs; the Gaussian families integrate against the expected
#: sufficient statistics of the same nodes, which is a different readout of
#: the same object, not a different object.
READOUTS = {
    "node_beliefs": lambda engine, state: engine.node_beliefs(state),
    "gaussian_node_stats": lambda engine, state: engine.gaussian_node_stats(state),
    "recurrence_weight_stats": lambda engine, state: engine.recurrence_weight_stats(
        state
    ),
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
    semantic_identity: str
    parameter_forms: tuple[str, ...]
    #: key into READOUTS: which engine readout supplies this family's stats
    readout: str

    def persistent_parameters(
        self, role_shapes: dict[str, tuple[int, ...]]
    ) -> tuple[PersistentParameter, ...]:
        """Name every persistent array and derive its shape from checked target roles."""

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

    def pool(
        self, records: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Pool factor-local statistic records for one posterior update."""

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
    return np.asarray(
        efrules.DIRICHLET.mean_stats_from_concentration(
            np.asarray(counts), axis=axis
        )
    )


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

    semantic_identity = LEARNING_IDENTITIES["dirichlet"]
    parameter_forms = ("expected", "counts")

    def persistent_parameters(self, role_shapes):
        return (PersistentParameter("alpha", next(iter(role_shapes.values()))),)

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
        return {
            "alpha": ef.DIRICHLET.posterior(params["alpha"], stats["n"])
        }

    def pool(self, records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {"n": np.sum(np.stack([record["n"] for record in records]), axis=0)}

    def rollover(self, params: dict[str, np.ndarray], ftype: str) -> dict[str, np.ndarray]:
        return {self._ROLE[ftype]: _elog(params["alpha"], axis=self._AXIS[ftype])}

    def raw(self, params: dict[str, np.ndarray], ftype: str) -> dict[str, np.ndarray]:
        return {self._ROLE[ftype]: params["alpha"]}

    def mean(self, params: dict[str, np.ndarray], ftype: str) -> dict[str, np.ndarray]:
        return {
            self._ROLE[ftype]: ef.DIRICHLET.mean_from_concentration(
                params["alpha"], axis=self._AXIS[ftype]
            )
        }

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
        if not bool(
            ef.DIRICHLET.is_proper_concentration(
                alpha, axis=tuple(range(alpha.ndim))
            )
        ):
            raise ValueError(f"learnable {learnable!r}: Dirichlet counts must be > 0")

    # A bare array is the natural spelling for a one-parameter family, so the
    # runtime accepts it and normalizes here rather than making every caller
    # write {"alpha": ...}.
    def as_params(self, prior: object) -> dict[str, np.ndarray]:
        if isinstance(prior, dict):
            return {k: np.asarray(v, dtype=float).copy() for k, v in prior.items()}
        return {"alpha": np.asarray(prior, dtype=float).copy()}


from cffg.gaussian_learning import MNIWFamily, NIWFamily
from cffg.hybrid_learning import ComponentMNIWFamily, PGWeightsFamily


FAMILIES: dict[str, Family] = {
    "dirichlet": Dirichlet(),
    "niw": NIWFamily(),
    "mniw": MNIWFamily(),
    "component_mniw": ComponentMNIWFamily(),
    "pg_weights": PGWeightsFamily(),
}

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
        *,
        only: tuple[str, ...] | None = None,
    ):
        engine_plans = {p.name: p for p in engine.arena.learn_plans}
        if only is None:
            names = tuple(engine_plans)
        else:
            unknown = sorted(set(only) - set(engine_plans))
            if unknown:
                raise KeyError(
                    f"engine does not declare requested learnables {unknown}"
                )
            names = tuple(dict.fromkeys(only))
        self.plans = {name: engine_plans[name] for name in names}
        self._restricted = only is not None
        self.params: dict[str, dict[str, np.ndarray]] = {}
        initial_params: dict[str, MappingProxyType] = {}
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
            initial = {}
            for key, value in params.items():
                frozen = value.copy()
                frozen.setflags(write=False)
                initial[key] = frozen
            initial_params[name] = MappingProxyType(initial)
            self.params[name] = {
                key: value.copy() for key, value in initial.items()
            }
        self.initial_params = MappingProxyType(initial_params)

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
        self,
        engine: "Engine",
        result: "RunResult",
        only: tuple[str, ...] | None = None,
    ) -> None:
        """Add one converged result's pooled statistics to the posterior."""
        self._update(engine, result, only, replace=False)

    def replace(
        self,
        engine: "Engine",
        result: "RunResult",
        only: tuple[str, ...] | None = None,
    ) -> None:
        """Recompute selected posteriors from immutable initial priors."""
        self._update(engine, result, only, replace=True)

    def _update(
        self,
        engine: "Engine",
        result: "RunResult",
        only: tuple[str, ...] | None,
        *,
        replace: bool,
    ) -> None:
        from cffg.engine import RunResult

        if not isinstance(result, RunResult):
            raise TypeError(
                "learning updates require the status-bearing RunResult returned "
                "by Engine.run"
            )
        state = result.require_converged()
        plans = self._selected_plans(engine, only)
        readouts: dict[str, dict] = {}
        pending: dict[str, dict[str, np.ndarray]] = {}
        for plan in plans:
            name = plan.name
            if name not in self.params:
                raise KeyError(f"engine declares {name!r}, which this Learnables lacks")
            family = FAMILIES[plan.family]
            base = self.initial_params[name] if replace else self.params[name]
            candidate = {key: value.copy() for key, value in base.items()}
            if plan.sources:
                if family.readout not in READOUTS:
                    raise KeyError(
                        f"learnable {name!r}: unknown readout {family.readout!r}"
                    )
                if family.readout not in readouts:
                    readouts[family.readout] = READOUTS[family.readout](engine, state)
                per_node = readouts[family.readout]
                records = []
                for source in plan.sources:
                    try:
                        records.append(
                            family.stats(plan.ftype, per_node[source])
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"learnable {name!r}, source {source!r}: {exc}"
                        ) from exc
                candidate = family.fold(candidate, family.pool(records))
            family.validate_prior(name, candidate, plan.shapes)
            pending[name] = {
                key: np.asarray(value).copy() for key, value in candidate.items()
            }
        self.params.update(pending)

    def _selected_plans(
        self, engine: "Engine", only: tuple[str, ...] | None
    ) -> list["LearnPlan"]:
        plans = {plan.name: plan for plan in engine.arena.learn_plans}
        if only is None:
            names = (
                tuple(name for name in plans if name in self.plans)
                if self._restricted
                else tuple(plans)
            )
        else:
            unknown = sorted(set(only) - set(plans))
            if unknown:
                raise KeyError(
                    f"engine does not declare requested learnables {unknown}"
                )
            names = tuple(dict.fromkeys(only))
        selected = [plans[name] for name in names]
        for plan in selected:
            original = self.plans.get(plan.name)
            if original is None:
                raise KeyError(
                    f"engine declares {plan.name!r}, which this Learnables lacks"
                )
            if (plan.family, plan.ftype, plan.semantic_identity) != (
                original.family, original.ftype, original.semantic_identity
            ):
                raise ValueError(
                    f"learnable {plan.name!r} has an incompatible plan in this engine"
                )
            # Different windows may read different blobs owned by the same
            # posterior, such as analytic A in perception and raw Ac counts
            # in functional planning. Parameter roles and shapes are the
            # cross-window contract; blob identity stays graph-local.
            role_shapes = {
                (target.role, plan.shapes[target.blob]) for target in plan.targets
            }
            original_role_shapes = {
                (target.role, original.shapes[target.blob])
                for target in original.targets
            }
            if role_shapes != original_role_shapes:
                raise ValueError(
                    f"learnable {plan.name!r} has an incompatible plan in this engine"
                )
        return selected

    def write(
        self,
        engine: "Engine",
        params: dict,
        only: tuple[str, ...] | None = None,
    ) -> dict:
        """Return `params` with every learnable's targets in this engine
        refreshed. Expected targets get the rolled-over parameters the rules
        consume; counts targets get the raw hyperparameters. A target names
        the parameter role its blob fills, so a family with several
        parameters writes each one where it belongs."""
        from cffg.compiled import NEG
        from cffg.params import _RuntimeWrite, _apply_runtime_writes

        writes: list[_RuntimeWrite] = []
        for plan in self._selected_plans(engine, only):
            name = plan.name
            if name not in self.params:
                raise KeyError(f"engine declares {name!r}, which this Learnables lacks")
            family = FAMILIES[plan.family]
            family.validate_prior(name, self.params[name], plan.shapes)
            by_form: dict[str, dict[str, np.ndarray]] = {}
            for tgt in plan.targets:
                if tgt.form not in by_form:
                    by_form[tgt.form] = (
                        family.rollover(self.params[name], plan.ftype)
                        if tgt.form == "expected"
                        else family.raw(self.params[name], plan.ftype)
                    )
                value = by_form[tgt.form][tgt.role]
                if tgt.layout == "tm":
                    value = value.transpose(1, 2, 0)  # [U, K', K] -> [K', K, U]
                if tgt.transform == "covariance_to_precision":
                    value = np.linalg.inv(np.asarray(value))
                value = np.asarray(value, dtype=params[tgt.key].dtype)
                if plan.family == "dirichlet" and tgt.form == "expected":
                    # NEG's bit pattern is reserved for exact log-zero. A
                    # proper positive Dirichlet can produce any finite
                    # E[log theta], including (after a dtype cast) exactly
                    # that value. Move only this representational collision
                    # one dtype ULP toward zero; no magnitude floor or
                    # probability support is invented.
                    neg = np.asarray(NEG, dtype=value.dtype)
                    value = np.where(
                        value == neg,
                        np.nextafter(neg, np.asarray(0.0, dtype=value.dtype)),
                        value,
                    )
                if not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"learnable {name!r}: rollover role {tgt.role!r} is non-finite"
                    )
                writes.append(_RuntimeWrite(tgt.key, tgt.index, value))
        return _apply_runtime_writes(params, writes)

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

    def rollover(self, name: str, role: str | None = None) -> np.ndarray:
        """The point parameter consumed by the next inference run.

        This differs from :meth:`expected` when a conjugate family reports a
        posterior mean but inference consumes an expected natural parameter.
        """
        plan = self.plans[name]
        out = FAMILIES[plan.family].rollover(self.params[name], plan.ftype)
        if role is not None:
            return out[role]
        if len(out) != 1:
            raise ValueError(
                f"learnable {name!r} has roles {sorted(out)}; name one"
            )
        return next(iter(out.values()))

    def hyperparameter_snapshot(self) -> dict[str, np.ndarray]:
        """Encoded numeric values for diagnostics, not checkpoint restore."""
        return {
            _encode_state_key(name, key): arr.copy()
            for name, params in self.params.items()
            for key, arr in params.items()
        }

    def state_dict(self) -> dict[str, np.ndarray]:
        """Posterior arrays plus versioned semantics and persistent-state inventory."""
        data = self.hyperparameter_snapshot()
        if self.plans:
            data[PERSISTENT_MANIFEST_KEY] = encode_persistent_manifest(
                self.persistent_descriptors()
            )
        return data

    def persistent_descriptors(self):
        """Resolve complete immutable-prior/current-posterior array ownership."""
        return tuple(
            resolve_persistent_state(
                plan, FAMILIES[plan.family], self.initial_params[name], self.params[name]
            )
            for name, plan in sorted(self.plans.items())
        )

    def load_state_dict(
        self,
        data: dict[str, np.ndarray],
        *,
        exact: bool = False,
    ) -> None:
        """Restore complete posteriors for supplied owners; ``exact`` requires all owners."""
        descriptors = self.persistent_descriptors()
        manifest = data.get(PERSISTENT_MANIFEST_KEY)
        payload = (
            validate_persistent_manifest(manifest, descriptors, exact=exact)
            if manifest is not None else None
        )
        recurrence = checkpoint_context(data, self.plans, self.params)
        pending = {
            name: {key: value.copy() for key, value in params.items()}
            for name, params in self.params.items()
        }
        changed: set[str] = set()
        legacy_names: set[str] = set()
        seen: set[tuple[str, str]] = set()
        for flat, arr in data.items():
            if flat in (CHECKPOINT_MANIFEST_KEY, PERSISTENT_MANIFEST_KEY):
                continue
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
                legacy_names.add(name)
            if name not in self.params:
                raise KeyError(f"no learnable {name!r} in this graph")
            if key not in self.params[name]:
                raise KeyError(f"learnable {name!r} has no hyperparameter {key!r}")
            if payload is not None and (name, key) not in payload:
                raise KeyError(
                    f"learnable {name!r}: checkpoint hyperparameter {key!r} "
                    "has no persistent descriptor"
                )
            if (name, key) in seen:
                raise KeyError(
                    f"checkpoint names learnable {name!r} hyperparameter {key!r} twice"
                )
            seen.add((name, key))
            if payload is None:
                recurrence.require_identity(name)
            if np.iscomplexobj(arr):
                raise ValueError(
                    f"learnable {name!r}: checkpoint hyperparameter {key!r} "
                    "must be real"
                )
            if np.shape(arr) != self.params[name][key].shape:
                raise ValueError(
                    f"learnable {name!r}: checkpoint shape {np.shape(arr)} != "
                    f"{self.params[name][key].shape}"
                )
            pending[name][key] = (
                restore_persistent_array(name, key, arr, payload)
                if payload is not None else recurrence.restore_value(name, key, arr)
            )
            changed.add(name)
        recurrence.require_complete(seen)
        if payload is not None and seen != set(payload):
            raise KeyError(f"checkpoint is missing learnable hyperparameters {sorted(set(payload) - seen)}")
        for name in changed if payload is None else legacy_names:
            present = {key for candidate, key in seen if candidate == name}
            required = set(self.params[name])
            if len(required) > 1 and present != required:
                missing = sorted(required - present)
                raise KeyError(
                    f"legacy checkpoint incompletely names learnable {name!r}; "
                    f"missing hyperparameters {missing}. Rewrite it with a v2 "
                    "checkpoint"
                )
        if exact:
            expected = {
                (name, key)
                for name, params in self.params.items()
                for key in params
            }
            missing = sorted(expected - seen)
            if missing:
                labels = [f"{name}/{key}" for name, key in missing]
                raise KeyError(
                    f"checkpoint is missing learnable hyperparameters {labels}"
                )
        for name in changed:
            plan = self.plans[name]
            FAMILIES[plan.family].validate_prior(name, pending[name], plan.shapes)
        self.params.update({name: pending[name] for name in changed})
