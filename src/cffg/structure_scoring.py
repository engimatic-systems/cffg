"""Fit explicit candidates on one partition and compare held-out ordinary Bethe energies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import operator

import numpy as np

from cffg.colony import Colony
from cffg.compiler import compile_template
from cffg.core import ConstraintKind, PortRole, Template, expand_plates
from cffg.core.serialize import structure_hash
from cffg.engine import Engine
from cffg.learn import Learnables
from cffg.neural_artifacts import array_digest
from cffg.runtime_types import RunResult, RunStatus, State, _replace


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    template: Template
    arrays: dict[str, np.ndarray]
    priors: dict[str, object] | None = None


@dataclass(frozen=True)
class Partition:
    """Stable IDs of original artifacts, each with ordered independent graph bindings."""

    partition_id: str
    artifacts: Mapping[str, Sequence[dict[str, object]]]

    def events(self):
        for artifact, events in self.artifacts.items():
            for index, observations in enumerate(events):
                yield artifact, index, observations


@dataclass(frozen=True)
class EventScore:
    artifact: str
    index: int
    status: RunStatus
    inner_status: RunStatus
    energies: tuple[float, ...]
    iterations: int

    @property
    def usable(self):
        return (self.status is RunStatus.CONVERGED and self.inner_status is RunStatus.CONVERGED
                and bool(self.energies) and math.isfinite(self.energies[-1]))


@dataclass
class CandidateScore:
    candidate_id: str
    fit_partition: str
    evaluation_partition: str
    fit_artifacts: tuple[str, ...]
    evaluation_artifacts: tuple[str, ...]
    template_hash: str
    fitted_params_sha256: str | None = None
    schedule: str | None = None
    dtype: str | None = None
    approximation: str | None = None
    data_domains: dict[str, tuple[str, int]] = field(default_factory=dict)
    fit_events: list[EventScore] = field(default_factory=list)
    evaluation_events: list[EventScore] = field(default_factory=list)
    status: str = "pending"
    error: str | None = None
    total_bfe: float | None = None
    rank: int | None = None


def _cyclic(template):
    """A repeat within a connected component closes a cycle, including parallel edges."""
    parents = {}

    def root(key):
        while key in parents:
            parent = parents[key]
            parents[key] = parents.get(parent, parent)
            key = parents[key]
        return key

    for name, factor in template.factors.items():
        factor_root = root(("factor", name))
        for edge in factor.all_edges().values():
            edge_root = root(("edge", edge))
            if edge_root == factor_root:
                return True
            parents[edge_root] = factor_root
    return False


def _record(artifact, index, result: RunResult):
    return EventScore(artifact, index, result.status, result.inner_status,
                      tuple(result.free_energies), result.iterations)


def _unusable(event):
    if event.status is not RunStatus.CONVERGED:
        return event.status.value
    if event.inner_status is not RunStatus.CONVERGED:
        return event.inner_status.value
    return "undefined_energy"


def _fit(engine, base, priors, partition, row, n_iters, tol):
    learner = Learnables(engine, priors)
    if learner.plans:
        for artifact, index, observations in partition.events():
            params = learner.write(engine, base.params)
            observed = engine.bind_data(_replace(base, params=params), observations)
            result = engine.run(observed, n_iters, tol=tol, harden_last=n_iters)
            event = _record(artifact, index, result)
            row.fit_events.append(event)
            if not event.usable:
                row.status = _unusable(event)
                return None
            learner.accumulate(engine, result)
        base = _replace(base, params=learner.write(engine, base.params))
    row.fitted_params_sha256 = array_digest(base.params)
    return base


def _validate_round(candidates, fit, heldout, n_iters, tol):
    ids = [c.candidate_id for c in candidates]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(name, str) or not name for name in ids):
        raise ValueError("supply candidates with distinct nonempty IDs")
    if not fit.partition_id or not heldout.partition_id or fit.partition_id == heldout.partition_id:
        raise ValueError("fit and evaluation require distinct nonempty partition IDs")
    for partition in (fit, heldout):
        if any(not isinstance(key, str) or not key for key in partition.artifacts):
            raise ValueError("every retained artifact requires a nonempty string ID")
        for events in partition.artifacts.values():
            if (not isinstance(events, Sequence) or isinstance(events, (str, bytes))
                    or any(not isinstance(event, Mapping) for event in events)):
                raise ValueError("each artifact must contain a replayable sequence of observation bindings")
    if fit.artifacts.keys() & heldout.artifacts.keys():
        raise ValueError("an original artifact cannot occur in both partitions")
    if n_iters < 0 or (tol is not None and (not math.isfinite(tol) or tol < 0)):
        raise ValueError("n_iters and tolerance must be nonnegative, with finite tolerance")


def compare_candidates(candidates: Sequence[Candidate], fit: Partition, heldout: Partition, *,
                       n_iters=100, tol=1e-6, grouped=True, schedule="auto") -> dict[str, CandidateScore]:
    """Learn in artifact/event order, freeze parameters and sum final held-out ledger values.

    Bindings use the same public DATA port names and measurement units for every
    candidate. Each event starts from fresh messages. A candidate without declared
    learnables has fixed parameters. Caller-supplied artifact IDs identify the
    original data across partitions, independently of the values observed.
    """
    n_iters = operator.index(n_iters)
    _validate_round(candidates, fit, heldout, n_iters, tol)
    if schedule not in ("auto", "two_pass", "flood"):
        raise ValueError("expected schedule auto, two_pass or flood")
    colony = Colony(schedule=schedule) if grouped else None
    scores = {}
    ready: dict[int, tuple[CandidateScore, Engine, State]] = {}
    for index, candidate in enumerate(candidates):
        key = structure_hash(candidate.template)
        row = CandidateScore(candidate.candidate_id, fit.partition_id, heldout.partition_id,
                             tuple(fit.artifacts), tuple(heldout.artifacts), key)
        scores[candidate.candidate_id] = row
        iid = None
        try:
            flat = expand_plates(candidate.template)
            if (any(c.kind is ConstraintKind.P_SUBSTITUTION for c in flat.constraints)
                    or any(p.role is PortRole.GOAL for p in flat.ports.values())):
                raise ValueError("model evidence requires an ordinary evaluation graph without goals or P-substitution")
            if colony is None:
                engine = Engine(compile_template(flat, arrays=candidate.arrays), schedule=schedule)
                base = engine.init_state()
            else:
                iid = colony.spawn(candidate.template, candidate.arrays)
                instance = colony.instances[iid]
                engine, base = colony.engines[instance.key], instance.state
            if (engine.has_adaptive_gaussian or engine.arena.gfe_groups
                    or engine.arena.gfe_tm_groups or engine.arena.hybrid_gfe_groups):
                raise ValueError("adaptive surrogate and generalized energies are ineligible for Bethe evidence")
            row.data_domains = {p.name: flat.variables[p.variable].domain_key
                                for p in flat.ports.values() if p.role is PortRole.DATA}
            if _cyclic(flat):
                row.approximation = "Loopy Bethe approximation to negative log evidence."
            row.schedule = engine.schedule.kind
            row.dtype = base.messages.dtype.name
            trained = _fit(engine, base, candidate.priors, fit, row, n_iters, tol)
            if trained is not None:
                ready[index if iid is None else iid] = row, engine, trained
                continue
        except (ValueError, TypeError, LookupError, OSError, ArithmeticError) as exc:
            row.status, row.error = "invalid", str(exc)
        if iid is not None:
            colony.kill(iid)

    observed_ports = set()
    for artifact, index, observations in heldout.events():
        observed_ports.update(observations)
        singles = {}
        for iid, (row, engine, base) in list(ready.items()):
            try:
                state = engine.bind_data(base, observations)
            except (TypeError, LookupError, ValueError) as exc:
                row.status, row.error = "invalid", f"{artifact}[{index}] {exc}"
                del ready[iid]
                if colony is not None:
                    colony.kill(iid)
                continue
            if colony is None:
                singles[iid] = engine.run(state, n_iters, tol=tol, harden_last=n_iters)
            else:
                colony.instances[iid].state = state
        results = singles if colony is None else colony.run(n_iters, tol=tol)
        for iid, result in results.items():
            row = ready[iid][0]
            row.evaluation_events.append(_record(artifact, index, result))

    _rank((row for row, _, _ in ready.values()), observed_ports)
    return scores


def _rank(rows, observed_ports):
    eligible = []
    for row in rows:
        failed = next((e for e in row.evaluation_events if not e.usable), None)
        if failed is not None:
            row.status = _unusable(failed)
        elif not row.evaluation_events:
            row.status = "empty_evaluation"
        else:
            try:
                row.total_bfe = math.fsum(e.energies[-1] for e in row.evaluation_events)
            except OverflowError:
                row.status = "total_out_of_range"
                continue
            row.status = "converged"
            eligible.append(row)
    for port in observed_ports:
        if len({row.data_domains[port] for row in eligible}) > 1:
            raise ValueError(f"held-out DATA port {port!r} has incomparable observation domains")
    # Anchor each tie at the group's lowest raw total. Summation and state-label
    # reordering can differ by a few working-precision ulps per event.
    anchor = None
    for index, row in enumerate(sorted(eligible, key=lambda r: r.total_bfe), start=1):
        eps = float(np.finfo(row.dtype).eps)
        allowance = math.fsum(8 * eps * max(1.0, abs(e.energies[-1])) for e in row.evaluation_events)
        if anchor is None or row.total_bfe - anchor.total_bfe > allowance + anchor_allowance:
            anchor, anchor_allowance = row, allowance
            anchor.rank = index
        row.rank = anchor.rank
