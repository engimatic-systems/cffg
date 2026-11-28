"""Request structural alternatives and compare their predictions on a common data partition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from cffg.structure_operators import RejectedEdit, StructuralCandidate, propose_operator
from cffg.structure_proposals import ProposalError, model_identity, record_proposal, replay_proposal
from cffg.structure_scoring import Candidate, CandidateScore, Partition, compare_candidates


class ProposalTrigger:
    """Request proposals once per streak of poor, completed evaluations of one model."""

    def __init__(self, threshold: float, consecutive: int):
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("threshold must be a finite positive number of nats")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be a finite positive number of nats")
        if isinstance(consecutive, bool) or not isinstance(consecutive, Integral) or consecutive < 1:
            raise ValueError("consecutive must be a positive integer")
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)
        self.candidate_id = None
        self.streak = 0

    def observe(self, score: CandidateScore) -> bool:
        """Count one evaluation, resetting on model change, failure or recovery."""
        if self.candidate_id != score.candidate_id:
            self.candidate_id = score.candidate_id
            self.streak = 0
        total = score.total_bfe
        if score.status != "converged" or total is None or not math.isfinite(total) or total <= self.threshold:
            self.streak = 0
            return False
        self.streak += 1
        return self.streak == self.consecutive


@dataclass(frozen=True)
class ProposalFailure:
    index: int
    code: str
    detail: str


@dataclass(frozen=True)
class SearchResult:
    incumbent_id: str
    candidates: dict[str, StructuralCandidate]
    scores: dict[str, CandidateScore]
    failures: tuple[ProposalFailure, ...]
    selected_id: str | None
    reason: str


def source_candidate(source_id, models) -> StructuralCandidate:
    """Record an incumbent from a retained (Template, arrays) source with no edits."""
    record = record_proposal(
        source_id=source_id, parent_id=None, proposal_source="incumbent",
        steps=(), models=models,
    )
    return StructuralCandidate(record, {})


def _snapshot(proposal):
    return StructuralCandidate(
        deepcopy(proposal.record),
        {key: np.array(value, copy=True) for key, value in proposal.inputs.items()},
    )


def _decision(scores, parent):
    leaders = [key for key, row in scores.items() if row.rank == 1]
    if not leaders:
        return None, "no_ranked_model"
    if parent in leaders:
        return parent, "incumbent_tied" if len(leaders) > 1 else "incumbent_best"
    if len(leaders) == 1:
        return leaders[0], "selected_challenger"
    return None, "tied_challengers"


def search_round(
    source_id: str, models: Mapping, proposals: Iterable, fit: Partition, heldout: Partition,
    *, incumbent: StructuralCandidate | None = None, priors: Mapping | None = None,
    n_iters=100, tol=1e-6, grouped=True, schedule="auto",
) -> SearchResult:
    """Compare a retained source and its children, passing unchanged ledgers to the decision.

    Proposals may be local operations, replayable StructuralCandidate records made by
    another producer, or RejectedEdit results. Optional learner priors are keyed by
    candidate ID. When continuing, supply the accepted incumbent record and register
    its replayed result in models under source_id. Retain all source models for replay.
    """
    priors = {} if priors is None else priors
    root = source_candidate(source_id, models) if incumbent is None else _snapshot(incumbent)
    parent = root.record["candidate_id"]
    template, arrays = replay_proposal(root.record, models, inputs=root.inputs)
    if model_identity(*models[source_id]) != root.record["result"]:
        raise ValueError("the proposed source differs from the replayed incumbent")
    records = {parent: root}
    candidates = [Candidate(parent, template, arrays, priors.get(parent))]
    failures = []
    for i, item in enumerate(proposals):
        proposal = item if isinstance(item, (StructuralCandidate, RejectedEdit)) else propose_operator(
            source_id, models, item, parent_id=parent,
        )
        if isinstance(proposal, RejectedEdit):
            failures.append(ProposalFailure(i, proposal.code, proposal.detail))
            continue
        try:
            proposal = _snapshot(proposal)
            record = proposal.record
            template, arrays = replay_proposal(record, models, inputs=proposal.inputs)
            if record["parent_id"] != parent or record["models"][record["source_id"]] != root.record["result"]:
                raise ProposalError("the recorded parent and source must match the incumbent")
            key = record["candidate_id"]
        except (TypeError, ValueError, LookupError, AttributeError) as exc:
            failures.append(ProposalFailure(i, "invalid_record", str(exc)))
            continue
        if key in records:
            failures.append(ProposalFailure(i, "duplicate_candidate", key))
            continue
        records[key] = proposal
        candidates.append(Candidate(key, template, arrays, priors.get(key)))
    scores = compare_candidates(
        candidates, fit, heldout, n_iters=n_iters, tol=tol, grouped=grouped, schedule=schedule,
    )
    selected, reason = _decision(scores, parent)
    return SearchResult(parent, records, scores, tuple(failures), selected, reason)
