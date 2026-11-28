"""Schedule the released interaction reducer on completed Gameworld actions.

The caller supplies a dedicated reduction key and the original seeded dormant
interaction model. Call after_action after observing the applied action,
including the last action of a run. A game reset preserves this history.
"""

from collections import Counter

import numpy as np

from cffg.component_bank_device import from_host, to_host
from cffg.experiments.axiom_reduction import ReductionState, reduce_interactions


PERIOD = 500
SAMPLE_COUNT = 2_000
PAIR_COUNT = 2_000


def _completed(step):
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ValueError("expected a nonnegative integer count of completed actions")
    return int(step)


def _check_history(history, last_step):
    if history.last_step != last_step:
        raise ValueError(f"reduction history ends at {history.last_step}, expected {last_step}")
    expected = last_step // PERIOD * SAMPLE_COUNT
    if len(history.records.features) != expected or len(history.records.categories) != expected:
        raise ValueError(f"reduction at {last_step} requires {expected} generated records")


def start(device_bank, reduction_key):
    """Initialize history for a fresh actor without transferring the learning bank."""
    return ReductionState.start(device_bank, reduction_key)


def archive(history):
    """Return exact NumPy checkpoint fields for the random key and generated tape."""
    _check_history(history, history.last_step)
    return history.state_dict()


def restore(arrays, device_bank, *, step):
    """Validate the saved history against the exact resumed scene and real clock."""
    completed = _completed(step)
    history = ReductionState.from_state_dict(arrays, device_bank)
    _check_history(history, completed - completed % PERIOD)
    return history


def after_action(device_bank, history, *, step, dormant_initial):
    """Return (bank, history, receipt). Ordinary ticks return the original objects.

    Away from the 500-action boundary this performs only host clock/shape checks.
    At the boundary the complete bank is validated and converted once. A failure
    raises before returning an updated model or advancing the reduction history.
    Repeating an already completed boundary returns the committed state unchanged.
    """
    completed = _completed(step)
    boundary = completed - completed % PERIOD
    if completed != boundary or history.last_step == completed:
        _check_history(history, boundary)
        return device_bank, history, None

    _check_history(history, boundary - PERIOD)
    if dormant_initial is None:
        raise ValueError("scheduled reduction requires the seeded dormant interaction model")
    bank = to_host(device_bank)
    result = reduce_interactions(bank, history, step=completed, dormant_initial=dormant_initial,
                                 sample_count=SAMPLE_COUNT, pair_count=PAIR_COUNT)
    if result.status != "ok":
        raise ArithmeticError(f"interaction reduction at completed action {completed}: {result.status}")
    _check_history(result.state, boundary)
    decisions = result.decisions
    statuses = Counter(event.status for event in decisions)
    receipt = {
        "step": completed, "status": result.status, "sample_count": SAMPLE_COUNT,
        "records_total": len(result.state.records.features), "pairs": len(decisions),
        "accepted": statuses["accepted"], "full_rescores": sum(e.full_rescore for e in decisions),
        "active_before": sum(i != 0 for i in bank.logical_ids),
        "active_after": sum(i != 0 for i in result.bank.logical_ids),
        "status_counts": dict(statuses), "initial_score": result.initial_score,
        "final_score": result.final_score, "decisions": [e._asdict() for e in decisions],
    }
    updated = device_bank if bank is result.bank else from_host(result.bank)
    return updated, result.state, receipt
