"""Held-out next-landmark evidence on a conditional, two-DATA-port CFFG tree.

A row identifies a source place and an applied action. Both candidates must use
the same public outcome alphabet and the same ordered held-out transitions.
Only their source-place assignments may differ. Log scores are in nats.
"""

from functools import lru_cache
import math
import operator

import equinox as eqx
import jax
import numpy as np

from cffg import Engine, RunStatus, compile_template, set_params
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec


def _index(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("a row, symbol or dimension must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError("a row, symbol or dimension must be an integer") from exc


def _events(events, n_rows, n_outcomes):
    for row, outcome in events:
        row, outcome = _index(row), _index(outcome)
        if not 0 <= row < n_rows or not 0 <= outcome < n_outcomes:
            raise ValueError(f"event {(row, outcome)} is outside the fixed table {(n_rows, n_outcomes)}")
        yield row, outcome


def fit_counts(exploration_events, *, n_rows, n_outcomes):
    """Count exploration events before smoothing, including unvisited rows."""
    shape = _index(n_rows), _index(n_outcomes)
    if min(shape) < 1:
        raise ValueError("both table dimensions must be positive")
    counts = np.zeros(shape, dtype=np.int64)
    for event in _events(exploration_events, *shape):
        counts[event] += 1
    return counts


def _counts(table):
    values = np.array(table, copy=True)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("counts must be a nonempty row-by-outcome table")
    if values.dtype.kind not in "iuf" or np.any(values < 0):
        raise ValueError("counts must be nonnegative integers")
    if values.dtype.kind == "f" and (not np.isfinite(values).all()
                                    or np.any(values != np.floor(values))):
        raise ValueError("fractional and nonfinite counts are invalid")
    return values


@lru_cache(maxsize=8)
def _tree(n_rows, n_outcomes, dtype):
    # Conditioning on both endpoints leaves Z = P(outcome | row).
    # A source prior would add its surprisal to every event.
    cards = {"row": n_rows, "outcome": n_outcomes}
    template = Template(
        "next_landmark",
        variables={name: VariableSpec(name, card) for name, card in cards.items()},
        factors={"next": FactorSpec("next", "matrix_cat", edges={"in": "row", "out": "outcome"},
                                    param_blobs={"matrix": "likelihood"})},
        ports={name: PortSpec(name, name, PortRole.DATA) for name in cards},
        blobs={"likelihood": BlobRef("likelihood", "likelihood.npy")},
    )
    initial = np.full((n_outcomes, n_rows), 1.0 / n_outcomes, dtype=dtype)
    engine = Engine(compile_template(template, arrays={"likelihood": initial}))
    if engine.schedule.kind != "two_pass":
        raise RuntimeError(f"expected exact two-pass tree, got {engine.schedule.kind}")
    return engine


def _condition(engine, matrix):
    state = engine.init_state()
    params = set_params(engine, state.params, {"next": {"matrix": matrix}})
    return eqx.tree_at(lambda s: s.params, state, params)


def _event_energy(engine, base, row, outcome):
    observed = engine.bind_data(base, {"row": row, "outcome": outcome})
    result = engine.run(observed, 1)
    if (result.status is not RunStatus.CONVERGED or result.inner_status is not RunStatus.CONVERGED
            or result.iterations != 1 or len(result.free_energies) != 1):
        raise ArithmeticError(f"event {(row, outcome)}: {result.status.value}, "
                              f"inner={result.inner_status.value}, iterations={result.iterations}")
    energy = float(result.free_energies[0])
    if not math.isfinite(energy):
        raise ArithmeticError(f"nonfinite free energy for event {(row, outcome)}")
    return energy


def score(counts, events, *, dtype="float64"):
    """Score every event natively against a frozen Laplace(+1) predictive table."""
    counts = _counts(counts)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    n_rows, n_outcomes = counts.shape
    events = list(_events(events, n_rows, n_outcomes))
    # Keep the oracle in count space even when the native CPT rounds a probability to one.
    totals = [sum(map(int, row)) + n_outcomes for row in counts]
    matrix = np.array([[(int(c) + 1) / total for c in row]
                       for row, total in zip(counts, totals, strict=True)], dtype=dtype).T
    if not np.isfinite(matrix).all() or np.any(matrix <= 0):
        raise ArithmeticError(f"Laplace support is unrepresentable at {dtype.name}")
    energies, oracle, bounds = [], [], []
    with jax.enable_x64(dtype == np.float64):
        engine = _tree(n_rows, n_outcomes, dtype)
        base = _condition(engine, matrix)
        if base.messages.dtype != dtype:
            raise ArithmeticError("native message dtype differs from requested precision")
        for row, outcome in events:
            energy = _event_energy(engine, base, row, outcome)
            mass = int(counts[row, outcome]) + 1
            expected = math.log1p((totals[row] - mass) / mass)
            # Allow the rounding of the CPT, its logarithm and the final exact-tree ledger.
            bound = 64.0 * float(np.finfo(dtype).eps) * max(1.0, expected)
            if abs(energy - expected) > bound:
                raise ArithmeticError(f"event {(row, outcome)}: native BFE {energy}, "
                                      f"predictive surprisal {expected}, tolerance {bound}")
            energies.append(energy)
            oracle.append(expected)
            bounds.append(bound)
    total = math.fsum(energies)
    return dict(status="ok", schedule="two_pass", dtype=dtype.name, n_rows=n_rows,
                n_outcomes=n_outcomes, n_events=len(events), outcomes=[y for _, y in events],
                energies=energies, oracle_energies=oracle, total_bfe=total,
                mean_bfe=total / len(events) if events else None,
                max_abs_error=max((abs(a - b) for a, b in zip(energies, oracle, strict=True)), default=0.),
                total_error_bound=math.fsum(bounds))


def select(appearance_score, split_score):
    """Select the split only when held-out native total BFE is distinctly lower.

    The caller supplies scores for the same frozen traversal. Row IDs may
    differ by candidate; the public next-symbol codes must agree in order.
    """
    a, b = appearance_score, split_score
    if a["status"] != "ok" or b["status"] != "ok":
        raise ValueError("both candidates must have qualified evidence")
    if not a["n_events"] or any(a[key] != b[key] for key in ("dtype", "n_outcomes", "n_events", "outcomes")):
        raise ValueError("selection needs matching precision and a common nonempty validation tape")
    gain = a["total_bfe"] - b["total_bfe"]
    bounds = [record["total_error_bound"] for record in (a, b)]
    if not math.isfinite(gain) or any(not math.isfinite(v) or v < 0 for v in bounds):
        raise ValueError("invalid total BFE or precision bound")
    tolerance = math.fsum(bounds)
    return dict(selected="split" if gain > tolerance else "appearance", gain=gain,
                tie_tolerance=tolerance,
                reason="precision_tie" if abs(gain) <= tolerance else "lower_bfe")
