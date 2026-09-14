"""Bounded independent dense solver for the frozen tiny D12 reference only.

This module depends on the independent evaluator, never on CFFG mathematics.
Quotient/xi steps use residual backtracking. A finite-difference Newton fallback
has a fixed budget; improper trials are rejected and failed attempts retained.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy

from hybrid_joint_reference import (
    EPS, TAU, SPEC_PATH, ReferenceFailure, evaluate, fixed_drive, load_case,
    specification, tilted, validate, zero_messages,
)
from hybrid_separator_reference import reflected_mixture_gap


class Coordinate:
    """Fixed support, one scalar gauge removed, symmetric precision, active xi."""

    def __init__(self, model):
        self.model = model
        self.entries = []
        t, _, d = model.shape
        for n in range(t - 1):
            for side in range(2):
                labels = np.flatnonzero(model.support[n + side])
                for i in labels:
                    if i != labels[0]:
                        self.entries.append((n, side, 0, (i,)))
                    for a in range(d):
                        self.entries.append((n, side, 1, (i, a)))
                    for a in range(d):
                        for b in range(a, d):
                            self.entries.append((n, side, 2, (i, a, b)))
        self.message_size = len(self.entries)
        self.auxiliary_indices = list(zip(*np.nonzero(model.auxiliary_support)))

    def pack(self, messages, xi):
        return np.array([messages[n][s].arrays()[a][index] for n, s, a, index in self.entries]
                        + [xi[index] for index in self.auxiliary_indices])

    def unpack(self, vector):
        if vector.shape != (self.message_size + len(self.auxiliary_indices),):
            raise ReferenceFailure("invalid packed coordinate shape")
        messages = zero_messages(self.model)
        for value, (n, s, a, index) in zip(vector, self.entries):
            messages[n][s].arrays()[a][index] = value
            if a == 2:
                i, row, col = index
                messages[n][s].precision[i, col, row] = value
        xi = np.zeros_like(self.model.offsets)
        for value, index in zip(vector[self.message_size:], self.auxiliary_indices):
            xi[index] = value
        return messages, xi

    def evaluate(self, vector):
        return evaluate(self.model, *self.unpack(vector))

    def residual(self, vector, evaluation=None):
        e = self.evaluate(vector) if evaluation is None else evaluation
        return self.pack(e["outgoing"], e["auxiliary_target"]) - vector


def jacobian(function, vector):
    """Frozen central difference, one-sided at xi/properness boundaries."""
    value = function(vector)
    columns = []
    for i, x in enumerate(vector):
        step = EPS**(1 / 3) * (1 + abs(x))
        offset = np.zeros_like(vector)
        offset[i] = step
        try:
            plus = function(vector + offset)
        except ReferenceFailure:
            plus = None
        try:
            minus = function(vector - offset)
        except ReferenceFailure:
            minus = None
        if plus is not None and minus is not None:
            columns.append((plus - minus) / (2 * step))
        elif plus is not None:
            columns.append((plus - value) / step)
        elif minus is not None:
            columns.append((value - minus) / step)
        else:
            raise ReferenceFailure("no feasible finite-difference stencil")
    return np.column_stack(columns)


@dataclass
class Result:
    status: str
    reason: str
    coordinate: np.ndarray | None
    evaluation: dict | None
    iterations: int
    evaluations: int
    rejected_trials: int
    newton_steps: int
    elapsed_seconds: float
    attempts: list


def solve(model, initial_xi=0., *, max_iterations=None):
    """Complete joint solve using only the committed reference strategy.

    ``max_iterations`` may reduce the frozen budget to test exhaustion; increasing
    it requires a prospectively committed specification change. No rejected
    trial changes the accepted coordinate. Every returned successful result is
    evaluated again at that coordinate with all certificates checked.
    """
    spec = specification()["solver"]
    budget = spec["max_iterations"] if max_iterations is None else max_iterations
    if not isinstance(budget, int) or not 0 <= budget <= spec["max_iterations"]:
        raise ValueError("iteration override must reduce the frozen budget")
    start = time.monotonic()
    attempts, evaluation, vector = [], None, None
    iterations = evaluations = rejected = newton_steps = 0

    def finish(status, reason):
        return Result(status, reason, vector, evaluation, iterations, evaluations,
                      rejected, newton_steps, time.monotonic() - start, attempts)

    try:
        xi = np.broadcast_to(np.asarray(initial_xi, dtype=float), model.offsets.shape).copy()
        # Inactive auxiliary coordinates are fixed, regardless of active start.
        xi[~model.auxiliary_support] = 0
        validate(model, xi)
        coordinate = Coordinate(model)
        _, _, _, messages = tilted(model, zero_messages(model), xi)
        vector = coordinate.pack(messages, xi)

        def at(value):
            nonlocal evaluations
            evaluations += 1
            return coordinate.evaluate(value)

        evaluation = at(vector)
        evaluation["certificates"]["relative_energy_change"] = np.inf
        for iteration in range(budget + 1):
            # Establish a stable, unchanged evaluation only after all other
            # complete certificates pass; it cannot hide a stationary residual.
            certificates = evaluation["certificates"]
            remaining = [v for k, v in certificates.items() if k != "relative_energy_change"]
            if max(remaining) <= 32 * EPS:
                previous = evaluation["free_energy"]
                evaluation = at(vector)
                change = abs(evaluation["free_energy"] - previous) / (1 + abs(previous))
                evaluation["certificates"]["relative_energy_change"] = change
                if max(evaluation["certificates"].values()) <= TAU:
                    return finish("converged", "complete returned-coordinate certificates passed")
            if iteration == budget:
                return finish("exhausted", "iteration budget exhausted")
            residual = coordinate.residual(vector, evaluation)
            scale = 1 + abs(vector)
            merit = np.linalg.norm(residual / scale)
            if not np.isfinite(merit):
                raise ReferenceFailure("nonfinite complete residual")
            accepted = False
            for direction_name in spec["directions"]:
                if direction_name.startswith("simultaneous"):
                    direction = residual
                    name = "fixed_point"
                else:
                    name = "newton"
                    newton_steps += 1

                    def residual_at(value):
                        return coordinate.residual(value, at(value))

                    derivative = jacobian(residual_at, vector)
                    direction = np.linalg.lstsq(derivative / scale[:, None], -residual / scale,
                                               rcond=EPS * max(derivative.shape))[0]
                for trial in range(spec["line_search_trials"]):
                    damping = spec["backtrack_factor"]**trial
                    proposal = vector + damping * direction
                    try:
                        candidate = at(proposal)
                        trial_merit = np.linalg.norm(coordinate.residual(proposal, candidate) / scale)
                        if not np.isfinite(trial_merit):
                            raise ReferenceFailure("nonfinite trial residual")
                        reason = "insufficient residual decrease"
                        accepted = trial_merit <= (1 - spec["armijo"] * damping) * merit
                    except ReferenceFailure as error:
                        accepted = False
                        reason = str(error)
                        trial_merit = None
                    if accepted:
                        previous = evaluation["free_energy"]
                        vector, evaluation = proposal, candidate
                        evaluation["certificates"]["relative_energy_change"] = abs(evaluation["free_energy"] - previous) / (1 + abs(evaluation["free_energy"]))
                        iterations = iteration + 1
                        break
                    rejected += 1
                    attempts.append(dict(iteration=iteration, direction=name, trial=trial,
                                         damping=damping, reason=reason, merit=trial_merit))
                if accepted:
                    break
            if not accepted:
                return finish("exhausted", "both declared directions exhausted their line searches")
    except (ReferenceFailure, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        attempts.append(dict(stage="initialization" if evaluation is None else "iteration", reason=str(error)))
        return finish("diverged", str(error))


def output_arrays(evaluation):
    """Stable public comparison groups, including physical and centered arrays."""
    groups = {"messages": np.concatenate([x.ravel() for bank in evaluation["messages"]
                                          for n in bank for x in n.arrays()]),
              "xi": evaluation["xi"].ravel(),
              "energy": np.array([evaluation["free_energy"], evaluation["dual_energy"]])}
    for name in ("regions", "separators"):
        for field in ("probability", "displacement", "mean", "covariance", "second"):
            groups[f"{name}_{field}"] = np.concatenate([getattr(b, field).ravel() for b in evaluation[name]])
    for field in ("target", "exposure", "centered_linear", "centered_precision", "linear", "precision"):
        groups[f"learning_{field}"] = np.concatenate([s[field].ravel() for s in evaluation["learning"]])
    return groups


def stopping_allowance(model, vector):
    """Prospective local first-order residual propagation, separate from arithmetic.

    For two nearby roots with scaled residual <= tau, the linearized output
    error is bounded by 2*tau*||Dg DR^-1||_infinity. This is a local diagnostic,
    not a global uniqueness proof or an approximation allowance. No candidate
    discrepancy is an input. Report the coordinate inverse norm as well.
    """
    coordinate = Coordinate(model)
    e = coordinate.evaluate(vector)
    groups = output_arrays(e)
    sizes = [len(g) for g in groups.values()]
    values = np.concatenate(list(groups.values()))
    residual_scale = 1 + abs(vector)
    derivative = jacobian(lambda x: coordinate.residual(x) / residual_scale, vector)
    inverse = np.linalg.inv(derivative)
    output_derivative = jacobian(lambda x: np.concatenate(list(output_arrays(coordinate.evaluate(x)).values())) / (1 + abs(values)), vector)
    sensitivity = output_derivative @ inverse
    start, allowances = 0, {}
    for name, size in zip(groups, sizes):
        norm = float(np.linalg.norm(sensitivity[start:start + size], ord=np.inf))
        allowances[name] = dict(sensitivity_norm=norm, scaled_allowance=2 * TAU * max(1., norm))
        start += size
    return dict(coordinate_inverse_norm=float(np.linalg.norm(inverse, ord=np.inf)), groups=allowances)


def report():
    """Reproducible tiny checkpoint; never runs moderate cases or recovery fits."""
    records = []
    for name in specification()["cases"]:
        model = load_case(name)
        for initial in specification()["initializations"]["xi"]:
            result = solve(model, initial)
            e = result.evaluation
            record = {k: getattr(result, k) for k in ("status", "reason", "iterations", "evaluations", "rejected_trials", "newton_steps", "elapsed_seconds", "attempts")}
            record.update(case=name, initial_xi=initial)
            if e is not None:
                record.update(message_coordinate=result.coordinate.tolist(), xi=e["xi"].tolist(),
                              free_energy=e["free_energy"], dual_energy=e["dual_energy"],
                              certificates={k: float(v) if np.isfinite(v) else None
                                            for k, v in e["certificates"].items()},
                              proper_active_beliefs=e["proper_active_beliefs"],
                              learning={k: [s[k].tolist() for s in e["learning"]] for k in
                                        ("linear", "precision", "centered_linear", "centered_precision")})
            if result.status == "converged":
                record["stopping_propagation"] = stopping_allowance(model, result.coordinate)
            records.append(record)
    conditioned = fixed_drive(**{k: np.asarray(v, dtype=float)
                                for k, v in specification()["fixed_drive"].items()})
    falsifier = reflected_mixture_gap(full=True)
    return dict(spec_sha256=hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
                source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("hybrid_joint_reference.py", "hybrid_joint_solver_reference.py",
                                            "hybrid_separator_reference.py")},
                versions=dict(numpy=np.__version__, scipy=scipy.__version__),
                records=records, fixed_auxiliary_approximation_context=specification()["comparison"]["fixed_auxiliary_approximation_ceilings"],
                conditioned_fixed_drive={k: np.asarray(v).tolist() for k, v in conditioned.items()},
                temporal_falsifier={k: np.asarray(v).tolist() for k, v in falsifier.items()},
                approximation_scope="Temporal falsifier and prior fixed-auxiliary ceilings remain separate from joint EC convergence; no universal exactness claim.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.output, "w") as stream:
        json.dump(report(), stream, indent=2, allow_nan=False)
        stream.write("\n")
