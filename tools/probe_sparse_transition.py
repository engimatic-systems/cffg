"""Measure indexed transition atoms inside the frozen public Hallway runs.

The probe replaces the transition's three local operations during Colony
tracing. It keeps the compiled controller, public result construction, dense
reference arms, and every original update and ledger evaluation.
"""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

from benchmark_ordinary_iteration import (
    DAMPING,
    MEMBERS,
    SWEEPS,
    hallway,
    hardware,
    make_runner,
    measured,
    results_record,
)
from cffg import colony, efrules, ledger
from cffg.compiled import NEG
from cffg.compiled_iteration import CompiledIteration
from cffg.compiler import compile_template
from cffg.engine import Engine
from cffg.execution import synchronous
from cffg.execution.contracts import ExecutionDelta
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.iteration_types import IterationReason
from cffg.kernel_runtime import discrete_potential, parameters
from cffg.rules import _norm_pad, bp_transition_mixture, energy_transition_mixture
from cffg.schedule_program import make_support_check


@dataclass(frozen=True)
class Atoms:
    indices: tuple[np.ndarray, ...]
    log_factor: np.ndarray
    present: np.ndarray

    def inputs(self, rows, roles):
        factors = jnp.arange(self.log_factor.shape[0])
        broadcast = (slice(None),) + (None,) * (self.log_factor.ndim - 1)
        indices = factors[broadcast]
        return tuple(rows[indices, role, self.indices[role]] for role in roles)


@dataclass(frozen=True)
class IndexedTransition:
    cards: tuple[int, ...]
    joint: Atoms
    outputs: tuple[Atoms, ...]

    @classmethod
    def pack(cls, dense):
        cards = dense.shape[1:]
        if len(cards) != 3 or min(cards) < 2:
            raise ValueError("the probe requires three transition roles with at least two values each")
        zero = dense.dtype.type(NEG)
        supports = [np.argwhere(factor != zero) for factor in dense]
        if (not np.isfinite(dense).all() or np.any(dense > 0)
                or any(not len(coords) for coords in supports)
                or np.any(dense[dense != zero] < zero / 4)):
            raise ValueError("unsupported transition potentials for the indexed probe")

        def pack_groups(groups):
            bins = len(groups[0])
            width = max(len(coords) for group in groups for coords in group)
            shape = (len(dense), bins, width) if bins > 1 else (len(dense), width)
            coordinates = np.zeros((*shape, len(cards)), dtype=np.int32)
            values = np.full(shape, NEG, dtype=dense.dtype)
            present = np.zeros(shape, dtype=bool)
            for factor, group in enumerate(groups):
                for index, coords in enumerate(group):
                    selection = (factor, index) if bins > 1 else (factor,)
                    selected = (*selection, slice(len(coords)))
                    coordinates[selected] = coords
                    values[selected] = dense[(factor, *coords.T)]
                    present[selected] = True
            return Atoms(tuple(np.moveaxis(coordinates, -1, 0)), values, present)

        joint = pack_groups([[coords] for coords in supports])
        outputs = tuple(pack_groups([
            [coords[coords[:, role] == value] for value in range(card)]
            for coords in supports
        ]) for role, card in enumerate(cards))
        return cls(cards, joint, outputs)

    def messages(self, incoming):
        # If each incoming maximum is within a quarter ULP of zero relative
        # to NEG, a forbidden maximizer rounds back to NEG at both additions.
        # If that coordinate is allowed, its score dwarfs the forbidden terms.
        # Near NEG the entire log multiplicity fits below half an ULP.
        quarter_ulp = abs(np.spacing(np.dtype(incoming.dtype).type(NEG))) / 4
        admitted = jnp.asarray(True)
        for role, card in enumerate(self.cards):
            row = incoming[:, role, :card]
            peak = row.max(axis=-1)
            admitted &= jnp.all((peak >= -quarter_ulp) & (peak <= 0))
        rows = []
        for role, atoms in enumerate(self.outputs):
            other = tuple(index for index in range(len(self.cards)) if index != role)
            first, second = atoms.inputs(incoming, other)
            logits = jnp.where(atoms.present, atoms.log_factor + first + second, NEG)
            logits = jnp.concatenate((logits, jnp.full((*logits.shape[:-1], 1), NEG, logits.dtype)), axis=-1)
            output = logsumexp(logits, axis=-1)
            rows.append(_norm_pad(output, self.cards[role], max(self.cards)))
        return jnp.where(admitted, jnp.stack(rows, axis=1), jnp.nan)

    def energy(self, incoming):
        out, source, selector = self.joint.inputs(incoming, range(len(self.cards)))
        factor = self.joint.log_factor
        joint = jnp.where(self.joint.present, factor + out + source + selector, -jnp.inf)
        maxima = jnp.max(incoming, axis=-1)
        admitted = jnp.all(maxima <= 0) & jnp.all(joint.max(axis=-1) > NEG / 2)
        # The guard proves the omitted probabilities underflow in both dtypes.
        return jnp.where(admitted, efrules.CATEGORICAL.energy(joint, factor), jnp.nan)

    def zero_support(self, support):
        factors = np.arange(self.joint.present.shape[0])[:, None]
        allowed = jnp.asarray(self.joint.present)
        for role, values in enumerate(support):
            allowed &= values[factors, self.joint.indices[role]]
        return ~jnp.all(jnp.any(allowed, axis=-1))


class ColonyProbe:
    """Install a separate controller while both dense reference arms stay stock."""

    def __init__(self):
        self.model = None
        self.retries = 0
        self.reference = colony.controller_for
        self.execute = synchronous._execute_discrete
        self.energy = ledger._disc_energy
        self.setup_s = None

    def controller(self, engine):
        if self.model is not None:
            if engine is not self.engine:
                raise ValueError("the probe expects one Hallway engine")
            return self
        start = time.perf_counter()
        self.engine = engine
        self.dense = self.reference(engine)
        arena = engine.arena
        if (engine.schedule.kind != "two_pass" or arena.semiring != "sum"
                or engine._prime is not None or arena.constraint_groups):
            raise ValueError("candidate fixture no longer runs ordinary sum-product flooding")
        # Separate raw functions keep sparse tracing out of the original
        # engine's JIT caches, including if a dense retry is needed.
        self.indexed = CompiledIteration(engine, (
            synchronous.make_sweep(arena), None, ledger.make_belief_energy(arena), None, None,
        ))
        self.group, = (group for group in arena.groups if group.type == "transition_mixture")
        packed = parameters(self.group, arena.log_params)
        self.paths = tuple(path for _, path in self.group.parameter_paths)
        self.expected = {path: np.asarray(arena.log_params[path]) for path in self.paths}
        self.model = IndexedTransition.pack(np.asarray(discrete_potential(self.group, packed)))
        executors = replace(SCHEDULED_EXECUTORS, discrete_zero_support=self.zero_support)
        support = make_support_check(arena, executors)
        self.indexed.bad_support = lambda state: support(state.messages, state.params)
        self.validated = {}
        self.setup_s = time.perf_counter() - start
        return self

    def execute_group(self, group, incoming, params, reduction):
        if group is self.group:
            return ExecutionDelta(self.model.messages(incoming))
        return self.execute(group, incoming, params, reduction)

    def energy_group(self, group, incoming, params):
        if group is self.group:
            return self.model.energy(incoming)
        return self.energy(group, incoming, params)

    def zero_support(self, group, supports, params):
        if group is self.group:
            return self.model.zero_support(supports)
        return SCHEDULED_EXECUTORS.discrete_zero_support(group, supports, params)

    def run_batch(self, states, **controls):
        for state in states:
            for path in self.paths:
                leaf = state.params[path]
                if id(leaf) in self.validated:
                    continue
                if not np.array_equal(np.asarray(leaf), self.expected[path]):
                    raise ValueError("candidate and the bound transition parameters differ")
                self.validated[id(leaf)] = leaf
        with ExitStack() as stack:
            stack.enter_context(patch.object(synchronous, "_execute_discrete", self.execute_group))
            stack.enter_context(patch.object(ledger, "_disc_energy", self.energy_group))
            outcomes = self.indexed.run_batch(states, **controls)
        if any(outcome.reason is not IterationReason.BUDGET for outcome in outcomes):
            self.retries += 1
            return self.dense.run_batch(states, **controls)
        return outcomes


def qualify_local():
    errors = {}
    for dtype in (np.float32, np.float64):
        rng = np.random.default_rng(19)
        cards = (3, 4, 2)
        dense = np.full((2, *cards), NEG, dtype=dtype)
        for factor, coordinates in enumerate((
            [(0, 0, 0), (0, 2, 1), (1, 3, 0)],
            [(0, 1, 0), (1, 1, 1), (2, 2, 0)],
        )):
            for coord in coordinates:
                dense[(factor, *coord)] = -rng.uniform(0, 4)
        indexed = IndexedTransition.pack(dense)
        incoming = -rng.uniform(0, 8, (2, 3, max(cards))).astype(dtype)
        incoming[0, 1, 2:] = NEG
        incoming[1, 0, 2] = dtype(NEG) / 2
        incoming = jnp.asarray(incoming)
        actual = np.asarray(jax.jit(indexed.messages)(incoming))
        expected = np.asarray(jax.jit(lambda x: bp_transition_mixture(x, dense, cards, max(cards)))(incoming))
        np.testing.assert_array_equal(actual == dtype(NEG), expected == dtype(NEG))
        message_error = float(np.max(abs(actual - expected) / (1 + abs(expected))))
        actual = np.asarray(jax.jit(indexed.energy)(incoming))
        expected = np.asarray(jax.jit(lambda x: energy_transition_mixture(x, dense, cards))(incoming))
        energy_error = float(np.max(abs(actual - expected) / (1 + abs(expected))))
        limit = 32 * np.finfo(dtype).eps
        if max(message_error, energy_error) > limit:
            raise ValueError(f"indexed local rule disagreement in {dtype}")
        supported = tuple(incoming[:, role, :card] != dtype(NEG) for role, card in enumerate(cards))
        if indexed.zero_support(supported) or not indexed.zero_support((
            supported[0], supported[1], jnp.zeros_like(supported[2]),
        )):
            raise ValueError("empty support guard disagrees")
        hostile = incoming.at[0, 0, 0].set(1)
        if not np.isnan(indexed.messages(hostile)).all() or not np.isnan(indexed.energy(hostile)).all():
            raise ValueError("out-of-range positive messages require dense replay")
        errors[np.dtype(dtype).name] = {"messages": message_error, "energy": energy_error}
    return errors


def compare(reference, candidate, decoder):
    worst = {}
    for old, new in zip(reference, candidate, strict=True):
        for field in old.__dataclass_fields__:
            if field in ("state", "free_energies", "message_deltas", "residual"):
                continue
            expected, actual = getattr(old, field), getattr(new, field)
            if isinstance(expected, float) and np.isnan(expected) and np.isnan(actual):
                continue
            if expected != actual:
                raise ValueError(f"public result field differs: {field}")
        for field in ("free_energies", "message_deltas", "residual", "messages"):
            a, b = (getattr(value.state if field == "messages" else value, field)
                    for value in (old, new))
            a, b = np.asarray(a), np.asarray(b)
            if field == "messages" and not np.array_equal(a == NEG, b == NEG):
                raise ValueError("the exact support of a directed message differs")
            finite = np.isfinite(a)
            np.testing.assert_array_equal(b[~finite], a[~finite], err_msg=field)
            if not np.isfinite(b[finite]).all():
                raise ValueError(f"{field} became nonfinite")
            a, b = a[finite], b[finite]
            error = float(np.max(np.abs(a - b) / (1 + np.abs(a)), initial=0.0))
            if not np.isfinite(error) or error > 1e-10:
                raise ValueError(f"{field} scaled difference {error:.4g}")
            worst[field] = max(worst.get(field, 0), error)
        old_marginals, new_marginals = (decoder.marginals(run.state) for run in (old, new))
        for edge, expected in old_marginals.items():
            error = float(np.max(np.abs(expected - new_marginals[edge])))
            if not np.isfinite(error) or error > 1e-10:
                raise ValueError(f"{edge} posterior difference {error:.4g}")
            worst["posterior"] = max(worst.get("posterior", 0), error)
    return worst


def run_arm(arm, repetitions, probe=None):
    reset, run, setup, fixture = make_runner(arm)
    timings = {}
    outputs = {}
    for budget in (1, SWEEPS):
        reset()
        cold, result = measured(lambda: run(budget))
        results_record(result, budget)
        samples = []
        for _ in range(repetitions):
            reset()
            elapsed, result = measured(lambda: run(budget))
            results_record(result, budget)
            samples.append(elapsed)
        timings[str(budget)] = {"cold_s": cold, "samples_s": samples, "median_s": statistics.median(samples)}
        outputs[budget] = result
    result = {"timings": timings, "setup": setup, "fixture": fixture}
    if probe is not None:
        result.update({"indexed_setup_s": probe.setup_s, "dense_retries": probe.retries})
    return result, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("repetitions must be positive")
    if not jax.config.x64_enabled or jax.default_backend() != "cpu":
        raise ValueError("set JAX_ENABLE_X64=1 and JAX_PLATFORMS=cpu")
    local_errors = qualify_local()
    single, _ = run_arm("single", args.repetitions)
    sequential, oracle = run_arm("sequential", args.repetitions)
    dense, reference = run_arm("colony", args.repetitions)
    probe = ColonyProbe()
    with patch.object(colony, "controller_for", probe.controller):
        indexed, candidate = run_arm("colony", args.repetitions, probe)
    medians = [row["timings"][str(SWEEPS)]["median_s"] for row in (sequential, dense, indexed)]
    print("Unverified 60-sweep warm medians (sequential, dense, indexed):", medians, flush=True)
    template, arrays, _ = hallway()
    decoder = Engine(compile_template(template, arrays=arrays))
    parity = {str(budget): {"dense_colony": compare(oracle[budget], reference[budget], decoder),
                            "indexed_colony": compare(oracle[budget], candidate[budget], decoder)}
              for budget in (1, SWEEPS)}
    receipt = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hardware": hardware(), "jax": jax.__version__, "members": MEMBERS,
        "damp": DAMPING, "sweeps": SWEEPS, "tol": None,
        "single": single, "sequential": sequential, "dense_colony": dense, "indexed_colony": indexed,
        "local_scaled_parity_errors": local_errors,
        "max_scaled_parity_errors": parity,
        "sequential_over_indexed": medians[0] / medians[2],
        "dense_colony_over_indexed": medians[1] / medians[2],
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"seconds": medians, "speedup": receipt["sequential_over_indexed"],
                      "dense_retries": indexed["dense_retries"]}))


if __name__ == "__main__":
    main()
