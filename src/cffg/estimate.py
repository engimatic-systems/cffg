"""Resource estimation for a compiled model (brief item 7).

Three sources, in order of trustworthiness:
- **Analytic memory** — exact, from every persistent State arena plus the
  parameter stacks; sizes are static post-compile.
- **XLA cost analysis** — flops and bytes-accessed per sweep from the
  compiled step's own cost model (`lower(...).compile().cost_analysis()`).
  Backend-dependent; None when the backend won't say.
- **Calibration** — a short measured run; `predict(n_iters)` extrapolates.
  Approximate by design ("fit a curve, if nothing else" — the brief).
"""

from __future__ import annotations

import time
from statistics import median
from dataclasses import dataclass, field

import jax

from cffg.engine import Engine, RunStatus, State
from cffg.plate_storage import UnknownStorageLayout as UnknownStorageLayout
from cffg.plate_storage import estimate_plate_storage as estimate_plate_storage
from cffg.resource_analysis import executable_reports, state_storage


@dataclass
class Estimate:
    arena_bytes: dict[str, int]
    param_bytes: int
    flops_per_sweep: float | None
    bytes_accessed_per_sweep: float | None
    seconds_per_sweep: float | None
    calibration_status: RunStatus | None
    declared_work: dict[str, dict[str, int]] = field(default_factory=dict)

    parameter_leaves: dict[str, int] = field(default_factory=dict)
    executables: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    @property
    def message_bytes(self) -> int:
        """Discrete log-message bytes (the pre-segmentation compatibility field)."""
        return self.arena_bytes["messages"]

    @property
    def arena_state_bytes(self) -> int:
        return sum(self.arena_bytes.values())

    @property
    def state_bytes(self) -> int:
        return self.arena_state_bytes + self.param_bytes

    @property
    def prediction_unit(self) -> str | None:
        """Scheduled work unit used by predict_seconds and the empirical latency."""
        return self.calibration.get("unit")

    def predict_seconds(self, n_iters: int) -> float | None:
        """Predict seconds for n_iters completed prediction_unit iterations."""
        if n_iters < 0:
            raise ValueError("A prediction requires a nonnegative work count")
        if self.calibration.get("unit") == "two_pass" or self.calibration_status in (
            RunStatus.DIVERGED,
            RunStatus.INFEASIBLE,
        ):
            return None
        if self.seconds_per_sweep is None:
            return None
        return n_iters * self.seconds_per_sweep

    def summary(self) -> str:
        flops = f"{self.flops_per_sweep:.3g}" if self.flops_per_sweep else "n/a"
        secs = (
            f"{self.seconds_per_sweep * 1e3:.3f} ms"
            if self.seconds_per_sweep
            else "n/a"
        )
        diagnostic = self.executables.get("flooding_step", {}).get("cost_diagnostic")
        if diagnostic:
            flops = (
                f"{diagnostic['kind']} ({diagnostic['type']}: {diagnostic['message']})"
            )
        traffic = self.bytes_accessed_per_sweep
        return (
            f"exact persistent state {self.state_bytes / 1024:.1f} KiB "
            f"(arenas {self.arena_state_bytes / 1024:.1f}, "
            f"params {self.param_bytes / 1024:.1f}) · "
            f"backend-reported flooding sweep {flops} FLOPs, {traffic} bytes · "
            + (
                f"empirical {self.calibration['warm_median_seconds'] * 1000:.3f} ms/two_pass run"
                if self.prediction_unit == "two_pass"
                else f"empirical {secs}/{self.prediction_unit or 'uncalibrated unit'}"
            )
        )


def estimate(
    engine: Engine,
    *,
    calibrate: bool = True,
    sample_iters: int = 10,
    state: State | None = None,
    repetitions: int = 5,
    batch_size: int | None = None,
) -> Estimate:
    """Analyze the supplied state (optionally batched) and this engine's compiled schedule."""
    if sample_iters < 1 or repetitions < 1:
        raise ValueError("Use a positive iteration budget and repetition count")
    if state is None:
        state = engine.init_state()
    storage = state_storage(state, batch_size)
    reports = executable_reports(
        engine, state, batch_size=batch_size, time_units=calibrate
    )
    flood = reports["flooding_step"]["cost"] or {}
    seconds = None
    status = None
    calibration = {}
    if calibrate:
        members = (
            [
                jax.tree.map(lambda x: x[i], state)
                for i in range(state.messages.shape[0])
            ]
            if state.messages.ndim > 1
            else None
        )
        if batch_size is not None:
            members = [state] * batch_size

        harden_last = sample_iters // 2 if len(engine.arena.delta_pairs) else 0
        exact = engine._can_run_exact(sample_iters, 0.0, harden_last)
        trust_region = bool(
            engine.arena.gfe_groups
            or engine.arena.gfe_tm_groups
            or engine.arena.hybrid_gfe_groups
            or engine.has_adaptive_gaussian
        )
        sequential = exact or trust_region

        def measure():
            start = time.perf_counter()
            if members is None:
                rows = [engine.run(state, sample_iters)]
            else:
                from cffg.compiled_iteration import controller_for
                from cffg.run_results import finalize_iteration

                if sequential:
                    rows = [engine.run(member, sample_iters) for member in members]
                else:
                    outcomes = controller_for(engine).run_batch(
                        members,
                        n_iters=sample_iters,
                        damp=0.0,
                        tol=None,
                        harden_last=sample_iters // 2
                        if len(engine.arena.delta_pairs)
                        else 0,
                    )
                    rows = [
                        finalize_iteration(o, factor_names=engine._failure_names)
                        for o in outcomes
                    ]
            jax.block_until_ready([r.state for r in rows])
            return time.perf_counter() - start, rows

        first, rows = measure()
        warm = []
        for _ in range(max(5, repetitions)):
            elapsed, rows = measure()
            warm.append(elapsed)
        status = next(
            (
                r.status
                for r in rows
                if r.status in (RunStatus.INFEASIBLE, RunStatus.DIVERGED)
            ),
            rows[0].status,
        )
        count = max(r.iterations for r in rows)
        unit = (
            "two_pass"
            if exact
            else "trust_region_iteration"
            if trust_region
            else "flooding_step"
            if engine.schedule.kind in ("flood", "two_pass")
            else engine.schedule.kind
        )
        calibration = {
            "unit": unit,
            "executed_units": count,
            "first_run_seconds_including_lazy_compilation": first,
            "execution": "single"
            if members is None
            else "sequential_batch"
            if sequential
            else "vmapped_batch",
            "batch_size": storage["batch_size"],
            "warm_seconds": warm,
            "warm_median_seconds": median(warm),
            "status": status.value,
        }
        if (
            not exact
            and count
            and status not in (RunStatus.DIVERGED, RunStatus.INFEASIBLE)
        ):
            seconds = median(warm) / count
    return Estimate(
        arena_bytes=storage["arenas"],
        param_bytes=storage["parameter_bytes"],
        flops_per_sweep=flood.get("flops"),
        bytes_accessed_per_sweep=flood.get("bytes accessed"),
        seconds_per_sweep=seconds,
        calibration_status=status,
        parameter_leaves=storage["parameter_leaves"],
        executables=reports,
        calibration=calibration,
        declared_work={
            g.name: {"members": len(g.members), **g.kernel.binding.work}
            for g in engine.arena.gauss_groups
            if g.kernel is not None and g.kernel.binding.work
        },
    )
