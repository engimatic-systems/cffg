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
from dataclasses import dataclass

import jax

from cffg.engine import Engine, RunStatus, State


@dataclass
class Estimate:
    arena_bytes: dict[str, int]
    param_bytes: int
    flops_per_sweep: float | None
    bytes_accessed_per_sweep: float | None
    seconds_per_sweep: float | None
    calibration_status: RunStatus | None

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

    def predict_seconds(self, n_iters: int) -> float | None:
        if self.seconds_per_sweep is None:
            return None
        return n_iters * self.seconds_per_sweep

    def summary(self) -> str:
        flops = f"{self.flops_per_sweep:.3g}" if self.flops_per_sweep else "n/a"
        secs = f"{self.seconds_per_sweep * 1e3:.3f} ms" if self.seconds_per_sweep else "n/a"
        return (
            f"state {self.state_bytes / 1024:.1f} KiB "
            f"(arenas {self.arena_state_bytes / 1024:.1f}, "
            f"params {self.param_bytes / 1024:.1f}) · "
            f"{flops} flops/sweep · {secs}/sweep"
        )


def _cost_analysis(engine: Engine, state: State) -> tuple[float | None, float | None]:
    try:
        lowered = jax.jit(lambda m, p: engine._step(m, p, 0.0, True)).lower(
            state.messages, state.params
        )
        cost = lowered.compile().cost_analysis()
        if isinstance(cost, (list, tuple)):
            cost = cost[0] if cost else None
        if not cost:
            return None, None
        return cost.get("flops"), cost.get("bytes accessed")
    except Exception:
        return None, None


def estimate(engine: Engine, *, calibrate: bool = True, sample_iters: int = 10) -> Estimate:
    state = engine.init_state()
    arena_bytes = {
        name: int(getattr(state, name).nbytes)
        for name in (
            "messages", "g_eta", "g_lam", "g_obs", "g_mask",
            "ef_eta", "ef_obs", "ef_mask",
        )
    }
    param_bytes = int(sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(state.params)))
    flops, bytes_accessed = _cost_analysis(engine, state)

    seconds = None
    if calibrate:
        warm = engine.step(state)  # compile + first dispatch
        warm.messages.block_until_ready()
        t0 = time.perf_counter()
        result = engine.run(warm, sample_iters)
        result.state.messages.block_until_ready()
        seconds = (time.perf_counter() - t0) / sample_iters
        calibration_status = result.status
    else:
        calibration_status = None

    return Estimate(
        arena_bytes=arena_bytes,
        param_bytes=param_bytes,
        flops_per_sweep=flops,
        bytes_accessed_per_sweep=bytes_accessed,
        seconds_per_sweep=seconds,
        calibration_status=calibration_status,
    )
