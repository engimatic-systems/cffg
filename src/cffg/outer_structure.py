"""Immutable structural inputs to outer charts, without runtime parameters."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from cffg._immutable import FrozenArray, ImmutableRecord
from cffg.compiled import Arena, EdgeInfo, GoalParameterPlan


@dataclass(frozen=True)
class OuterStructure(ImmutableRecord):
    edges: Mapping[str, EdgeInfo]
    goal_parameter_plans: tuple[GoalParameterPlan, ...]
    message_starts: FrozenArray
    g_eta_starts: FrozenArray
    g_lam_starts: FrozenArray

    @classmethod
    def from_arena(cls, arena: Arena) -> "OuterStructure":
        return cls(
            arena.edges,
            arena.goal_parameter_plans,
            arena.message_starts,
            arena.g_eta_starts,
            arena.g_lam_starts,
        )

    def message_indices(self, slots: np.ndarray | int, card: int) -> np.ndarray:
        base = self.message_starts[np.asarray(slots, dtype=np.int64)]
        return base[..., None] + np.arange(card, dtype=np.int64)

    def g_eta_indices(self, slots: np.ndarray | int, dim: int) -> np.ndarray:
        base = self.g_eta_starts[np.asarray(slots, dtype=np.int64)]
        return base[..., None] + np.arange(dim, dtype=np.int64)

    def g_lam_indices(self, slots: np.ndarray | int, dim: int) -> np.ndarray:
        base = self.g_lam_starts[np.asarray(slots, dtype=np.int64)]
        block = np.arange(dim * dim, dtype=np.int64).reshape(dim, dim)
        return base[..., None, None] + block
