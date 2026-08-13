"""Canonical environments and their CFFG templates (brief item 5)."""

from cffg.models.factored import (
    CouplingSpec,
    ObservationSpec,
    StateBlockSpec,
    build_factored_corridor,
    build_factored_rollout,
)
from cffg.models.rslds import RSLDSLearning, rslds, vague_rslds_learning

__all__ = [
    "CouplingSpec",
    "ObservationSpec",
    "RSLDSLearning",
    "StateBlockSpec",
    "build_factored_corridor",
    "build_factored_rollout",
    "rslds",
    "vague_rslds_learning",
]
