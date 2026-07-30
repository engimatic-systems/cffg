"""Canonical environments and their CFFG templates (brief item 5)."""

from cffg.models.factored import (
    CouplingSpec,
    ObservationSpec,
    StateBlockSpec,
    TransitionNoveltySpec,
    build_factored_corridor,
    build_factored_rollout,
)

__all__ = [
    "CouplingSpec",
    "ObservationSpec",
    "StateBlockSpec",
    "TransitionNoveltySpec",
    "build_factored_corridor",
    "build_factored_rollout",
]
