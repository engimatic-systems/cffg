"""Canonical environments and their CFFG templates (brief item 5)."""

from cffg.models.factored import (
    CouplingSpec,
    ObservationSpec,
    StateBlockSpec,
    build_factored_corridor,
    build_factored_rollout,
)
from cffg.models.rslds import RSLDSLearning, rslds, vague_rslds_learning
from cffg.models.rslds_fit import (
    PCAARHMMInitializer,
    RSLDSFitPriors,
    RSLDSFitResult,
    RSLDSInitialization,
    RSLDSInitializer,
    fit_rslds,
    vague_rslds_fit_priors,
)

__all__ = [
    "CouplingSpec",
    "ObservationSpec",
    "PCAARHMMInitializer",
    "RSLDSFitPriors",
    "RSLDSFitResult",
    "RSLDSInitialization",
    "RSLDSInitializer",
    "RSLDSLearning",
    "StateBlockSpec",
    "build_factored_corridor",
    "build_factored_rollout",
    "fit_rslds",
    "rslds",
    "vague_rslds_fit_priors",
    "vague_rslds_learning",
]
