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
    RSLDSFitPriors,
    RSLDSFitPhase,
    RSLDSFitResult,
    RSLDSInferenceAttempt,
    fit_rslds,
    vague_rslds_fit_priors,
)
from cffg.models.rslds_init import (
    PCAARHMMInitializer,
    RSLDSInitialization,
    RSLDSInitializer,
)

__all__ = [
    "CouplingSpec",
    "ObservationSpec",
    "PCAARHMMInitializer",
    "RSLDSFitPriors",
    "RSLDSFitPhase",
    "RSLDSFitResult",
    "RSLDSInitialization",
    "RSLDSInferenceAttempt",
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
