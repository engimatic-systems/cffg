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
    RSLDSFitWork,
    RSLDSInferenceAttempt,
    fit_rslds,
    vague_rslds_fit_priors,
)
from cffg.models.rslds_init import (
    PCAARHMMInitializer,
    RSLDSInitialization,
    RSLDSInitializer,
)
from cffg.models.rslds_stability import (
    RSLDSStabilityEvidence,
    RSLDSStabilityOutcome,
    RSLDSStabilityPolicy,
    encode_stability_evidence,
    evaluate_rslds_stability,
    validate_stability_evidence,
)

__all__ = [
    "CouplingSpec",
    "ObservationSpec",
    "PCAARHMMInitializer",
    "RSLDSFitPriors",
    "RSLDSFitPhase",
    "RSLDSFitResult",
    "RSLDSFitWork",
    "RSLDSInitialization",
    "RSLDSInferenceAttempt",
    "RSLDSInitializer",
    "RSLDSLearning",
    "RSLDSStabilityEvidence",
    "RSLDSStabilityOutcome",
    "RSLDSStabilityPolicy",
    "StateBlockSpec",
    "build_factored_corridor",
    "build_factored_rollout",
    "encode_stability_evidence",
    "evaluate_rslds_stability",
    "fit_rslds",
    "rslds",
    "vague_rslds_fit_priors",
    "vague_rslds_learning",
    "validate_stability_evidence",
]
