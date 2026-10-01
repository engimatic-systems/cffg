"""Supported model builders, environments, fitting, and learning helpers."""

from cffg.models.factored import (
    CouplingSpec,
    ObservationSpec,
    StateBlockSpec,
    build_factored_corridor,
    build_factored_rollout,
)
from cffg.models.gridworld import CorridorEnv, build_gridworld
from cffg.models.kalman import lgssm
from cffg.models.rollout import build_rollout
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
from cffg.models.tmaze import build_tmaze

__all__ = [
    "CorridorEnv",
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
    "build_gridworld",
    "build_rollout",
    "build_tmaze",
    "fit_rslds",
    "lgssm",
    "rslds",
    "vague_rslds_fit_priors",
    "vague_rslds_learning",
]
