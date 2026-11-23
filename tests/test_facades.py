"""Public package catalogues point to their canonical implementation owners."""

import cffg.models as models
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
    RSLDSFitPhase,
    RSLDSFitPriors,
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


def test_model_facade_is_the_complete_supported_builder_catalogue():
    expected = {
        "CorridorEnv": CorridorEnv,
        "CouplingSpec": CouplingSpec,
        "ObservationSpec": ObservationSpec,
        "PCAARHMMInitializer": PCAARHMMInitializer,
        "RSLDSFitPriors": RSLDSFitPriors,
        "RSLDSFitPhase": RSLDSFitPhase,
        "RSLDSFitResult": RSLDSFitResult,
        "RSLDSInitialization": RSLDSInitialization,
        "RSLDSInferenceAttempt": RSLDSInferenceAttempt,
        "RSLDSInitializer": RSLDSInitializer,
        "RSLDSLearning": RSLDSLearning,
        "StateBlockSpec": StateBlockSpec,
        "build_factored_corridor": build_factored_corridor,
        "build_factored_rollout": build_factored_rollout,
        "build_gridworld": build_gridworld,
        "build_rollout": build_rollout,
        "build_tmaze": build_tmaze,
        "fit_rslds": fit_rslds,
        "lgssm": lgssm,
        "rslds": rslds,
        "vague_rslds_fit_priors": vague_rslds_fit_priors,
        "vague_rslds_learning": vague_rslds_learning,
    }

    assert models.__all__ == list(expected)
    for name, owner in expected.items():
        assert getattr(models, name) is owner
