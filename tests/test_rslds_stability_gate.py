"""Fail-closed Foster eligibility at the observed-only rSLDS boundary."""

from __future__ import annotations

import dataclasses
import json

import jax
import numpy as np
import pytest

from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus
from cffg.learn import Learnables
from cffg.models import (
    PCAARHMMInitializer,
    RSLDSInitialization,
    RSLDSStabilityOutcome,
    RSLDSStabilityPolicy,
    encode_stability_evidence,
    evaluate_rslds_stability,
    fit_rslds,
    restore_stability_gated_learning,
    runtime_rslds_point,
    validate_stability_evidence,
    vague_rslds_fit_priors,
)
from cffg.models.coordinate_fit import CoordinateResidual
from cffg.models.rslds import rslds
from cffg.models.rslds_fit import _install
from cffg.models.rslds_stability import POINT_FIELDS
from cffg.switching_stability import (
    StabilityStatus,
    common_quadratic_certificate,
)


POLICY = RSLDSStabilityPolicy(
    foster_rate=0.8,
    tail_radius=2.0,
    compact_splits=2,
    maximum_tail_cones=8,
)


@pytest.mark.parametrize(
    "arguments",
    (
        (np.nan, 2.0, 2, 8),
        (0.8, 0.0, 2, 8),
        (0.8, 2.0, 0, 8),
        (0.8, 2.0, 2, False),
    ),
)
def test_policy_requires_explicit_proper_settings(arguments):
    with pytest.raises(ValueError):
        RSLDSStabilityPolicy(*arguments)


def _observations(steps: int = 8, dtype=np.float64) -> np.ndarray:
    time = np.linspace(0.0, 2.5 * np.pi, steps, dtype=dtype)
    return np.stack(
        (
            np.sin(time) + 0.2 * np.sin(3.0 * time) + 0.4,
            np.cos(time) - 0.15 * np.sin(2.0 * time) - 0.2,
        ),
        axis=1,
    ).astype(dtype)


def _point(
    dynamics=(0.4, 0.5),
    *,
    dtype=np.float64,
    regimes: int = 2,
) -> dict[str, np.ndarray]:
    dimension = 1
    dynamics = np.asarray(dynamics, dtype=dtype)
    if dynamics.size == 1:
        dynamics = np.full(regimes, dynamics.item(), dtype=dtype)
    return {
        "initial_regime_probability": np.concatenate(
            (
                np.ones(1, dtype=dtype),
                np.zeros(regimes - 1, dtype=dtype),
            )
        ),
        "dynamics_matrix": dynamics.reshape(regimes, dimension, dimension),
        "dynamics_offset": np.linspace(
            0.01, -0.02, regimes, dtype=dtype
        ).reshape(regimes, dimension),
        "dynamics_covariance": np.broadcast_to(
            np.eye(dimension, dtype=dtype) * 0.02,
            (regimes, dimension, dimension),
        ).copy(),
        "recurrence_weights": np.zeros(
            (regimes, regimes - 1, dimension), dtype=dtype
        ),
        "recurrence_offsets": np.zeros((regimes, regimes - 1), dtype=dtype),
        "emission_matrix": np.ones((1, dimension), dtype=dtype),
        "emission_offset": np.zeros(1, dtype=dtype),
        "emission_covariance": np.eye(1, dtype=dtype) * 0.1,
    }


def _engine(points: dict[str, np.ndarray], steps: int = 3) -> Engine:
    dimension = points["dynamics_matrix"].shape[-1]
    template, arrays = rslds(
        n_steps=steps,
        **points,
        initial_state_mean=np.zeros(
            dimension, dtype=points["dynamics_matrix"].dtype
        ),
        initial_state_covariance=np.eye(
            dimension, dtype=points["dynamics_matrix"].dtype
        ),
    )
    return Engine(compile_template(template, arrays=arrays))


@dataclasses.dataclass(frozen=True)
class _FixedInitializer:
    value: RSLDSInitialization

    def __call__(self, observations, *, regimes, latent_dimension, key):
        del observations, regimes, latent_dimension, key
        return self.value


def _fit(
    initializer,
    *,
    dtype=np.float64,
    outer_iterations: int = 1,
    outer_tolerance: float = 2.0,
):
    observations = _observations(dtype=dtype)
    return fit_rslds(
        observations,
        regimes=2,
        latent_dimension=1,
        initializer=initializer,
        priors=vague_rslds_fit_priors(2, 1, 2),
        max_outer_iterations=outer_iterations,
        max_inner_iterations=100,
        inner_tolerance=1e-5,
        outer_tolerance=outer_tolerance,
        key=jax.random.key(7),
        stability_policy=POLICY,
    )


def test_truth_free_policy_certifies_initializer_and_nontrivial_proposal():
    base = _point()
    engine = _engine(base)
    initializer = evaluate_rslds_stability(
        engine, base, POLICY, owner="initializer", coordinate="coordinate/0"
    )
    proposal = evaluate_rslds_stability(
        engine,
        _point((0.45, 0.55)),
        POLICY,
        owner="proposal",
        coordinate="coordinate/1",
    )

    assert initializer.outcome is RSLDSStabilityOutcome.INITIAL_CERTIFIED
    assert proposal.outcome is RSLDSStabilityOutcome.PROPOSAL_ACCEPTED
    assert initializer.accepted and proposal.accepted
    assert initializer.certificate.status is StabilityStatus.SATISFIED
    assert proposal.certificate.status is StabilityStatus.SATISFIED
    assert initializer.dtype == np.dtype(np.float64).str
    assert [row["name"] for row in initializer.point_identity] == list(POINT_FIELDS)
    assert len(initializer.statistic_ownership) == 4


def test_identity_metric_retains_multidimensional_recurrence_advantage():
    regimes = 5
    dimension = 2
    dtype = np.dtype(np.float64)
    matrix = np.stack(
        [np.eye(dimension, dtype=dtype) * value for value in (0.4,) * 4 + (1.2,)]
    )
    gate_weights = np.asarray(
        ((-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0)),
        dtype=dtype,
    )
    point = {
        "initial_regime_probability": np.full(regimes, 1.0 / regimes, dtype=dtype),
        "dynamics_matrix": matrix,
        "dynamics_offset": np.zeros((regimes, dimension), dtype=dtype),
        "dynamics_covariance": np.broadcast_to(
            np.eye(dimension, dtype=dtype) * 0.01,
            (regimes, dimension, dimension),
        ).copy(),
        "recurrence_weights": np.broadcast_to(
            gate_weights,
            (regimes, regimes - 1, dimension),
        ).copy(),
        "recurrence_offsets": np.full((regimes, regimes - 1), 4.0, dtype=dtype),
        "emission_matrix": np.eye(dimension, dtype=dtype),
        "emission_offset": np.zeros(dimension, dtype=dtype),
        "emission_covariance": np.eye(dimension, dtype=dtype) * 0.1,
    }
    policy = RSLDSStabilityPolicy(0.5, 4.0, 2, 8)

    common = common_quadratic_certificate(
        matrix,
        point["dynamics_offset"],
        point["dynamics_covariance"],
        np.eye(dimension, dtype=dtype),
    )
    recurrence = evaluate_rslds_stability(
        _engine(point),
        point,
        policy,
        owner="proposal",
        coordinate="coordinate/1",
    )

    assert common.status is StabilityStatus.VIOLATED
    assert recurrence.certificate.status is StabilityStatus.SATISFIED
    assert recurrence.outcome is RSLDSStabilityOutcome.PROPOSAL_ACCEPTED
    assert recurrence.certificate.compact_box_count == 4
    assert np.any(recurrence.certificate.exponentially_suppressed)


def test_unverified_contraction_ceiling_is_rejected_not_safe():
    base = _point()
    evidence = evaluate_rslds_stability(
        _engine(base),
        _point((1.1, 0.5)),
        POLICY,
        owner="proposal",
        coordinate="coordinate/1",
    )

    assert evidence.outcome is RSLDSStabilityOutcome.CERTIFICATE_REJECTED
    assert evidence.certificate.status is StabilityStatus.VIOLATED
    assert not evidence.accepted
    assert np.max(evidence.certificate.mode_contraction) == POLICY.foster_rate
    assert "neither contractive nor exponentially suppressed" in (
        evidence.certificate.reason
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_checkpoint_authenticates_exact_point_policy_and_d10_authority(dtype):
    point = _point(dtype=dtype)
    engine = _engine(point)
    evidence = evaluate_rslds_stability(
        engine, point, POLICY, owner="initializer", coordinate="coordinate/0"
    )
    checkpoint = encode_stability_evidence(evidence)

    restored = validate_stability_evidence(
        checkpoint,
        engine=engine,
        points=point,
        policy=POLICY,
        owner="initializer",
        coordinate="coordinate/0",
    )

    assert restored.outcome is RSLDSStabilityOutcome.INITIAL_CERTIFIED
    assert restored.dtype == np.dtype(dtype).str
    assert point["initial_regime_probability"][1] == 0.0


def test_checkpoint_rejects_reorder_tamper_d9_point_and_policy_mismatch():
    point = _point()
    engine = _engine(point)
    evidence = evaluate_rslds_stability(
        engine, point, POLICY, owner="initializer", coordinate="coordinate/0"
    )
    checkpoint = encode_stability_evidence(evidence)
    document = json.loads(checkpoint.item())

    mutations = []
    reordered = dict(document)
    reordered["point_identity"] = list(reversed(reordered["point_identity"]))
    mutations.append(reordered)
    d9 = dict(document)
    d9["d10_identity"] = {
        **d9["d10_identity"],
        "executable_identity": "cffg.recurrence.d9-structured-pair/1.0.0",
    }
    mutations.append(d9)
    hidden = dict(document)
    hidden["unregistered"] = True
    mutations.append(hidden)
    bound = json.loads(checkpoint.item())
    bound["certificate"]["additive_bound"] += 1.0
    mutations.append(bound)
    ownership = json.loads(checkpoint.item())
    ownership["statistic_ownership"] = list(
        reversed(ownership["statistic_ownership"])
    )
    mutations.append(ownership)
    for mutation in mutations:
        encoded = np.asarray(
            json.dumps(mutation, sort_keys=True, separators=(",", ":"))
        )
        with pytest.raises(ValueError, match="does not match"):
            validate_stability_evidence(
                encoded,
                engine=engine,
                points=point,
                policy=POLICY,
                owner="initializer",
                coordinate="coordinate/0",
            )

    noncanonical = np.asarray(json.dumps(document))
    with pytest.raises(ValueError, match="canonically encoded"):
        validate_stability_evidence(
            noncanonical,
            engine=engine,
            points=point,
            policy=POLICY,
            owner="initializer",
            coordinate="coordinate/0",
        )
    changed = {name: value.copy() for name, value in point.items()}
    changed["emission_offset"][0] = 0.25
    with pytest.raises(ValueError, match="does not match"):
        validate_stability_evidence(
            checkpoint,
            engine=engine,
            points=changed,
            policy=POLICY,
            owner="initializer",
            coordinate="coordinate/0",
        )

    template, arrays = rslds(
        n_steps=3,
        **point,
        initial_state_mean=np.zeros(1),
        initial_state_covariance=np.eye(1),
    )
    legacy = dataclasses.replace(
        template,
        learnables={
            name: declaration
            for name, declaration in template.learnables.items()
            if name != "emission"
        },
    )
    legacy_engine = Engine(compile_template(legacy, arrays=arrays))
    with pytest.raises(ValueError, match="current D10 dispatch"):
        validate_stability_evidence(
            checkpoint,
            engine=legacy_engine,
            points=point,
            policy=POLICY,
            owner="initializer",
            coordinate="coordinate/0",
        )
    changed_policy = dataclasses.replace(POLICY, tail_radius=3.0)
    with pytest.raises(ValueError, match="does not match"):
        validate_stability_evidence(
            checkpoint,
            engine=engine,
            points=point,
            policy=changed_policy,
            owner="initializer",
            coordinate="coordinate/0",
        )


def test_checkpoint_rejects_json_type_and_signed_zero_tampering():
    point = _point()
    engine = _engine(point)
    evidence = evaluate_rslds_stability(
        engine, point, POLICY, owner="initializer", coordinate="coordinate/0"
    )
    checkpoint = encode_stability_evidence(evidence)

    mutations = []
    accepted = json.loads(checkpoint.item())
    accepted["accepted"] = 1
    mutations.append(accepted)
    integer = json.loads(checkpoint.item())
    integer["policy"]["compact_splits"] = 2.0
    mutations.append(integer)
    shape = json.loads(checkpoint.item())
    shape["point_identity"][0]["shape"][0] = 2.0
    mutations.append(shape)
    signed_zero = json.loads(checkpoint.item())
    signed_zero["certificate"]["decay_rate"]["values"][0][0][0] = -0.0
    mutations.append(signed_zero)

    for mutation in mutations:
        encoded = np.asarray(
            json.dumps(mutation, sort_keys=True, separators=(",", ":"))
        )
        with pytest.raises(ValueError, match="does not match"):
            validate_stability_evidence(
                encoded,
                engine=engine,
                points=point,
                policy=POLICY,
                owner="initializer",
                coordinate="coordinate/0",
            )


@pytest.mark.parametrize(
    ("mutation", "status"),
    (
        ("nonfinite", StabilityStatus.DIVERGED),
        ("improper", StabilityStatus.INFEASIBLE),
        ("reordered", StabilityStatus.INFEASIBLE),
    ),
)
def test_complete_point_is_fail_closed(mutation, status):
    base = _point()
    point = {name: value.copy() for name, value in base.items()}
    if mutation == "nonfinite":
        point["emission_matrix"][0, 0] = np.nan
    elif mutation == "improper":
        point["dynamics_covariance"][0, 0, 0] = 0.0
    else:
        point = {name: point[name] for name in reversed(POINT_FIELDS)}

    evidence = evaluate_rslds_stability(
        _engine(base), point, POLICY, owner="proposal", coordinate="coordinate/1"
    )

    assert evidence.certificate.status is status
    assert not evidence.accepted


def test_k8_certificate_has_bounded_structural_evidence():
    point = _point((0.2,), regimes=8)
    evidence = evaluate_rslds_stability(
        _engine(point),
        point,
        POLICY,
        owner="initializer",
        coordinate="coordinate/0",
    )

    assert evidence.accepted
    assert evidence.certificate.orthant_signs.shape == (2, 1)
    assert evidence.certificate.mode_contraction.shape == (2, 8, 8)
    assert evidence.certificate.compact_box_count == 2


@pytest.mark.parametrize("tier", ("native", "shared"))
def test_gated_fit_certifies_both_initializer_tiers_and_accepts_progress(tier):
    observations = _observations()
    native = PCAARHMMInitializer()
    initialized = native(
        observations,
        regimes=2,
        latent_dimension=1,
        key=jax.random.key(7),
    )
    initializer = native if tier == "native" else _FixedInitializer(initialized)

    result = _fit(initializer)

    assert result.gate_termination is None
    assert [phase.outcome for phase in result.stability_phases] == [
        RSLDSStabilityOutcome.INITIAL_CERTIFIED,
        RSLDSStabilityOutcome.PROPOSAL_ACCEPTED,
    ]
    assert result.work.d10_map_evaluations == 1
    assert result.work.certificate_attempts == 2
    assert result.work.accepted_proposals == 1
    assert result.work.rejected_proposals == 0
    assert result.residuals == result.evaluated_residuals
    assert result.rejected_residuals == ()
    assert result.phases[0].stability_evidence is result.stability_phases[1]
    assert all(
        not np.array_equal(result.parameters[name], initialized.arrays()[name])
        for name in POINT_FIELDS
    )


def test_float32_gate_certifies_and_returns_the_exact_installed_runtime_point():
    with jax.enable_x64(False):
        result = _fit(PCAARHMMInitializer(), dtype=np.float32)
        returned = runtime_rslds_point(result.engine, result.state)

    assert result.gate_termination is None
    assert result.stability_phases[-1].outcome is (
        RSLDSStabilityOutcome.PROPOSAL_ACCEPTED
    )
    assert all(
        evidence.dtype == np.dtype(np.float32).str
        for evidence in result.stability_phases
    )
    for name, value in result.parameters.items():
        assert value.dtype == np.float32
        np.testing.assert_array_equal(value, returned[name])
    restored = validate_stability_evidence(
        encode_stability_evidence(result.stability_phases[-1]),
        engine=result.engine,
        points=result.parameters,
        policy=POLICY,
        owner="proposal",
        coordinate="coordinate/1",
    )
    assert restored.accepted


def test_float32_installed_rounding_can_flip_a_boundary_certificate():
    coefficient = np.sqrt(0.8) - 1e-9
    policy = dataclasses.replace(POLICY, foster_rate=0.8)
    point64 = _point((coefficient, coefficient), dtype=np.float64)
    engine64 = _engine(point64)
    runtime64 = runtime_rslds_point(engine64, engine64.init_state())
    evidence64 = evaluate_rslds_stability(
        engine64,
        runtime64,
        policy,
        owner="proposal",
        coordinate="coordinate/1",
    )
    with jax.enable_x64(False):
        point32 = _point((coefficient, coefficient), dtype=np.float32)
        engine32 = _engine(point32)
        runtime32 = runtime_rslds_point(engine32, engine32.init_state())
        evidence32 = evaluate_rslds_stability(
            engine32,
            runtime32,
            policy,
            owner="proposal",
            coordinate="coordinate/1",
        )

    assert evidence64.certificate.status is StabilityStatus.SATISFIED
    assert evidence32.certificate.status is StabilityStatus.VIOLATED
    assert runtime64["dynamics_matrix"].dtype == np.float64
    assert runtime32["dynamics_matrix"].dtype == np.float32
    assert float(runtime32["dynamics_matrix"][0, 0, 0]) != coefficient


def test_current_certified_rejected_proposal_is_terminal_and_atomic(monkeypatch):
    initializer = PCAARHMMInitializer()
    runtime_calls = []
    original_runtime_point = runtime_rslds_point

    def unstable_candidate(engine, state):
        points = original_runtime_point(engine, state)
        if runtime_calls:
            points["dynamics_matrix"][:] = np.asarray(
                [[[1.1]], [[0.5]]], dtype=points["dynamics_matrix"].dtype
            )
        runtime_calls.append(True)
        return points

    monkeypatch.setattr(
        "cffg.models.rslds_fit.runtime_rslds_point", unstable_candidate
    )

    result = _fit(initializer)

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.inner_status is RunStatus.CONVERGED
    assert result.gate_termination is RSLDSStabilityOutcome.CERTIFICATE_REJECTED
    assert result.work.d10_map_evaluations == 1
    assert result.work.certificate_attempts == 2
    assert result.work.accepted_proposals == 0
    assert result.work.rejected_proposals == 1
    assert result.residuals == ()
    assert len(result.evaluated_residuals) == 1
    assert result.rejected_residuals == result.evaluated_residuals
    assert len(result.free_energies) == 1
    assert not result.phases[-1].coordinate_replaced
    assert result.phases[-1].stability_evidence is result.stability_phases[-1]
    initial_runtime = original_runtime_point(result.engine, result.state)
    for name, value in result.parameters.items():
        np.testing.assert_array_equal(value, initial_runtime[name])


def test_rejection_after_one_hit_cannot_count_as_progress_or_convergence(monkeypatch):
    initializer = PCAARHMMInitializer()
    original_runtime_point = runtime_rslds_point
    runtime_calls = 0
    proposed = []

    def candidate_sequence(engine, state):
        nonlocal runtime_calls
        points = original_runtime_point(engine, state)
        runtime_calls += 1
        if runtime_calls == 1:
            return points
        if proposed:
            points["dynamics_matrix"][:] = np.asarray(
                [[[1.1]], [[0.5]]], dtype=points["dynamics_matrix"].dtype
            )
        proposed.append({name: value.copy() for name, value in points.items()})
        return points

    residual_values = iter((0.05, 0.2))

    def residual_sequence(*_):
        value = next(residual_values)
        return CoordinateResidual(
            maximum=value,
            root_mean_square=value,
            per_family={"forced": value},
            per_array={"forced/value": value},
            vector=(value,),
        )

    monkeypatch.setattr(
        "cffg.models.rslds_fit.runtime_rslds_point", candidate_sequence
    )
    monkeypatch.setattr("cffg.models.rslds_fit.coordinate_residual", residual_sequence)

    result = _fit(initializer, outer_iterations=2, outer_tolerance=0.1)

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.certified_coordinate is None
    assert result.gate_termination is RSLDSStabilityOutcome.CERTIFICATE_REJECTED
    assert result.residuals == pytest.approx((0.05,))
    assert result.evaluated_residuals == pytest.approx((0.05, 0.2))
    assert result.rejected_residuals == pytest.approx((0.2,))
    assert result.work.d10_map_evaluations == 2
    assert result.work.certificate_attempts == 3
    assert result.work.accepted_proposals == 1
    assert result.work.rejected_proposals == 1
    for name, value in result.parameters.items():
        np.testing.assert_array_equal(value, proposed[0][name])


def test_uncertified_initializer_stops_before_inner_solve(monkeypatch):
    observations = _observations()
    native = PCAARHMMInitializer()
    initialized = native(
        observations,
        regimes=2,
        latent_dimension=1,
        key=jax.random.key(7),
    )
    unstable = dataclasses.replace(
        initialized,
        dynamics_matrix=np.asarray([[[1.1]], [[1.1]]], dtype=observations.dtype),
        recurrence_weights=np.zeros((2, 1, 1), dtype=observations.dtype),
        recurrence_offsets=np.zeros((2, 1), dtype=observations.dtype),
    )
    runs = []
    monkeypatch.setattr(Engine, "run", lambda *args, **kwargs: runs.append(True))

    result = _fit(_FixedInitializer(unstable))

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.inner_status is None
    assert result.gate_termination is (
        RSLDSStabilityOutcome.INITIAL_CERTIFICATE_REJECTED
    )
    assert result.iterations == 0
    assert result.phases == ()
    assert len(result.stability_phases) == 1
    assert result.work.certificate_attempts == 1
    assert result.work.d10_map_evaluations == 0
    assert result.work.accepted_proposals == 0
    assert runs == []


def test_gated_restore_authenticates_before_exposing_a_pending_write():
    observations = np.linspace(-0.2, 0.2, 3, dtype=np.float64)[:, None]
    point = _point()
    engine = _engine(point)
    priors = vague_rslds_fit_priors(2, 1, 1)
    learning = Learnables(engine, priors=priors.learning_priors())
    current = engine.bind_data(
        engine.init_state(),
        {f"y{step}": row for step, row in enumerate(observations)},
    )
    inference = engine.run(current, 200, tol=1e-5)
    assert inference.status is RunStatus.CONVERGED
    learning.replace(engine, inference)
    staged = _install(engine, observations, learning)
    staged_point = runtime_rslds_point(engine, staged)
    evidence = evaluate_rslds_stability(
        engine,
        staged_point,
        POLICY,
        owner="proposal",
        coordinate="coordinate/1",
    )
    assert evidence.accepted
    learning_checkpoint = learning.state_dict()
    checkpoint_snapshot = {
        name: value.copy() for name, value in learning_checkpoint.items()
    }

    restored = restore_stability_gated_learning(
        observations,
        engine=engine,
        priors=priors,
        learning_checkpoint=learning_checkpoint,
        stability_checkpoint=encode_stability_evidence(evidence),
        stability_policy=POLICY,
        coordinate="coordinate/1",
    )

    for name, value in restored.parameters.items():
        np.testing.assert_array_equal(value, staged_point[name])
    for name, value in checkpoint_snapshot.items():
        np.testing.assert_array_equal(learning_checkpoint[name], value)

    with pytest.raises(ValueError, match="checkpoint"):
        restore_stability_gated_learning(
            observations,
            engine=engine,
            priors=priors,
            learning_checkpoint=learning_checkpoint,
            stability_checkpoint=None,
            stability_policy=POLICY,
            coordinate="coordinate/1",
        )
    with pytest.raises(ValueError, match="does not match"):
        restore_stability_gated_learning(
            observations,
            engine=engine,
            priors=priors,
            learning_checkpoint=learning_checkpoint,
            stability_checkpoint=encode_stability_evidence(evidence),
            stability_policy=dataclasses.replace(POLICY, tail_radius=3.0),
            coordinate="coordinate/1",
        )


def test_default_fit_remains_ungated():
    observations = _observations()
    result = fit_rslds(
        observations,
        regimes=2,
        latent_dimension=1,
        initializer=PCAARHMMInitializer(),
        priors=vague_rslds_fit_priors(2, 1, 2),
        max_outer_iterations=1,
        max_inner_iterations=100,
        inner_tolerance=1e-5,
        outer_tolerance=2.0,
        key=jax.random.key(7),
    )

    assert result.stability_phases == ()
    assert result.work.certificate_attempts == 0
    assert result.work.accepted_proposals == 1
