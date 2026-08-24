"""Fail-closed Foster eligibility evidence for observed-only rSLDS fitting."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import json
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from cffg import gauss
from cffg.learning_checkpoint import (
    D10_COORDINATE_IDENTITY,
    D10_DYNAMICS_STATISTIC_IDENTITY,
    D10_EMISSION_STATISTIC_IDENTITY,
    D10_EXECUTABLE_IDENTITY,
    D10_INITIAL_STATISTIC_IDENTITY,
    D10_OBJECTIVE_IDENTITY,
    D10_RECURRENCE_STATISTIC_IDENTITY,
    array_sha256,
    d10_checkpoint_identity,
    learning_statistic_identity,
)
from cffg.switching_stability import (
    RecurrenceCertificateEvidence,
    StabilityStatus,
    recurrence_stability_certificate,
)
from cffg.structured_hybrid_runtime import component_model


RSLDS_STABILITY_GATE_IDENTITY = "cffg.rslds.foster-proposal-gate/1.0.0"
RSLDS_STABILITY_GATE_CHECKPOINT_VERSION = "1.0.0"
IDENTITY_LYAPUNOV_POLICY = "cffg.rslds.identity-mode-lyapunov/1.0.0"

POINT_FIELDS = (
    "initial_regime_probability",
    "dynamics_matrix",
    "dynamics_offset",
    "dynamics_covariance",
    "recurrence_weights",
    "recurrence_offsets",
    "emission_matrix",
    "emission_offset",
    "emission_covariance",
)

_EXPECTED_STATISTICS = {
    ("dirichlet", "categorical"): D10_INITIAL_STATISTIC_IDENTITY,
    (
        "component_mniw",
        "gaussian_mixture_linear",
    ): D10_DYNAMICS_STATISTIC_IDENTITY,
    ("pg_weights", "recurrence_link"): D10_RECURRENCE_STATISTIC_IDENTITY,
    ("mniw", "gaussian_linear"): D10_EMISSION_STATISTIC_IDENTITY,
}


class RSLDSStabilityOutcome(str, Enum):
    """Fit-owned outcome at one current-point or proposal gate."""

    INITIAL_CERTIFIED = "initial_certified"
    INITIAL_CERTIFICATE_REJECTED = "initial_certificate_rejected"
    INITIAL_CERTIFICATE_INFEASIBLE = "initial_certificate_infeasible"
    INITIAL_CERTIFICATE_DIVERGED = "initial_certificate_diverged"
    PROPOSAL_ACCEPTED = "proposal_accepted"
    CERTIFICATE_REJECTED = "certificate_rejected"
    CERTIFICATE_INFEASIBLE = "certificate_infeasible"
    CERTIFICATE_DIVERGED = "certificate_diverged"


@dataclass(frozen=True)
class RSLDSStabilityPolicy:
    """Explicit truth-free settings for the recurrence Foster authority.

    Each mode uses the identity Lyapunov matrix. Callers must explicitly
    register every numerical setting; the fitter has no data-tuned defaults.
    """

    foster_rate: float
    tail_radius: float
    compact_splits: int
    maximum_tail_cones: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.foster_rate, (bool, np.bool_))
            or not np.isfinite(self.foster_rate)
            or not 0.0 < self.foster_rate < 1.0
        ):
            raise ValueError("foster_rate must be finite and in (0, 1)")
        if (
            isinstance(self.tail_radius, (bool, np.bool_))
            or not np.isfinite(self.tail_radius)
            or self.tail_radius <= 0.0
        ):
            raise ValueError("tail_radius must be finite and positive")
        for name in ("compact_splits", "maximum_tail_cones"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "foster_rate", float(self.foster_rate))
        object.__setattr__(self, "tail_radius", float(self.tail_radius))

    def document(self) -> dict[str, object]:
        return {
            "lyapunov_policy": IDENTITY_LYAPUNOV_POLICY,
            "foster_rate": self.foster_rate,
            "tail_radius": self.tail_radius,
            "compact_splits": self.compact_splits,
            "maximum_tail_cones": self.maximum_tail_cones,
        }


@dataclass(frozen=True)
class RSLDSStabilityEvidence:
    """Authenticated host evidence for one point eligibility decision."""

    owner: str
    coordinate: str
    outcome: RSLDSStabilityOutcome
    accepted: bool
    dtype: str
    point_identity: tuple[dict[str, object], ...]
    policy: RSLDSStabilityPolicy
    d10_identity: dict[str, object]
    statistic_ownership: tuple[dict[str, object], ...]
    certificate: RecurrenceCertificateEvidence


def _invalid_certificate(status: StabilityStatus, reason: str):
    return RecurrenceCertificateEvidence(status=status, reason=reason)


def _complete_point_error(
    points: Mapping[str, Any],
    d10_identity: Mapping[str, object],
) -> tuple[StabilityStatus, str] | None:
    if tuple(points) != POINT_FIELDS:
        return StabilityStatus.INFEASIBLE, (
            "complete rSLDS point fields are missing, extra, or reordered"
        )
    arrays = {name: np.asarray(points[name]) for name in POINT_FIELDS}
    if any(array.dtype.kind != "f" for array in arrays.values()):
        return StabilityStatus.INFEASIBLE, "complete rSLDS point must be floating"
    if len({array.dtype.str for array in arrays.values()}) != 1:
        return StabilityStatus.INFEASIBLE, (
            "complete rSLDS point arrays must share one dtype"
        )
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        return StabilityStatus.DIVERGED, "complete rSLDS point is non-finite"

    component = d10_identity["component"]
    regimes = int(component["cardinality"])
    dimension = int(component["dimension"])
    observation_dimensions = tuple(component["observation_dimensions"])
    if not observation_dimensions or len(set(observation_dimensions)) != 1:
        return StabilityStatus.INFEASIBLE, (
            "D10 stability gate requires one shared observation dimension"
        )
    observation_dimension = int(observation_dimensions[0])
    shapes = {
        "initial_regime_probability": (regimes,),
        "dynamics_matrix": (regimes, dimension, dimension),
        "dynamics_offset": (regimes, dimension),
        "dynamics_covariance": (regimes, dimension, dimension),
        "recurrence_weights": (regimes, regimes - 1, dimension),
        "recurrence_offsets": (regimes, regimes - 1),
        "emission_matrix": (observation_dimension, dimension),
        "emission_offset": (observation_dimension,),
        "emission_covariance": (observation_dimension, observation_dimension),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            return StabilityStatus.INFEASIBLE, (
                f"complete rSLDS point {name!r} shape does not match D10 topology"
            )
    initial = arrays["initial_regime_probability"]
    probability_tolerance = (
        64.0 * regimes * np.finfo(initial.dtype).eps
    )
    if (
        np.any(initial < 0.0)
        or not np.sum(initial) > 0.0
        or abs(float(np.sum(initial)) - 1.0) > probability_tolerance
    ):
        return StabilityStatus.INFEASIBLE, (
            "initial regime probabilities must be nonnegative and normalized"
        )
    try:
        for covariance in arrays["dynamics_covariance"]:
            tolerance = (
                64.0
                * covariance.shape[0]
                * np.finfo(covariance.dtype).eps
                * max(1.0, float(np.max(np.abs(covariance))))
            )
            if not np.allclose(covariance, covariance.T, atol=tolerance, rtol=0.0):
                raise np.linalg.LinAlgError("asymmetric dynamics covariance")
            np.linalg.cholesky(covariance)
        emission_covariance = arrays["emission_covariance"]
        tolerance = (
            64.0
            * emission_covariance.shape[0]
            * np.finfo(emission_covariance.dtype).eps
            * max(1.0, float(np.max(np.abs(emission_covariance))))
        )
        if not np.allclose(
            emission_covariance,
            emission_covariance.T,
            atol=tolerance,
            rtol=0.0,
        ):
            raise np.linalg.LinAlgError("asymmetric emission covariance")
        np.linalg.cholesky(emission_covariance)
    except np.linalg.LinAlgError:
        return StabilityStatus.INFEASIBLE, (
            "dynamics and emission covariances must be positive definite"
        )
    return None


def _point_identity(points: Mapping[str, Any]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": name,
            "dtype": np.asarray(points[name]).dtype.str,
            "shape": list(np.asarray(points[name]).shape),
            "sha256": array_sha256(points[name]),
        }
        for name in POINT_FIELDS
    )


def _target_document(target) -> dict[str, object]:
    return {
        "blob": target.blob,
        "role": target.role,
        "form": target.form,
        "layout": target.layout,
        "transform": target.transform,
    }


def _d10_authority(engine) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    identity = d10_checkpoint_identity(engine)
    if identity is None:
        raise ValueError("rSLDS stability gate requires current D10 dispatch")
    if (
        identity.get("executable_identity") != D10_EXECUTABLE_IDENTITY
        or identity.get("objective_identity") != D10_OBJECTIVE_IDENTITY
        or identity.get("coordinate_identity") != D10_COORDINATE_IDENTITY
    ):
        raise ValueError("rSLDS stability gate requires current D10 identities")
    plans = sorted(engine.arena.learn_plans, key=lambda plan: plan.name)
    actual = [(plan.family, plan.ftype) for plan in plans]
    if len(plans) != 4 or set(actual) != set(_EXPECTED_STATISTICS):
        raise ValueError(
            "rSLDS stability gate requires exactly one owner for all four D10 families"
        )
    ownership = []
    for plan in plans:
        statistic = learning_statistic_identity(identity, plan.family, plan.ftype)
        if statistic != _EXPECTED_STATISTICS[(plan.family, plan.ftype)]:
            raise ValueError("rSLDS stability gate statistic identity mismatch")
        ownership.append(
            {
                "name": plan.name,
                "family": plan.family,
                "factor_type": plan.ftype,
                "statistic_identity": statistic,
                "sources": list(plan.sources),
                "parameter_blobs": [
                    {"name": name, "shape": list(shape)}
                    for name, shape in sorted(plan.shapes.items())
                ],
                "targets": [
                    _target_document(target)
                    for target in sorted(
                        plan.targets,
                        key=lambda target: (
                            target.blob,
                            target.role,
                            target.form,
                            target.layout,
                            target.transform,
                        ),
                    )
                ],
            }
        )
    return dict(identity), tuple(ownership)


def _outcome(owner: str, status: StabilityStatus) -> RSLDSStabilityOutcome:
    if owner == "initializer":
        return {
            StabilityStatus.SATISFIED: RSLDSStabilityOutcome.INITIAL_CERTIFIED,
            StabilityStatus.VIOLATED: (
                RSLDSStabilityOutcome.INITIAL_CERTIFICATE_REJECTED
            ),
            StabilityStatus.INFEASIBLE: (
                RSLDSStabilityOutcome.INITIAL_CERTIFICATE_INFEASIBLE
            ),
            StabilityStatus.DIVERGED: (
                RSLDSStabilityOutcome.INITIAL_CERTIFICATE_DIVERGED
            ),
        }[status]
    if owner == "proposal":
        return {
            StabilityStatus.SATISFIED: RSLDSStabilityOutcome.PROPOSAL_ACCEPTED,
            StabilityStatus.VIOLATED: RSLDSStabilityOutcome.CERTIFICATE_REJECTED,
            StabilityStatus.INFEASIBLE: RSLDSStabilityOutcome.CERTIFICATE_INFEASIBLE,
            StabilityStatus.DIVERGED: RSLDSStabilityOutcome.CERTIFICATE_DIVERGED,
        }[status]
    raise ValueError("stability evidence owner must be 'initializer' or 'proposal'")


def evaluate_rslds_stability(
    engine,
    points: Mapping[str, Any],
    policy: RSLDSStabilityPolicy,
    *,
    owner: str,
    coordinate: str,
) -> RSLDSStabilityEvidence:
    """Evaluate one complete installed or proposed D10 point."""
    if not isinstance(policy, RSLDSStabilityPolicy):
        raise TypeError("stability policy must be RSLDSStabilityPolicy")
    if not isinstance(coordinate, str) or not coordinate:
        raise ValueError("stability coordinate must be a nonempty string")
    identity, ownership = _d10_authority(engine)
    error = _complete_point_error(points, identity)
    if error is None:
        matrix = np.asarray(points["dynamics_matrix"])
        metric = np.broadcast_to(
            np.eye(matrix.shape[-1], dtype=matrix.dtype), matrix.shape
        ).copy()
        certificate = recurrence_stability_certificate(
            matrix,
            points["dynamics_offset"],
            points["dynamics_covariance"],
            points["recurrence_weights"],
            points["recurrence_offsets"],
            metric,
            foster_rate=policy.foster_rate,
            tail_radius=policy.tail_radius,
            compact_splits=policy.compact_splits,
            maximum_tail_cones=policy.maximum_tail_cones,
        )
    else:
        certificate = _invalid_certificate(*error)
    outcome = _outcome(owner, certificate.status)
    point_identity = _point_identity(points) if tuple(points) == POINT_FIELDS else ()
    dtypes = {row["dtype"] for row in point_identity}
    dtype = next(iter(dtypes)) if len(dtypes) == 1 else "mixed-or-invalid"
    return RSLDSStabilityEvidence(
        owner=owner,
        coordinate=coordinate,
        outcome=outcome,
        accepted=certificate.status is StabilityStatus.SATISFIED,
        dtype=dtype,
        point_identity=point_identity,
        policy=policy,
        d10_identity=identity,
        statistic_ownership=ownership,
        certificate=certificate,
    )


def runtime_rslds_point(engine, state) -> dict[str, np.ndarray]:
    """Materialize the exact D10 point consumed by one staged runtime state."""
    identity, _ = _d10_authority(engine)
    plan = engine.arena.structured_hybrid_plans[0]
    model, input_ok = component_model(
        engine.arena,
        plan,
        state.g_obs,
        state.g_mask,
        state.params,
    )
    if not bool(np.asarray(input_ok)):
        raise ValueError("staged D10 runtime point is improper or non-finite")

    log_initial = np.asarray(model.log_initial)
    initial_probability = np.asarray(jax.nn.softmax(model.log_initial))
    initial_probability = np.where(np.isfinite(log_initial), initial_probability, 0.0)

    topology = plan.topology

    def emission(role: str) -> np.ndarray:
        rows = [
            np.asarray(
                state.params[
                    engine.arena.member_of[name].parameter(role).key
                ][engine.arena.member_of[name].parameter(role).index]
            )
            for name in topology.emission_members
        ]
        if any(not np.array_equal(rows[0], row) for row in rows[1:]):
            raise ValueError("staged D10 emission parameters are not tied")
        return rows[0].copy()

    emission_precision = emission("cov")
    emission_covariance = np.asarray(
        gauss._pd_inverse(jnp.asarray(emission_precision))
    )
    point = {
        "initial_regime_probability": initial_probability.copy(),
        "dynamics_matrix": np.asarray(model.dynamics_matrix).copy(),
        "dynamics_offset": np.asarray(model.dynamics_offset).copy(),
        "dynamics_covariance": np.asarray(model.dynamics_covariance).copy(),
        "recurrence_weights": np.asarray(model.recurrence_weights).copy(),
        "recurrence_offsets": np.asarray(model.recurrence_offsets).copy(),
        "emission_matrix": emission("matrix"),
        "emission_offset": emission("offset"),
        "emission_covariance": emission_covariance.copy(),
    }
    error = _complete_point_error(point, identity)
    if error is not None:
        raise ValueError(error[1])
    return point


def _array_document(value: Any) -> dict[str, object]:
    array = np.asarray(value)
    if array.dtype.kind not in "bif" or not np.all(np.isfinite(array)):
        raise ValueError("stability certificate arrays must be finite numeric values")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _certificate_document(evidence: RecurrenceCertificateEvidence) -> dict[str, object]:
    document: dict[str, object] = {}
    for field in fields(evidence):
        value = getattr(evidence, field.name)
        if isinstance(value, Enum):
            document[field.name] = value.value
        elif isinstance(value, np.ndarray):
            document[field.name] = _array_document(value)
        elif isinstance(value, (np.floating, np.integer)):
            document[field.name] = value.item()
        else:
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("stability certificate scalars must be finite")
            document[field.name] = value
    return document


def stability_evidence_document(
    evidence: RSLDSStabilityEvidence,
) -> dict[str, object]:
    """Return the strict JSON-compatible evidence document."""
    return {
        "schema_version": RSLDS_STABILITY_GATE_CHECKPOINT_VERSION,
        "gate_identity": RSLDS_STABILITY_GATE_IDENTITY,
        "owner": evidence.owner,
        "coordinate": evidence.coordinate,
        "outcome": evidence.outcome.value,
        "accepted": evidence.accepted,
        "dtype": evidence.dtype,
        "point_identity": list(evidence.point_identity),
        "policy": evidence.policy.document(),
        "d10_identity": evidence.d10_identity,
        "statistic_ownership": list(evidence.statistic_ownership),
        "certificate": _certificate_document(evidence.certificate),
    }


def encode_stability_evidence(evidence: RSLDSStabilityEvidence) -> np.ndarray:
    """Encode one canonical scalar JSON checkpoint value."""
    return np.asarray(
        json.dumps(
            stability_evidence_document(evidence),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def validate_stability_evidence(
    value: Any,
    *,
    engine,
    points: Mapping[str, Any],
    policy: RSLDSStabilityPolicy,
    owner: str,
    coordinate: str,
) -> RSLDSStabilityEvidence:
    """Authenticate evidence against the exact current point before resume."""
    encoded = np.asarray(value)
    if encoded.shape != () or encoded.dtype.kind not in "US":
        raise ValueError("stability gate checkpoint must be scalar JSON text")
    raw = encoded.item()
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("stability gate checkpoint is not UTF-8") from error
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("stability gate checkpoint is not valid JSON") from error
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    if raw != canonical:
        raise ValueError("stability gate checkpoint is not canonically encoded")
    expected = evaluate_rslds_stability(
        engine,
        points,
        policy,
        owner=owner,
        coordinate=coordinate,
    )
    expected_raw = json.dumps(
        stability_evidence_document(expected),
        sort_keys=True,
        separators=(",", ":"),
    )
    if raw != expected_raw:
        raise ValueError(
            "stability gate checkpoint does not match the current D10 point, "
            "policy, ownership, or certificate"
        )
    return expected


__all__ = [
    "IDENTITY_LYAPUNOV_POLICY",
    "POINT_FIELDS",
    "RSLDS_STABILITY_GATE_CHECKPOINT_VERSION",
    "RSLDS_STABILITY_GATE_IDENTITY",
    "RSLDSStabilityEvidence",
    "RSLDSStabilityOutcome",
    "RSLDSStabilityPolicy",
    "encode_stability_evidence",
    "evaluate_rslds_stability",
    "runtime_rslds_point",
    "stability_evidence_document",
    "validate_stability_evidence",
]
