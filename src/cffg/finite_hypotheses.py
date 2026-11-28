"""Exact Bayesian updating on a fixed, ordered set of parameter vectors."""

from __future__ import annotations

import json
import math
from threading import RLock

import jax
import numpy as np
from scipy.special import logsumexp

from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.serialize import to_dict
from cffg.ef import resolved_correlation
from cffg.gaussian_taps import negative_log_density
from cffg.neural_artifacts import array_digest, read_artifact, write_artifact

__all__ = ["FiniteHypotheses", "ImpossibleEvidence"]


class ImpossibleEvidence(ValueError):
    """No hypothesis with positive prior support can explain this observation."""


def _conditional_template(dimension):
    edges = ("output", "point", "copy")
    return Template(
        "finite_hypotheses_observed_gaussian",
        variables={edge: VariableSpec(edge, dim=dimension) for edge in edges},
        factors={
            "prediction": FactorSpec("prediction", "gaussian", {"out": "output"},
                                     param_blobs={"mean": "mean", "cov": "cov"}),
            "tap": FactorSpec("tap", "equality", dict(zip(("a", "b", "c"), edges))),
        },
        ports={"observation": PortSpec("observation", "point", PortRole.DATA),
               "copy": PortSpec("copy", "copy", PortRole.GLUE)},
        blobs={key: BlobRef(key, f"{key}.npy") for key in ("mean", "cov")},
    )


# Each bound equality coordinate contributes the Gaussian node's point density.
# The equality and adjacent fixed edge have zero entropy. Batch over readings;
# the density already handles the leading candidate axis.
_densities = jax.jit(jax.vmap(negative_log_density, in_axes=(0, 0, None)))


def _numeric(value, label):
    arr = np.asarray(value)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{label} must be real numeric data")
    return arr


def _finite(value, dtype, label):
    with np.errstate(over="ignore"):
        result = _numeric(value, label).astype(dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite at {np.dtype(dtype).name} precision")
    return result


def _precision(covariance, dtype):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            resolved_correlation(covariance)
            precision = np.linalg.inv(covariance.astype(np.float64)).astype(dtype)
            resolved_correlation(precision)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("covariance must be positive definite at the working precision") from error
    return precision


def _mask(observed, rows):
    if observed is None:
        return np.ones(rows, dtype=bool)
    mask = np.asarray(observed)
    if mask.shape != (rows,) or mask.dtype != bool:
        raise ValueError(f"observed must be a boolean mask of shape ({rows},)")
    return mask


def _prior(prior, size):
    mass = _finite(prior, np.float64, "prior")
    if mass.shape != (size,) or np.any(mass < 0) or not np.any(mass > 0):
        raise ValueError(f"prior must contain {size} nonnegative masses with positive total")
    logs = np.full(size, -np.inf)
    np.log(mass, out=logs, where=mass > 0)
    logs -= np.max(logs)
    logs -= logsumexp(logs)
    return logs


def _condition(logs, scores):
    live = np.isfinite(logs) & np.isfinite(scores)
    if not np.any(live):
        raise ImpossibleEvidence("zero evidence on the current hypothesis support")
    # Remove the common likelihood first. A huge common score can otherwise
    # erase all prior log odds when it is added to each component.
    offset = np.max(scores[live])
    joint = logs[live] + (scores[live] - offset)
    anchor = np.max(joint)
    centered = joint - anchor
    normalizer = logsumexp(centered)
    updated = np.full_like(logs, -np.inf)
    updated[live] = centered - normalizer
    if not np.isfinite(updated[live]).all():
        raise FloatingPointError("live log weights exceed the carry precision")
    increment = math.fsum((offset, anchor, normalizer))
    return updated, increment


_FORMAT = "cffg.finite_hypotheses"
_FIELDS = {"candidates", "covariance", "log_prior", "log_weights", "support"}


def _digest(meta, arrays):
    document = np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False))
    return array_digest({**arrays, "metadata": document})


class FiniteHypotheses:
    """Posterior on K fixed candidate rows [K,P], with independent Gaussian readings.

    Covariance is [D,D] shared or [K,D,D] per candidate. It is the total
    conditional covariance of a reading given a candidate. Candidate parameters
    and covariance are owned snapshots. Supply fresh predicted means for each
    observation. The likelihood dtype defaults to the active JAX float dtype.
    Log weights and accumulated log evidence use host float64.
    """

    def __init__(self, candidates, covariance, *, prior=None, dtype=None):
        self._lock = RLock()
        values = _numeric(candidates, "candidates")
        if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
            raise ValueError("candidates must be a nonempty finite [K,P] array")
        self._candidates = values.copy()
        self._size = len(values)
        if dtype is None:
            dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        self._dtype = np.dtype(dtype)
        if self._dtype not in (np.float32, np.float64):
            raise ValueError("likelihood dtype must be float32 or float64")
        cov = _finite(covariance, self._dtype, "covariance")
        if (cov.ndim not in (2, 3) or cov.shape[-2] != cov.shape[-1] or cov.shape[-1] < 1
                or (cov.ndim == 3 and len(cov) != self._size)):
            raise ValueError("covariance must have shape [D,D] or [K,D,D]")
        self._covariance = cov
        self._dimension = cov.shape[-1]
        matrices = cov[None] if cov.ndim == 2 else cov
        self._precision = np.stack([_precision(m, self._dtype) for m in matrices])
        self.reset(prior=np.ones(self._size) if prior is None else prior)

    @property
    def candidates(self):
        return self._candidates.copy()

    @property
    def log_weights(self):
        return self._log_weights.copy()

    @property
    def weights(self):
        """Probability readout. A live component can round to zero here."""
        return np.exp(self._log_weights)

    @property
    def support(self):
        return np.isfinite(self._log_weights)

    @property
    def log_evidence(self):
        return self._log_evidence

    @property
    def n_observed(self):
        return self._n_observed

    def reset(self, *, prior=None):
        """Clear evidence and restore the prior. Supplying masses replaces the reset prior."""
        with self._lock:
            if prior is not None:
                self._log_prior = _prior(prior, self._size)
            self._log_weights = self._log_prior.copy()
            self._log_evidence = 0.
            self._n_observed = 0

    def conditional_model(self, index, prediction):
        """Executable single-candidate likelihood. Bind port 'observation' to one [D] point."""
        if (isinstance(index, bool) or not isinstance(index, (int, np.integer))
                or not 0 <= index < self._size):
            raise ValueError("invalid candidate index")
        mean = _finite(prediction, self._dtype, "prediction")
        if mean.shape != (self._dimension,):
            raise ValueError("prediction must have shape [D]")
        cov = self._covariance if self._covariance.ndim == 2 else self._covariance[index]
        return _conditional_template(self._dimension), {"mean": mean, "cov": cov.copy()}

    def update(self, predictions, observations, *, observed=None):
        """Consume [B,K,D] predictions and [B,D] readings, or one [K,D], [D] pair.

        A false boolean mask entry omits that reading and its prediction values.
        Passing observations=None omits the entire batch. Returns its conditional
        log evidence. A failure leaves the entire batch uncommitted.
        """
        means = _numeric(predictions, "predictions")
        if means.ndim == 2:
            means = means[None]
        if means.ndim != 3 or means.shape[1:] != (self._size, self._dimension):
            raise ValueError("predictions must have shape [B,K,D] or [K,D]")
        mask = _mask(observed, len(means))
        if observations is None:
            if observed is not None and np.any(mask):
                raise ValueError("missing observations cannot be marked observed")
            return 0.
        points = _numeric(observations, "observations")
        if points.ndim == 1:
            points = points[None]
        if points.shape != (len(means), self._dimension):
            raise ValueError("observations must have shape [B,D] or [D]")
        if not mask.any():
            return 0.
        means = _finite(means[mask], self._dtype, "observed predictions")
        points = _finite(points[mask], self._dtype, "observed points")
        with jax.enable_x64(self._dtype == np.float64):
            scores = -np.asarray(_densities(points[:, None, :], means, self._precision))
        if not np.isfinite(scores).all():
            raise FloatingPointError("Gaussian log likelihood is nonfinite at the working precision")
        return self._apply(scores.astype(np.float64))

    def update_log_likelihood(self, log_likelihood, *, observed=None):
        """Condition on absolute log likelihoods [K] or [B,K]; -inf denotes impossible."""
        rows = _numeric(log_likelihood, "log likelihood")
        if rows.ndim == 1:
            rows = rows[None]
        if rows.ndim != 2 or rows.shape[1] != self._size:
            raise ValueError("log likelihood must have shape [K] or [B,K]")
        rows = rows[_mask(observed, len(rows))]
        with np.errstate(over="raise"):
            scores = rows.astype(np.float64)
        if not np.all(np.isfinite(scores) | np.isneginf(scores)):
            raise ValueError("log likelihoods must be finite or negative infinity")
        return self._apply(scores)

    def _apply(self, scores):
        with self._lock, np.errstate(over="raise", invalid="raise"):
            logs, evidence = self._log_weights, self._log_evidence
            increments = []
            for row in scores:
                logs, increment = _condition(logs, row)
                evidence = math.fsum((evidence, increment))
                increments.append(increment)
            batch_evidence = math.fsum(increments)
            self._log_weights, self._log_evidence = logs, evidence
            self._n_observed += len(scores)
        return batch_evidence

    def save(self, path):
        """Atomically save the conditional Template, candidate order, and committed carry."""
        with self._lock:
            arrays = dict(candidates=self._candidates, covariance=self._covariance,
                          log_prior=self._log_prior, log_weights=self._log_weights, support=self.support)
            meta = dict(format=_FORMAT, version=1, model=to_dict(_conditional_template(self._dimension)),
                        log_evidence=self.log_evidence, n_observed=self.n_observed)
            meta["sha256"] = _digest(meta, arrays)
            write_artifact(path, meta, arrays)

    @classmethod
    def load(cls, path):
        """Restore one version-1 bank. Unknown formats and inconsistent state are errors."""
        meta, arrays = read_artifact(path, format_name=_FORMAT, fields=_FIELDS)
        if set(meta) != {"format", "version", "model", "log_evidence", "n_observed", "sha256"}:
            raise ValueError("incomplete or unknown finite hypothesis metadata")
        if meta.pop("sha256") != _digest(meta, arrays):
            raise ValueError("finite hypothesis checkpoint digest mismatch")
        bank = cls(arrays["candidates"], arrays["covariance"], dtype=arrays["covariance"].dtype)
        if meta["model"] != to_dict(_conditional_template(bank._dimension)):
            raise ValueError("incompatible conditional Template")
        prior, logs, live = (arrays[name] for name in ("log_prior", "log_weights", "support"))
        for row in (prior, logs):
            if (row.shape != (bank._size,) or row.dtype != np.float64
                    or not np.all(np.isfinite(row) | np.isneginf(row))
                    or abs(logsumexp(row)) > 32 * np.finfo(row.dtype).eps):
                raise ValueError("invalid normalized checkpoint log masses")
        if (live.dtype != bool or not np.array_equal(live, np.isfinite(logs))
                or np.any(live & ~np.isfinite(prior))):
            raise ValueError("invalid checkpoint hypothesis support")
        count, evidence = meta["n_observed"], meta["log_evidence"]
        if (type(count) is not int or count < 0 or type(evidence) is not float
                or not math.isfinite(evidence) or (count == 0 and
                (evidence != 0. or not np.array_equal(prior, logs)))):
            raise ValueError("invalid checkpoint evidence or observation count")
        bank._log_prior, bank._log_weights = prior, logs
        bank._n_observed, bank._log_evidence = count, evidence
        return bank
