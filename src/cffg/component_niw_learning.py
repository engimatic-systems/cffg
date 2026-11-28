"""Declared NIW learning from conditional moments of Gaussian mixture components."""

from collections.abc import Mapping, Sequence

import numpy as np

from cffg import conj
from cffg.core.learning_identity import LEARNING_IDENTITIES
from cffg.gaussian_learning import _copy_params, _proper, _require_keys
from cffg.learning_schema import PersistentParameter


def _check_psi(learnable, psi):
    """Require resolvable positive definiteness independently of coordinate units."""
    diagonal = np.diagonal(psi, axis1=-2, axis2=-1)
    error = f"learnable {learnable!r}: component NIW psi must be positive definite"
    if np.any(diagonal <= 0):
        raise ValueError(error)
    dim = psi.shape[-1]
    if dim == 1:
        return
    scale = np.sqrt(diagonal.astype(np.float64))
    with np.errstate(over="ignore"):
        corr = (psi / scale[:, :, None]) / scale[:, None, :]
    eps = np.finfo(psi.dtype).eps
    if not np.isfinite(corr).all() or np.any(np.abs(corr) > 1 + dim * eps):
        raise ValueError(error)
    if np.any(np.abs(corr - corr.swapaxes(-1, -2)) > 16 * dim * eps):
        raise ValueError(f"learnable {learnable!r}: component NIW psi must be symmetric")
    # Reserve one source-dtype dot product's roundoff before accepting a pivot.
    guarded = .5 * (corr + corr.swapaxes(-1, -2))
    index = np.arange(dim)
    guarded[:, index, index] = 1 - dim * eps
    try:
        np.linalg.cholesky(guarded)
    except np.linalg.LinAlgError as exc:
        raise ValueError(error) from exc


class ComponentNIWFamily:
    """One NIW per selector value, with the node responsibility as occurrence weight."""

    name = "component_niw"
    readout = "centered_mixture_stats"
    semantic_identity = LEARNING_IDENTITIES[name]
    parameter_forms = ("expected",)
    _keys = ("mu", "kappa", "psi", "nu")

    def check(self, learnable, ftype, shapes, roles):
        if ftype != "gaussian_mixture" or set(roles.values()) != {"mean", "cov"}:
            raise ValueError(
                f"learnable {learnable!r}: component_niw/gaussian_mixture "
                "requires mean and cov roles"
            )
        by_role = {role: shapes[blob] for blob, role in roles.items()}
        mean = by_role["mean"]
        if len(mean) != 2 or min(mean) < 1 or by_role["cov"] != (*mean, mean[-1]):
            raise ValueError(
                f"learnable {learnable!r}: expected [K,d] and [K,d,d], got {by_role}"
            )

    def persistent_parameters(self, role_shapes):
        components, _ = role_shapes["mean"]
        return (
            PersistentParameter("mu", role_shapes["mean"]),
            PersistentParameter("kappa", (components,)),
            PersistentParameter("psi", role_shapes["cov"]),
            PersistentParameter("nu", (components,)),
        )

    def as_params(self, prior):
        if isinstance(prior, Mapping):
            params = _copy_params(prior)
        elif isinstance(prior, Sequence) and prior and all(isinstance(p, conj.NIW) for p in prior):
            params = {key: np.stack([getattr(p, key) for p in prior]) for key in self._keys}
        else:
            raise TypeError("component NIW requires a nonempty sequence of NIWs or a parameter mapping")
        return {key: a.astype(float) if a.dtype.kind in "iu" else a for key, a in params.items()}

    def validate_prior(self, learnable, params, shapes):
        _require_keys(learnable, params, set(self._keys), "component NIW")
        mu, kappa, psi, nu = (np.asarray(params[key]) for key in self._keys)
        if (
            mu.ndim != 2 or mu.shape not in shapes.values()
            or psi.shape != (*mu.shape, mu.shape[-1]) or psi.shape not in shapes.values()
            or kappa.shape != mu.shape[:1] or nu.shape != kappa.shape
        ):
            raise ValueError(f"learnable {learnable!r}: incompatible component NIW shapes")
        if any(a.dtype.kind != "f" or not np.isfinite(a).all() for a in (mu, kappa, psi, nu)):
            raise ValueError(f"learnable {learnable!r}: component NIW values must be finite and real")
        if np.any(kappa <= 0) or np.any(nu <= mu.shape[1] - 1):
            raise ValueError(f"learnable {learnable!r}: improper component NIW kappa or nu")
        _check_psi(learnable, psi)

    def stats(self, ftype, readout):
        _proper(readout, ftype)
        keys = ("responsibility", "component_mean", "component_covariance")
        return {key: np.asarray(readout[key]) for key in keys}

    def pool(self, records):
        return {key: np.stack([row[key] for row in records]) for key in records[0]}

    def fold(self, params, stats):
        weights = stats["responsibility"]
        means = stats["component_mean"]
        covariances = stats["component_covariance"]
        components, dimension = params["mu"].shape
        if (
            weights.ndim != 2 or weights.shape[1] != components
            or means.shape != (*weights.shape, dimension)
            or covariances.shape != (*means.shape, dimension)
        ):
            raise ValueError(
                "component NIW requires one weight and one conditional Gaussian per component"
            )
        if weights.dtype.kind not in "fiu" or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("component responsibilities must be finite and nonnegative")
        active = np.flatnonzero(np.any(weights > 0, axis=0))
        if not len(active):
            return _copy_params(params)
        posterior = {key: np.array(value, dtype=np.float64, copy=True) for key, value in params.items()}
        for index in active:
            prior = conj.NIW(*(params[key][index] for key in self._keys))
            updated = prior.update_centered(means[:, index], covariances[:, index], weights[:, index])
            for key in self._keys:
                posterior[key][index] = getattr(updated, key)
        return posterior

    def rollover(self, params, ftype):
        return {"mean": params["mu"].copy(), "cov": params["psi"] / params["nu"][:, None, None]}

    def mean(self, params, ftype):
        denominator = params["nu"] - params["mu"].shape[1] - 1
        if np.any(denominator <= 0):
            raise ValueError("posterior-mean covariance undefined for nu <= d+1")
        return {"mean": params["mu"].copy(), "cov": params["psi"] / denominator[:, None, None]}

    def raw(self, params, ftype):
        raise ValueError("component NIW has no raw-count runtime target")
