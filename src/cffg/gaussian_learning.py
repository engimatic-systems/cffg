"""NIW and MNIW adapters for the declared-learning protocol."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cffg import conj


def _copy_params(params: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in params.items()}


def _canonical_identity(family, declared: str | None) -> str:
    if declared is None:
        return family.semantic_identity
    if declared != family.semantic_identity:
        raise ValueError(
            f"semantic identity {declared!r} is incompatible with family "
            f"{family.name!r}; expected {family.semantic_identity!r}"
        )
    return declared


def _require_keys(
    learnable: str, params: Mapping[str, object], expected: set[str], family: str
) -> None:
    if set(params) != expected:
        raise ValueError(
            f"learnable {learnable!r}: {family} hyperparameters are "
            f"{sorted(expected)}, got {sorted(params)}"
        )


def _scalar(learnable: str, key: str, value: object) -> float:
    array = np.asarray(value)
    if array.shape != () or not np.isfinite(array):
        raise ValueError(
            f"learnable {learnable!r}: hyperparameter {key!r} must be a finite scalar"
        )
    return float(array)


def _proper(readout: Mapping[str, object], factor_type: str) -> None:
    if "proper" not in readout or np.shape(readout["proper"]) != ():
        raise ValueError(f"{factor_type} statistic has no scalar proper flag")
    if not bool(readout["proper"]):
        raise ValueError(f"{factor_type} statistic is improper")


def _finite_stat(readout: Mapping[str, object], key: str, factor_type: str) -> np.ndarray:
    if key not in readout:
        raise KeyError(f"{factor_type} statistic has no {key!r} value")
    value = np.asarray(readout[key])
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError(f"{factor_type} statistic {key!r} must be finite and numeric")
    return value


class NIWFamily:
    """Normal-inverse-Wishart learning for a Gaussian prior factor."""

    name = "niw"
    readout = "gaussian_node_stats"
    semantic_identity = "cffg.gaussian.niw-node-moments/1.0.0"
    _PARAMS = {"mu", "kappa", "psi", "nu"}

    def canonical_identity(self, declared):
        return _canonical_identity(self, declared)

    def check(self, learnable, ftype, shapes, roles) -> None:
        if ftype != "gaussian":
            raise ValueError(
                f"learnable {learnable!r}: NIW does not support factor type "
                f"{ftype!r}; supported pair is niw/gaussian"
            )
        if set(roles.values()) != {"mean", "cov"}:
            raise ValueError(
                f"learnable {learnable!r}: niw/gaussian requires mean and cov "
                f"roles, got {sorted(roles.values())}"
            )
        by_role = {role: shapes[blob] for blob, role in roles.items()}
        mean_shape = by_role["mean"]
        cov_shape = by_role["cov"]
        if len(mean_shape) != 1 or cov_shape != mean_shape + mean_shape:
            raise ValueError(
                f"learnable {learnable!r}: niw/gaussian blob shapes must be "
                f"(d,) and (d,d), got {by_role}"
            )

    def stats(self, ftype, readout) -> dict[str, np.ndarray]:
        _proper(readout, ftype)
        return {
            "mean": _finite_stat(readout, "mean", ftype),
            "second": _finite_stat(readout, "second", ftype),
        }

    def pool(self, records) -> dict[str, np.ndarray]:
        return {
            "mean": np.stack([record["mean"] for record in records]),
            "second": np.stack([record["second"] for record in records]),
        }

    def fold(self, params, stats) -> dict[str, np.ndarray]:
        prior = self._posterior("posterior", params)
        means = np.asarray(stats["mean"])
        seconds = np.asarray(stats["second"])
        if means.ndim != 2 or means.shape[1:] != prior.mu.shape:
            raise ValueError(
                f"NIW mean statistics must have shape [n,{prior.mu.shape[0]}], "
                f"got {means.shape}"
            )
        if seconds.shape != (means.shape[0], *prior.psi.shape):
            raise ValueError(
                f"NIW second statistics shape {seconds.shape} != "
                f"{(means.shape[0], *prior.psi.shape)}"
            )
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(seconds)):
            raise ValueError("NIW statistics must be finite")
        return self.as_params(prior.update(means, seconds))

    def rollover(self, params, ftype) -> dict[str, np.ndarray]:
        mean, cov = self._posterior("posterior", params).rollover()
        return {"mean": mean, "cov": cov}

    def raw(self, params, ftype) -> dict[str, np.ndarray]:
        raise ValueError("NIW has no raw-count runtime target")

    def mean(self, params, ftype) -> dict[str, np.ndarray]:
        posterior = self._posterior("posterior", params)
        return {"mean": posterior.mu.copy(), "cov": posterior.mean_cov()}

    def as_params(self, prior) -> dict[str, np.ndarray]:
        if isinstance(prior, conj.NIW):
            return {
                "mu": np.asarray(prior.mu).copy(),
                "kappa": np.asarray(prior.kappa).copy(),
                "psi": np.asarray(prior.psi).copy(),
                "nu": np.asarray(prior.nu).copy(),
            }
        if not isinstance(prior, Mapping):
            raise TypeError("an NIW prior must be cffg.conj.NIW or a parameter mapping")
        return _copy_params(prior)

    def validate_prior(self, learnable, params, shapes) -> None:
        posterior = self._posterior(learnable, params)
        available = set(shapes.values())
        if posterior.mu.shape not in available or posterior.psi.shape not in available:
            raise ValueError(
                f"learnable {learnable!r}: NIW prior shapes "
                f"{posterior.mu.shape, posterior.psi.shape} do not match blobs "
                f"{sorted(available)}"
            )

    def _posterior(self, learnable, params) -> conj.NIW:
        _require_keys(learnable, params, self._PARAMS, "NIW")
        mu = np.asarray(params["mu"])
        psi = np.asarray(params["psi"])
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(psi)):
            raise ValueError(f"learnable {learnable!r}: NIW arrays must be finite")
        return conj.NIW(
            mu.copy(),
            _scalar(learnable, "kappa", params["kappa"]),
            psi.copy(),
            _scalar(learnable, "nu", params["nu"]),
        )


class MNIWFamily:
    """Matrix-normal-inverse-Wishart learning for Gaussian-linear factors."""

    name = "mniw"
    readout = "gaussian_node_stats"
    semantic_identity = "cffg.gaussian.mniw-node-moments/1.0.0"
    _PARAMS = {"m", "lam", "psi", "nu"}

    def canonical_identity(self, declared):
        return _canonical_identity(self, declared)

    def check(self, learnable, ftype, shapes, roles) -> None:
        if ftype != "gaussian_linear":
            raise ValueError(
                f"learnable {learnable!r}: MNIW does not support factor type "
                f"{ftype!r}; supported pair is mniw/gaussian_linear"
            )
        if set(roles.values()) != {"matrix", "offset", "cov"}:
            raise ValueError(
                f"learnable {learnable!r}: mniw/gaussian_linear requires matrix, "
                f"offset, and cov roles, got {sorted(roles.values())}"
            )
        by_role = {role: shapes[blob] for blob, role in roles.items()}
        matrix_shape = by_role["matrix"]
        if len(matrix_shape) != 2:
            raise ValueError(
                f"learnable {learnable!r}: MNIW matrix must be rank 2, got "
                f"{matrix_shape}"
            )
        output_dim, _ = matrix_shape
        if by_role["offset"] != (output_dim,) or by_role["cov"] != (
            output_dim,
            output_dim,
        ):
            raise ValueError(
                f"learnable {learnable!r}: incompatible MNIW blob shapes {by_role}"
            )

    def stats(self, ftype, readout) -> dict[str, np.ndarray]:
        mask = np.asarray(readout.get("mask", 1.0))
        if mask.shape != () or not np.isfinite(mask) or float(mask) not in (0.0, 1.0):
            raise ValueError(f"{ftype} statistic mask must be scalar zero or one")
        if float(mask) == 0.0:
            return {
                "Syy": np.zeros_like(np.asarray(readout["Syy"])),
                "Syx": np.zeros_like(np.asarray(readout["Syx"])),
                "Sxx": np.zeros_like(np.asarray(readout["Sxx"])),
                "n": np.asarray(0.0),
            }
        _proper(readout, ftype)
        return {
            "Syy": _finite_stat(readout, "Syy", ftype),
            "Syx": _finite_stat(readout, "Syx", ftype),
            "Sxx": _finite_stat(readout, "Sxx", ftype),
            "n": np.asarray(1.0),
        }

    def pool(self, records) -> dict[str, np.ndarray]:
        return {
            key: np.sum(np.stack([record[key] for record in records]), axis=0)
            for key in ("Syy", "Syx", "Sxx", "n")
        }

    def fold(self, params, stats) -> dict[str, np.ndarray]:
        prior = self._posterior("posterior", params)
        syy = np.asarray(stats["Syy"])
        syx = np.asarray(stats["Syx"])
        sxx = np.asarray(stats["Sxx"])
        n = np.asarray(stats["n"])
        output_dim, augmented_dim = prior.m.shape
        expected = {
            "Syy": (output_dim, output_dim),
            "Syx": (output_dim, augmented_dim),
            "Sxx": (augmented_dim, augmented_dim),
        }
        actual = {"Syy": syy.shape, "Syx": syx.shape, "Sxx": sxx.shape}
        if actual != expected or n.shape != ():
            raise ValueError(
                f"MNIW statistic shapes {actual}, n={n.shape} != {expected}, n=()"
            )
        if (
            not np.all(np.isfinite(syy))
            or not np.all(np.isfinite(syx))
            or not np.all(np.isfinite(sxx))
            or not np.isfinite(n)
            or float(n) < 0.0
        ):
            raise ValueError("MNIW statistics and occurrence count must be finite")
        return self.as_params(prior.update(syy, syx, sxx, float(n)))

    def rollover(self, params, ftype) -> dict[str, np.ndarray]:
        matrix, offset, cov = self._posterior("posterior", params).rollover()
        return {"matrix": matrix, "offset": offset, "cov": cov}

    def raw(self, params, ftype) -> dict[str, np.ndarray]:
        raise ValueError("MNIW has no raw-count runtime target")

    def mean(self, params, ftype) -> dict[str, np.ndarray]:
        posterior = self._posterior("posterior", params)
        output_dim = posterior.psi.shape[0]
        denominator = posterior.nu - output_dim - 1
        if denominator <= 0:
            raise ValueError("posterior-mean covariance undefined for nu <= d+1")
        return {
            "matrix": posterior.m[:, :-1].copy(),
            "offset": posterior.m[:, -1].copy(),
            "cov": posterior.psi / denominator,
        }

    def as_params(self, prior) -> dict[str, np.ndarray]:
        if isinstance(prior, conj.MNIW):
            return {
                "m": np.asarray(prior.m).copy(),
                "lam": np.asarray(prior.lam).copy(),
                "psi": np.asarray(prior.psi).copy(),
                "nu": np.asarray(prior.nu).copy(),
            }
        if not isinstance(prior, Mapping):
            raise TypeError("an MNIW prior must be cffg.conj.MNIW or a parameter mapping")
        return _copy_params(prior)

    def validate_prior(self, learnable, params, shapes) -> None:
        posterior = self._posterior(learnable, params)
        output_dim, augmented_dim = posterior.m.shape
        required = {
            (output_dim, augmented_dim - 1),
            (output_dim,),
            (output_dim, output_dim),
        }
        if not required <= set(shapes.values()):
            raise ValueError(
                f"learnable {learnable!r}: MNIW prior shape {posterior.m.shape} "
                f"does not match blobs {sorted(shapes.values())}"
            )

    def _posterior(self, learnable, params) -> conj.MNIW:
        _require_keys(learnable, params, self._PARAMS, "MNIW")
        m = np.asarray(params["m"])
        lam = np.asarray(params["lam"])
        psi = np.asarray(params["psi"])
        if (
            not np.all(np.isfinite(m))
            or not np.all(np.isfinite(lam))
            or not np.all(np.isfinite(psi))
        ):
            raise ValueError(f"learnable {learnable!r}: MNIW arrays must be finite")
        if m.ndim != 2:
            raise ValueError(f"learnable {learnable!r}: MNIW m must be rank 2")
        return conj.MNIW(
            m.copy(),
            lam.copy(),
            psi.copy(),
            _scalar(learnable, "nu", params["nu"]),
        )
