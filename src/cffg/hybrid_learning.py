"""Declared-learning adapters for hybrid factor families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from cffg import conj
from cffg.gaussian_learning import (
    _copy_params,
    _finite_stat,
    _proper,
    _require_keys,
)


def _diagnostic_scalar(readout: Mapping[str, object], key: str) -> str:
    value = np.asarray(readout.get(key, np.nan))
    if value.shape != () or not np.issubdtype(value.dtype, np.number):
        return "missing"
    return format(float(value), ".17g")


def _stack_mniw(posteriors: Sequence[conj.MNIW]) -> dict[str, np.ndarray]:
    return {
        "m": np.stack([posterior.m for posterior in posteriors]),
        "lam": np.stack([posterior.lam for posterior in posteriors]),
        "psi": np.stack([posterior.psi for posterior in posteriors]),
        "nu": np.asarray([posterior.nu for posterior in posteriors]),
    }


class ComponentMNIWFamily:
    """One MNIW posterior per Gaussian-mixture selector component."""

    name = "component_mniw"
    readout = "gaussian_node_stats"
    _PARAMS = {"m", "lam", "psi", "nu"}

    def check(self, learnable, ftype, shapes, roles) -> None:
        if ftype != "gaussian_mixture_linear":
            raise ValueError(
                f"learnable {learnable!r}: component MNIW does not support "
                f"factor type {ftype!r}; supported pair is "
                "component_mniw/gaussian_mixture_linear"
            )
        if set(roles.values()) != {"matrix", "offset", "cov"}:
            raise ValueError(
                f"learnable {learnable!r}: "
                "component_mniw/gaussian_mixture_linear requires matrix, "
                f"offset, and cov roles, got {sorted(roles.values())}"
            )
        by_role = {role: shapes[blob] for blob, role in roles.items()}
        matrix_shape = by_role["matrix"]
        if len(matrix_shape) != 3:
            raise ValueError(
                f"learnable {learnable!r}: component MNIW matrix must have "
                f"shape [K,dy,dx], got {matrix_shape}"
            )
        components, output_dim, _ = matrix_shape
        if components < 2:
            raise ValueError(
                f"learnable {learnable!r}: component MNIW requires K>=2"
            )
        if by_role["offset"] != (components, output_dim) or by_role["cov"] != (
            components,
            output_dim,
            output_dim,
        ):
            raise ValueError(
                f"learnable {learnable!r}: incompatible component MNIW blob "
                f"shapes {by_role}"
            )

    def stats(self, ftype, readout) -> dict[str, np.ndarray]:
        _proper(readout, ftype)
        responsibility = _finite_stat(readout, "responsibility", ftype)
        if responsibility.ndim != 1 or np.any(responsibility < 0.0):
            raise ValueError(
                f"{ftype} responsibility must be a nonnegative component vector"
            )
        return {
            "Syy": _finite_stat(readout, "Syy", ftype),
            "Syx": _finite_stat(readout, "Syx", ftype),
            "Sxx": _finite_stat(readout, "Sxx", ftype),
            "n": responsibility,
        }

    def pool(self, records) -> dict[str, np.ndarray]:
        return {
            key: np.sum(np.stack([record[key] for record in records]), axis=0)
            for key in ("Syy", "Syx", "Sxx", "n")
        }

    def fold(self, params, stats) -> dict[str, np.ndarray]:
        priors = self._posteriors("posterior", params)
        syy = np.asarray(stats["Syy"])
        syx = np.asarray(stats["Syx"])
        sxx = np.asarray(stats["Sxx"])
        count = np.asarray(stats["n"])
        components = len(priors)
        output_dim, augmented_dim = priors[0].m.shape
        expected = {
            "Syy": (components, output_dim, output_dim),
            "Syx": (components, output_dim, augmented_dim),
            "Sxx": (components, augmented_dim, augmented_dim),
        }
        actual = {"Syy": syy.shape, "Syx": syx.shape, "Sxx": sxx.shape}
        if actual != expected or count.shape != (components,):
            raise ValueError(
                f"component MNIW statistic shapes {actual}, n={count.shape} != "
                f"{expected}, n={(components,)}"
            )
        if (
            not np.all(np.isfinite(syy))
            or not np.all(np.isfinite(syx))
            or not np.all(np.isfinite(sxx))
            or not np.all(np.isfinite(count))
            or np.any(count < 0.0)
        ):
            raise ValueError(
                "component MNIW statistics and occurrence counts must be finite "
                "and nonnegative"
            )

        posteriors = []
        for component, prior in enumerate(priors):
            if count[component] == 0.0:
                if any(
                    np.any(value[component] != 0.0)
                    for value in (syy, syx, sxx)
                ):
                    raise ValueError(
                        "zero-responsibility component has nonzero MNIW statistics"
                    )
                posteriors.append(prior)
                continue
            posteriors.append(
                prior.update(
                    syy[component],
                    syx[component],
                    sxx[component],
                    float(count[component]),
                )
            )
        return _stack_mniw(posteriors)

    def rollover(self, params, ftype) -> dict[str, np.ndarray]:
        values = [posterior.rollover() for posterior in self._posteriors("posterior", params)]
        return {
            "matrix": np.stack([value[0] for value in values]),
            "offset": np.stack([value[1] for value in values]),
            "cov": np.stack([value[2] for value in values]),
        }

    def raw(self, params, ftype) -> dict[str, np.ndarray]:
        raise ValueError("component MNIW has no raw-count runtime target")

    def mean(self, params, ftype) -> dict[str, np.ndarray]:
        posteriors = self._posteriors("posterior", params)
        return {
            "matrix": np.stack([posterior.m[:, :-1] for posterior in posteriors]),
            "offset": np.stack([posterior.m[:, -1] for posterior in posteriors]),
            "cov": np.stack([posterior.mean_cov() for posterior in posteriors]),
        }

    def as_params(self, prior) -> dict[str, np.ndarray]:
        if isinstance(prior, Sequence) and not isinstance(prior, (str, bytes, Mapping)):
            posteriors = tuple(prior)
            if not posteriors or not all(
                isinstance(posterior, conj.MNIW) for posterior in posteriors
            ):
                raise TypeError(
                    "a component MNIW prior must be a nonempty sequence of "
                    "cffg.conj.MNIW values"
                )
            return _stack_mniw(posteriors)
        if not isinstance(prior, Mapping):
            raise TypeError(
                "a component MNIW prior must be a sequence of cffg.conj.MNIW "
                "values or a parameter mapping"
            )
        return _copy_params(prior)

    def validate_prior(self, learnable, params, shapes) -> None:
        posteriors = self._posteriors(learnable, params)
        components = len(posteriors)
        output_dim, augmented_dim = posteriors[0].m.shape
        required = {
            (components, output_dim, augmented_dim - 1),
            (components, output_dim),
            (components, output_dim, output_dim),
        }
        if not required <= set(shapes.values()):
            raise ValueError(
                f"learnable {learnable!r}: component MNIW prior shapes do not "
                f"match blobs {sorted(shapes.values())}"
            )

    def _posteriors(self, learnable, params) -> tuple[conj.MNIW, ...]:
        _require_keys(learnable, params, self._PARAMS, "component MNIW")
        m = np.asarray(params["m"])
        lam = np.asarray(params["lam"])
        psi = np.asarray(params["psi"])
        nu = np.asarray(params["nu"])
        if m.ndim != 3:
            raise ValueError(
                f"learnable {learnable!r}: component MNIW m must be rank 3"
            )
        components, output_dim, augmented_dim = m.shape
        if (
            lam.shape != (components, augmented_dim, augmented_dim)
            or psi.shape != (components, output_dim, output_dim)
            or nu.shape != (components,)
        ):
            raise ValueError(
                f"learnable {learnable!r}: incompatible component MNIW "
                f"hyperparameter shapes m={m.shape}, lam={lam.shape}, "
                f"psi={psi.shape}, nu={nu.shape}"
            )
        if not all(np.all(np.isfinite(value)) for value in (m, lam, psi, nu)):
            raise ValueError(
                f"learnable {learnable!r}: component MNIW values must be finite"
            )
        return tuple(
            conj.MNIW(m[index], lam[index], psi[index], float(nu[index]))
            for index in range(components)
        )


class PGWeightsFamily:
    """Pair-derived Gaussian learning for recurrent stick weights."""

    name = "pg_weights"
    readout = "recurrence_weight_stats"
    _PARAMS = {"eta", "precision"}

    def check(self, learnable, ftype, shapes, roles) -> None:
        if ftype != "recurrence_link":
            raise ValueError(
                f"learnable {learnable!r}: Pólya–Gamma weights do not support "
                f"factor type {ftype!r}; supported pair is "
                "pg_weights/recurrence_link"
            )
        if set(roles.values()) != {"weights", "offsets"}:
            raise ValueError(
                f"learnable {learnable!r}: pg_weights/recurrence_link requires "
                f"weights and offsets roles, got {sorted(roles.values())}"
            )
        by_role = {role: shapes[blob] for blob, role in roles.items()}
        weights_shape = by_role["weights"]
        if len(weights_shape) != 3:
            raise ValueError(
                f"learnable {learnable!r}: recurrence weights must have shape "
                f"[K,K-1,d], got {weights_shape}"
            )
        regimes, sticks, dimension = weights_shape
        if (
            regimes < 2
            or sticks != regimes - 1
            or dimension < 1
            or by_role["offsets"] != (regimes, sticks)
        ):
            raise ValueError(
                f"learnable {learnable!r}: incompatible Pólya–Gamma blob "
                f"shapes {by_role}"
            )

    def stats(self, ftype, readout) -> dict[str, np.ndarray]:
        if "proper" not in readout or np.shape(readout["proper"]) != ():
            _proper(readout, ftype)
        if not bool(readout["proper"]):
            failures = []
            if not bool(readout.get("probability_ok", True)):
                failures.append(
                    "probability_ok=false "
                    f"(minimum={_diagnostic_scalar(readout, 'probability_minimum')}, "
                    "max_sum_error="
                    f"{_diagnostic_scalar(readout, 'probability_sum_error')}, "
                    f"tolerance={_diagnostic_scalar(readout, 'probability_tolerance')})"
                )
            if not bool(readout.get("drive_ok", True)):
                failures.append(
                    "drive_ok=false "
                    "(min_eigenvalue="
                    f"{_diagnostic_scalar(readout, 'drive_min_eigenvalue')}, "
                    "symmetry_error="
                    f"{_diagnostic_scalar(readout, 'drive_symmetry_error')}, "
                    f"tolerance={_diagnostic_scalar(readout, 'drive_tolerance')})"
                )
            if not bool(readout.get("parameter_ok", True)):
                failures.append("parameter_ok=false")
            if not bool(readout.get("second_ok", True)):
                failures.append(
                    "second_ok=false "
                    f"(minimum={_diagnostic_scalar(readout, 'logit_second_minimum')}, "
                    f"tolerance={_diagnostic_scalar(readout, 'logit_second_tolerance')})"
                )
            if not bool(readout.get("finite_output_ok", True)):
                failures.append("finite_output_ok=false")
            detail = f": {'; '.join(failures)}" if failures else ""
            raise ValueError(f"{ftype} statistic is improper{detail}")
        return {
            "linear": _finite_stat(readout, "linear", ftype),
            "quadratic": _finite_stat(readout, "quadratic", ftype),
        }

    def pool(self, records) -> dict[str, np.ndarray]:
        return {
            key: np.sum(np.stack([record[key] for record in records]), axis=0)
            for key in ("linear", "quadratic")
        }

    def fold(self, params, stats) -> dict[str, np.ndarray]:
        posterior = self._posterior("posterior", params).update(
            np.asarray(stats["linear"]), np.asarray(stats["quadratic"])
        )
        return self.as_params(posterior)

    def rollover(self, params, ftype) -> dict[str, np.ndarray]:
        weights, offsets = self._posterior("posterior", params).rollover()
        return {"weights": weights, "offsets": offsets}

    def raw(self, params, ftype) -> dict[str, np.ndarray]:
        raise ValueError("Pólya–Gamma weights have no raw-count runtime target")

    def mean(self, params, ftype) -> dict[str, np.ndarray]:
        weights, offsets = self._posterior("posterior", params).rollover()
        return {"weights": weights, "offsets": offsets}

    def as_params(self, prior) -> dict[str, np.ndarray]:
        if isinstance(prior, conj.PGWeights):
            return {
                "eta": np.asarray(prior.eta).copy(),
                "precision": np.asarray(prior.precision).copy(),
            }
        if not isinstance(prior, Mapping):
            raise TypeError(
                "a Pólya–Gamma weight prior must be cffg.conj.PGWeights or a "
                "parameter mapping"
            )
        return _copy_params(prior)

    def validate_prior(self, learnable, params, shapes) -> None:
        posterior = self._posterior(learnable, params)
        required = {
            (posterior.regimes, posterior.regimes - 1, posterior.dimension),
            (posterior.regimes, posterior.regimes - 1),
        }
        if not required <= set(shapes.values()):
            raise ValueError(
                f"learnable {learnable!r}: Pólya–Gamma prior shape "
                f"{posterior.eta.shape} does not match blobs "
                f"{sorted(shapes.values())}"
            )

    def _posterior(self, learnable, params) -> conj.PGWeights:
        _require_keys(learnable, params, self._PARAMS, "Pólya–Gamma")
        eta = np.asarray(params["eta"])
        precision = np.asarray(params["precision"])
        return conj.PGWeights(eta.copy(), precision.copy())
