"""Recurrent switching linear dynamical systems built from CFFG factors.

The model uses one normalized state-dependent transition,
``recurrence_link(z[t+1], z[t], x[t])``. Its source-specific offsets carry the
ordinary transition baseline; a separate ``matrix_cat`` would multiply two
conditionals without accounting for their state-dependent normalizer.

Learning is an explicit between-trial variational-EM boundary. Inference uses
point parameters, factor-local soft statistics update NumPy conjugate
posteriors, and :class:`RSLDSLearning` installs their rollover values into a
fresh state without recompiling the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from cffg import conj
from cffg.core.ir import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.engine import Engine, State, _replace


def rslds(
    n_steps: int,
    dynamics_matrix: np.ndarray,
    dynamics_offset: np.ndarray,
    dynamics_covariance: np.ndarray,
    recurrence_weights: np.ndarray,
    recurrence_offsets: np.ndarray,
    emission_matrix: np.ndarray,
    emission_covariance: np.ndarray,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
    initial_regime_probability: np.ndarray,
) -> tuple[Template, dict[str, np.ndarray]]:
    """Return a serialized rSLDS template and its arrays.

    ``n_steps`` counts latent states and observations, so a model has
    ``x0..x{n_steps-1}``, ``z0..z{n_steps-1}``, and DATA ports with matching
    ``y`` names. Parameters are tied across time by shared blob names.
    """
    if n_steps < 2:
        raise ValueError("rslds requires at least two latent states")

    a = np.asarray(dynamics_matrix)
    b = np.asarray(dynamics_offset)
    q = np.asarray(dynamics_covariance)
    weights = np.asarray(recurrence_weights)
    offsets = np.asarray(recurrence_offsets)
    h = np.asarray(emission_matrix)
    r = np.asarray(emission_covariance)
    m0 = np.asarray(initial_state_mean)
    p0 = np.asarray(initial_state_covariance)
    pi = np.asarray(initial_regime_probability)

    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(f"dynamics_matrix must have shape [K,d,d], got {a.shape}")
    regimes, dimension, _ = a.shape
    observation_dimension = h.shape[0] if h.ndim == 2 else -1
    expected = {
        "dynamics_offset": (regimes, dimension),
        "dynamics_covariance": (regimes, dimension, dimension),
        "recurrence_weights": (regimes, regimes - 1, dimension),
        "recurrence_offsets": (regimes, regimes - 1),
        "emission_matrix": (observation_dimension, dimension),
        "emission_covariance": (observation_dimension, observation_dimension),
        "initial_state_mean": (dimension,),
        "initial_state_covariance": (dimension, dimension),
        "initial_regime_probability": (regimes,),
    }
    actual = {
        "dynamics_offset": b.shape,
        "dynamics_covariance": q.shape,
        "recurrence_weights": weights.shape,
        "recurrence_offsets": offsets.shape,
        "emission_matrix": h.shape,
        "emission_covariance": r.shape,
        "initial_state_mean": m0.shape,
        "initial_state_covariance": p0.shape,
        "initial_regime_probability": pi.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} must have shape {shape}, got {actual[name]}")
    if regimes < 2 or observation_dimension < 1:
        raise ValueError(
            "rslds requires at least two regimes and one observation dimension"
        )

    arrays = {
        "dynamics_matrix": a,
        "dynamics_offset": b,
        "dynamics_covariance": q,
        "recurrence_weights": weights,
        "recurrence_offsets": offsets,
        "emission_matrix": h,
        "emission_offset": np.zeros(observation_dimension, dtype=h.dtype),
        "emission_covariance": r,
        "initial_state_mean": m0,
        "initial_state_covariance": p0,
        "initial_regime_probability": pi,
    }
    template = Template(name=f"rslds_{n_steps}")
    for name in arrays:
        template.blobs[name] = BlobRef(name, f"blobs/{name}.safetensors")

    def variable(name: str, *, dim: int = 0, cardinality: int = 0) -> str:
        template.variables[name] = VariableSpec(name, dim=dim, cardinality=cardinality)
        return name

    for step in range(n_steps):
        variable(f"x{step}", dim=dimension)
        variable(f"z{step}", cardinality=regimes)

    template.factors["x0_prior"] = FactorSpec(
        "x0_prior",
        "gaussian",
        edges={"out": "x0"},
        param_blobs={"mean": "initial_state_mean", "cov": "initial_state_covariance"},
    )
    template.factors["z0_prior"] = FactorSpec(
        "z0_prior",
        "categorical",
        edges={"out": "z0"},
        param_blobs={"statistic": "initial_regime_probability"},
    )

    for step in range(n_steps - 1):
        variable(f"x{step}_switch", dim=dimension)
        variable(f"x{step}_rest", dim=dimension)
        variable(f"x{step}_dynamics", dim=dimension)
        variable(f"x{step}_emission", dim=dimension)
        template.factors[f"x{step}_split"] = FactorSpec(
            f"x{step}_split",
            "equality",
            edges={"a": f"x{step}", "b": f"x{step}_switch", "c": f"x{step}_rest"},
        )
        template.factors[f"x{step}_rest_split"] = FactorSpec(
            f"x{step}_rest_split",
            "equality",
            edges={
                "a": f"x{step}_rest",
                "b": f"x{step}_dynamics",
                "c": f"x{step}_emission",
            },
        )

        if step:
            variable(f"z{step}_switch", cardinality=regimes)
            variable(f"z{step}_dynamics", cardinality=regimes)
            template.factors[f"z{step}_split"] = FactorSpec(
                f"z{step}_split",
                "equality",
                edges={
                    "a": f"z{step}",
                    "b": f"z{step}_switch",
                    "c": f"z{step}_dynamics",
                },
            )

        next_step = step + 1
        recurrence_in = "z0" if step == 0 else f"z{step}_switch"
        selector = (
            f"z{next_step}" if next_step == n_steps - 1 else f"z{next_step}_dynamics"
        )
        template.factors[f"switch{next_step}"] = FactorSpec(
            f"switch{next_step}",
            "recurrence_link",
            edges={
                "out": f"z{next_step}",
                "in": recurrence_in,
                "drive": f"x{step}_switch",
            },
            param_blobs={
                "weights": "recurrence_weights",
                "offsets": "recurrence_offsets",
            },
        )
        template.factors[f"dynamics{next_step}"] = FactorSpec(
            f"dynamics{next_step}",
            "gaussian_mixture_linear",
            edges={
                "out": f"x{next_step}",
                "in": f"x{step}_dynamics",
                "selector": selector,
            },
            param_blobs={
                "matrix": "dynamics_matrix",
                "offset": "dynamics_offset",
                "cov": "dynamics_covariance",
            },
        )

    for step in range(n_steps):
        variable(f"y{step}", dim=observation_dimension)
        emission_in = f"x{step}" if step == n_steps - 1 else f"x{step}_emission"
        template.factors[f"emission{step}"] = FactorSpec(
            f"emission{step}",
            "gaussian_linear",
            edges={"out": f"y{step}", "in": emission_in},
            param_blobs={
                "matrix": "emission_matrix",
                "offset": "emission_offset",
                "cov": "emission_covariance",
            },
        )
        template.ports[f"y{step}"] = PortSpec(f"y{step}", f"y{step}", PortRole.DATA)

    return template, arrays


@dataclass(frozen=True)
class RSLDSLearning:
    """Three conjugate posteriors at the rSLDS between-trial boundary."""

    initial_alpha: np.ndarray
    dynamics: tuple[conj.MNIW, ...]
    recurrence: conj.PGWeights

    def __post_init__(self):
        alpha = np.asarray(self.initial_alpha, dtype=np.float64)
        regimes = len(self.dynamics)
        if alpha.shape != (regimes,) or regimes < 2:
            raise ValueError(
                f"initial_alpha must match at least two dynamics regimes, got {alpha.shape}"
            )
        if not np.all(np.isfinite(alpha)) or not np.all(alpha > 0):
            raise ValueError("initial_alpha must be finite and positive")
        if self.recurrence.regimes != regimes:
            raise ValueError("recurrence and dynamics regime counts must match")
        dimensions = {posterior.m.shape[1] - 1 for posterior in self.dynamics}
        output_dimensions = {posterior.m.shape[0] for posterior in self.dynamics}
        if dimensions != {self.recurrence.dimension} or output_dimensions != dimensions:
            raise ValueError("all dynamics and recurrence dimensions must match")
        object.__setattr__(self, "initial_alpha", alpha.copy())

    @property
    def regimes(self) -> int:
        return len(self.dynamics)

    @property
    def dimension(self) -> int:
        return self.recurrence.dimension

    def update(self, engine: Engine, state: State) -> RSLDSLearning:
        """Fold one converged sequence's soft factor-local statistics."""
        regime = engine.marginals(state).get("z0")
        if regime is None or regime.shape != self.initial_alpha.shape:
            raise ValueError("engine is not an rslds model with a z0 marginal")
        if not np.all(np.isfinite(regime)) or np.any(regime < 0) or regime.sum() <= 0:
            raise ValueError("rSLDS initial-regime belief must be proper")
        mixture = engine.gaussian_node_stats(state)
        dynamics_rows = [
            mixture[name] for name in sorted(mixture) if name.startswith("dynamics")
        ]
        if not dynamics_rows or any(not bool(row["proper"]) for row in dynamics_rows):
            raise ValueError("rSLDS dynamics beliefs must all be proper")

        updated_dynamics = []
        for index, posterior in enumerate(self.dynamics):
            updated_dynamics.append(
                posterior.update(
                    syy=np.sum([row["Syy"][index] for row in dynamics_rows], axis=0),
                    syx=np.sum([row["Syx"][index] for row in dynamics_rows], axis=0),
                    sxx=np.sum([row["Sxx"][index] for row in dynamics_rows], axis=0),
                    n=float(
                        np.sum([row["responsibility"][index] for row in dynamics_rows])
                    ),
                )
            )

        recurrence_rows = engine.recurrence_weight_stats(state)
        recurrence_rows = [
            recurrence_rows[name]
            for name in sorted(recurrence_rows)
            if name.startswith("switch")
        ]
        if not recurrence_rows or any(
            not bool(row["proper"]) for row in recurrence_rows
        ):
            raise ValueError("rSLDS recurrence beliefs must all be proper")
        recurrence = self.recurrence.update(
            np.stack([row["linear"] for row in recurrence_rows]),
            np.stack([row["quadratic"] for row in recurrence_rows]),
        )
        return RSLDSLearning(
            initial_alpha=self.initial_alpha + np.asarray(regime),
            dynamics=tuple(updated_dynamics),
            recurrence=recurrence,
        )

    def parameter_arrays(self) -> dict[str, np.ndarray]:
        """Point parameters matching the blob names returned by :func:`rslds`."""
        rolled = [posterior.rollover() for posterior in self.dynamics]
        weights, offsets = self.recurrence.rollover()
        return {
            "initial_regime_probability": self.initial_alpha / self.initial_alpha.sum(),
            "dynamics_matrix": np.stack([row[0] for row in rolled]),
            "dynamics_offset": np.stack([row[1] for row in rolled]),
            "dynamics_covariance": np.stack([row[2] for row in rolled]),
            "recurrence_weights": weights,
            "recurrence_offsets": offsets,
        }

    def install(self, engine: Engine, state: State) -> State:
        """Install rollover values into a fresh state without recompilation."""
        fresh = engine.init_state()
        if any(
            not np.array_equal(np.asarray(value), np.asarray(expected))
            for value, expected in (
                (state.messages, fresh.messages),
                (state.g_eta, fresh.g_eta),
                (state.g_lam, fresh.g_lam),
                (state.ef_eta, fresh.ef_eta),
            )
        ):
            raise ValueError(
                "rSLDS learning parameters require a fresh inference state"
            )
        values = self.parameter_arrays()
        params = dict(state.params)

        try:
            prior_key, prior_row = engine.arena.member_of["z0_prior"]
        except KeyError:
            raise ValueError("engine is not an rslds model with z0_prior") from None
        probability = values["initial_regime_probability"]
        params[prior_key] = (
            params[prior_key]
            .at[prior_row]
            .set(jnp.log(jnp.asarray(probability, dtype=params[prior_key].dtype)))
        )

        a = values["dynamics_matrix"]
        b = values["dynamics_offset"]
        w = np.linalg.inv(values["dynamics_covariance"])
        found_dynamics = False
        for group in engine.arena.mix_groups:
            rows = [
                i for i, name in enumerate(group.members) if name.startswith("dynamics")
            ]
            if not rows:
                continue
            found_dynamics = True
            for suffix, value in (("A", a), ("b", b), ("W", w)):
                key = f"{group.name}:{suffix}"
                cast = jnp.asarray(value, dtype=params[key].dtype)
                params[key] = (
                    params[key]
                    .at[jnp.asarray(rows)]
                    .set(jnp.broadcast_to(cast, (len(rows),) + cast.shape))
                )

        weights = values["recurrence_weights"]
        offsets = values["recurrence_offsets"]
        found_recurrence = False
        for group in engine.arena.recurrence_groups:
            rows = [
                i for i, name in enumerate(group.members) if name.startswith("switch")
            ]
            if not rows:
                continue
            found_recurrence = True
            for suffix, value in (("weights", weights), ("offsets", offsets)):
                key = f"{group.name}:{suffix}"
                cast = jnp.asarray(value, dtype=params[key].dtype)
                params[key] = (
                    params[key]
                    .at[jnp.asarray(rows)]
                    .set(jnp.broadcast_to(cast, (len(rows),) + cast.shape))
                )
        if not found_dynamics or not found_recurrence:
            raise ValueError(
                "engine does not contain the rslds dynamics and recurrence factors"
            )
        return _replace(state, params=params)


def vague_rslds_learning(
    regimes: int,
    dimension: int,
    *,
    initial_strength: float = 1.0,
    dynamics_scale: float = 1.0,
    recurrence_strength: float = 1e-3,
) -> RSLDSLearning:
    """Construct proper weak posteriors for an rSLDS trial loop."""
    if not np.isfinite(initial_strength) or initial_strength <= 0:
        raise ValueError("initial_strength must be finite and positive")
    return RSLDSLearning(
        initial_alpha=np.full(regimes, initial_strength, dtype=np.float64),
        dynamics=tuple(
            conj.vague_mniw(dimension, dimension, scale=dynamics_scale)
            for _ in range(regimes)
        ),
        recurrence=conj.vague_pg_weights(
            regimes, dimension, strength=recurrence_strength
        ),
    )
