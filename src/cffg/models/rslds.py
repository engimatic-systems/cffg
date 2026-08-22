"""Recurrent switching linear dynamical systems built from CFFG factors.

The model uses one normalized state-dependent transition,
``recurrence_link(z[t+1], z[t], x[t])``. Its source-specific offsets carry the
ordinary transition baseline; a separate ``matrix_cat`` would multiply two
conditionals without accounting for their state-dependent normalizer.

Learning is an explicit between-trial variational-EM boundary. Inference uses
point parameters, factor-local soft statistics update NumPy conjugate
posteriors, and :class:`RSLDSLearning` delegates their declared updates and
writes to :class:`cffg.learn.Learnables` without recompiling the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cffg import conj
from cffg.core.ir import (
    BlobRef,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.engine import Engine, RunResult, State, _replace
from cffg.learn import Learnables
from cffg.learning_checkpoint import (
    D10_RECURRENCE_STATISTIC_IDENTITY,
    RECURRENCE_PRIOR_IDENTITY,
    RECURRENCE_STATISTIC_IDENTITY,
)


INITIAL_REGIME_LEARNABLE = "initial_regime"
DYNAMICS_LEARNABLE = "dynamics"
RECURRENCE_LEARNABLE = "recurrence"
EMISSION_LEARNABLE = "emission"
RSLDS_MODEL_LEARNABLES = (
    INITIAL_REGIME_LEARNABLE,
    DYNAMICS_LEARNABLE,
    RECURRENCE_LEARNABLE,
)


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
    *,
    emission_offset: np.ndarray | None = None,
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
    emission_bias = (
        np.zeros(h.shape[0], dtype=h.dtype)
        if emission_offset is None and h.ndim == 2
        else np.asarray(emission_offset)
    )
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
        "emission_offset": (observation_dimension,),
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
        "emission_offset": emission_bias.shape,
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
        "emission_offset": emission_bias,
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

    template.learnables[INITIAL_REGIME_LEARNABLE] = LearnableSpec(
        INITIAL_REGIME_LEARNABLE,
        "dirichlet",
        ("initial_regime_probability",),
        ("z0_prior",),
    )
    template.learnables[DYNAMICS_LEARNABLE] = LearnableSpec(
        DYNAMICS_LEARNABLE,
        "component_mniw",
        ("dynamics_matrix", "dynamics_offset", "dynamics_covariance"),
        tuple(f"dynamics{step}" for step in range(1, n_steps)),
    )
    template.learnables[RECURRENCE_LEARNABLE] = LearnableSpec(
        RECURRENCE_LEARNABLE,
        "pg_weights",
        ("recurrence_weights", "recurrence_offsets"),
        tuple(f"switch{step}" for step in range(1, n_steps)),
    )
    template.learnables[EMISSION_LEARNABLE] = LearnableSpec(
        EMISSION_LEARNABLE,
        "mniw",
        ("emission_matrix", "emission_offset", "emission_covariance"),
    )

    return template, arrays


@dataclass(frozen=True)
class RSLDSLearning:
    """Compatibility facade for the three-family between-trial boundary.

    The historical API keeps the tied emission fixed. Statistics,
    hyperparameter state, checkpoint encoding, and runtime writes delegate to
    a restricted :class:`Learnables` instance over the serialized rSLDS
    declarations. Direct construction declares the supplied recurrence block
    as a fresh prior; updates and restored checkpoints carry the engine's
    D9 or D10 statistic identity explicitly.
    """

    initial_alpha: np.ndarray
    dynamics: tuple[conj.MNIW, ...]
    recurrence: conj.PGWeights
    recurrence_statistic_identity: str = RECURRENCE_PRIOR_IDENTITY

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
        if self.recurrence_statistic_identity not in {
            RECURRENCE_PRIOR_IDENTITY,
            RECURRENCE_STATISTIC_IDENTITY,
            D10_RECURRENCE_STATISTIC_IDENTITY,
        }:
            raise ValueError(
                "rSLDS recurrence state has an unknown or legacy statistic "
                "identity"
            )
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

    def _runtime(self, engine: Engine) -> Learnables:
        learning = Learnables(
            engine,
            priors={
                INITIAL_REGIME_LEARNABLE: self.initial_alpha,
                DYNAMICS_LEARNABLE: self.dynamics,
                RECURRENCE_LEARNABLE: self.recurrence,
            },
            only=RSLDS_MODEL_LEARNABLES,
        )
        if self.recurrence_statistic_identity not in {
            RECURRENCE_PRIOR_IDENTITY,
            learning.recurrence_statistic_identity,
        }:
            raise ValueError(
                "rSLDS recurrence checkpoint identity does not match this "
                "engine's executable statistic semantics"
            )
        return learning

    @classmethod
    def _from_runtime(cls, learning: Learnables) -> RSLDSLearning:
        dynamics = learning.params[DYNAMICS_LEARNABLE]
        return cls(
            initial_alpha=np.asarray(
                learning.params[INITIAL_REGIME_LEARNABLE]["alpha"]
            ),
            dynamics=tuple(
                conj.MNIW(
                    dynamics["m"][index],
                    dynamics["lam"][index],
                    dynamics["psi"][index],
                    float(dynamics["nu"][index]),
                )
                for index in range(dynamics["m"].shape[0])
            ),
            recurrence=conj.PGWeights(
                learning.params[RECURRENCE_LEARNABLE]["eta"],
                learning.params[RECURRENCE_LEARNABLE]["precision"],
            ),
            recurrence_statistic_identity=(
                learning.recurrence_statistic_identity
            ),
        )

    def update(self, engine: Engine, result: RunResult) -> RSLDSLearning:
        """Accumulate one distinct, converged sequence through Learnables."""
        learning = self._runtime(engine)
        learning.accumulate(engine, result)
        return self._from_runtime(learning)

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
            not np.array_equal(
                np.asarray(value),
                np.asarray(expected, dtype=np.asarray(value).dtype),
            )
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
        params = self._runtime(engine).write(engine, state.params)
        return engine.reset_structured_hybrid(_replace(state, params=params))

    def state_dict(self, engine: Engine) -> dict[str, np.ndarray]:
        """Return hyperparameters plus the structured-statistic manifest."""
        return self._runtime(engine).state_dict()

    def load_state_dict(
        self, engine: Engine, data: dict[str, np.ndarray]
    ) -> RSLDSLearning:
        """Restore only a checkpoint identified as current D9 or D10."""
        learning = self._runtime(engine)
        learning.load_state_dict(data)
        return self._from_runtime(learning)


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
