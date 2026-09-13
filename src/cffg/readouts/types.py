"""Readout value records and named dependencies, without public orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cffg.runtime_types import State


@dataclass(frozen=True)
class EfMarginal:
    """A scalar EF edge belief: standard parameters, e.g. (shape, rate) for
    gamma, (a, b) for beta."""

    family: str
    params: tuple[float, ...]
    mean: float
    proper: bool


@dataclass(frozen=True)
class DirichletMarginal:
    """One in-graph table posterior in canonical serialized layout."""

    alpha: np.ndarray
    mean: np.ndarray
    expected_log: np.ndarray
    proper: bool


type Marginals = Callable[[State], dict[str, np.ndarray]]
type ParameterBeliefs = Callable[[State], dict[str, DirichletMarginal]]
type PairBeliefs = Callable[[State], dict[str, dict[str, np.ndarray]]]
