"""Canonical parameter initializers: the warm-start vocabulary (cff-wx28).

Models rarely want uniform or random parameters. They want *structured*
canonical shapes — a diagonal-dominant "sticky" transition, a Gaussian band
along the diagonal, a one-hot goal, shift dynamics for a counter/corridor —
and Dirichlet learners want those same shapes as counts with a confidence
knob. This module is a registry of pure functions producing exactly that.

Two ways in:

- **Programmatic**: ``make("banded", (6, 6), sigma=1.5)`` → ndarray.
- **In the template**: ``BlobRef("B", "B.npy", init=spec("sticky", p_stay=0.8))``
  — the blob then needs no file: the compiler materializes it from the spec
  (resolution order: explicit arrays > existing file > init spec), so a
  serialized template is fully self-contained, and a continual-learning
  checkpoint written to the file path overrides the warm start on reload.

Conventions:

- Matrix kinds produce **column-stochastic** arrays (columns = conditioning
  states, matching the engine's factor convention); vector kinds produce
  distributions. Extra leading axes broadcast: asking `sticky` for shape
  (U, K, K) gives U stacked copies — the transition_mixture layout.
- Every probability kind accepts ``strength`` (+ optional ``floor`` > 0,
  default 0.05): if given, the result is ``strength * P + floor`` — a
  **Dirichlet count** warm start whose confidence is the strength. One knob
  turns any template into an informed prior for learn_a / learn_b.
- Anything random takes an explicit ``seed`` (reproducibility invariant,
  AGENTS.md); same spec ⇒ same array, forever.
- Spec values serialize as strings (canonical sorted tuple, like constraint
  details); ``make`` parses them back, so specs survive JSON round trips
  and stay hashable.
"""

from __future__ import annotations

import numpy as np

_FLOOR = 0.05


def _normalize_cols(m: np.ndarray) -> np.ndarray:
    return m / m.sum(axis=-2, keepdims=True)


def _square(shape: tuple[int, ...], name: str) -> tuple[int, tuple[int, ...]]:
    """Matrix kinds need trailing (K, K); leading axes are stacked copies."""
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(f"init {name!r} needs a trailing square (K, K) shape, got {shape}")
    return shape[-1], shape[:-2]


def _uniform(shape):
    if len(shape) == 1:
        return np.full(shape, 1.0 / shape[0])
    return np.full(shape, 1.0 / shape[-2])  # column-stochastic


def _one_hot(shape, index=0):
    v = np.zeros(shape)
    if len(shape) == 1:
        v[int(index)] = 1.0
        return v
    v[..., int(index), :] = 1.0  # every column maps to `index`
    return v


def _constant(shape, value=0.0):
    return np.full(shape, float(value))


def _zeros(shape):
    return np.zeros(shape)


def _eye(shape, scale=1.0):
    k, lead = _square(shape, "eye")
    return np.broadcast_to(float(scale) * np.eye(k), (*lead, k, k)).copy()


def _sticky(shape, p_stay=0.9):
    """Diagonal-dominant transition: stay w.p. p_stay, else uniform elsewhere."""
    k, lead = _square(shape, "sticky")
    p = float(p_stay)
    m = np.full((k, k), (1.0 - p) / (k - 1) if k > 1 else 0.0)
    np.fill_diagonal(m, p if k > 1 else 1.0)
    return np.broadcast_to(m, (*lead, k, k)).copy()


def _banded(shape, sigma=1.0, wrap=False):
    """A Gaussian band along the diagonal: column z puts mass ∝
    exp(−(z' − z)²/2σ²) on z', with distance wrapping on a ring if `wrap`.
    The canonical 'roughly stays put, drifts locally' dynamics template."""
    k, lead = _square(shape, "banded")
    idx = np.arange(k)
    d = np.abs(idx[:, None] - idx[None, :])
    if wrap:
        d = np.minimum(d, k - d)
    m = _normalize_cols(np.exp(-0.5 * (d / float(sigma)) ** 2))
    return np.broadcast_to(m, (*lead, k, k)).copy()


def _shift(shape, offset=1, p_move=1.0, wrap=False):
    """Move by `offset` w.p. p_move (clamped at edges, or wrapping), else stay:
    counters, corridors, conveyor dynamics."""
    k, lead = _square(shape, "shift")
    p = float(p_move)
    m = np.zeros((k, k))
    for z in range(k):
        t = (z + int(offset)) % k if wrap else min(max(z + int(offset), 0), k - 1)
        m[t, z] += p
        m[z, z] += 1.0 - p
    return np.broadcast_to(m, (*lead, k, k)).copy()


def _random_stochastic(shape, seed=0, concentration=1.0):
    """Seeded Dirichlet columns; low concentration ⇒ near-deterministic draws."""
    rng = np.random.default_rng(int(seed))
    if len(shape) == 1:
        return rng.dirichlet(np.full(shape[0], float(concentration)))
    rows = shape[-2]
    flat = rng.dirichlet(np.full(rows, float(concentration)),
                         size=int(np.prod(shape) // rows))
    return np.moveaxis(flat.reshape(*shape[:-2], shape[-1], rows), -1, -2).copy()


def _random_spd(shape, seed=0, scale=1.0):
    """Seeded symmetric positive-definite matrices (covariances)."""
    k, lead = _square(shape, "random_spd")
    rng = np.random.default_rng(int(seed))
    out = np.empty((*lead, k, k)) if lead else np.empty((k, k))
    flat = out.reshape(-1, k, k)
    for i in range(flat.shape[0]):
        a = rng.normal(size=(k, k))
        flat[i] = float(scale) * (a @ a.T / k + np.eye(k))
    return out


# Probability-template kinds: `strength` converts any of them to counts.
_PROB_KINDS = {
    "uniform": _uniform,
    "one_hot": _one_hot,
    "sticky": _sticky,
    "banded": _banded,
    "shift": _shift,
    "random_stochastic": _random_stochastic,
}
_RAW_KINDS = {
    "constant": _constant,
    "zeros": _zeros,
    "eye": _eye,
    "random_spd": _random_spd,
}
INITIALIZERS = {**_PROB_KINDS, **_RAW_KINDS}


def make(kind: str, shape: tuple[int, ...], **params) -> np.ndarray:
    """Materialize an initializer. Params may arrive as strings (from a
    serialized spec); numeric-looking ones are parsed. For probability kinds,
    `strength` (with optional `floor`) yields Dirichlet counts instead."""
    if kind not in INITIALIZERS:
        raise ValueError(f"unknown initializer {kind!r}; have {sorted(INITIALIZERS)}")
    parsed = {k: _parse(v) for k, v in params.items()}
    strength = parsed.pop("strength", None)
    floor = parsed.pop("floor", _FLOOR)
    if strength is not None and kind not in _PROB_KINDS:
        raise ValueError(f"init {kind!r} is not a probability template; strength is meaningless")
    arr = INITIALIZERS[kind](tuple(int(s) for s in shape), **parsed)
    if strength is not None:
        if floor <= 0:
            raise ValueError("counts need floor > 0 (Dirichlet counts must be positive)")
        return float(strength) * arr + float(floor)
    return arr


def spec(kind: str, **params) -> tuple[tuple[str, str], ...]:
    """Canonical serialized form of an init: sorted (key, value-as-string)
    pairs prefixed by the kind — hashable, JSON-safe, order-insensitive."""
    if kind not in INITIALIZERS:
        raise ValueError(f"unknown initializer {kind!r}; have {sorted(INITIALIZERS)}")
    return (("kind", kind), *sorted((k, repr(v)) for k, v in params.items()))


def from_spec(s: tuple[tuple[str, str], ...], shape: tuple[int, ...]) -> np.ndarray:
    d = dict(s)
    kind = d.pop("kind")
    return make(kind, shape, **d)


def _parse(v):
    if not isinstance(v, str):
        return v
    try:
        f = float(v.strip("'\""))
        return int(f) if f == int(f) and "." not in v and "e" not in v.lower() else f
    except ValueError:
        s = v.strip("'\"")
        return {"True": True, "False": False}.get(s, s)
