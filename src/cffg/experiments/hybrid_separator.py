"""D12 hybrid EC equations; an experiment, with no production binding.

A transition region owns recurrence and switching dynamics jointly. Its two
separators preserve per-regime Gaussian moments. Gaussian normalizer constants
are part of each regime's natural coordinate: only a common scalar is gauge.
The constrained EC objective is not a bound on global negative log evidence.
"""

from dataclasses import dataclass

import numpy as np

from .regime_pair import gaussian_moments, logsumexp, quadratic_bound


@dataclass(frozen=True)
class Natural:
    constant: np.ndarray
    linear: np.ndarray
    precision: np.ndarray

    def __add__(self, other):
        if any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(self.arrays(), other.arrays())):
            raise ValueError("natural coordinate shapes or dtypes disagree")
        return Natural(*(a + b for a, b in zip(self.arrays(), other.arrays())))

    def arrays(self):
        return self.constant, self.linear, self.precision


@dataclass(frozen=True)
class Belief:
    probability: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    log_normalizer: float

    @property
    def second(self):
        return self.covariance + self.mean[..., :, None] * self.mean[..., None, :]

    def entropy(self):
        value = self.probability.dtype.type(0)
        dim = self.mean.shape[-1]
        for index in zip(*np.nonzero(self.probability)):
            p = self.probability[index]
            _, determinant = np.linalg.slogdet(self.covariance[index])
            value += p * (-np.log(p) + (dim * (1 + np.log(2 * np.pi)) + determinant) / 2)
        return value

    def expectation(self, natural):
        active = self.probability > 0
        energy = natural.constant[active] + np.einsum("...d,...d->...", natural.linear[active], self.mean[active])
        energy -= np.einsum("...ab,...ab->...", natural.precision[active], self.second[active]) / 2
        return np.sum(self.probability[active] * energy)


def validate(natural):
    a, h, j = natural.arrays()
    if a.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("coordinates must use float32 or float64")
    if h.ndim != a.ndim + 1 or h.shape[:-1] != a.shape or h.shape[-1] < 1:
        raise ValueError("natural coordinate shapes disagree")
    if j.shape != h.shape + (h.shape[-1],) or any(x.dtype != a.dtype for x in (h, j)):
        raise ValueError("natural coordinate shapes or dtypes disagree")
    if np.isnan(a).any() or np.isposinf(a).any() or not all(np.isfinite(x).all() for x in (h, j)):
        raise ValueError("non-finite natural coordinate")
    if not np.array_equal(j, np.swapaxes(j, -1, -2)):
        raise ValueError("precision must be symmetric")
    if not np.isfinite(a).any():
        raise ValueError("empty supported region")


def normalize(natural):
    """Only active aggregate/tilted blocks must be proper; messages may be signed."""
    validate(natural)
    a, h, j = natural.arrays()
    mean, covariance = np.zeros_like(h), np.zeros_like(j)
    scores = np.full_like(a, -np.inf)
    for index in zip(*np.nonzero(np.isfinite(a))):
        mean[index], covariance[index], mass = gaussian_moments(h[index], j[index])
        covariance[index] = (covariance[index] + covariance[index].T) / 2
        scores[index] = a[index] + mass
    normalizer = logsumexp(scores)
    if not np.isfinite(normalizer):
        raise ValueError("non-finite objective/normalizer")
    probability = np.exp(scores - normalizer)
    if np.any(np.isfinite(scores) & (probability == 0)):
        raise ValueError("positive support underflow; rescale the declared problem")
    return Belief(probability, mean, covariance, float(normalizer))


def canonical(belief):
    """Normalized per-label density; keep the label-dependent Gaussian scale."""
    p, m, s = belief.probability, belief.mean, belief.covariance
    a, h, j = np.full_like(p, -np.inf), np.zeros_like(m), np.zeros_like(s)
    for index in zip(*np.nonzero(p)):
        _, inverse, _ = gaussian_moments(np.zeros_like(m[index]), s[index])
        inverse = (inverse + inverse.T) / 2
        j[index], h[index] = inverse, inverse @ m[index]
        _, _, mass = gaussian_moments(h[index], j[index])
        a[index] = np.log(p[index]) - mass
    return Natural(a, h, j)


def quotient(belief, cavity):
    """Projection divided by cavity, on a fixed supported face; no precision clamp."""
    projected = canonical(belief)
    active = belief.probability > 0
    if np.any(active & ~np.isfinite(cavity.constant)):
        raise ValueError("support changed; rebuild active coordinates")
    a = np.full_like(projected.constant, -np.inf)
    a[active] = projected.constant[active] - cavity.constant[active]
    a[active] -= a[active][0]
    h = np.where(active[..., None], projected.linear - cavity.linear, 0)
    j = np.where(active[..., None, None], projected.precision - cavity.precision, 0)
    return Natural(a, h, j)


def collapse_label(region, axis):
    """Drop one discrete label while retaining all continuous moments."""
    probability = region.probability.sum(axis=axis)
    dim = region.mean.shape[-1]
    mean = np.zeros((len(probability), dim), dtype=probability.dtype)
    covariance = np.zeros((len(probability), dim, dim), dtype=probability.dtype)
    for i in np.flatnonzero(probability):
        p = region.probability[i, :] if axis == 1 else region.probability[:, i]
        m = region.mean[i, :] if axis == 1 else region.mean[:, i]
        s = region.covariance[i, :] if axis == 1 else region.covariance[:, i]
        conditional = p / probability[i]
        mean[i] = np.einsum("k,kd->d", conditional, m)
        centered = m - mean[i]
        covariance[i] = np.einsum("k,kab->ab", conditional, s)
        covariance[i] += np.einsum("k,ka,kb->ab", conditional, centered, centered)
    return Belief(probability, mean, covariance, 0)


def marginal(region, side):
    """Moment projection conditional on each endpoint label."""
    dim = region.mean.shape[-1] // 2
    selection = slice(side * dim, (side + 1) * dim)
    return collapse_label(Belief(region.probability, region.mean[..., selection],
                                region.covariance[..., selection, selection], 0), 1 - side)


def lift(left, right):
    """Embed two hybrid endpoint cavities in a pair region without merging labels."""
    validate(left)
    validate(right)
    if left.linear.shape != right.linear.shape or left.constant.ndim != 1:
        raise ValueError("endpoint shapes disagree")
    card, dim = left.linear.shape
    h = np.zeros((card, card, 2 * dim), dtype=left.linear.dtype)
    j = np.zeros((card, card, 2 * dim, 2 * dim), dtype=left.linear.dtype)
    h[..., :dim], h[..., dim:] = left.linear[:, None], right.linear[None, :]
    j[..., :dim, :dim], j[..., dim:, dim:] = left.precision[:, None], right.precision[None, :]
    return Natural(left.constant[:, None] + right.constant[None, :], h, j)


def transition(weights, offsets, c, matrix, bias, noise):
    """JJ recurrence times N(x_next; A_j x + b_j, Q_j), D12.2.

    Indexing dynamics by destination is a declared model convention. A source
    or pair-indexed dynamics extension changes this factor construction only.
    Normalized dynamics constants are retained for the objective and evidence.
    """
    weights, offsets, c, matrix, bias, noise = map(np.asarray, (weights, offsets, c, matrix, bias, noise))
    if weights.ndim != 3:
        raise ValueError("weights require [K,K-1,d]")
    card, sticks, dim = weights.shape
    if card < 2 or sticks != card - 1 or dim < 1 or any((
        offsets.shape != (card, sticks), c.shape != offsets.shape,
        matrix.shape != (card, dim, dim), bias.shape != (card, dim), noise.shape != matrix.shape,
    )):
        raise ValueError("transition shapes disagree")
    values = (weights, offsets, c, matrix, bias, noise)
    if weights.dtype not in (np.dtype("float32"), np.dtype("float64")) or any(x.dtype != weights.dtype for x in values):
        raise ValueError("transition dtypes disagree")
    if not all(np.isfinite(x).all() for x in values) or (c < 0).any():
        raise ValueError("invalid transition coordinate")
    if not np.array_equal(noise, noise.swapaxes(-1, -2)):
        raise ValueError("noise covariance must be symmetric")
    rh, rj, ra = quadratic_bound(weights, offsets, c)
    h = np.zeros((card, card, 2 * dim), dtype=weights.dtype)
    jmat = np.zeros((card, card, 2 * dim, 2 * dim), dtype=weights.dtype)
    a = ra.copy()
    for k in range(card):
        _, inv, _ = gaussian_moments(np.zeros(dim, dtype=weights.dtype), noise[k])
        a[:, k] -= (dim * np.log(2 * np.pi) + np.linalg.slogdet(noise[k])[1] + bias[k] @ inv @ bias[k]) / 2
        h[:, k, :dim] = rh[:, k] - matrix[k].T @ inv @ bias[k]
        h[:, k, dim:] = inv @ bias[k]
        jmat[:, k, :dim, :dim] = rj[:, k] + matrix[k].T @ inv @ matrix[k]
        jmat[:, k, :dim, dim:] = -matrix[k].T @ inv
        jmat[:, k, dim:, :dim] = -inv @ matrix[k]
        jmat[:, k, dim:, dim:] = inv
    # Preserve mathematical symmetry across independent floating-point contractions.
    jmat = (jmat + jmat.swapaxes(-1, -2)) / 2
    return Natural(a, h, jmat)


def source_region(region):
    """The recurrence triplet marginal of the joint transition belief."""
    dim = region.mean.shape[-1] // 2
    return Belief(region.probability, region.mean[..., :dim], region.covariance[..., :dim, :dim], 0)


def chain_energy(regions, separators, factors, sites):
    """D12.3: transition entropy once, internal separator correction once.

    Canonical unary sites own their expected log potential and no extra
    entropy. This is the full EC functional even away from consistency; only
    feasible beliefs or a stationary dual point receive an objective claim.
    """
    value = sum(-b.entropy() - b.expectation(f) for b, f in zip(regions, factors))
    value += sum(q.entropy() for q in separators[1:-1])
    value -= sum(q.expectation(s) for q, s in zip(separators, sites))
    if not np.isfinite(value):
        raise ValueError("non-finite objective")
    return float(value)


def evaluate(factors, sites, messages):
    """One O(T K² d³) fixed-auxiliary evaluation, with only local covariance."""
    if not factors or len(sites) != len(factors) + 1 or len(messages) != len(factors):
        raise ValueError("chain lengths disagree")
    totals = list(sites)
    for t, (left, right) in enumerate(messages):
        totals[t], totals[t + 1] = totals[t] + left, totals[t + 1] + right
    separators = [normalize(total) for total in totals]
    regions, targets, cavities = [], [], []
    for t, factor in enumerate(factors):
        left = sites[t] if t == 0 else sites[t] + messages[t - 1][1]
        right = sites[t + 1] if t == len(factors) - 1 else sites[t + 1] + messages[t + 1][0]
        region = normalize(factor + lift(left, right))
        regions.append(region)
        targets.append((quotient(marginal(region, 0), left), quotient(marginal(region, 1), right)))
        cavities.append((left, right))
    energy = chain_energy(regions, separators, factors, sites)
    # At stationarity this dual equals the feasible primal energy above.
    dual = -sum(b.log_normalizer for b in regions)
    for t, q in enumerate(separators):
        if t not in (0, len(separators) - 1):
            dual += q.log_normalizer
    if not np.isfinite(dual):
        raise ValueError("non-finite dual objective")
    return regions, separators, targets, energy, float(dual)


def initial_messages(sites):
    def zero(site):
        return Natural(np.where(np.isfinite(site.constant), 0, -np.inf).astype(site.constant.dtype),
                       np.zeros_like(site.linear), np.zeros_like(site.precision))
    return [(zero(a), zero(b)) for a, b in zip(sites[:-1], sites[1:])]


def message_distance(messages, targets):
    maximum = 0.0
    for pair, target in zip(messages, targets):
        for old, new in zip(pair, target):
            validate(old)
            validate(new)
            if not np.array_equal(np.isfinite(old.constant), np.isfinite(new.constant)):
                raise ValueError("support changed; rebuild active coordinates")
            for a, b in zip(old.arrays(), new.arrays()):
                active = np.isfinite(a)
                maximum = max(maximum, float(np.max(abs(b[active] - a[active]) / (1 + abs(a[active])))))
    return maximum


def damp(messages, targets, fraction):
    result = []
    for pair, target in zip(messages, targets):
        endpoints = []
        for old, new in zip(pair, target):
            a = old.constant.copy()
            active = np.isfinite(a)
            a[active] += fraction * (new.constant[active] - a[active])
            h = old.linear + fraction * (new.linear - old.linear)
            j = old.precision + fraction * (new.precision - old.precision)
            endpoints.append(Natural(a, h, j))
        result.append(tuple(endpoints))
    return result


@dataclass(frozen=True)
class ChainResult:
    regions: list
    separators: list
    messages: list
    free_energy: float
    dual_energy: float
    residual: float
    status: str
    iterations: int


def solve(factors, sites, *, max_iterations=256, max_backtracks=24):
    """Damped fixed-JJ EC solve; finite residual and FE stability certify success.

    Backtracking decreases the fixed-point residual and rejects improper
    proposals. EC need not be convex, and this direction need not be descent:
    failure to find progress is exhaustion, never a claim of convergence.
    The experiment's caller supplies held auxiliaries; D12 separately states
    the complete auxiliary residual for a joint solve.
    """
    if max_iterations < 1 or max_backtracks < 1:
        raise ValueError("iteration budgets must be positive")
    messages = initial_messages(sites)
    for t in range(len(factors)):
        left = sites[t] if t == 0 else sites[t] + messages[t - 1][1]
        right = sites[t + 1]
        region = normalize(factors[t] + lift(left, right))
        messages[t] = (quotient(marginal(region, 0), left), quotient(marginal(region, 1), right))
    values = evaluate(factors, sites, messages)
    tolerance = 64 * np.finfo(sites[0].constant.dtype).eps ** 0.75
    previous, status = None, "budget_exhausted"
    for iteration in range(1, max_iterations + 1):
        regions, separators, targets, energy, dual = values
        residual = message_distance(messages, targets)
        delta = np.inf if previous is None else abs(energy - previous) / (1 + abs(previous))
        if max(residual, delta) <= tolerance:
            status = "converged"
            break
        accepted = False
        for backtrack in range(max_backtracks):
            proposed = damp(messages, targets, 0.5 ** backtrack)
            try:
                next_values = evaluate(factors, sites, proposed)
                next_residual = message_distance(proposed, next_values[2])
            except ValueError:
                continue
            if next_residual < residual or max(residual, next_residual) <= tolerance:
                previous, messages, values = energy, proposed, next_values
                accepted = True
                break
        if not accepted:
            break
    regions, separators, targets, energy, dual = values
    return ChainResult(regions, separators, messages, energy, dual,
                       message_distance(messages, targets), status, iteration)


def gaussian_adapter(hybrid_cavity, gaussian_cavity):
    """D12 identity adapter from (i,x) to an ordinary Gaussian copy u=x.

    Its reduced belief lives on (i,x), not on a density containing a Dirac
    delta. It owns zero factor energy and minus its labeled entropy. Only
    unconditional moments cross the Gaussian port; association stays hybrid.
    """
    validate(gaussian_cavity)
    if gaussian_cavity.constant.shape != ():
        raise ValueError("adapter requires one ordinary Gaussian cavity")
    a = np.broadcast_to(gaussian_cavity.constant, hybrid_cavity.constant.shape)
    h = np.broadcast_to(gaussian_cavity.linear, hybrid_cavity.linear.shape)
    j = np.broadcast_to(gaussian_cavity.precision, hybrid_cavity.precision.shape)
    belief = normalize(hybrid_cavity + Natural(a, h, j))
    p, m, s = belief.probability, belief.mean, belief.covariance
    mean = np.einsum("k,kd->d", p, m)
    centered = m - mean
    covariance = np.einsum("k,kab->ab", p, s) + np.einsum("k,ka,kb->ab", p, centered, centered)
    projected = Belief(np.array([1], dtype=p.dtype), mean[None], covariance[None], 0)
    ordinary = Natural(gaussian_cavity.constant[None], gaussian_cavity.linear[None], gaussian_cavity.precision[None])
    message = quotient(projected, ordinary)
    return belief, quotient(belief, hybrid_cavity), Natural(*(x[0] for x in message.arrays()))


def auxiliary_residual(regions, weights, offsets, c):
    """The complete JJ residual uses xi=c²; c=0 alone is not stationarity."""
    from .regime_pair import auxiliary_target

    if not (len(regions) == len(weights) == len(offsets) == len(c)) or np.shape(offsets) != np.shape(c):
        raise ValueError("auxiliary coordinate shapes disagree")
    values = []
    for b, r, bias, aux in zip(regions, weights, offsets, c):
        target = auxiliary_target(source_region(b), r, bias)
        values.append(aux * aux - target * target)
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise ValueError("diverged: non-finite auxiliary residual")
    return result


def categorical_adapter(hybrid_cavity, log_cavity):
    """Identity adapter to an ordinary categorical copy v=i, on active support."""
    if log_cavity.shape != hybrid_cavity.constant.shape or log_cavity.dtype != hybrid_cavity.constant.dtype:
        raise ValueError("categorical adapter shapes or dtypes disagree")
    embedded = Natural(log_cavity, np.zeros_like(hybrid_cavity.linear), np.zeros_like(hybrid_cavity.precision))
    belief = normalize(hybrid_cavity + embedded)
    active = belief.probability > 0
    outgoing = np.full_like(log_cavity, -np.inf)
    outgoing[active] = np.log(belief.probability[active]) - log_cavity[active]
    outgoing -= logsumexp(outgoing)
    return belief, quotient(belief, hybrid_cavity), outgoing
