"""Host expected log scores and incremental context winners for rMM reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import numpy as np
from scipy.special import digamma, gammaln

from cffg.experiments.axiom_evidence import release_view


_likelihood_view = jax.jit(release_view)


class Records(NamedTuple):
    features: np.ndarray
    categories: np.ndarray  # integer columns: own, other, used/unused, action, reward, switch


def validate_records(records, width, cardinalities):
    x, cats = map(np.asarray, records)
    if (x.ndim != 2 or x.shape[1] != width or x.dtype not in (np.float32, np.float64)
            or not np.isfinite(x).all()):
        raise ValueError("invalid generated continuous features")
    if (cats.shape != (len(x), 6) or cats.dtype.kind not in "iu" or len(cardinalities) != 6
            or np.any(cats < 0) or np.any(cats >= np.asarray(cardinalities))):
        raise ValueError("generated records require six in-range category indices")


def _effective(high, low):
    return np.asarray(high, dtype=np.float64) + np.asarray(low, dtype=np.float64)


def mixing(bank, prior):
    mass = _effective(bank.alpha, bank.carry.alpha)
    base = _effective(prior.alpha, prior.carry.alpha)
    # Compare the increments before adding the low words to a large count.
    excess = ((np.asarray(bank.alpha, float) - np.asarray(prior.alpha, float))
              + (np.asarray(bank.carry.alpha, float) - np.asarray(prior.carry.alpha, float)))
    return mass, base, excess


def mixing_kl(mass, base, excess):
    total = mass.sum()
    value = (gammaln(total) - gammaln(base.sum())
             + (gammaln(base) - gammaln(mass) + excess * (digamma(mass) - digamma(total))).sum())
    if not np.isfinite(value):
        raise ValueError("nonfinite Dirichlet mixing KL")
    return float(value)


@dataclass
class Tables:
    mean: np.ndarray
    precision: np.ndarray
    constant: np.ndarray
    log_heads: tuple[np.ndarray, ...]
    probability: tuple[np.ndarray, ...]

    @classmethod
    def build(cls, bank, *, minimum):
        b = _likelihood_view(bank, minimum=minimum)
        mu, psi, kappa, nu = (_effective(getattr(b, k), getattr(b.carry, k))
                              for k in ("mu", "psi", "kappa", "nu"))
        d = mu.shape[1]
        root = np.linalg.cholesky(psi)
        ident = np.broadcast_to(np.eye(d), psi.shape)
        inverse = np.linalg.solve(root, ident)
        precision = (inverse.swapaxes(-1, -2) @ inverse) * nu[:, None, None]
        logdet = 2 * np.log(np.diagonal(root, axis1=-2, axis2=-1)).sum(axis=-1)
        elog = digamma((nu[:, None] - np.arange(d)) / 2).sum(axis=1)
        constant = .5 * (elog + d*np.log(2.) - logdet - d*np.log(2*np.pi) - d/kappa)
        heads = tuple(_effective(a, l) for a, l in zip(b.categories, b.carry.categories, strict=True))
        if len(heads) != 6:
            raise ValueError("rMM reduction requires six categorical heads")
        log_heads = tuple(digamma(h) - digamma(h.sum(axis=1, keepdims=True)) for h in heads)
        probabilities = tuple(h / h.sum(axis=1, keepdims=True) for h in heads)
        if not all(np.isfinite(a).all() for a in (mu, precision, constant, *log_heads)):
            raise ValueError("invalid release-projected likelihood")
        return cls(mu, precision, constant, log_heads, probabilities)

    def install(self, ids, new):
        for full, rows in zip((self.mean, self.precision, self.constant, *self.log_heads, *self.probability),
                              (new.mean, new.precision, new.constant, *new.log_heads, *new.probability),
                              strict=True):
            full[ids] = rows

    def gaussian(self, points, ids):
        delta = points[:, None] - self.mean[ids]
        return self.constant[ids] - .5*np.einsum("nki,kij,nkj->nk", delta, self.precision[ids], delta)

    def context(self, tape, ids):
        scores = self.gaussian(tape.features, ids)
        for h, logs in enumerate(self.log_heads[:4]):
            scores += logs[ids][:, tape.categories[:, h]].T
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite context scores")
        return scores

    def output(self, categories, winners, evidence):
        nats = (self.log_heads[-2][winners, categories[:, -2]]
                + self.log_heads[-1][winners, categories[:, -1]])
        return np.where(evidence[winners], nats, 0.)


def soft_neighbors(tables, sources, block=64):
    """Nearest row by Gaussian point and all six categorical mean likelihoods."""
    k = len(tables.mean)
    nearest = np.empty(len(sources), dtype=np.intp)
    for off in range(0, len(sources), block):
        src = sources[off:off+block]
        best = np.full(len(src), -np.inf)
        ids = np.zeros(len(src), dtype=np.intp)
        for start in range(0, k, block):
            dst = np.arange(start, min(start+block, k))
            scores = tables.gaussian(tables.mean[src], dst)
            for mean, logs in zip(tables.probability, tables.log_heads, strict=True):
                # Multinomial base measures depend only on the query and cancel in argmax.
                scores += mean[src] @ logs[dst].T
            scores[src[:, None] == dst] = -np.inf
            which = scores.argmax(axis=1)
            maximum = scores[np.arange(len(src)), which]
            better = maximum > best  # scan ascending indices; retain the first exact tie
            ids = np.where(better, dst[which], ids)
            best = np.maximum(best, maximum)
        nearest[off:off+len(src)] = ids
    return nearest


class _Leaders(NamedTuple):
    scores: np.ndarray
    ids: np.ndarray


def _rank(scores, ids, count=3):
    """Return up to count leaders per record, breaking equal scores by slot index."""
    scores = scores.copy()
    n, k = scores.shape
    ids = np.broadcast_to(ids, scores.shape)
    width = min(count, k)
    result = _Leaders(np.empty((n, width)), np.empty((n, width), dtype=np.intp))
    row = np.arange(n)
    for pos in range(width):
        best = scores.max(axis=1)
        chosen = np.where(scores == best[:, None], ids, np.iinfo(np.intp).max).argmin(axis=1)
        result.scores[:, pos], result.ids[:, pos] = best, ids[row, chosen]
        scores[row, chosen] = -np.inf
    return result


def _unite(one, two):
    if one is None:
        return two
    return _rank(np.concatenate((one.scores, two.scores), axis=1),
                 np.concatenate((one.ids, two.ids), axis=1))


def _fixed_bounds(tables, tape, ids):
    """Bound context scores of fixed diagonal Gaussians in the working arithmetic."""
    # Every term of a diagonal squared deviation is nonnegative. Limit the
    # operand magnitudes so the eager finite-score check can never overflow
    # in the quadratic or the four score additions on rows we skip.
    limit = np.finfo(np.float64).max ** .25
    if not len(ids) or np.max(np.abs(tape.features), initial=0.) >= limit:
        return np.zeros(len(ids), bool), -np.inf
    precision = tables.precision[ids]
    d = precision.shape[-1]
    diag = np.diagonal(precision, axis1=-2, axis2=-1)
    eligible = (precision[:, ~np.eye(d, dtype=bool)] == 0).all(axis=1)
    eligible &= ((diag > 0) & (diag < limit)).all(axis=1)
    eligible &= (np.abs(tables.mean[ids]) < limit).all(axis=1)
    bound = tables.constant[ids].copy()
    eligible &= np.abs(bound) < limit
    for head in tables.log_heads[:4]:
        logs = head[ids]
        eligible &= (np.abs(logs) < limit).all(axis=1)
        bound += logs.max(axis=1)
    return eligible, bound[eligible].max(initial=-np.inf)


class _FixedWinners:
    """Remember the best dormant row for each record that ever needs it."""

    def __init__(self, ids, upper, tables, tape, block):
        self.ids, self.upper, self.tables, self.tape, self.block = ids, upper, tables, tape, block
        n = len(tape.features)
        self.score = np.full(n, -np.inf)
        self.winner = np.full(n, -1, dtype=np.intp)

    def resolve(self, scores, ids):
        if not len(self.ids):
            return ids
        missing = np.flatnonzero((scores <= self.upper) & (self.winner < 0))
        for n in range(0, len(missing), 512):
            rows = missing[n:n+512]
            tape = Records(self.tape.features[rows], self.tape.categories[rows])
            for start in range(0, len(self.ids), self.block):
                slots = self.ids[start:start+self.block]
                context = self.tables.context(tape, slots)
                local = context.argmax(axis=1)
                highest = context[np.arange(len(rows)), local]
                better = highest > self.score[rows]
                self.score[rows[better]] = highest[better]
                self.winner[rows[better]] = slots[local[better]]
        wins = (self.score > scores) | ((self.score == scores) & (self.winner < ids))
        return np.where(wins, self.winner, ids)


class Competition:
    """Keep the top three so replacing two columns cannot hide the best untouched row."""

    def __init__(self, tables, tape, mutable_ids, *, active, block=64):
        self.tables, self.tape = tables, tape
        mutable = np.asarray(sorted(set(map(int, mutable_ids))), dtype=np.intp)
        fixed = np.setdiff1d(np.arange(len(tables.mean)), mutable)
        dormant = fixed[~active[fixed]]
        eligible, upper = _fixed_bounds(tables, tape, dormant)
        deferred = dormant[eligible]
        self.dormant = _FixedWinners(deferred, upper, tables, tape, block)
        fixed = np.setdiff1d(fixed, deferred, assume_unique=True)
        self.fixed = None
        for start in range(0, len(fixed), block):
            self.fixed = _unite(self.fixed, self._block(fixed[start:start+block]))
        self.groups = [mutable[i:i+block] for i in range(0, len(mutable), block)]
        self.group_for = {int(slot): group for group, ids in enumerate(self.groups) for slot in ids}
        self.leaders = [self._block(ids) for ids in self.groups]
        self._refresh()

    def _block(self, ids):
        n = len(self.tape.features)
        width = min(3, len(ids))
        result = _Leaders(np.empty((n, width)), np.empty((n, width), dtype=np.intp))
        for start in range(0, n, 512):
            sl = slice(start, start+512)
            batch = Records(self.tape.features[sl], self.tape.categories[sl])
            top = _rank(self.tables.context(batch, ids), ids)
            result.scores[sl], result.ids[sl] = top
        return result

    def _refresh(self):
        best = self.fixed
        for top in self.leaders:
            best = _unite(best, top)
        self.best = best

    def winners(self):
        if self.best is None:
            return self.dormant.resolve(np.full(len(self.tape.features), -np.inf),
                                        np.zeros(len(self.tape.features), dtype=np.intp))
        return self.dormant.resolve(self.best.scores[:, 0], self.best.ids[:, 0])

    def trial_winners(self, pair, new):
        keep = ~np.isin(self.best.ids, pair)
        # At most two rows were replaced, so one of the first three survives
        # whenever the bank contains any unaffected row.
        row = np.arange(len(keep))
        col = keep.argmax(axis=1)
        other = np.where(keep.any(axis=1), self.best.scores[row, col], -np.inf)
        other_id = self.best.ids[row, col]
        scores = new.context(self.tape, np.arange(2))
        choices = np.column_stack((other_id, np.broadcast_to(pair, scores.shape)))
        best = _rank(np.column_stack((other, scores)), choices, 1)
        return self.dormant.resolve(best.scores[:, 0], best.ids[:, 0])

    def accept(self, pair, new):
        self.tables.install(pair, new)
        for group in sorted({self.group_for[int(slot)] for slot in pair}):
            self.leaders[group] = self._block(self.groups[group])
        self._refresh()
