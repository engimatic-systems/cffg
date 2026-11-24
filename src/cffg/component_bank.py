"""Host-side birth and merge transactions for a fixed-capacity mixed learner.

References belong to one bank lineage. A successful proposal is a new immutable
bank, ready for evaluation. Keep it to accept the proposal or keep the original
to discard it. The caller owns candidate selection and any random keys it uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg.mixed_learning import MixedBatch, update as update_mixed
from cffg.mixed_scoring import validate_bank
from cffg.mixed_state import MixedBank, MixedCarry


class ComponentRef(NamedTuple):
    slot: int
    logical_id: int
    generation: int


class SlotStatus(Enum):
    OK = "ok"
    FULL = "full"
    STALE_REFERENCE = "stale_reference"
    SAME_COMPONENT = "same_component"
    INVALID_POSTERIOR = "invalid_posterior"
    ID_EXHAUSTED = "id_exhausted"


class SlotResult(NamedTuple):
    bank: ComponentBank
    status: SlotStatus
    component: ComponentRef | None


_MAX_ID = np.iinfo(np.int64).max


def _supported(bank):
    """Give standalone validation a support point when all lifecycle slots are free."""
    if isinstance(bank.active, jax.core.Tracer):
        raise TypeError("component lifecycle transactions require a concrete host bank")
    if np.any(np.asarray(bank.active)):
        return bank
    return bank._replace(active=jnp.asarray(bank.active).at[0].set(True))


def _snapshot(bank, *, exact):
    record = _supported(bank).state_dict(exact=exact)
    record["active"] = np.array(bank.active, copy=True)
    return record


def _restore_bank(record, mixed_version):
    if np.asarray(record.get("version")).shape != () or record.get("version") != mixed_version:
        raise ValueError(f"component lifecycle requires mixed-bank version {mixed_version}")
    active = np.asarray(record.get("active"))
    if active.ndim != 1 or active.dtype != np.bool_ or not len(active):
        raise ValueError("invalid component mask")
    if active.any():
        return MixedBank.from_state_dict(record)
    temporary = active.copy()
    temporary[0] = True
    bank = MixedBank.from_state_dict({**record, "active": temporary})
    return bank._replace(active=jnp.array(active, copy=True))


def _count_fields(bank):
    for name in ("kappa", "nu", "alpha"):
        yield name, getattr(bank, name), getattr(bank.carry, name)
    for i, (values, lows) in enumerate(zip(bank.categories, bank.carry.categories, strict=True)):
        yield f"category_{i}", values, lows


def _check_counts(prior, posterior, *, compact_posterior=None):
    """Validate count ordering and align equivalent midpoint encodings in a compact copy."""
    for base_field, field in zip(_count_fields(prior), _count_fields(posterior), strict=True):
        name, base, base_low = base_field
        _, value, low = field
        base, base_low, value, low = map(np.asarray, (base, base_low, value, low))
        # Opposite low words can encode the identical midpoint at adjacent high words.
        shared = (low != 0) & (low == -base_low) & ((base - value) == low + low)
        if np.any(((value < base) & ~shared) | ((value == base) & (low < base_low))):
            raise ValueError("posterior count is below its reset prior")
        if compact_posterior is not None:
            # Equal counts must still have zero excess after both low words are dropped.
            np.copyto(compact_posterior[name], base, where=shared)


def _validate_layout(prior, posterior):
    lhs, left_tree = jax.tree.flatten(prior)
    rhs, right_tree = jax.tree.flatten(posterior)
    if left_tree != right_tree or any(
        a.shape != b.shape or a.dtype != b.dtype for a, b in zip(lhs, rhs)
    ):
        raise ValueError("posterior layout differs from the reset prior")
    _check_counts(prior, posterior)
    dormant = ~np.asarray(posterior.active)
    if dormant.any():
        for field in (*MixedCarry._fields, "carry"):
            for a, b in zip(jax.tree.leaves(getattr(prior, field)),
                            jax.tree.leaves(getattr(posterior, field)), strict=True):
                if not np.array_equal(np.asarray(a)[dormant], np.asarray(b)[dormant]):
                    raise ValueError("dormant component differs from its reset prior")


def _integer(state, key, shape):
    value = np.asarray(state.get(key))
    if value.shape != shape or value.dtype != np.int64 or np.any(value < 0):
        raise ValueError(f"invalid component snapshot {key}")
    return value


def _exact(high, low):
    return Fraction(float(high)) + Fraction(float(low))


def _nearest(value, dtype):
    """Round a rational directly to the source dtype, including float32 midpoints."""
    with np.errstate(over="ignore"):
        rounded = dtype.type(float(value))
    if not np.isfinite(rounded):
        raise ArithmeticError("component posterior overflow")
    error = value - Fraction(float(rounded))
    if dtype == np.float32 and error:
        with np.errstate(over="ignore"):
            neighbor = np.nextafter(rounded, dtype.type(np.inf if error > 0 else -np.inf))
        if np.isfinite(neighbor) and abs(value - Fraction(float(neighbor))) < abs(error):
            rounded = neighbor
    return rounded


def _split(value, dtype, original_high, original_low):
    original = _exact(original_high, original_low)
    if value == original:
        return original_high, original_low
    high = _nearest(value, dtype)
    low = _nearest(value - Fraction(float(high)), dtype)
    represented = _exact(high, low)
    # A rounded residual can land exactly on a half ULP. Normalize both words
    # so reloading the expansion reproduces its stored primary value.
    high = _nearest(represented, dtype)
    low = dtype.type(float(represented - Fraction(float(high))))
    if abs(represented - value) > Fraction(float(np.finfo(dtype).eps)) * abs(value - original):
        raise ArithmeticError("the two-word posterior cannot resolve the increment")
    return high, low


def _values(row):
    def combine(hi, lo):
        return np.vectorize(_exact, otypes=[object])(np.asarray(hi), np.asarray(lo))
    return jax.tree.map(combine, MixedCarry(*(getattr(row, k) for k in MixedCarry._fields)),
                        row.carry)


def _merged_row(receiver, donor, reset):
    """Subtract the donor prior once in centered NIW sufficient statistics.

    All arithmetic before rounding concerns just three rows. Exact rational
    temporaries retain both source words even when the two large Gaussian
    second moments in a prior subtraction nearly cancel.
    """
    r, d, p = map(_values, (receiver, donor, reset))

    def transfer(a, b, base):
        increment = b - base
        if np.any(increment < 0):
            raise ArithmeticError("donor count is below its reset prior")
        return a + increment

    kappa = transfer(r.kappa, d.kappa, p.kappa)
    nu = transfer(r.nu, d.nu, p.nu)
    alpha = transfer(r.alpha, d.alpha, p.alpha)
    categories = tuple(transfer(*triple) for triple in zip(
        r.categories, d.categories, p.categories, strict=True
    ))
    donor_delta = d.mu - r.mu
    prior_delta = p.mu - r.mu
    offset = (d.kappa * donor_delta - p.kappa * prior_delta) / kappa
    psi = (r.psi + (d.psi - p.psi)
           + d.kappa * np.outer(donor_delta, donor_delta)
           - p.kappa * np.outer(prior_delta, prior_delta)
           - kappa * np.outer(offset, offset))
    values = MixedCarry(r.mu + offset, kappa, psi, nu, alpha, categories)
    dtype = receiver.mu.dtype
    leaves, layout = jax.tree.flatten(values)
    hi, lo = [], []
    primary = MixedCarry(*(getattr(receiver, key) for key in MixedCarry._fields))
    split = np.vectorize(lambda x, h, l: _split(x, dtype, h, l), otypes=[dtype, dtype])
    for new, old_high, old_low in zip(leaves, jax.tree.leaves(primary),
                                     jax.tree.leaves(receiver.carry), strict=True):
        high, low = split(new, np.asarray(old_high), np.asarray(old_low))
        hi.append(jnp.asarray(high))
        lo.append(jnp.asarray(low))
    fields = layout.unflatten(hi)
    return MixedBank(*fields, receiver.active, carry=layout.unflatten(lo))


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ComponentBank:
    """An immutable mixed posterior, reset priors and slot identities.

    Construct with the reset prior and initial active mask. Call update to learn
    from observations on this layout. Both this constructor and the
    snapshot codec accept an empty active set. All operations are host
    transactions on one bank, outside JIT and vmap.
    """

    reset_prior: MixedBank
    posterior: MixedBank
    logical_ids: tuple[int, ...]  # zero denotes an unoccupied slot
    generations: tuple[int, ...]
    next_id: int

    def __init__(self, reset_prior: MixedBank):
        if not bool(validate_bank(_supported(reset_prior))):
            raise ValueError("invalid component reset prior")
        prior = jax.tree.map(lambda a: jnp.array(a, copy=True), reset_prior)
        active = np.asarray(prior.active)
        ids = tuple(map(int, np.where(active, np.cumsum(active), 0)))
        self._initialize(prior, prior, ids, tuple(map(int, active)), int(active.sum()) + 1)

    def _initialize(self, prior, posterior, ids, generations, next_id):
        object.__setattr__(self, "reset_prior", prior)
        object.__setattr__(self, "posterior", posterior)
        object.__setattr__(self, "logical_ids", ids)
        object.__setattr__(self, "generations", generations)
        object.__setattr__(self, "next_id", next_id)

    @classmethod
    def _assembled(cls, prior, posterior, ids, generations, next_id):
        result = object.__new__(cls)
        result._initialize(prior, posterior, ids, generations, next_id)
        return result

    def update(self, batch: MixedBatch, weights: jax.Array) -> SlotResult:
        """Apply the public mixed learner to this bank and its current occupants."""
        posterior, valid = update_mixed(self.posterior, batch, weights)
        if not bool(valid):
            return SlotResult(self, SlotStatus.INVALID_POSTERIOR, None)
        new = self._assembled(self.reset_prior, posterior, self.logical_ids,
                              self.generations, self.next_id)
        return SlotResult(new, SlotStatus.OK, None)

    def replace_posterior(self, posterior: MixedBank, *, previous: MixedBank) -> SlotResult:
        """Install a caller's proposal computed from this exact source snapshot.

        Pass the posterior object used to compute the proposal as previous.
        This prevents installing old results after a merge, birth or update.
        """
        if previous is not self.posterior:
            return SlotResult(self, SlotStatus.STALE_REFERENCE, None)
        if not np.array_equal(previous.active, posterior.active):
            return SlotResult(self, SlotStatus.INVALID_POSTERIOR, None)
        try:
            if not bool(validate_bank(_supported(posterior))):
                return SlotResult(self, SlotStatus.INVALID_POSTERIOR, None)
            _validate_layout(self.reset_prior, posterior)
        except (TypeError, ValueError):
            return SlotResult(self, SlotStatus.INVALID_POSTERIOR, None)
        installed = jax.tree.map(lambda a: jnp.array(a, copy=True), posterior)
        new = self._assembled(self.reset_prior, installed, self.logical_ids,
                              self.generations, self.next_id)
        return SlotResult(new, SlotStatus.OK, None)

    def reference(self, slot: int) -> ComponentRef | None:
        if not isinstance(slot, (int, np.integer)) or not 0 <= slot < len(self.logical_ids):
            raise ValueError("slot outside this bank")
        logical_id = self.logical_ids[slot]
        return ComponentRef(int(slot), logical_id, self.generations[slot]) if logical_id else None

    def _live(self, ref):
        return (isinstance(ref, ComponentRef) and isinstance(ref.slot, (int, np.integer))
                and 0 <= ref.slot < len(self.logical_ids) and self.reference(ref.slot) == ref)

    def birth(self) -> SlotResult:
        """Activate the first free slot at its immutable per-slot prior."""
        try:
            slot = self.logical_ids.index(0)
        except ValueError:
            return SlotResult(self, SlotStatus.FULL, None)
        if self.next_id >= _MAX_ID:
            return SlotResult(self, SlotStatus.ID_EXHAUSTED, None)
        ids, generations = list(self.logical_ids), list(self.generations)
        ids[slot] = self.next_id
        generations[slot] += 1
        posterior = self.posterior._replace(active=self.posterior.active.at[slot].set(True))
        new = self._assembled(self.reset_prior, posterior, tuple(ids), tuple(generations),
                              self.next_id + 1)
        return SlotResult(new, SlotStatus.OK, new.reference(slot))

    def propose_merge(self, receiver: ComponentRef, donor: ComponentRef) -> SlotResult:
        """Transfer the donor's evidence into the receiver and release the donor."""
        if not self._live(receiver) or not self._live(donor):
            return SlotResult(self, SlotStatus.STALE_REFERENCE, None)
        if receiver == donor:
            return SlotResult(self, SlotStatus.SAME_COMPONENT, None)

        i, j = receiver.slot, donor.slot
        r = jax.tree.map(lambda a: a[i], self.posterior)
        d = jax.tree.map(lambda a: a[j], self.posterior)
        reset = jax.tree.map(lambda a: a[j], self.reset_prior)
        try:
            merged = _merged_row(r, d, reset)
        except ArithmeticError:
            return SlotResult(self, SlotStatus.INVALID_POSTERIOR, None)
        if not bool(validate_bank(jax.tree.map(lambda a: a[None], merged))):
            return SlotResult(self, SlotStatus.INVALID_POSTERIOR, None)
        dormant = reset._replace(active=jnp.asarray(False))
        posterior = jax.tree.map(lambda a, b, c: a.at[i].set(b).at[j].set(c),
                                 self.posterior, merged, dormant)
        ids = list(self.logical_ids)
        ids[j] = 0
        new = self._assembled(self.reset_prior, posterior, tuple(ids),
                              self.generations, self.next_id)
        return SlotResult(new, SlotStatus.OK, receiver)

    def state_dict(self, *, exact=False) -> dict[str, np.ndarray]:
        """Save the reset prior, posterior and identities in independent host arrays.

        The compact default omits both sets of low words. A count at the same
        midpoint as its reset prior uses the prior's primary word. Set exact=True
        to retain every low word for bitwise continuation on the same runtime.
        """
        state = {
            "version": np.array(1 if exact else 2, np.int32),
            "logical_ids": np.array(self.logical_ids, np.int64),
            "generations": np.array(self.generations, np.int64),
            "next_id": np.array(self.next_id, np.int64),
        }
        for prefix, bank in (("prior", self.reset_prior), ("posterior", self.posterior)):
            record = _snapshot(bank, exact=exact)
            if prefix == "posterior" and not exact:
                _check_counts(self.reset_prior, self.posterior, compact_posterior=record)
            state.update({f"{prefix}/{k}": v for k, v in record.items()})
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray]) -> ComponentBank:
        """Restore a v1 exact or v2 compact lifecycle snapshot and validate both banks."""
        metadata = {k for k in state if not k.startswith(("prior/", "posterior/"))}
        if metadata != {"version", "next_id", "generations", "logical_ids"}:
            raise ValueError("unexpected or missing component lifecycle metadata")
        version = np.asarray(state["version"])
        if version.shape != () or version.dtype != np.int32 or version.item() not in (1, 2):
            raise ValueError("unknown component lifecycle version")

        mixed_version = 3 if version == 1 else 2

        def decode(prefix):
            record = {k.removeprefix(prefix): state[k] for k in state if k.startswith(prefix)}
            return _restore_bank(record, mixed_version)
        prior, posterior = decode("prior/"), decode("posterior/")
        _validate_layout(prior, posterior)
        shape = posterior.active.shape
        ids = _integer(state, "logical_ids", shape)
        generations = _integer(state, "generations", shape)
        next_id = int(_integer(state, "next_id", ()))
        active = np.asarray(posterior.active)
        live = ids[active]
        initially_active = np.asarray(prior.active)
        initial_ids = np.cumsum(initially_active)
        first_occupants = active & initially_active & (generations == 1)
        if (np.any((ids != 0) != active) or len(np.unique(live)) != len(live)
                or sum(map(int, generations)) + 1 != next_id
                or np.any(ids >= next_id) or np.any(generations[active] > live)
                or np.any(generations[active | initially_active] == 0)
                or np.any(ids[first_occupants] != initial_ids[first_occupants])
                or np.any(active & ~first_occupants & (ids <= initially_active.sum()))):
            raise ValueError("component identity, generation or next ID is inconsistent")
        return cls._assembled(prior, posterior, tuple(map(int, ids)),
                              tuple(map(int, generations)), next_id)
