"""Indexed sum-product contractions for sparse built-in transition tables."""

from dataclasses import dataclass
from math import prod
from weakref import WeakValueDictionary

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

from cffg import efrules, kernels, rules
from cffg.compiled import NEG
from cffg.execution.contracts import ExecutionDelta
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.kernel_runtime import discrete_energy


_MIN_GROUP_ATOMS = 4096
_MIN_TERM_REDUCTION = 8


def _eligible_group(group):
    return (
        group.kernel is kernels._BUILTIN_TRANSITION_KERNEL
        and group.mode == "bp"
        and min(group.cards) >= 2
        and len(group.members) * prod(group.cards) >= _MIN_GROUP_ATOMS
    )


def has_indexable_transition(arena):
    return any(_eligible_group(group) for group in arena.groups)


@dataclass(frozen=True)
class _Atoms:
    indices: tuple[np.ndarray, ...]
    log_factor: np.ndarray
    present: np.ndarray

    def inputs(self, rows, roles):
        factors = jnp.arange(self.log_factor.shape[0])
        broadcast = (slice(None),) + (None,) * (self.log_factor.ndim - 1)
        return tuple(rows[factors[broadcast], role, self.indices[role]] for role in roles)


@dataclass(frozen=True)
class _IndexedTransition:
    cards: tuple[int, ...]
    joint: _Atoms
    outputs: tuple[_Atoms, ...]

    @classmethod
    def pack(cls, dense):
        cards = dense.shape[1:]
        zero = dense.dtype.type(NEG)
        nonzero = dense != zero
        if np.count_nonzero(nonzero) * _MIN_TERM_REDUCTION > dense.size:
            return None
        if (not np.isfinite(dense).all() or np.any(dense > 0)
                or np.any(dense[nonzero] < zero / 4)):
            return None
        supports = [np.argwhere(row) for row in nonzero]
        if any(not len(coords) for coords in supports):
            return None

        def pack_groups(groups, bins=None):
            width = max(len(coords) for group in groups for coords in group)
            shape = (len(dense), width) if bins is None else (len(dense), bins, width)
            coordinates = np.zeros((*shape, len(cards)), dtype=np.int32)
            values = np.full(shape, NEG, dtype=dense.dtype)
            present = np.zeros(shape, dtype=bool)
            for factor, group in enumerate(groups):
                for index, coords in enumerate(group):
                    selection = (factor,) if bins is None else (factor, index)
                    selected = (*selection, slice(len(coords)))
                    coordinates[selected] = coords
                    values[selected] = dense[(factor, *coords.T)]
                    present[selected] = True
            return _Atoms(tuple(np.moveaxis(coordinates, -1, 0)), values, present)

        joint = pack_groups([[coords] for coords in supports])
        outputs = tuple(pack_groups([
            [coords[coords[:, role] == value] for value in range(card)]
            for coords in supports
        ], card) for role, card in enumerate(cards))
        work = joint.log_factor.size + sum(atoms.log_factor.size for atoms in outputs)
        if work * _MIN_TERM_REDUCTION > 4 * dense.size:
            return None
        return cls(cards, joint, outputs)

    def messages(self, incoming):
        # Near NEG, log multiplicity rounds away. A forbidden term with
        # maximizing operands rounds exactly to NEG at both additions. If
        # that term is allowed, its score already dominates the floor.
        quarter_ulp = abs(np.spacing(np.dtype(incoming.dtype).type(NEG))) / 4
        valid = jnp.asarray(True)
        for role, card in enumerate(self.cards):
            peak = incoming[:, role, :card].max(axis=-1)
            valid &= jnp.all((peak >= -quarter_ulp) & (peak <= 0))
        rows = []
        for role, atoms in enumerate(self.outputs):
            other = tuple(index for index in range(len(self.cards)) if index != role)
            first, second = atoms.inputs(incoming, other)
            logits = jnp.where(atoms.present, atoms.log_factor + first + second, NEG)
            floor = jnp.full((*logits.shape[:-1], 1), NEG, dtype=logits.dtype)
            output = logsumexp(jnp.concatenate((logits, floor), axis=-1), axis=-1)
            rows.append(rules._norm_pad(output, self.cards[role], max(self.cards)))
        return jnp.where(valid, jnp.stack(rows, axis=1), jnp.nan)

    def energy(self, incoming):
        out, source, selector = self.joint.inputs(incoming, range(len(self.cards)))
        factor = self.joint.log_factor
        joint = jnp.where(self.joint.present, factor + out + source + selector, -jnp.inf)
        peak = joint.max(axis=-1)
        valid = jnp.all(incoming <= 0) & jnp.all(peak > NEG / 2)
        # The omitted probabilities underflow. Preserve the ledger's atom
        # cutoff even if the different reduction lengths round differently.
        log_partition = logsumexp(joint, axis=-1)
        log_probability = joint - log_partition[:, None]
        cutoff = jnp.log(jnp.asarray(efrules._CATEGORICAL_TERM_CUTOFF, joint.dtype))
        eps = jnp.finfo(joint.dtype).eps
        error = 8 * eps * (joint.shape[-1] + 1 + jnp.abs(log_partition))
        valid &= ~jnp.any(jnp.abs(log_probability - cutoff) <= error[:, None])
        return jnp.where(valid, efrules.CATEGORICAL.energy(joint, factor), jnp.nan)

    def zero_support(self, incoming):
        factors = np.arange(self.joint.present.shape[0])[:, None]
        support = jnp.asarray(self.joint.present)
        for role, values in enumerate(incoming):
            support &= values[factors, self.joint.indices[role]]
        return ~jnp.all(jnp.any(support, axis=-1))


class IndexedTransitions:
    """Bind indexed plans to one engine's immutable, shared log potentials."""

    def __init__(self, arena, dtype):
        self._groups = {}
        self._values = {}
        self._matches = WeakValueDictionary()
        self._mismatches = WeakValueDictionary()
        for group in arena.groups:
            if not _eligible_group(group):
                continue
            path = dict(group.parameter_paths)["logf"]
            dense = np.asarray(arena.log_params[path], dtype=dtype)
            if (plan := _IndexedTransition.pack(dense)) is not None:
                self._groups[group.name] = (group, plan)
                self._values[path] = dense

    def __bool__(self):
        return bool(self._groups)

    def matches(self, states):
        # A caller can supply a new State or replace a params entry. Identity
        # is safe only after checking that exact immutable device array.
        if (rules.BP_UPDATES["transition_mixture"] is not rules.bp_transition_mixture
                or rules.NODE_ENERGIES["transition_mixture"] is not rules.energy_transition_mixture):
            return False
        for path, expected in self._values.items():
            for state in states:
                value = state.params.get(path)
                if not isinstance(value, jax.Array) or value.dtype != expected.dtype:
                    return False
                key = (path, id(value))
                if self._matches.get(key) is value:
                    continue
                if self._mismatches.get(key) is value:
                    return False
                if not np.array_equal(value, expected):
                    self._mismatches[key] = value
                    return False
                self._matches[key] = value
        return True

    def _plan(self, group):
        selected = self._groups.get(group.name)
        return selected[1] if selected and selected[0] is group else None

    def message(self, group, incoming, params, reduction):
        if (plan := self._plan(group)) is not None:
            return ExecutionDelta(plan.messages(incoming))
        return SCHEDULED_EXECUTORS.discrete(group, incoming, params, reduction)

    def energy(self, group, incoming, params):
        if (plan := self._plan(group)) is not None:
            return plan.energy(incoming)
        return discrete_energy(group, incoming, params)

    def zero_support(self, group, incoming, params):
        if (plan := self._plan(group)) is not None:
            return plan.zero_support(incoming)
        return SCHEDULED_EXECUTORS.discrete_zero_support(group, incoming, params)
