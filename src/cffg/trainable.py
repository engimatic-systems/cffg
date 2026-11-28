"""Tied point-parameter blobs and differentiable injection into packed rows."""

from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter

import jax.numpy as jnp
import numpy as np


def _packed_rows(rows, params):
    """Canonical blob addresses are sorted by packed key, then row number."""
    for key, members in groupby(rows, key=itemgetter(0)):
        indices = np.fromiter((i for _, i in members), dtype=np.intp)
        count = params[key].shape[0]
        if indices[0] < 0 or indices[-1] >= count:
            raise ValueError(f"trainable row outside packed parameter {key!r}")
        yield key, None if len(indices) == count else indices


@dataclass(frozen=True)
class WeightBlob:
    name: str
    start: int
    size: int
    rows: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class MeanParameters:
    """One coordinate vector per serialized blob, shared by every reader."""

    blobs: tuple[WeightBlob, ...]

    @property
    def size(self):
        return sum(blob.size for blob in self.blobs)

    @classmethod
    def from_engine(cls, engine, params, factors=None):
        supported = {
            name for group in engine.arena.gauss_groups
            if group.kernel is not None and group.kernel.binding.supports_point_training
            for name in group.members
        }
        selected = supported if factors is None else set(factors)
        if not selected or not selected <= supported:
            raise ValueError("trainable factors must name point-trainable Gaussian means")
        names = {engine.arena.member_of[name].parameter("parameters").blob for name in selected}
        blobs, start = [], 0
        for name in sorted(names):
            addresses = [address for member in engine.arena.member_of.values()
                         for address in member.parameters if address.blob == name]
            if any(address.role != "parameters" or address.transform != "identity" or address.permutation
                   for address in addresses):
                raise ValueError(f"trainable blob {name!r} has an incompatible parameter reader")
            rows = tuple(sorted({(address.key, address.index) for address in addresses}))
            reference = None
            for key, indices in _packed_rows(rows, params):
                packed = params[key]
                values = np.asarray(packed if indices is None else packed[indices])
                if reference is None:
                    reference = values[0]
                    if reference.ndim != 1 or reference.size == 0 or not np.isfinite(reference).all():
                        raise ValueError(f"trainable blob {name!r} requires a nonempty finite flat vector")
                if values.shape[1:] != reference.shape or not (values == reference).all():
                    raise ValueError(f"trainable blob {name!r} has inconsistent tied rows")
            blobs.append(WeightBlob(name, start, reference.size, rows))
            start += reference.size
        return cls(tuple(blobs))

    def extract(self, params):
        return jnp.concatenate([params[blob.rows[0][0]][blob.rows[0][1]] for blob in self.blobs])

    def inject(self, weights, params):
        if weights.shape != (self.size,):
            raise ValueError("trainable vector does not match the canonical blob layout")
        result = dict(params)
        for blob in self.blobs:
            value = weights[blob.start:blob.start + blob.size]
            for key, indices in _packed_rows(blob.rows, params):
                packed = result[key]
                if indices is None:
                    result[key] = jnp.broadcast_to(value.astype(packed.dtype), packed.shape)
                else:
                    result[key] = packed.at[indices].set(value)
        return result

    def values(self, weights):
        return {blob.name: weights[blob.start:blob.start + blob.size] for blob in self.blobs}


NeuralParameters = MeanParameters
