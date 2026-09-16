"""Canonical neural weight blobs and differentiable injection into packed rows."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class WeightBlob:
    name: str
    start: int
    size: int
    rows: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class NeuralParameters:
    """One coordinate vector per serialized blob, shared by every reader."""

    blobs: tuple[WeightBlob, ...]

    @property
    def size(self):
        return sum(blob.size for blob in self.blobs)

    @classmethod
    def from_engine(cls, engine, params, factors=None):
        supported = {
            name for group in engine.arena.gauss_groups
            if group.kernel is not None and "network_parameters" in group.kernel.binding.work
            for name in group.members
        }
        selected = supported if factors is None else set(factors)
        if not selected or not selected <= supported:
            raise ValueError("trainable factors must name compiled neural Gaussian means")
        names = {engine.arena.member_of[name].parameter("parameters").blob for name in selected}
        blobs, start = [], 0
        for name in sorted(names):
            addresses = [address for member in engine.arena.member_of.values()
                         for address in member.parameters if address.blob == name]
            if any(address.role != "parameters" or address.transform != "identity" or address.permutation
                   for address in addresses):
                raise ValueError(f"trainable blob {name!r} has an incompatible parameter reader")
            rows = tuple(sorted({(address.key, address.index) for address in addresses}))
            values = [np.asarray(params[key][index]) for key, index in rows]
            reference = values[0]
            if reference.ndim != 1 or not np.isfinite(reference).all():
                raise ValueError(f"trainable blob {name!r} requires finite flat weights")
            if any(not np.array_equal(reference, value) for value in values[1:]):
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
            for key, index in blob.rows:
                result[key] = result[key].at[index].set(value)
        return result

    def values(self, weights):
        return {blob.name: weights[blob.start:blob.start + blob.size] for blob in self.blobs}
