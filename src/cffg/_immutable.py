"""Host-side publication of read-only compiler and kernel declaration data.

Mutable builders remain private. Publication snapshots caller-owned buffers
and containers; already published values reuse their storage. Callbacks are
trusted opaque objects, not data to copy or a security boundary to enforce.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from types import MappingProxyType

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class FrozenMap(Mapping):
    """An insertion-ordered, copy-compatible mapping with no mutable aliases."""

    __slots__ = ("_data",)

    def __init__(self, values=()):
        object.__setattr__(self, "_data", MappingProxyType({
            key: freeze(value) for key, value in dict(values).items()
        }))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"FrozenMap({dict(self._data)!r})"

    def __setattr__(self, name, value):
        raise TypeError("published mappings are read-only")

    def __delattr__(self, name):
        raise TypeError("published mappings are read-only")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class FrozenArray(NDArrayOperatorsMixin):
    """Owned numeric data; NumPy exports are detached working arrays.

    NumPy's ``ufunc.at`` can bypass WRITEABLE=False, even on bytes-backed
    arrays. Never expose owned storage via a NumPy array/buffer protocol.
    JAX conversion also detaches storage: CPU NumPy views can alias JAX arrays.
    This is an ownership boundary, not protection against private-field access.
    """

    __slots__ = ("_array",)

    def __init__(self, value):
        if isinstance(value, FrozenArray):
            array = value._array
        else:
            source = np.asarray(value)
            if source.dtype.hasobject:
                raise TypeError("published arrays cannot contain Python objects")
            array = np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)
        object.__setattr__(self, "_array", array)

    @classmethod
    def _from_owned(cls, array):
        result = object.__new__(cls)
        object.__setattr__(result, "_array", array)
        return result

    def __setattr__(self, name, value):
        raise TypeError("published array metadata is read-only")

    def __delattr__(self, name):
        raise TypeError("published array metadata is read-only")

    def __setitem__(self, key, value):
        raise TypeError("published arrays are read-only")

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("published arrays cannot export shared NumPy storage")
        return np.array(self._array, dtype=dtype, copy=True)

    def __jax_array__(self):
        import jax.numpy as jnp

        return jnp.array(self._array, copy=True)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "at" and isinstance(inputs[0], FrozenArray):
            raise TypeError("published arrays are read-only")
        if any(isinstance(value, FrozenArray) for value in kwargs.get("out", ())):
            raise TypeError("published arrays are read-only")
        inputs = tuple(value._array if isinstance(value, FrozenArray) else value for value in inputs)
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getitem__(self, key):
        result = self._array[key]
        return self._from_owned(result) if isinstance(result, np.ndarray) else result

    def __len__(self):
        return len(self._array)

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    def __bool__(self):
        return bool(self._array)

    def __int__(self):
        return int(self._array)

    def __float__(self):
        return float(self._array)

    def __index__(self):
        return self._array.__index__()

    def __repr__(self):
        return f"FrozenArray({self._array!r})"

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def size(self):
        return self._array.size

    @property
    def nbytes(self):
        return self._array.nbytes

    @property
    def T(self):
        return self._from_owned(self._array.T)

    def reshape(self, *shape, **kwargs):
        return self._from_owned(self._array.reshape(*shape, **kwargs))

    def ravel(self, order="C"):
        return self._from_owned(self._array.ravel(order))

    def tolist(self):
        return self._array.tolist()

    def tobytes(self, order="C"):
        return self._array.tobytes(order)

    def copy(self, order="C"):
        return self._array.copy(order)

    def astype(self, dtype, **kwargs):
        return self.copy().astype(dtype, **kwargs)

    def setflags(self, write=None, align=None, uic=None):
        raise TypeError("published array metadata is read-only")

    def resize(self, *args, **kwargs):
        raise TypeError("published arrays cannot be resized")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def freeze(value):
    """Publish owned value data without modifying the caller's objects."""
    if isinstance(value, (ImmutableRecord, FrozenMap, FrozenArray)):
        return value
    if isinstance(value, np.ndarray):
        return FrozenArray(value)
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (tuple, list)):
        result = tuple(freeze(item) for item in value)
        if isinstance(value, tuple) and all(a is b for a, b in zip(value, result, strict=True)):
            return value
        return result
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    return value


class ImmutableRecord:
    """Publication and copy behavior for frozen value dataclasses."""

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, freeze(getattr(self, field.name)))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
