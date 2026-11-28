"""Row-major integer addressing for copies of a plate fragment."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from numbers import Integral


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class PlateGrid:
    """Copy numbers and coordinates, with the last coordinate varying fastest."""

    dims: tuple[int, ...]
    count: int = field(init=False)
    strides: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        try:
            raw_dims = tuple(self.dims)
        except TypeError as err:
            raise TypeError(
                f"plate dims must be a sequence of integers, got {self.dims!r}"
            ) from err
        dims = tuple(_integer(n, "plate dimension") for n in raw_dims)
        if not dims or any(n < 1 for n in dims):
            raise ValueError(
                f"plate dims {dims} must all be >= 1 and contain at least one axis"
            )
        count = 1
        strides = []
        for n in reversed(dims):
            strides.append(count)
            count *= n
        if count > (1 << 63) - 1:
            raise ValueError(f"plate dims {dims} exceed the signed 64-bit copy count")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "strides", tuple(reversed(strides)))

    def ordinal(self, coordinate: tuple[int, ...]) -> int:
        if len(coordinate) != len(self.dims):
            raise ValueError(
                f"coordinate {coordinate} must match the axes of plate dims {self.dims}"
            )
        number = 0
        for raw, n, stride in zip(coordinate, self.dims, self.strides):
            index = _integer(raw, "copy coordinate")
            if not 0 <= index < n:
                raise ValueError(
                    f"coordinate {coordinate} outside plate dims {self.dims}"
                )
            number += index * stride
        return number

    def _checked_ordinal(self, ordinal: int) -> int:
        ordinal = _integer(ordinal, "copy number")
        if not 0 <= ordinal < self.count:
            raise ValueError(f"copy number {ordinal} outside plate dims {self.dims}")
        return ordinal

    def coordinate(self, ordinal: int) -> tuple[int, ...]:
        ordinal = self._checked_ordinal(ordinal)
        return tuple(
            (ordinal // stride) % n for stride, n in zip(self.strides, self.dims)
        )

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        # Advance one coordinate in-place. A range in itertools.product would
        # first be copied into an axis-sized tuple, even for just one lookup.
        coordinate = [0] * len(self.dims)
        for _ in range(self.count):
            yield tuple(coordinate)
            for axis in range(len(coordinate) - 1, -1, -1):
                coordinate[axis] += 1
                if coordinate[axis] < self.dims[axis]:
                    break
                coordinate[axis] = 0

    def chain_problems(self, axis: int, offset: int) -> list[str]:
        """Check both fields so a model editor can report all invalid inputs."""
        problems = []
        if (
            isinstance(axis, bool)
            or not isinstance(axis, Integral)
            or not 0 <= axis < len(self.dims)
        ):
            problems.append(
                f"chain axis {axis!r} must be an integer in range for dims {self.dims}"
            )
        if isinstance(offset, bool) or not isinstance(offset, Integral) or offset < 1:
            problems.append(f"chain offset {offset!r} must be an integer >= 1")
        return problems

    def _checked_chain(self, axis: int, offset: int) -> tuple[int, int]:
        if problems := self.chain_problems(axis, offset):
            raise ValueError(", ".join(problems))
        return int(axis), int(offset)

    def neighbor(self, ordinal: int, axis: int, offset: int = 1) -> int | None:
        """Return the connected copy number or None at the exposed boundary."""
        axis, offset = self._checked_chain(axis, offset)
        ordinal = self._checked_ordinal(ordinal)
        stride = self.strides[axis]
        position = (ordinal // stride) % self.dims[axis]
        if position + offset >= self.dims[axis]:
            return None
        return ordinal + offset * stride

    def chain(
        self, axis: int, offset: int
    ) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Yield connected pairs in copy order."""
        axis, offset = self._checked_chain(axis, offset)
        if offset >= self.dims[axis]:
            return iter(())
        return (
            (index, index[:axis] + (index[axis] + offset,) + index[axis + 1 :])
            for index in self
            if index[axis] + offset < self.dims[axis]
        )
