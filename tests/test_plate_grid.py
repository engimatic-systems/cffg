"""Compact plate indices, independent row-major oracles, and public admission."""

from itertools import islice

import numpy as np
import pytest
from test_plates import pixel_fragment

from cffg.core import Chain, Expose, ModelError, PlateSpec, Template, expand_plates
from cffg.core.plate_grid import PlateGrid


def test_plate_addresses_and_chain_boundaries_match_row_major_enumeration():
    for dims in ((1,), (4,), (2, 4), (2, 3, 4)):
        coordinates = list(np.ndindex(dims))
        grid = PlateGrid(dims)
        assert grid.count == len(coordinates)
        assert list(grid) == coordinates
        assert [grid.coordinate(i) for i in range(grid.count)] == coordinates
        assert [grid.ordinal(index) for index in coordinates] == list(range(grid.count))
        for axis, n in enumerate(dims):
            for offset in (1, 2, n + 1):
                expected = []
                for i, index in enumerate(coordinates):
                    next_index = list(index)
                    next_index[axis] += offset
                    if next_index[axis] < n:
                        expected.append((index, tuple(next_index)))
                        assert grid.neighbor(i, axis, offset) == coordinates.index(
                            tuple(next_index)
                        )
                    else:
                        assert grid.neighbor(i, axis, offset) is None
                assert list(grid.chain(axis, offset)) == expected


def test_plate_direct_address_near_signed_limit_without_materialization():
    grid = PlateGrid((np.uint64(2**31), 2**31 - 1))
    assert grid.coordinate(grid.count - 1) == (2**31 - 1, 2**31 - 2)
    assert grid.ordinal((2**31 - 1, 2**31 - 2)) == grid.count - 1
    assert grid.neighbor(grid.count - 2, 1) == grid.count - 1
    assert list(islice(grid, 2)) == [(0, 0), (0, 1)]
    assert list(grid.chain(0, 2**31)) == []
    for coordinate in ((0,), (0, -1), (0, 0.5), (True, 0), (2**31, 0)):
        with pytest.raises((TypeError, ValueError)):
            grid.ordinal(coordinate)
    for ordinal in (-1, grid.count, 2.0, False):
        with pytest.raises((TypeError, ValueError)):
            grid.coordinate(ordinal)


def test_bad_plate_dimensions_and_chain_specs_fail_before_copying(monkeypatch):
    from cffg.core import plates

    def forbid_copy(*args):
        pytest.fail("invalid grid reached copy allocation")

    monkeypatch.setattr(plates, "_scope_copy", forbid_copy)
    for dims in ((-1,), (0,), (3.5,), (True,), (2**32, 2**31)):
        template = Template(
            "bad", plates={"cells": PlateSpec(pixel_fragment(), dims, (Expose(),))}
        )
        with pytest.raises(ModelError, match="plate 'cells'") as error:
            expand_plates(template)
        if dims == (2**32, 2**31):
            assert str(dims) in str(error.value)
            assert "64-bit" in str(error.value)
    for axis, offset in ((False, 1), (0.0, 1), (0, 1.5), (0, 0), (2, 1)):
        plate = PlateSpec(pixel_fragment(), (2, 3), (Chain("x", "x", axis, offset),))
        with pytest.raises(ModelError, match="chain (axis|offset)"):
            expand_plates(Template("bad", plates={"cells": plate}))
    invalid = PlateSpec(pixel_fragment(), (2, 3), (Chain("x", "x", 2, 0),))
    with pytest.raises(ModelError) as error:
        expand_plates(Template("bad", plates={"cells": invalid}))
    assert "chain axis" in str(error.value) and "chain offset" in str(error.value)
