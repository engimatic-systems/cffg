"""Independent structural counts and inventories of real initialized State leaves."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from cffg import kernels
from cffg.compiler import compile_template
from cffg.core import Broadcast, Chain, PlateSpec, Template
from cffg.core.plate_grid import PlateGrid
from cffg.engine import Engine
from cffg.estimate import UnknownStorageLayout, estimate_plate_storage
from cffg.resource_analysis import state_storage
from test_ef_engine import gamma_poisson
from test_plates import K, M, pixel_fragment, selector_fragment, sensor_fragment


def inventory(template, result, arrays=None):
    state = Engine(compile_template(template, arrays=arrays), schedule="flood").init_state()
    actual = state_storage(state, batch_size=result["batch_size"])
    for field in actual:
        assert result[field] == actual[field], field


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_named_payloads_grid_and_private_broadcast(dtype):
    pixel = Template("grid", plates={"xy": PlateSpec(pixel_fragment(), (64, 64))})
    bank = Template("bank", plates={
        "b": PlateSpec(selector_fragment(), (7,), (Broadcast("s", "hub"),), ("w",))
    })
    with jax.enable_x64(dtype is np.float64):
        report = estimate_plate_storage(pixel, source_dtype=np.float32)
        scale = np.dtype(dtype).itemsize
        assert report["arenas"]["messages"] == 4096 * 2 * (K + M) * scale
        assert report["source_bytes"] == 4 * (K + M * K)
        assert report["parameter_bytes"] == 4096 * (K + M * K) * scale
        inventory(pixel, report)
        report = estimate_plate_storage(bank, batch_size=3)
        assert report["source_blobs"]["b/{index}/w"]["copies"] == 7
        assert report["source_bytes"] == 7 * K * 8
        assert report["parameter_bytes"] == 3 * 7 * K * scale
        assert report["topology"] == dict(edges=13, factors=13, ports=1,
                                          chain_bonds=0, broadcast_equalities=6)
        inventory(bank, report)


def test_offset_chain_affine_broadcast_and_scalar_ef(hmm_slice):
    # Each strip of five slices has three offset-two connections.
    model = Template("chains", plates={"xy": PlateSpec(
        hmm_slice, (3, 5), (Chain("tail", "head", axis=1, offset=2),)
    )})
    report = estimate_plate_storage(model)
    assert report["topology"]["chain_bonds"] == 9
    assert report["topology"]["edges"] == 15 * len(hmm_slice.variables) - 9
    assert report["topology"]["ports"] == 15 * len(hmm_slice.ports) - 18
    arrays = {"A": np.ones((M, K)) / M, "B": np.ones((K, K)) / K}
    inventory(model, report, arrays)

    for size in (1, 2, 5):
        sensors = Template("sensors", plates={"s": PlateSpec(
            sensor_fragment(3), (size,), (Broadcast("st", "hub"),)
        )})
        report = estimate_plate_storage(sensors)
        assert report["topology"]["broadcast_equalities"] == size - 1
        assert report["arenas"]["g_obs"] == size * 3 * 8
        inventory(sensors, report)

    ef, arrays = gamma_poisson()
    inventory(ef, estimate_plate_storage(ef), arrays)


def test_direct_count_overflow_and_refusal_never_expand_or_pack(monkeypatch):
    frag = selector_fragment()
    template = Template("bank", plates={"b": PlateSpec(
        frag, (10**9,), (Broadcast("s", "hub"),), ("w",)
    )})

    def forbidden(*args, **kwargs):
        raise AssertionError("copies or packs materialized")
    monkeypatch.setattr(PlateGrid, "__iter__", forbidden)
    monkeypatch.setattr(kernels, "packed_parameters", forbidden)
    report = estimate_plate_storage(template, dtype="float32")
    assert report["arenas"]["messages"] == 2 * (2 * 10**9 - 1) * K * 4
    with pytest.raises(OverflowError):
        estimate_plate_storage(template, batch_size=10**9)
    with pytest.raises(ValueError, match="dimension"):
        estimate_plate_storage(template, batch_size=True)
    with pytest.raises(ValueError, match="dtype"):
        estimate_plate_storage(template, dtype=np.int32)
    template.plates["b"] = replace(template.plates["b"], dims=(2**64,))
    with pytest.raises(ValueError, match="64-bit"):
        estimate_plate_storage(template)
    template.plates["b"] = replace(template.plates["b"], dims=(1,))
    original = kernels.KERNELS["categorical"]
    replacement = replace(original, binding=replace(original.binding, pack_params=forbidden))
    monkeypatch.setitem(kernels.KERNELS, "categorical", replacement)
    with pytest.raises(UnknownStorageLayout, match="categorical"):
        estimate_plate_storage(template)


def test_conflicting_tie_and_reused_ports():
    frag = pixel_fragment()
    factor = frag.factors["obs"]
    frag.factors["obs"] = replace(factor, param_blobs={"matrix": "d"})
    with pytest.raises(ValueError, match="incompatible shapes"):
        estimate_plate_storage(Template("tie", plates={"p": PlateSpec(frag, (3,))}))
    bad = PlateSpec(selector_fragment(), (7,), (Broadcast("s", "x"), Broadcast("s", "y")))
    with pytest.raises(ValueError, match="consume"):
        estimate_plate_storage(Template("duplicate", plates={"b": bad}))
