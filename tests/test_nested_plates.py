"""A room owns its shared sensor calibration and each sensor owns a separate prior."""

import itertools
from dataclasses import replace

import jax
import numpy as np
import pytest

from cffg import Engine, compile_template
from cffg.colony import Colony
from cffg.core import (
    BlobRef, Broadcast, Chain, Expose, FactorSpec, PlateSpec, PortRole, PortSpec,
    Template, VariableSpec, expand_plates,
)
from cffg.core.compose import merge
from cffg.core.edits import SetPlateDims, apply_edits
from cffg.core.plates import expanded_blob_refs
from cffg.core.serialize import from_json, structure_hash, to_dict, to_json
from test_engine_contracts import assert_result_equal
from test_plate_lowering import assert_plan_equal
from test_plates import pixel_fragment, selector_fragment


def rooms(dims):
    sensor = pixel_fragment()
    sensor.blobs = {"A": BlobRef("A", "cal/{index}.npy"), "d": BlobRef("d", "prior/{index}.npy")}
    room = Template("room", plates={"sensor": PlateSpec(sensor, (2,), (Expose(),), ("d",))})
    return Template("study", plates={"room": PlateSpec(room, dims)})


def room_reference(dims, dtype):
    flat = Template("study")
    arrays, readings = {}, {}
    nll = 0.0
    for coordinates in itertools.product(*(range(n) for n in dims)):
        index = "/".join(map(str, coordinates))
        room = "room/" + index
        matrix = room + "/A"
        flat.blobs[matrix] = BlobRef(matrix, f"cal/{index}.npy")
        response = np.array([[.25, .5, 1], [.75, .5, 0]], dtype=dtype)
        arrays[matrix] = response[::-1] if sum(coordinates) % 2 else response
        for k in range(2):
            cell = f"{room}/sensor/{k}"
            z, x = cell + "/z", cell + "/x"
            prior, emission = cell + "/prior", cell + "/obs"
            d = cell + "/d"
            flat.variables.update({z: VariableSpec(z, 3), x: VariableSpec(x, 2)})
            flat.factors[prior] = FactorSpec(prior, "categorical", {"out": z}, param_blobs={"statistic": d})
            flat.factors[emission] = FactorSpec(emission, "matrix_cat", {"in": z, "out": x}, param_blobs={"matrix": matrix})
            flat.ports[x] = PortSpec(x, x, PortRole.DATA)
            flat.blobs[d] = BlobRef(d, f"prior/{index}/{k}.npy")
            arrays[d] = np.array([0, .25, .75] if k else [.5, 0, .5], dtype=dtype)
            reading = k if sum(coordinates) or k else None
            readings[x] = reading
            if reading is not None:
                nll -= np.log(arrays[matrix][reading] @ arrays[d])
    return flat, arrays, readings, nll


@pytest.mark.parametrize("x64", [False, True])
def test_room_and_sensor_identity_beliefs_and_evidence(x64, monkeypatch):
    dims = (2, 1, 2)
    template = rooms(dims)
    flat, arrays, readings, nll = room_reference(dims, np.float64 if x64 else np.float32)
    restored = from_json(to_json(template))
    assert "/sensor/0" not in to_json(restored)
    assert structure_hash(restored) == structure_hash(template)
    assert to_dict(expand_plates(restored)) == to_dict(flat)
    assert expanded_blob_refs(restored) == flat.blobs
    with jax.enable_x64(x64):
        arena = compile_template(flat, arrays=arrays)
        engine = Engine(arena)
        expected = engine.run(engine.bind_data(engine.init_state(), readings), 12, tol=1e-6)
        expected.require_converged()
        assert float(engine.bfe(expected.state)) == pytest.approx(nll, rel=2e-6 if not x64 else 1e-12)

        def no_expansion(*args, **kwargs):
            pytest.fail("compact compilation created per-copy source declarations")

        monkeypatch.setattr("cffg.core.plates._scope_copy", no_expansion)
        for spec in (VariableSpec, FactorSpec, PortSpec):
            monkeypatch.setattr(spec, "__init__", no_expansion)
        assert_plan_equal(compile_template(restored, arrays=arrays), arena)
        colony = Colony()
        for _ in range(2):
            colony.spawn(restored, arrays, observations=readings)
        assert len(colony.engines) == 1
        for result in colony.run(12, tol=1e-6).values():
            assert_result_equal(result, expected)


def test_resize_outer_and_inner_preserves_rules():
    original = rooms((1,))
    outer = apply_edits(original, [SetPlateDims("room", (2, 1, 2))])
    spec = outer.plates["room"]
    inner = apply_edits(spec.fragment, [SetPlateDims("sensor", (3,))])
    resized = replace(outer, plates={"room": replace(spec, fragment=inner)})
    assert len({structure_hash(t) for t in (original, outer, resized)}) == 3
    expected = {f"room/{i}/0/{j}/sensor/{k}/x" for i in range(2) for j in range(2) for k in range(3)}
    assert set(expand_plates(resized).ports) == expected
    assert set(expand_plates(original).ports) == {"room/0/sensor/0/x", "room/0/sensor/1/x"}


def test_outer_chain_continues_inner_chain(hmm_slice):
    block = Template("block", plates={"t": PlateSpec(hmm_slice, (2,), (Chain("tail", "head"),))})
    source = Template("sequence", plates={"block": PlateSpec(block, (2,), (Chain("t/1/tail", "t/0/head"),))})
    flat = expand_plates(source)
    times = [f"block/{b}/t/{i}" for b in range(2) for i in range(2)]
    for before, after in zip(times[:-1], times[1:], strict=True):
        assert flat.factors[after + "/trans"].edges["in"] == before + "/z_out"
    assert set(flat.ports) == {times[0] + "/head", times[-1] + "/tail", *(t + "/sense" for t in times)}
    arrays = {name: np.full((3, 3), 1 / 3) if name.endswith("/B") else np.full((2, 3), .5) for name in flat.blobs}
    assert_plan_equal(compile_template(source, arrays=arrays), compile_template(flat, arrays=arrays))


def test_two_levels_of_broadcast_preserve_one_global_hub():
    group = Template("group", plates={"vote": PlateSpec(selector_fragment(), (3,), (Broadcast("s", "choice"),))})
    source = Template("voting", plates={"group": PlateSpec(group, (2,), (Broadcast("vote/choice", "winner"),))})
    arrays = {"group/0/w": np.array([.25, .25, .5]), "group/1/w": np.array([.2, .3, .5])}
    flat = expand_plates(source)
    assert set(flat.ports) == {"group/winner"}
    arena = compile_template(source, arrays=arrays)
    assert_plan_equal(arena, compile_template(flat, arrays=arrays))
    engine = Engine(arena)
    state = engine.run(engine.init_state(), 40, tol=1e-8).require_converged()
    weights = (arrays["group/0/w"] * arrays["group/1/w"]) ** 3
    np.testing.assert_allclose(engine.marginals(state)["group/winner"], weights / weights.sum())
    assert engine.bfe(state) == pytest.approx(-np.log(weights.sum()))

def test_cyclic_source_rejected_before_hashing_and_composition():
    template = Template("cycle")
    template.plates["self"] = PlateSpec(template, (1,))
    for operation in (lambda: structure_hash(template), lambda: Colony().spawn(template, {}),
                      lambda: merge(template, Template("empty"), "joined")):
        with pytest.raises(ValueError, match="cyclic plate fragment"):
            operation()
