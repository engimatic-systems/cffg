"""Compact compilation has the same addresses and inference as the public expansion."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import jax
import numpy as np
import pytest

from cffg import Engine, compile_template, kernels
from cffg.colony import Colony
from cffg._immutable import FrozenArray
from cffg.core import (
    Broadcast, Chain, ConstraintKind, ConstraintSpec, Expose, FactorSpec, LearnableSpec,
    PlateSpec, PortSpec, Template, VariableSpec, expand_plates,
)
from cffg.core.serialize import to_dict
from cffg.kernels import FactorKernel
from cffg.engine import RunStatus
from test_discrete_kernels import declaration, model
from test_engine_contracts import assert_result_equal
from test_plates import pixel_fragment, selector_fragment, sensor_fragment


def assert_plan_equal(actual, expected):
    if actual is expected:
        return
    assert type(actual) is type(expected)
    if isinstance(actual, (FrozenArray, np.ndarray)):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif is_dataclass(actual):
        for field in fields(actual):
            assert_plan_equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(actual, Mapping):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_plan_equal(actual[key], expected[key])
    elif isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected, strict=True):
            assert_plan_equal(x, y)
    else:
        assert actual == expected


def plate(fragment, dims, *bonds):
    return Template("compact", plates={"p": PlateSpec(fragment, dims, bonds=bonds)})


def test_compact_layout_across_domains(hmm_slice):
    t = plate(pixel_fragment(), (2, 3), Expose())
    s = plate(selector_fragment(), (7,), Broadcast("s", "hub"))
    g = plate(sensor_fragment(), (3,), Broadcast("st", "hub"))
    c = plate(hmm_slice, (2, 3), Chain("tail", "head", axis=1))
    for compact in (t, s, g, c):
        flat = expand_plates(compact)
        arrays = {}
        if compact is c:
            for name in flat.blobs:
                arrays[name] = np.ones((3, 3)) / 3 if name == "B" else np.ones((2, 3)) / 2
        for semiring in (("sum",) if compact is g else ("sum", "max")):
            direct = compile_template(flat, arrays=arrays, semiring=semiring)
            indexed = compile_template(compact, arrays=arrays, semiring=semiring)
            assert_plan_equal(indexed, direct)
            assert Engine(indexed).schedule.describe() == Engine(direct).schedule.describe()


@pytest.mark.parametrize("use_x64", [False, True])
def test_sparse_plate_observations_match_flat_inference(use_x64):
    compact = plate(pixel_fragment(), (2,), Expose())
    dtype = np.float64 if use_x64 else np.float32
    arrays = {"d": np.asarray([0, .25, .75], dtype=dtype),
              "A": np.asarray([[1, 1, 0], [0, 0, 1]], dtype=dtype)}
    with jax.enable_x64(use_x64):
        results = []
        for model in (compact, expand_plates(compact)):
            engine = Engine(compile_template(model, arrays=arrays))
            state = engine.bind_data(engine.init_state(), {"p/0/x": 0, "p/1/x": None})
            assert engine.run(state, 0, tol=1e-6).status is RunStatus.BUDGET_EXHAUSTED
            result = engine.run(state, 10, tol=1e-6)
            result.require_converged()
            assert result.state.messages.dtype == dtype
            results.append(result)
        assert_result_equal(*results)


def test_no_copied_ir_at_cold_compile_and_cached_spawn(monkeypatch):
    fragment = pixel_fragment()
    fragment.constraints = [ConstraintSpec(ConstraintKind.MEAN_FIELD, "obs")]
    fragment.learnables["sensor"] = LearnableSpec("sensor", "dirichlet", ("A",))
    t = plate(fragment, (2, 3), Expose())
    flat = expand_plates(t)
    y = {"p/1/2/x": 0}
    a = {"d": np.asarray([.4, .2, .4], dtype=np.float32),
         "A": np.asarray([[.7, .2, .9], [.3, .8, .1]], dtype=np.float32)}
    direct = Engine(compile_template(flat, arrays=a))
    baseline = direct.run(direct.bind_data(direct.init_state(), y), 30, tol=1e-6)
    baseline.require_converged()

    def forbidden(*args, **kwargs):
        pytest.fail("compact compilation copied an IR declaration or ran public expansion")

    for cls in (VariableSpec, FactorSpec, PortSpec, ConstraintSpec):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr("cffg.compiler.expand_plates", forbidden)
    monkeypatch.setattr("cffg.core.plates._scope_copy", forbidden)
    colony = Colony()
    ids = [colony.spawn(t, a, observations=y) for _ in range(2)]
    assert len(colony.engines) == 1
    for result in colony.run(30, tol=1e-6).values():
        assert_result_equal(result, baseline)
    assert ids == [0, 1]


def test_registered_callback_receives_concrete_source_records():
    kernel = declaration()
    source, arrays = model()
    compact = plate(source, (2,))
    seen = []

    def inspect_template(t, members, data):
        assert isinstance(t, Template)
        assert all(isinstance(f, FactorSpec) for f in t.factors.values())
        assert all(isinstance(v, VariableSpec) and v.name == name for name, v in t.variables.items())
        assert to_dict(t) == expected
        assert list(t.variables) == expected_names
        seen.append(tuple(members))
        return kernel.pack_params(t, members, data)

    binding = replace(kernel.binding, pack_params=inspect_template)
    kernels.register_kernel(FactorKernel(kernel.factor_type, kernel.impl_id, kernel.version,
                                         hostile=kernel.hostile, binding=binding))
    try:
        flat = expand_plates(compact)
        expected = to_dict(flat)
        expected_names = list(flat.variables)
        assert_plan_equal(compile_template(compact, arrays=arrays), compile_template(flat, arrays=arrays))
        assert seen == [("p/0/interaction", "p/1/interaction")] * 2
    finally:
        kernels.unregister_kernel(kernel.factor_type.name)
