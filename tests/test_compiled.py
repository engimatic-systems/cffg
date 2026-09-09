"""Published compiler values are shared blueprints, not runtime state."""

import subprocess
import sys
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_compiler_contracts import IDENTITIES, model

from cffg import compiled, compiler, kernels, set_params
from cffg._immutable import FrozenArray, FrozenMap, ImmutableRecord, freeze
from cffg.compiled import Arena, EdgeInfo, GroupPlan
from cffg.engine import Engine


def test_compatibility_exports_have_one_owner():
    records = {
        name: value for name, value in vars(compiled).items()
        if isinstance(value, type) and is_dataclass(value)
    }
    assert len(records) == 20
    for name, value in records.items():
        assert getattr(compiler, name) is value
        assert value.__module__ == "cffg.compiled"
        assert issubclass(value, ImmutableRecord)
        assert value.__dataclass_params__.frozen
    assert compiler.NEG == compiled.NEG
    assert compiler.PARAMETER_TRANSFORMS is compiled.PARAMETER_TRANSFORMS


def test_value_owner_imports_without_compiler_runtime_or_jax():
    # Bypass the public facade, which deliberately exports Engine and Agent.
    source = """
import importlib
import importlib.abc
import sys
import types

class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        forbidden = ('jax', 'equinox', 'cffg.compiler', 'cffg.engine',
                     'cffg.schedule', 'cffg.outer', 'cffg.learn', 'cffg.params',
                     'cffg.agent', 'cffg.models', 'cffg.estimator')
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('compiled values import ' + fullname)

package = types.ModuleType('cffg')
package.__path__ = [sys.argv[1]]
sys.modules['cffg'] = package
sys.meta_path.insert(0, NoRuntime())
values = importlib.import_module('cffg.compiled')
assert values.Arena.__module__ == 'cffg.compiled'
"""
    subprocess.run(
        [sys.executable, "-c", source, str(Path(compiled.__file__).parent)],
        check=True, capture_output=True, text=True,
    )


def test_publication_snapshots_inputs_and_reuses_published_storage():
    slots = np.array([[0, 1]], dtype=np.int64)
    members = ["factor"]
    plan = GroupPlan("group", "equality", ("a", "b"), (2, 2), members, slots, slots)
    edge = EdgeInfo("x", 2, 0, 1, False)
    edges = {"x": edge}
    groups = [plan]
    params = {"group": np.array([[0.0, -np.inf]], dtype=np.float32)}
    arena = Arena("test", 2, np.array([0, 2, 4]), edges, groups, {}, log_params=params)

    slots[:] = 9
    members.append("new")
    groups.clear()
    edges.clear()
    params["group"][:] = 7
    np.testing.assert_array_equal(plan.gather, [[0, 1]])
    assert plan.members == ("factor",)
    assert arena.groups == (plan,)
    assert arena.edges["x"] is edge
    np.testing.assert_array_equal(arena.log_params["group"], [[0, -np.inf]])
    assert slots.flags.writeable and params["group"].flags.writeable
    assert arena.groups[0] is plan
    assert freeze(plan.gather) is plan.gather
    assert replace(plan, mode="vmp").gather is plan.gather
    other = replace(arena, template_name="other")
    assert other.log_params is arena.log_params
    assert other.message_starts is arena.message_starts
    assert other.groups is arena.groups


@pytest.mark.parametrize("name", IDENTITIES)
def test_every_published_field_is_read_only(name):
    template, arrays = model(name)
    arena = compiler.compile_template(template, arrays=arrays)

    def check(value):
        if isinstance(value, kernels.FactorKernel):
            assert value is kernels.kernel_for(value.factor_type.name)
            check(value.binding)
        elif isinstance(value, ImmutableRecord):
            for field in fields(value):
                item = getattr(value, field.name)
                with pytest.raises(FrozenInstanceError):
                    setattr(value, field.name, item)
                check(item)
        elif isinstance(value, FrozenArray):
            with pytest.raises(TypeError):
                value.setflags(write=True)
            with pytest.raises(TypeError):
                value[...] = 0
        elif isinstance(value, Mapping):
            assert isinstance(value, FrozenMap)
            with pytest.raises(TypeError):
                value["new"] = 0
            for item in value.values():
                check(item)
        elif isinstance(value, (tuple, frozenset)):
            for item in value:
                check(item)
        else:
            assert not isinstance(value, (list, dict, set, np.ndarray))

    check(arena)


@pytest.mark.parametrize("source", [
    np.array(7, dtype=np.int64),
    np.empty((0, 3), dtype=np.float32),
    np.arange(12, dtype=np.int64).reshape(3, 4),
    np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
    np.arange(24, dtype=np.float32).reshape(4, 6)[::-1, ::2],
    np.arange(8, dtype=">i8"),
    np.array([0, 0x80000000, 0x7FC00123, 0xFF800000], dtype=np.uint32).view(np.float32),
    np.array([0, 0x8000000000000000, 0x7FF8000000000123], dtype=np.uint64).view(np.float64),
], ids=["scalar", "empty", "C", "F", "strided", "endian", "f32-bits", "f64-bits"])
def test_array_publication_preserves_shape_dtype_and_bits(source):
    published = freeze(source)
    assert published.shape == source.shape
    assert published.dtype == source.dtype
    assert published.tobytes() == source.tobytes()
    assert source.flags.writeable
    assert not np.shares_memory(source, published)
    with pytest.raises(TypeError, match="metadata"):
        published.shape = (source.size,)
    with pytest.raises(TypeError, match="metadata"):
        published.dtype = np.uint8
    with pytest.raises(TypeError, match="resized"):
        published.resize(source.shape)


@pytest.mark.parametrize("mutate", [
    lambda x: x.__setitem__(0, 9),
    lambda x: x.flat.__setitem__(0, 9),
    lambda x: x.fill(9),
    lambda x: x.put([0], [9]),
    lambda x: x.sort(),
    lambda x: np.copyto(x, np.zeros_like(x)),
    lambda x: np.add(x, 1, out=x),
    lambda x: np.add.at(x, [0], 10),
], ids=["item", "flat", "fill", "put", "sort", "copyto", "ufunc-out", "ufunc-at"])
def test_array_writes_and_views_are_rejected(mutate):
    published = freeze(np.array([3, 2, 1], dtype=np.int64))
    for view in (published, published[::2]):
        with pytest.raises((TypeError, AttributeError)):
            mutate(view)
        with pytest.raises(TypeError):
            view.setflags(write=True)
        mutate(np.asarray(view))
    np.testing.assert_array_equal(published, [3, 2, 1])
    with pytest.raises(ValueError, match="shared NumPy storage"):
        np.array(published, copy=False)


def test_working_numpy_outputs_are_resealed_on_publication():
    published = freeze(np.arange(4))
    for working in (published.copy(), published + 1, np.asarray(published).copy()):
        expected = working.tobytes()
        sealed = freeze(working)
        assert isinstance(sealed, FrozenArray)
        assert not np.shares_memory(working, sealed)
        working[:] = 99
        assert sealed.tobytes() == expected
    readonly_view = np.arange(4)
    readonly_view.flags.writeable = False
    sealed = freeze(readonly_view)
    readonly_view.flags.writeable = True
    readonly_view[:] = 99
    np.testing.assert_array_equal(sealed, np.arange(4))


def test_external_bytes_and_readonly_memoryviews_are_not_trusted():
    raw = np.arange(4).tobytes()
    original = np.frombuffer(raw, dtype=np.int64)
    published = freeze(original)
    try:
        np.add.at(original, [0], 10)
    except ValueError:
        pass
    np.testing.assert_array_equal(published, np.arange(4))
    mutable = bytearray(np.arange(4).tobytes())
    original = np.frombuffer(memoryview(mutable).toreadonly(), dtype=np.int64)
    published = freeze(original)
    mutable[:] = bytes(len(mutable))
    np.testing.assert_array_equal(published, np.arange(4))
    with pytest.raises(TypeError, match="Python objects"):
        freeze(np.array([{}], dtype=object))


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_jax_conversion_does_not_export_owned_storage(dtype):
    published = freeze(np.arange(4, dtype=dtype))
    converted = jnp.asarray(published)
    converted.block_until_ready()
    alias = np.asarray(converted)
    # Some NumPy versions ignore writeability for ufunc.at. If fixed, this
    # will raise; either way, the published source must remain unchanged.
    try:
        np.add.at(alias, [0], 10)
    except ValueError:
        pass
    np.testing.assert_array_equal(published, np.arange(4, dtype=dtype))
    np.testing.assert_array_equal(jax.jit(lambda: jnp.asarray(published) + 1)(), np.arange(4) + 1)


def test_declarations_and_copy_protocol_preserve_handles_not_mutable_aliases():
    kernel = kernels.kernel_for("categorical")
    parameters = dict(kernel.binding.parameters)
    roles = list(kernel.binding.roles)
    binding = replace(kernel.binding, parameters=parameters, roles=roles)
    parameters.clear()
    roles.clear()
    assert binding.parameters == kernel.binding.parameters
    assert binding.roles == kernel.binding.roles
    with pytest.raises(TypeError):
        del binding.parameters._data
    template, arrays = model("hmm")
    arena = compiler.compile_template(template, arrays=arrays)
    for value in (kernel, binding, arena, arena.message_starts, arena.log_params):
        assert copy(value) is value or isinstance(value, np.ndarray)
        assert deepcopy(value) is value
    # asdict is inspection, not a template/checkpoint serializer. Read-only
    # mapping leaves remain mappings; ndarray leaves keep their protected data.
    snapshot = asdict(arena)
    assert snapshot["log_params"] is arena.log_params
    assert snapshot["message_starts"] is arena.message_starts
    assert snapshot["groups"][0]["kernel"]["binding"]["parameters"] == binding.parameters


@pytest.mark.parametrize("schedule", ["flood", "auto"])
def test_shared_arena_keeps_runtime_updates_and_vmapped_states_independent(schedule):
    template, arrays = model("hmm")
    arena = compiler.compile_template(template, arrays=arrays)
    engine = Engine(arena, schedule=schedule)
    left, right = engine.init_state(), engine.init_state()
    assert left.params is not right.params
    before = {key: np.array(value) for key, value in arena.log_params.items()}
    address = arena.member_of["prior/init"].parameter("statistic")
    probabilities = np.zeros_like(arrays["prior/d"])
    probabilities[0] = 1
    updated_params = set_params(
        engine, left.params, {"prior/init": {"statistic": probabilities}}
    )
    left = eqx.tree_at(lambda state: state.params, left, updated_params)
    port = next(iter(arena.data_slots))
    left = engine.bind_data(left, {port: 0})
    right = engine.bind_data(right, {port: 1})
    assert not np.array_equal(left.messages, right.messages)
    assert not np.array_equal(left.params[address.key], right.params[address.key])
    batch = jax.tree.map(lambda a, b: jnp.stack([a, b]), left, right)
    stepped = jax.vmap(engine.step)(batch)
    assert not np.array_equal(stepped.messages[0], stepped.messages[1])
    for index, state in enumerate((left, right)):
        np.testing.assert_allclose(stepped.messages[index], engine.step(state).messages)
    for key, expected in before.items():
        np.testing.assert_array_equal(arena.log_params[key], expected)
