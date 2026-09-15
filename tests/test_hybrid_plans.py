"""D12 metadata is inspectable while every production execution path is closed."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest
from test_hybrid_declarations import declared

from cffg import compiled
from cffg._immutable import FrozenArray
from cffg.compiled import Arena, MemberAddress, ParameterAddress
from cffg.compiler import compile_template
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.engine import Engine
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.execution.synchronous import make_sweep
from cffg.execution_availability import ExecutionUnavailableError
from cffg.hybrid_plans import (
    HybridAdapterPlan,
    HybridBufferRequirement,
    HybridComponentPlan,
    HybridRegionPlan,
    HybridSeparatorPlan,
)
from cffg.inventory import analytic_inventory
from cffg.ledger import ObjectiveLedger, make_belief_energy
from cffg.lifecycle import StateLifecycle
from cffg.schedule import Schedule, build_schedule
from cffg.schedule_program import (
    build_schedule_program,
    make_exact_unit,
    make_propagate,
    make_support_check,
)


def plan_fixture(**changes):
    """Hand-published metadata, not evidence of compiler assembly support."""
    spec = declared().hybrid_components["association"]
    state_copies = ("xr", "xd", "xp")
    plan = HybridComponentPlan(
        spec,
        (
            HybridSeparatorPlan("left", 3, 2, ("z0",), state_copies),
            HybridSeparatorPlan("middle", 3, 2, ("z1",), state_copies),
            HybridSeparatorPlan("right", 3, 2, ("z1",), ("y",)),
        ),
        tuple(HybridRegionPlan(name, np.asarray(indices, np.int32))
              for name, indices in (("switch", (0, 1)), ("channel", (1, 2)),
                                    ("label_prior", (0,)), ("state_prior", (0,)))),
        (HybridAdapterPlan("readout", 2, "gaussian", (10, 11), 10),),
    )
    return replace(plan, **changes)


def arena_fixture(plan=None):
    return Arena(
        "inspection", 0, np.zeros(1, np.int64), {}, (), {},
        hybrid_components=(plan or plan_fixture(),),
        log_params={"weights": np.zeros((1, 3, 2, 2), np.float32)},
        member_of={"r": MemberAddress("recurrence_link", (
            ParameterAddress("weights", "w", "weights", 0, "identity"),
        ))},
    )


def test_identity_ownership_and_input_snapshots():
    plan = plan_fixture()
    indices = np.array([0, 1], np.int32)
    region = HybridRegionPlan("switch", indices)
    members = [region, *plan.regions[1:]]
    plan = replace(plan, regions=members)
    arena = arena_fixture(plan)
    indices[:] = -1
    members.clear()
    np.testing.assert_array_equal(plan.regions[0].separator_indices, [0, 1])
    assert arena.hybrid_components[0] is plan
    assert plan.declaration == declared().hybrid_components["association"]
    assert plan.declaration.separators[0].active_support == (0, 2)
    assert plan.members == ("r", "d", "pz", "px", "eq")
    assert plan.parameter_members == ("r", "d", "pz", "px")
    assert plan.schedule_stage == "deferred"
    address = arena.member_of[plan.parameter_members[0]].parameter("weights")
    assert (address.blob, address.key, address.index) == ("w", "weights", 0)
    assert arena.log_params[address.key].shape == (1, 3, 2, 2)
    with pytest.raises(FrozenInstanceError):
        plan.declaration = None
    with pytest.raises(FrozenInstanceError):
        plan.separators[0].cardinality = 9
    with pytest.raises(TypeError):
        plan.regions[0].separator_indices[0] = 9
    with pytest.raises(TypeError, match="execution_available"):
        replace(plan, execution_available=True)


@pytest.mark.parametrize("change, message", [
    (lambda p: dict(declaration=replace(p.declaration, approximation="hybrid-regime-ec-v2")), "unknown approximation"),
    (lambda p: dict(declaration=replace(p.declaration, support_semantics="adaptive")), "support semantics"),
    (lambda p: dict(declaration=replace(p.declaration, separators=("broken",))), "separator plan identities"),
    (lambda p: dict(declaration=replace(p.declaration, equalities=("r",))), "ownership"),
    (lambda p: dict(regions=(replace(p.regions[0], name="renamed"), *p.regions[1:])), "region plan identities"),
    (lambda p: dict(regions=(replace(p.regions[0], separator_indices=np.array([1, 0])), *p.regions[1:])), "ordered separator indexes"),
    (lambda p: dict(regions=(replace(p.regions[0], separator_indices=np.array([0., 1.])), *p.regions[1:])), "ordered separator indexes"),
    (lambda p: dict(separators=(replace(p.separators[0], cardinality=2), *p.separators[1:])), "fixed label support"),
    (lambda p: dict(separators=(replace(p.separators[0], regime_variables=("other",)), *p.separators[1:])), "graph-copy identities"),
    (lambda p: dict(adapters=(replace(p.adapters[0], separator_index=0),)), "separator index"),
    (lambda p: dict(adapters=(replace(p.adapters[0], domain="categorical"),)), "domain/copy"),
    (lambda p: dict(adapters=(replace(p.adapters[0], scatter=12),)), "gather/scatter"),
    (lambda p: dict(coordinate_requirements=(HybridBufferRequirement("vector", (-1,)),)), "shape/dtype"),
    (lambda p: dict(scratch_requirements=(HybridBufferRequirement("status", (1,), "object"),)), "shape/dtype"),
])
def test_malformed_contracts_fail_by_component(change, message):
    plan = plan_fixture()
    with pytest.raises(ValueError, match=f"association.*{message}"):
        replace(plan, **change(plan))


def test_duplicate_and_untyped_arena_identities_are_rejected():
    arena = arena_fixture()
    with pytest.raises(ValueError, match="association.*duplicate compiled identity"):
        replace(arena, hybrid_components=arena.hybrid_components * 2)
    with pytest.raises(ValueError, match="association.*HybridComponentPlan"):
        replace(arena, hybrid_components=(arena.hybrid_components[0].declaration,))


def _forbidden(*args, **kwargs):
    pytest.fail("execution or JIT wrapping occurred before the availability check")


@pytest.mark.parametrize("factory", [
    lambda a: Engine(a),
    lambda a: Engine(a, Schedule(kind="flood", reason="supplied")),
    lambda a: StateLifecycle(a),
    lambda a: make_sweep(a),
    lambda a: make_sweep(a, True),
    lambda a: build_schedule(a),
    lambda a: build_schedule(a, "flood"),
    lambda a: build_schedule(a, "two_pass"),
    lambda a: build_schedule_program(a, "auto", SCHEDULED_EXECUTORS,
                                    make_sweep=_forbidden, free_energy=_forbidden),
    lambda a: build_schedule_program(a, Schedule(kind="flood", reason="supplied"),
                                    SCHEDULED_EXECUTORS, make_sweep=_forbidden,
                                    free_energy=_forbidden),
    lambda a: make_support_check(a),
    lambda a: make_exact_unit(a, _forbidden, _forbidden, _forbidden),
    lambda a: make_propagate(a, ()),
    lambda a: make_belief_energy(a),
    lambda a: ObjectiveLedger(a, _forbidden),
])
def test_standalone_execution_factories_reject_before_jit(factory, monkeypatch):
    arena = arena_fixture()
    monkeypatch.setattr(eqx, "filter_jit", _forbidden)
    monkeypatch.setattr(jax, "jit", _forbidden)
    with pytest.raises(ExecutionUnavailableError, match="association.*hybrid-regime-ec-v1.*execution unavailable"):
        factory(arena)


@pytest.mark.parametrize("dtype, itemsize", [(np.float32, 4), (np.float64, 8)])
def test_inventory_separates_actual_storage_from_future_requirements(dtype, itemsize, monkeypatch):
    plan = plan_fixture(
        coordinate_requirements=(HybridBufferRequirement("accepted", (19,)),),
        scratch_requirements=(HybridBufferRequirement("residual", (43,)),
                              HybridBufferRequirement("status", (), "int32")),
    )
    arena = arena_fixture(plan)
    monkeypatch.setattr(FrozenArray, "__array__", _forbidden)
    monkeypatch.setattr(jax, "jit", _forbidden)
    monkeypatch.setattr(eqx, "filter_jit", _forbidden)
    monkeypatch.setattr(Engine, "__init__", _forbidden)
    monkeypatch.setattr(StateLifecycle, "init_state", _forbidden)
    report = analytic_inventory(arena, dtype=dtype)
    # Six stored int32 port indexes plus four int64 prefix sentinels.
    assert report.stored_plan_bytes == 6 * 4 + 4 * 8
    assert report.stored_parameter_bytes == 12 * 4
    assert sum(a.elements for a in report.stored_plan_arrays) == 10
    assert all(a.elements == 0 for a in report.ordinary_state_requirements)
    hybrid, = report.hybrid_requirements
    assert hybrid.component == "association"
    assert hybrid.coordinate_bytes == 19 * itemsize
    assert hybrid.scratch_bytes == 43 * itemsize + 4
    assert report.physical_peak_bytes is report.compiled_peak_bytes is None
    with pytest.raises(FrozenInstanceError):
        report.dtype = "float16"


def test_unknown_empty_and_large_shape_only_requirements():
    unknown = analytic_inventory(plan_fixture(), dtype="float32")
    assert unknown.stored_parameter_arrays is unknown.ordinary_state_requirements is None
    assert unknown.stored_parameter_bytes is None
    assert unknown.hybrid_requirements[0].coordinate_bytes is None
    assert unknown.hybrid_requirements[0].scratch_bytes is None
    plan = plan_fixture(
        coordinate_requirements=(HybridBufferRequirement("accepted", (2**40, 100)),),
        scratch_requirements=(),
    )
    report = analytic_inventory(plan, dtype="float64")
    assert report.hybrid_requirements[0].coordinate_bytes == 2**40 * 100 * 8
    assert report.hybrid_requirements[0].scratch_bytes == 0
    assert report.stored_plan_bytes == unknown.stored_plan_bytes == 24
    with pytest.raises(ValueError, match="float32 or float64"):
        analytic_inventory(plan, dtype="float16")
    with pytest.raises(TypeError, match="dtype"):
        analytic_inventory(plan)


def test_shared_published_arrays_are_counted_once_and_aliases_are_reported():
    plan = plan_fixture()
    regions = (*plan.regions[:3], replace(plan.regions[3],
                                        separator_indices=plan.regions[2].separator_indices))
    report = analytic_inventory(replace(plan, regions=regions), dtype="float32")
    assert report.stored_plan_bytes == 20
    assert report.stored_plan_arrays[-1].aliases == ("plan.regions[3].separator_indices",)


@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_ordinary_models_keep_execution_and_inventory_paths(schedule, monkeypatch):
    template = Template("ordinary", variables={"z": VariableSpec("z", cardinality=2)},
                        factors={"prior": FactorSpec("prior", "categorical", {"out": "z"},
                                                    param_blobs={"statistic": "p"})},
                        blobs={"p": BlobRef("p", "p.npy")},
                        ports={"z": PortSpec("z", "z", PortRole.GLUE)})
    monkeypatch.setattr(HybridComponentPlan, "__post_init__", _forbidden)
    arena = compile_template(template, arrays={"p": np.array([0.25, 0.75])})
    assert arena.hybrid_components == ()
    report = analytic_inventory(arena, dtype="float32")
    assert report.hybrid_requirements == ()
    assert report.ordinary_state_requirements[0].logical_bytes == 4 * 4
    engine = Engine(arena, schedule)
    state = engine.step(engine.init_state())
    np.testing.assert_allclose(engine.marginals(state)["z"], [0.25, 0.75])
    assert np.isfinite(engine.bfe(state))


def test_inventory_owner_imports_and_runs_without_jax_or_numerical_modules():
    source = """
import importlib
import importlib.abc
import sys
import types

class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        forbidden = ('jax', 'equinox', 'cffg.compiler', 'cffg.engine',
                     'cffg.hybrid_joint', 'cffg.hybrid_joint_storage',
                     'cffg.hybrid_separator', 'cffg.lifecycle')
        if any(fullname == n or fullname.startswith(n + '.') for n in forbidden):
            raise AssertionError('inventory imports ' + fullname)

package = types.ModuleType('cffg')
package.__path__ = [sys.argv[1]]
sys.modules['cffg'] = package
sys.meta_path.insert(0, NoRuntime())
import numpy as np
from cffg.compiled import Arena
from cffg.core.ir import HybridComponentSpec, HybridRegionSpec, HybridSeparatorSpec
from cffg.hybrid_plans import HybridComponentPlan, HybridRegionPlan, HybridSeparatorPlan
from cffg.inventory import analytic_inventory
arena = Arena('empty', 0, np.zeros(1, np.int64), {}, (), {})
report = analytic_inventory(arena, dtype='float32')
assert report.hybrid_requirements == ()
assert report.stored_plan_bytes == 32
assert report.stored_parameter_bytes == 0
assert report.physical_peak_bytes is None
spec = HybridComponentSpec('inspect',
    (HybridSeparatorSpec('boundary', 'z', 'x', (0, 2)),),
    (HybridRegionSpec('prior', 'unary', 'p', ('boundary',)),))
plan = HybridComponentPlan(spec,
    (HybridSeparatorPlan('boundary', 3, 2, ('z',), ('x',)),),
    (HybridRegionPlan('prior', np.array([0], np.int32)),))
report = analytic_inventory(plan, dtype='float64')
assert report.stored_plan_bytes == 4
assert report.hybrid_requirements[0].coordinate_bytes is None
assert report.hybrid_requirements[0].scratch_bytes is None
"""
    subprocess.run([sys.executable, "-c", source, str(Path(compiled.__file__).parent)],
                   check=True, capture_output=True, text=True)
