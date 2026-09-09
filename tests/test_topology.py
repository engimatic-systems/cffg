"""Declared wiring stays separate from factor semantics and observation rows."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from test_compiler_contracts import model

from cffg import compiler
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    ModelError,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.plates import expand_plates
from cffg.core.validate import validate
from cffg.topology import build_topology, prefix_offsets


def topology(template):
    expanded = expand_plates(template)
    validate(expanded)
    return build_topology(expanded)


def test_declarations_and_layout_do_not_retain_mutable_template_aliases():
    template, _ = model("hmm")
    layout = topology(template)
    name = next(iter(template.factors))
    source = template.factors[name]
    expected_edges = dict(source.edges)
    expected_blobs = dict(source.param_blobs)
    source.edges.clear()
    source.param_blobs.clear()
    template.factors.clear()
    template.variables.clear()
    template.ports.clear()
    assert layout.factors[name].edges == expected_edges
    assert layout.factors[name].param_blobs == expected_blobs
    assert layout.variables and layout.ports
    assert all(layout.degrees[name] == len(ends) for name, ends in layout.endpoints.items())
    with pytest.raises(TypeError):
        layout.factors[name].edges["out"] = "changed"
    with pytest.raises(TypeError):
        layout.endpoints["changed"] = ()
    with pytest.raises(FrozenInstanceError):
        layout.n_slots = 0
    with pytest.raises(TypeError):
        layout.message_lengths[0] = 99


@pytest.mark.parametrize("name", ["hmm", "gaussian", "ef", "parameter_edge", "plate"])
def test_declared_slots_survive_downstream_lowering(name, monkeypatch):
    template, arrays = model(name)
    layout = topology(template)
    monkeypatch.setattr(compiler, "build_topology", lambda _: layout)
    arena = compiler.compile_template(template, arrays=arrays)
    assert (arena.n_slots, arena.n_gslots, arena.n_efslots) == (
        layout.n_slots, layout.n_gslots, layout.n_efslots,
    )
    np.testing.assert_array_equal(arena.message_starts, prefix_offsets(layout.message_lengths))
    np.testing.assert_array_equal(arena.g_eta_starts, prefix_offsets(layout.g_eta_lengths))
    np.testing.assert_array_equal(arena.g_lam_starts, prefix_offsets(layout.g_lam_lengths))
    for edge_name, raw in layout.edges.items():
        lowered = arena.edges[edge_name]
        assert (raw.slot0, raw.slot1, raw.card, raw.dim, raw.ef) == (
            lowered.slot0, lowered.slot1, lowered.card, lowered.dim, lowered.ef,
        )
        assert raw.param_shape == () and raw.param_axis == 0
        if lowered.param_shape:
            assert raw is not lowered
            assert raw.card == int(np.prod(lowered.param_shape))
        else:
            assert raw is lowered
    for ends in layout.endpoints.values():
        assert tuple(sorted(ends)) == ends
        if len(ends) == 2:
            a, b = ends
            assert layout.slot_of[a] == layout.opposite[b]
            assert layout.slot_of[b] == layout.opposite[a]


def test_data_candidate_order_does_not_determine_observation_row_order():
    template = Template(
        "different-orders",
        variables={name: VariableSpec(name, dim=1) for name in ("z", "a", "state")},
        factors={
            name: FactorSpec(
                name, "gaussian_linear", {"out": output, "in": "state"},
                param_blobs={"matrix": "A", "offset": "b", "cov": "V"},
            )
            for name, output in (("a_factor", "z"), ("z_factor", "a"))
        },
        blobs={name: BlobRef(name, f"{name}.npy") for name in ("A", "b", "V")},
        ports={name + "_data": PortSpec(name + "_data", name, PortRole.DATA) for name in ("z", "a")},
    )
    layout = topology(template)
    assert tuple(candidate.port for candidate in layout.data_candidates) == ("a_data", "z_data")
    arena = compiler.compile_template(
        template, arrays={"A": np.ones((1, 1)), "b": np.zeros(1), "V": np.eye(1)},
    )
    assert arena.gauss_data == {"z_data": (0, 1), "a_data": (1, 1)}


def admission_model():
    return Template(
        "admission-order",
        variables={"x": VariableSpec("x", 2)},
        factors={"f": FactorSpec("f", "categorical", {"out": "x"}, param_blobs={"statistic": "p"})},
        blobs={"p": BlobRef("p", "p.npy")},
        ports={"goal": PortSpec("goal", "x", PortRole.GOAL)},
        constraints=[ConstraintSpec(ConstraintKind.MOMENT_MATCHING, "f")],
    )


@pytest.mark.parametrize("cardinality", [2, 2**63])
def test_topology_returns_port_diagnostics_and_compiler_keeps_admission_order(cardinality):
    template = admission_model()
    template.variables["x"] = VariableSpec("x", cardinality)
    layout = topology(template)
    port_problem = "port 'goal' (goal) is unterminated: compose the missing factor before compiling"
    assert layout.port_problems == (port_problem,)
    assert layout.constraints == tuple(template.constraints)
    assert layout.data_candidates == ()
    assert layout.message_lengths == (cardinality, cardinality)
    with pytest.raises(compiler.CompileError) as exc:
        compiler.compile_template(template, arrays={"p": np.ones(2) / 2})
    assert str(exc.value) == (
        "cannot compile 'admission-order':\n  - factor 'f': legacy moment-matching syntax "
        "does not identify an incident edge endpoint or projection\n  - " + port_problem
    )


def test_core_validation_still_owns_malformed_incidence(monkeypatch):
    template = admission_model()
    template.variables["unused"] = VariableSpec("unused", 2)

    def unexpected(_):
        pytest.fail("topology ran before core validation")

    monkeypatch.setattr(compiler, "build_topology", unexpected)
    with pytest.raises(ModelError) as exc:
        compiler.compile_template(template)
    assert "variable 'unused': unused (degree 0)" in exc.value.problems


def test_conditioning_diagnostics_still_wait_until_after_array_resolution(monkeypatch):
    template = Template(
        "invalid-conditioning",
        variables={"x": VariableSpec("x", dim=1)},
        factors={"prior": FactorSpec("prior", "gaussian", {"out": "x"}, param_blobs={"mean": "m", "cov": "V"})},
        blobs={name: BlobRef(name, f"{name}.npy") for name in ("m", "V")},
        ports={"data": PortSpec("data", "x", PortRole.DATA)},
    )
    assert len(topology(template).data_candidates) == 1

    class ResolutionFailure(Exception):
        pass

    def fail_resolution(*args):
        raise ResolutionFailure

    with monkeypatch.context() as patch:
        patch.setattr(compiler, "_resolve_arrays", fail_resolution)
        with pytest.raises(ResolutionFailure):
            compiler.compile_template(template)
    with pytest.raises(compiler.CompileError) as exc:
        compiler.compile_template(template, arrays={"m": np.zeros(1), "V": np.eye(1)})
    assert str(exc.value) == (
        "cannot compile 'invalid-conditioning':\n  - port 'data': Gaussian DATA port "
        "on 'prior'/'out' requires a registered conditioning role"
    )


def test_constraint_detail_is_a_snapshot_even_for_list_input():
    template = admission_model()
    detail = [["projection", "unspecified"]]
    template.constraints[0] = replace(template.constraints[0], detail=detail)
    layout = topology(template)
    detail[0][1] = "changed"
    assert layout.constraints[0].detail == (("projection", "unspecified"),)


def test_parameter_diagnostic_keeps_its_original_endpoint_list_format():
    template, arrays = model("parameter_edge")
    del template.factors["theta_prior"]
    template.ports["theta"] = PortSpec("theta", "theta", PortRole.GLUE)
    assert topology(template).endpoints["theta"] == (("lik", "matrix"),)
    with pytest.raises(compiler.CompileError) as exc:
        compiler.compile_template(template, arrays=arrays)
    assert str(exc.value) == (
        f"cannot compile {template.name!r}:\n  - parameter edge 'theta': requires exactly "
        "one facing Dirichlet prior, got endpoints [('lik', 'matrix')]"
    )
