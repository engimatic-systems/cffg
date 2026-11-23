"""Endpoint meaning survives model transport and Figure 12 compression."""

from dataclasses import replace

import numpy as np
import pytest

from cffg.compiler import CompileError, compile_template
from cffg.core import (
    BlobRef, ConstraintKind, ConstraintSpec, FactorSpec, ModelError,
    MomentProjection, PlateSpec, PortRole, PortSpec, Template, VariableSpec,
    expand_plates, migrate_moment_matching, moment_matching,
    moment_matching_endpoint, validate,
)
from cffg.core.compose import merge, scoped
from cffg.core.edits import AddConstraint, action_from_dict, action_to_dict
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.viz import NodeEnd, derive, layout, present, render_svg
from cffg.viz.scene import Bead, Path


@pytest.fixture
def gaussian_pair():
    arrays = {"m": np.zeros(1), "cov": np.eye(1), "A": np.eye(1)}
    template = Template(
        "pair",
        variables={name: VariableSpec(name, dim=1) for name in ("x", "y")},
        factors={
            "prior": FactorSpec("prior", "gaussian", {"out": "x"},
                                param_blobs={"mean": "m", "cov": "cov"}),
            "map": FactorSpec("map", "gaussian_linear", {"in": "x", "out": "y"},
                              param_blobs={"matrix": "A", "offset": "m", "cov": "cov"}),
        },
        ports={"tail": PortSpec("tail", "y", PortRole.GLUE)},
        blobs={name: BlobRef(name, name + ".npy") for name in arrays},
    )
    return template, arrays


def declaration(factor="map", role="in"):
    return moment_matching(factor, role, MomentProjection.GAUSSIAN)


def test_both_ends_are_distinct_serialized_constraints(gaussian_pair):
    template, _ = gaussian_pair
    template.constraints = [declaration()]
    one_hash = structure_hash(template)
    template.constraints.append(declaration("prior", "out"))
    validate(template)
    restored = from_json(to_json(template))
    assert restored.constraints == sorted(template.constraints, key=lambda c: c.target)
    assert structure_hash(restored) == structure_hash(template) != one_hash
    assert {c.target for c in restored.constraints} == {"prior", "map"}
    for c in restored.constraints:
        endpoint = moment_matching_endpoint(c, restored)
        assert restored.factors[c.target].edges[endpoint.role] == "x"


@pytest.mark.parametrize("detail, message", [
    ((("role", "in"),), "exactly 'role' and 'projection'"),
    ((("projection", "gaussian_first_second"),), "exactly 'role' and 'projection'"),
    ((("role", "matrix"), ("projection", "gaussian_first_second")), "structural edge"),
    ((("role", "in"), ("projection", "unspecified")), "unknown moment projection"),
    ((("role", "in"), ("role", "out"), ("projection", "gaussian_first_second")), "duplicate"),
])
def test_invalid_details_cannot_be_validated_or_silently_serialized(gaussian_pair, detail, message):
    template, _ = gaussian_pair
    template.constraints = [ConstraintSpec(ConstraintKind.MOMENT_MATCHING, "map", detail)]
    with pytest.raises(ModelError, match=message):
        validate(template)
    with pytest.raises(ValueError, match=message):
        to_json(template)


def test_legacy_requires_explicit_migration(gaussian_pair):
    template, arrays = gaussian_pair
    legacy = ConstraintSpec(ConstraintKind.MOMENT_MATCHING, "map")
    template.constraints = [legacy]
    restored = from_json(to_json(template))
    assert restored.constraints == [legacy]
    with pytest.raises(CompileError, match="legacy moment-matching syntax"):
        compile_template(restored, arrays=arrays)
    with pytest.raises(ValueError, match="legacy factor-only syntax"):
        derive(restored)
    with pytest.raises(TypeError, match="projection"):
        migrate_moment_matching(legacy, template, role="in")
    restored.constraints = [migrate_moment_matching(
        legacy, restored, role="in", projection=MomentProjection.GAUSSIAN,
    )]
    assert restored.constraints == [declaration()]
    compile_template(restored, arrays=arrays)


def test_compiler_requires_declared_projection_capability(gaussian_pair, monkeypatch):
    from cffg import kernels

    template, arrays = gaussian_pair
    template.constraints = [declaration()]
    original = kernels.KERNELS["gaussian_linear"]
    replacement = kernels.FactorKernel(
        original.factor_type, original.impl_id, original.version, hostile=original.hostile,
        binding=replace(original.binding, moment_projections=frozenset()),
    )
    monkeypatch.setitem(kernels.KERNELS, "gaussian_linear", replacement)
    with pytest.raises(CompileError, match="no registered moment-matching rule"):
        compile_template(template, arrays=arrays)


def test_gaussian_projection_rejects_discrete_endpoint(hmm_slice):
    hmm_slice.constraints = [declaration("obs", "in")]
    with pytest.raises(ModelError, match="requires a Gaussian edge"):
        validate(hmm_slice)


def test_scope_plate_composition_and_edit_codec_retain_role(gaussian_pair):
    template, _ = gaussian_pair
    action = AddConstraint(declaration())
    assert action_from_dict(action_to_dict(action)) == action
    action.apply(template)
    copied = scoped(template, "left")
    assert copied.constraints == [replace(declaration(), target="left/map")]
    joined = merge(copied, scoped(template, "right"), "joined")
    validate(joined)
    assert {c.target for c in joined.constraints} == {"left/map", "right/map"}
    # A terminated prior fragment can be independently repeated in a plate.
    fragment = Template(
        "leaf", variables={"x": VariableSpec("x", dim=1)},
        factors={"prior": template.factors["prior"]},
        blobs={name: template.blobs[name] for name in ("m", "cov")},
        constraints=[declaration("prior", "out")],
        ports={"x": PortSpec("x", "x", PortRole.GLUE)},
    )
    from cffg.core import Expose
    repeated = Template("repeated", plates={
        "copies": PlateSpec(fragment, (3,), bonds=(Expose(),)),
    })
    flat = expand_plates(repeated)
    validate(flat)
    assert len(flat.constraints) == 3
    assert len({c.target for c in flat.constraints}) == 3
    assert all(moment_matching_endpoint(c, flat).role == "out" for c in flat.constraints)


@pytest.mark.parametrize("both", [False, True])
@pytest.mark.parametrize("level", [0, 4])
def test_figure12_marks_only_selected_half_edges(gaussian_pair, both, level):
    template, _ = gaussian_pair
    template.constraints = [declaration()]
    if both:
        template.constraints.append(declaration("prior", "out"))
    diagram = derive(template)
    wire = diagram.wires["var:x"]
    assert wire.bead == "E"
    selected = {end for end, _ in wire.moment_endpoints}
    assert NodeEnd("map", "in") in selected
    assert (NodeEnd("prior", "out") in selected) == both
    scene = layout(diagram, present(diagram, level))
    halves = [item for item in scene.items if isinstance(item, Path) and item.ref == "var:x"]
    assert len(halves) == 2
    assert sum(item.dashed for item in halves) == 1 + both
    beads = [item for item in scene.items if isinstance(item, Bead) and item.ref == "var:x"]
    assert len(beads) == 1 and beads[0].symbol == "E"
    assert 'stroke-dasharray="4 3"' in render_svg(template, level=level)


def test_gaussian_endpoint_bundle_and_inference_parity(gaussian_pair, tmp_path):
    from cffg.bundle import load_bundle, save_bundle
    from cffg.engine import Engine

    template, arrays = gaussian_pair
    plain = Engine(compile_template(template, arrays=arrays))
    template.constraints = [declaration(), declaration("prior", "out")]
    marked = Engine(compile_template(template, arrays=arrays))
    target = tmp_path / "moments.cffgz"
    save_bundle(target, template, arrays)
    restored = load_bundle(target)
    assert restored.template.constraints == from_json(to_json(template)).constraints
    assert structure_hash(restored.template) == structure_hash(template)
    # Gaussian full-density and first/second-moment consistency are equivalent.
    a = plain.run(plain.init_state(), 8, tol=1e-9)
    b = marked.run(marked.init_state(), 8, tol=1e-9)
    assert a.status == b.status
    np.testing.assert_allclose(a.state.g_eta, b.state.g_eta)
    np.testing.assert_allclose(a.state.g_lam, b.state.g_lam)
    np.testing.assert_allclose(a.free_energies, b.free_energies)
