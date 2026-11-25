"""Edit bundles are all-or-nothing; failures leave the original untouched."""

import pytest

from cffg.core import (
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    ModelError,
    PortRole,
    PortSpec,
    VariableSpec,
)
from cffg.core.edits import (
    AddConstraint,
    AddFactor,
    AddVariable,
    DeclarePort,
    EditError,
    RebindBlobPath,
    RemoveFactor,
    RemovePort,
    RemoveVariable,
    apply_edits,
)
from cffg.core.serialize import structure_hash, to_dict


def snapshot(t):
    return to_dict(t)


def test_valid_bundle_grows_the_chain(hmm_slice):
    """Extend the slice with a control prior on a new selector edge — the
    kind of coherent multi-part change bundles exist for."""
    grown = apply_edits(
        hmm_slice,
        [
            AddVariable(VariableSpec("u", 4)),
            AddFactor(
                FactorSpec("ctrl", "categorical", edges={"out": "u"}, param_blobs={"statistic": "B"})
            ),
            DeclarePort(PortSpec("control", "u", PortRole.DATA)),
        ],
    )
    assert "ctrl" in grown.factors and "control" in grown.ports
    assert "ctrl" not in hmm_slice.factors  # original untouched


def test_half_valid_bundle_rolls_back(hmm_slice):
    before = snapshot(hmm_slice)
    with pytest.raises(ModelError):
        apply_edits(
            hmm_slice,
            [
                AddVariable(VariableSpec("u", 4)),  # fine alone...
                # ...but no factor/port ever attaches u -> unused variable
            ],
        )
    assert snapshot(hmm_slice) == before


def test_edit_error_rolls_back(hmm_slice):
    before = snapshot(hmm_slice)
    with pytest.raises(EditError, match="does not exist"):
        apply_edits(hmm_slice, [RemoveFactor("obs"), RemoveFactor("obs")])
    assert snapshot(hmm_slice) == before


def test_remove_factor_alone_violates_port_invariant(hmm_slice):
    with pytest.raises(ModelError):
        apply_edits(hmm_slice, [RemoveFactor("obs")])
    # The coherent bundle: the whole observation branch goes, the equality node
    # becomes pointless, and the transition rewires straight to the tail edge.
    pruned = apply_edits(
        hmm_slice,
        [
            RemoveFactor("obs"),
            RemoveFactor("eq"),
            RemoveVariable("x"),  # also drops port "sense"
            RemoveVariable("z_obs"),
            RemoveVariable("z_mid"),
            RemoveFactor("trans"),
            AddFactor(
                FactorSpec(
                    "trans", "matrix_cat", edges={"out": "z_out", "in": "z_in"}, param_blobs={"matrix": "B"}
                )
            ),
        ],
    )
    assert set(pruned.variables) == {"z_in", "z_out"}
    assert set(pruned.ports) == {"head", "tail"}


def test_remove_variable_cascades_ports(hmm_slice):
    with pytest.raises(ModelError):
        # dropping x removes port sense too; obs factor now references a ghost
        apply_edits(hmm_slice, [RemoveVariable("x")])


def test_constraint_bundle_toggles_bfe_ablation(hmm_slice):
    """The one-bead GFE<->BFE flip from Part II §7.1, as a two-action bundle."""
    bead = ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "obs", (("substituted_role", "out"),))
    gfe = apply_edits(hmm_slice, [AddConstraint(bead)])
    assert structure_hash(gfe) != structure_hash(hmm_slice)

    from cffg.core.edits import RemoveConstraint

    bfe = apply_edits(gfe, [RemoveConstraint(bead)])
    assert structure_hash(bfe) == structure_hash(hmm_slice)


def test_rebind_blob_path_preserves_structure_hash(hmm_slice):
    moved = apply_edits(hmm_slice, [RebindBlobPath("A", "trained/A_v2.safetensors")])
    assert structure_hash(moved) == structure_hash(hmm_slice)
    assert moved.blobs["A"].path == "trained/A_v2.safetensors"


def test_remove_port_alone_dangles(hmm_slice):
    with pytest.raises(ModelError):
        apply_edits(hmm_slice, [RemovePort("sense")])


def _product_latent(k=3, dim=0):
    """Unnormalized product of two normalized densities on the same scoped edge."""
    from cffg.core import BlobRef, Template
    import numpy as np

    edge = "inner/state"
    t = Template("product", variables={edge: VariableSpec(edge, k if not dim else 0, dim)})
    arrays = {}
    for i, name in enumerate(("first", "second")):
        if dim:
            params = {"mean": f"m{i}", "cov": f"V{i}"}
            arrays.update({f"m{i}": np.arange(dim) + i, f"V{i}": (i + 1) * np.eye(dim)})
            kind = "gaussian"
        else:
            weight = np.arange(1, k + 1, dtype=float) ** (i + 1)
            params = {"statistic": f"p{i}"}
            arrays[f"p{i}"] = weight / weight.sum()
            kind = "categorical"
        t.factors[name] = FactorSpec(name, kind, {"out": edge}, param_blobs=params)
    t.blobs.update({name: BlobRef(name, f"{name}.npy") for name in arrays})
    return t, edge, arrays


@pytest.mark.parametrize("count", [1, 2, 16, 64])
def test_fan_out_minimal_scoped_tree(count):
    from cffg.core import fan_out, validate

    t, target, _ = _product_latent()
    t.constraints.append(ConstraintSpec(ConstraintKind.DELTA, target))
    before = snapshot(t)
    result, ports = fan_out(t, target, count)
    validate(result)
    assert len(set(ports)) == count
    assert all(result.ports[p].role is PortRole.GLUE for p in ports)
    assert len(result.factors) == len(t.factors) + count
    assert len(result.variables) == len(t.variables) + 2 * count
    assert result.constraints == t.constraints
    assert snapshot(t) == before


def test_observe_and_fanout_rollback():
    from cffg.core import FanOut, ObserveInterior, fan_out, observe_interior
    from cffg.core.edits import check_edits

    t, target, _ = _product_latent()
    for action in [FanOut(target, n) for n in (0, -1, 1.5, True)] + [
        FanOut("absent", 2),
        ObserveInterior("absent"),
    ]:
        before = snapshot(t)
        assert check_edits(t, [action])
        assert snapshot(t) == before
    for operation in (FanOut(target, 3), ObserveInterior(target)):
        result = apply_edits(t, [operation])
        before = snapshot(result)
        assert check_edits(result, [operation])
        assert snapshot(result) == before
    observed, port = observe_interior(t, target)
    assert check_edits(observed, [ObserveInterior(observed.ports[port].variable)])
    # A second-level branch on the interior hub keeps the DATA port and its domain.
    skeleton, branches = fan_out(observed, target, 2)
    assert len(branches) == 2 and skeleton.ports[port].role is PortRole.DATA


def _manual_tap(original, target, role):
    """A hand-written one-leaf equality tap, independent of edit names and topology."""
    import copy
    from dataclasses import replace

    model = copy.deepcopy(original)
    for name in ("other", "tap"):
        model.variables[name] = replace(model.variables[target], name=name)
    second = model.factors["second"]
    model.factors[second.name] = replace(second, edges={"out": "other"})
    model.factors["join"] = FactorSpec("join", "equality", {"a": "tap", "b": target, "c": "other"})
    model.ports["sensor"] = PortSpec("sensor", "tap", role)
    return model


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_discrete_observe_matches_product_and_unbound_model(dtype):
    import jax
    import numpy as np
    from cffg.compiler import compile_template
    from cffg.core import observe_interior
    from cffg.engine import Engine

    t, target, arrays = _product_latent()
    transformed, port = observe_interior(t, target)
    with jax.enable_x64(dtype == "float64"):
        arrays = {k: v.astype(dtype) for k, v in arrays.items()}
        tol = 64 * np.finfo(dtype).eps
        product = arrays["p0"] * arrays["p1"]
        manual = _manual_tap(t, target, PortRole.DATA)
        for template, data_port in ((t, None), (manual, "sensor"), (transformed, port)):
            engine = Engine(compile_template(template, arrays=arrays))
            state = engine.run(engine.init_state(), 2, tol=tol).require_converged()
            np.testing.assert_allclose(engine.marginals(state)[target], product / product.sum(), atol=tol)
            np.testing.assert_allclose(engine.bfe(state), -np.log(product.sum()), atol=tol)
            if data_port is not None:
                observed = engine.bind_data(state, {data_port: 1})
                clamped = engine.run(observed, 2, tol=tol).require_converged()
                np.testing.assert_array_equal(engine.marginals(clamped)[target], [0, 1, 0])
                np.testing.assert_allclose(engine.bfe(clamped), -np.log(product[1]), atol=tol)
                with pytest.raises(ValueError):
                    engine.bind_data(state, {data_port: 3})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fanout_vacuous_branches_preserve_gaussian_product(dtype):
    import jax
    import numpy as np
    from cffg.compiler import compile_template
    from cffg.core import fan_out
    from cffg.engine import Engine

    original, target, arrays = _product_latent(dim=2)
    branched, ports = fan_out(original, target, 3)
    with jax.enable_x64(dtype == "float64"):
        arrays = {k: v.astype(dtype) for k, v in arrays.items()}
        tol = 64 * np.finfo(dtype).eps if dtype == "float32" else 1e-10
        energies = []
        manual = _manual_tap(original, target, PortRole.GLUE)
        for template, names in (
            (original, [target]),
            (branched, [target, *ports]),
            (manual, [target, "tap"]),
        ):
            engine = Engine(compile_template(template, arrays=arrays))
            state = engine.run(engine.init_state(), 32, tol=tol).require_converged()
            beliefs = engine.gaussian_marginals(state)
            for name in names:
                mean, cov = beliefs[name]
                np.testing.assert_allclose(mean, [1 / 3, 4 / 3], atol=tol, rtol=0)
                np.testing.assert_allclose(cov, (2 / 3) * np.eye(2), atol=tol, rtol=0)
            energies.append(float(engine.bfe(state)))
        np.testing.assert_allclose(energies, energies[0], atol=tol, rtol=0)


def test_64_branch_cardinality_32_flooding():
    import jax
    import numpy as np
    from cffg.compiler import compile_template
    from cffg.core import fan_out
    from cffg.engine import Engine

    t, target, arrays = _product_latent(k=32)
    t, ports = fan_out(t, target, 64)
    with jax.enable_x64(False):
        arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
        tol = 64 * np.finfo(np.float32).eps
        eng = Engine(compile_template(t, arrays=arrays), schedule="flood")
        state = eng.run(eng.init_state(), 64, tol=1e-10).require_converged()
        expected = arrays["p0"] * arrays["p1"]
        expected /= expected.sum()
        marginals = eng.marginals(state)
        for name in [target, *ports]:
            np.testing.assert_allclose(marginals[name], expected, atol=tol, rtol=0)
        assert np.isfinite(eng.bfe(state))


@pytest.mark.parametrize("operation", ["fan_out", "observe_interior"])
@pytest.mark.parametrize("clash", ["factor", "variable", "port"])
def test_branch_collisions_preserve_input(operation, clash):
    from cffg.core import FanOut, ObserveInterior, validate
    from cffg.core.edits import check_edits

    t, target, _ = _product_latent()
    action = FanOut(target, 2) if operation == "fan_out" else ObserveInterior(target)
    prefix = f"{target}/_fanout" if operation == "fan_out" else f"{target}/_observe"
    port = action.ports[0] if isinstance(action, FanOut) else action.port
    var_name = f"{prefix}/peer" if clash == "variable" else "outside"
    factor_name = f"{prefix}/eq0" if clash == "factor" else "outside_prior"
    port_name = port if clash == "port" else "outside_port"
    t.variables[var_name] = VariableSpec(var_name, 3)
    t.factors[factor_name] = FactorSpec(
        factor_name, "categorical", {"out": var_name}, param_blobs={"statistic": "p0"}
    )
    t.ports[port_name] = PortSpec(port_name, var_name, PortRole.GLUE)
    validate(t)
    before = snapshot(t)
    problems = check_edits(t, [action])
    assert len(problems) == 1
    assert operation in problems[0] and target in problems[0]
    assert {"factor": factor_name, "variable": var_name, "port": port_name}[clash] in problems[0]
    assert snapshot(t) == before


def test_branch_rejects_parameter_and_unsupported_observation_domain():
    from cffg.core import BlobRef, FanOut, ObserveInterior, Template, validate
    from cffg.core.edits import check_edits
    from test_dirichlet_param_edges import _make_template

    parameterized = _make_template("categorical", (3,))
    validate(parameterized)
    for action in (FanOut("theta", 1), ObserveInterior("theta")):
        (problem,) = check_edits(parameterized, [action])
        assert "parameter" in problem and "theta" in problem
    ef = Template(
        "ef",
        variables={"s": VariableSpec("s", ef="gamma")},
        blobs={key: BlobRef(key, key + ".npy") for key in ("shape", "rate")},
        factors={
            name: FactorSpec(name, "gamma", {"out": "s"}, param_blobs={"shape": "shape", "rate": "rate"})
            for name in ("left", "right")
        },
    )
    validate(ef)
    (problem,) = check_edits(ef, [ObserveInterior("s")])
    assert "observe_interior('s')" in problem and "unsupported DATA domain" in problem
