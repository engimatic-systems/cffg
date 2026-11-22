"""Compiled incidence and complete-model coverage of the D13.6 recognizer."""

from dataclasses import replace

import numpy as np
import pytest

from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core import (
    BlobRef, FactorSpec, FunctionSpec, LearnableSpec, MomentProjection,
    PortRole, PortSpec, Template, VariableSpec, moment_matching,
)
from cffg.directed_chain_plan import recognize_directed_chain
from nonlinear_fixture import adaptive_pair


def small_chain(method="linearization"):
    arrays = dict(mean=np.zeros(1), cov=np.eye(1), theta=np.array([.1]),
                  A=np.ones((1, 1)), b=np.zeros(1), R=np.eye(1))
    template = Template("scalar", blobs={k: BlobRef(k, k + ".npy") for k in arrays})

    def variable(name):
        template.variables[name] = VariableSpec(name, dim=1)
        return name

    state = variable("start")
    template.factors["source"] = FactorSpec(
        "source", "gaussian", {"out": state}, param_blobs={"mean": "mean", "cov": "cov"})
    for i in range(3):
        emission_state = state
        if i < 2:
            emission_state, next_input = variable(f"measurement{i}"), variable(f"input{i}")
            roles = ("b", "a", "c") if i == 0 else ("c", "b", "a")
            equality = f"branch{i}"
            template.factors[equality] = FactorSpec(
                equality, "equality", dict(zip(roles, (state, emission_state, next_input))))
            state = variable(f"state{i+1}")
            transition = ("zeta", "alpha")[i]
            template.factors[transition] = FactorSpec(
                transition, "gaussian_nonlinear", {"in": next_input, "out": state},
                param_blobs={"parameters": "theta", "cov": "cov"},
                function=FunctionSpec("cffg.mean.cubic", "1.0.0", method=method))
            template.constraints.append(moment_matching(transition, "in", MomentProjection.GAUSSIAN))
        name = f"sensor{i}"
        datum = variable(f"value{i}")
        template.factors[name] = FactorSpec(
            name, "gaussian_linear", {"in": emission_state, "out": datum},
            param_blobs={"matrix": "A", "offset": "b", "cov": "R"})
        port = f"port{i}"
        template.ports[port] = PortSpec(port, datum, PortRole.DATA)
    return template, arrays


def assert_member(ref, name):
    assert ref.group.members[ref.row] == name
    assert ref.gather == tuple(ref.group.gather[ref.row])
    assert ref.scatter == tuple(ref.group.scatter[ref.row])


@pytest.mark.parametrize("roundtrip", [False, True])
def test_order_and_slot_maps_survive_opaque_renaming_and_bundle(roundtrip, tmp_path):
    template, arrays = small_chain()
    names = {k: k for k in template.factors}
    ports = list(template.ports)
    if roundtrip:
        # Reverse lexical and insertion order independently of the graph roles.
        names = {k: f"node{100-i}" for i, k in enumerate(sorted(template.factors))}
        variables = {k: f"edge{100-i}" for i, k in enumerate(sorted(template.variables))}
        template.variables = {v: VariableSpec(v, dim=1) for v in reversed(variables.values())}
        template.factors = {
            names[k]: replace(f, name=names[k], edges={r: variables[v] for r, v in f.edges.items()})
            for k, f in reversed(template.factors.items())}
        template.constraints = [replace(c, target=names[c.target]) for c in reversed(template.constraints)]
        ports = ["delta", "beta", "aardvark"]
        template.ports = {k: PortSpec(k, variables[f"value{i}"], PortRole.DATA)
                          for i, k in reversed(list(enumerate(ports)))}
        path = tmp_path / "renamed.cffgz"
        save_bundle(path, template, arrays)
        loaded = load_bundle(path)
        template, arrays = loaded.template, loaded.arrays

    arena = compile_template(template, arrays=arrays)
    if roundtrip:
        arena = replace(arena, gauss_groups=tuple(reversed(arena.gauss_groups)))
    plan = recognize_directed_chain(arena)
    assert plan.dim == 1
    assert_member(plan.prior, names["source"])
    assert len(plan.sites) == 3
    pred = plan.prior
    for i, site in enumerate(plan.sites):
        assert_member(site.emission, names[f"sensor{i}"])
        assert (site.obs_row, 1) == arena.gauss_data[ports[i]]
        if i == 2:
            assert site.equality is site.equality_scatter is site.transition is None
            assert site.emission.gather[1] == pred.scatter[0]
            break
        assert_member(site.transition, names[("zeta", "alpha")[i]])
        assert_member(site.equality, names[f"branch{i}"])
        assert site.equality_scatter == (pred.gather[0], site.emission.gather[1],
                                         site.transition.gather[1])
        assert set(site.equality_scatter) == set(site.equality.scatter)
        pred = site.transition


def test_retained_reactor_with_rectangular_goals():
    from test_cstr_runtime import fixture as reactor

    template, arrays = reactor.model()
    arena = compile_template(template, arrays=arrays)
    plan = recognize_directed_chain(arena)
    assert plan.dim == 4
    assert len(plan.sites) == 20
    assert [site.emission.group.observed_dim for site in plan.sites] == [3] * 4 + [2] * 16
    for i, site in enumerate(plan.sites[:-1]):
        assert_member(site.transition, f"tr{i+1}")
        assert site.obs_row == arena.gauss_data[f"e{i}"][0]
    assert np.linalg.matrix_rank(arrays["G_goal"]) < plan.dim


def test_interacting_nonlinear_sensor_cavities_require_the_coupled_solver():
    template, arrays = adaptive_pair()
    assert recognize_directed_chain(compile_template(template, arrays=arrays)) is None


@pytest.mark.parametrize("change", [
    "sigma_point", "uncertain_emissions", "unconditioned_emission",
    "reversed_transition", "disconnected_emission", "mixed_family",
])
def test_unsupported_model_or_dependency_returns_no_plan(change):
    template, arrays = small_chain("sigma_point" if change == "sigma_point" else "linearization")
    if change == "uncertain_emissions":
        template.learnables["shared"] = LearnableSpec("shared", "mniw", ("A", "b", "R"))
    elif change == "unconditioned_emission":
        template.ports["port1"] = replace(template.ports["port1"], role=PortRole.GLUE)
    elif change == "reversed_transition":
        f = template.factors["alpha"]
        template.factors[f.name] = replace(f, edges={"in": f.edges["out"], "out": f.edges["in"]})
    elif change == "disconnected_emission":
        for k in ("x", "y"):
            template.variables[k] = VariableSpec(k, dim=1)
        template.ports["extra"] = PortSpec("extra", "y", PortRole.DATA)
        template.ports["open"] = PortSpec("open", "x", PortRole.GLUE)
        template.factors["island"] = replace(template.factors["sensor0"], name="island",
                                             edges={"in": "x", "out": "y"})
    elif change == "mixed_family":
        template.variables["coin"] = VariableSpec("coin", cardinality=2)
        template.ports["coin"] = PortSpec("coin", "coin", PortRole.DATA)
        template.blobs["p"] = BlobRef("p", "p.npy")
        arrays["p"] = np.array([.4, .6])
        template.factors["toss"] = FactorSpec("toss", "categorical", {"out": "coin"},
                                              param_blobs={"statistic": "p"})
    arena = compile_template(template, arrays=arrays)
    assert recognize_directed_chain(arena) is None
