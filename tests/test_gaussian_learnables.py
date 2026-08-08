"""Declared NIW/MNIW learning, cadence, pooling, and persistence."""

import dataclasses

import jax
import numpy as np
import pytest

from cffg import conj
from cffg.compiler import CompileError, compile_template
from cffg.core import (
    BlobRef,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.engine import Engine, InferenceError, RunStatus, _replace
from cffg.learn import Learnables
from cffg.core.serialize import from_json, structure_hash, to_dict, to_json


def _blob(name: str) -> BlobRef:
    return BlobRef(name, f"blobs/{name}.npy")


def _learning_template(dimension: int = 2):
    template = Template(name="gaussian_learning")
    arrays = {
        "m0": np.zeros(dimension),
        "P0": np.eye(dimension) * 1.5,
        "A": np.eye(dimension) * 0.7,
        "b": np.zeros(dimension),
        "Q": np.eye(dimension) * 0.4,
        "H": np.eye(dimension),
        "h": np.zeros(dimension),
        "R": np.eye(dimension) * 0.2,
    }
    template.blobs = {name: _blob(name) for name in arrays}
    template.variables = {
        "x0": VariableSpec("x0", dim=dimension),
        "x1": VariableSpec("x1", dim=dimension),
        "y": VariableSpec("y", dim=dimension),
    }
    template.factors = {
        "prior": FactorSpec(
            "prior",
            "gaussian",
            edges={"out": "x0"},
            param_blobs={"mean": "m0", "cov": "P0"},
        ),
        "transition": FactorSpec(
            "transition",
            "gaussian_linear",
            edges={"out": "x1", "in": "x0"},
            param_blobs={"matrix": "A", "offset": "b", "cov": "Q"},
        ),
        "emission": FactorSpec(
            "emission",
            "gaussian_linear",
            edges={"out": "y", "in": "x1"},
            param_blobs={"matrix": "H", "offset": "h", "cov": "R"},
        ),
    }
    template.ports = {"observation": PortSpec("observation", "y", PortRole.DATA)}
    template.learnables = {
        "a_state": LearnableSpec("a_state", "niw", ("m0", "P0")),
        "z_dynamics": LearnableSpec("z_dynamics", "mniw", ("A", "b", "Q")),
    }
    return template, arrays


def _converged(dimension: int = 2):
    template, arrays = _learning_template(dimension)
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(
        engine.init_state(),
        {"observation": np.linspace(-0.5, 1.0, dimension)},
    )
    result = engine.run(state, 30, tol=1e-12)
    result.require_converged()
    return engine, result


def _assert_params(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-10)


def _niw_params(posterior: conj.NIW):
    return {
        "mu": posterior.mu,
        "kappa": np.asarray(posterior.kappa),
        "psi": posterior.psi,
        "nu": np.asarray(posterior.nu),
    }


def _mniw_params(posterior: conj.MNIW):
    return {
        "m": posterior.m,
        "lam": posterior.lam,
        "psi": posterior.psi,
        "nu": np.asarray(posterior.nu),
    }


def test_niw_and_mniw_match_direct_updates_and_rollover_by_role():
    engine, result = _converged()
    niw_prior = conj.vague_niw(2)
    mniw_prior = conj.vague_mniw(2, 2)
    learnables = Learnables(
        engine,
        priors={"a_state": niw_prior, "z_dynamics": mniw_prior},
    )
    stats = engine.gaussian_node_stats(result.state)
    direct_niw = niw_prior.update(
        stats["prior"]["mean"][None], stats["prior"]["second"][None]
    )
    direct_mniw = mniw_prior.update(
        stats["transition"]["Syy"],
        stats["transition"]["Syx"],
        stats["transition"]["Sxx"],
        1.0,
    )

    learnables.replace(engine, result)

    _assert_params(learnables.params["a_state"], _niw_params(direct_niw))
    _assert_params(learnables.params["z_dynamics"], _mniw_params(direct_mniw))
    np.testing.assert_allclose(
        learnables.expected("a_state", "cov"), direct_niw.mean_cov()
    )
    np.testing.assert_allclose(
        learnables.expected("z_dynamics", "matrix"), direct_mniw.m[:, :-1]
    )

    written = learnables.write(engine, engine.init_state().params)
    state_address = engine.arena.member_of["prior"]
    transition_address = engine.arena.member_of["transition"]
    niw_mean, niw_cov = direct_niw.rollover()
    matrix, offset, cov = direct_mniw.rollover()
    np.testing.assert_allclose(
        written[state_address.parameter("mean").key][state_address.parameter("mean").index],
        niw_mean,
    )
    np.testing.assert_allclose(
        written[state_address.parameter("cov").key][state_address.parameter("cov").index],
        np.linalg.inv(niw_cov),
    )
    np.testing.assert_allclose(
        written[transition_address.parameter("matrix").key][transition_address.parameter("matrix").index],
        matrix,
    )
    np.testing.assert_allclose(
        written[transition_address.parameter("offset").key][transition_address.parameter("offset").index],
        offset,
    )
    np.testing.assert_allclose(
        written[transition_address.parameter("cov").key][transition_address.parameter("cov").index],
        np.linalg.inv(cov),
    )


def _tied_emissions(count: int = 4, dimension: int = 2):
    template = Template(name="tied_emissions")
    arrays = {
        "input_mean": np.zeros(dimension),
        "input_cov": np.eye(dimension),
        "H": np.eye(dimension),
        "h": np.zeros(dimension),
        "R": np.eye(dimension),
    }
    template.blobs = {name: _blob(name) for name in arrays}
    for time in range(count):
        x, y = f"x{time}", f"y{time}"
        template.variables[x] = VariableSpec(x, dim=dimension)
        template.variables[y] = VariableSpec(y, dim=dimension)
        template.factors[f"input{time}"] = FactorSpec(
            f"input{time}",
            "gaussian",
            edges={"out": x},
            param_blobs={"mean": "input_mean", "cov": "input_cov"},
        )
        template.factors[f"emission{time}"] = FactorSpec(
            f"emission{time}",
            "gaussian_linear",
            edges={"out": y, "in": x},
            param_blobs={"matrix": "H", "offset": "h", "cov": "R"},
        )
        template.ports[f"observation{time}"] = PortSpec(
            f"observation{time}", y, PortRole.DATA
        )
    template.learnables["emission"] = LearnableSpec(
        "emission", "mniw", ("H", "h", "R")
    )
    return template, arrays


def test_tied_conditioned_sources_pool_once_and_replace_is_idempotent():
    template, arrays = _tied_emissions()
    engine = Engine(compile_template(template, arrays=arrays))
    data = {
        f"observation{time}": np.array([time - 1.5, 0.25 * time])
        for time in range(4)
    }
    result = engine.run(engine.bind_data(engine.init_state(), data), 30, tol=1e-12)
    result.require_converged()
    prior = conj.vague_mniw(2, 2)
    learnables = Learnables(engine, priors={"emission": prior})
    rows = engine.gaussian_node_stats(result.state)
    direct = prior.update(
        sum((rows[f"emission{i}"]["Syy"] for i in range(4)), np.zeros((2, 2))),
        sum((rows[f"emission{i}"]["Syx"] for i in range(4)), np.zeros((2, 3))),
        sum((rows[f"emission{i}"]["Sxx"] for i in range(4)), np.zeros((3, 3))),
        4.0,
    )

    learnables.replace(engine, result)
    first = {key: value.copy() for key, value in learnables.params["emission"].items()}
    _assert_params(first, _mniw_params(direct))
    learnables.replace(engine, result)
    _assert_params(learnables.params["emission"], first)

    accumulated = Learnables(engine, priors={"emission": prior})
    accumulated.accumulate(engine, result)
    accumulated.accumulate(engine, result)
    twice = direct.update(
        sum((rows[f"emission{i}"]["Syy"] for i in range(4)), np.zeros((2, 2))),
        sum((rows[f"emission{i}"]["Syx"] for i in range(4)), np.zeros((2, 3))),
        sum((rows[f"emission{i}"]["Sxx"] for i in range(4)), np.zeros((3, 3))),
        4.0,
    )
    _assert_params(accumulated.params["emission"], _mniw_params(twice))


def test_masked_conditioned_source_adds_no_evidence():
    template, arrays = _tied_emissions(count=2, dimension=1)
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), {"observation0": np.array([2.0])})
    result = engine.run(state, 30, tol=1e-12)
    result.require_converged()
    prior = conj.vague_mniw(1, 1)
    learnables = Learnables(engine, priors={"emission": prior})
    stats = engine.gaussian_node_stats(result.state)
    assert stats["emission1"]["mask"] == 0.0

    learnables.replace(engine, result)

    direct = prior.update(
        stats["emission0"]["Syy"],
        stats["emission0"]["Syx"],
        stats["emission0"]["Sxx"],
        1.0,
    )
    _assert_params(learnables.params["emission"], _mniw_params(direct))


def test_empty_sources_roll_back_to_immutable_initial_prior():
    template, arrays = _learning_template(1)
    template.learnables["z_dynamics"] = dataclasses.replace(
        template.learnables["z_dynamics"], sources=()
    )
    engine = Engine(compile_template(template, arrays=arrays))
    result = engine.run(
        engine.bind_data(engine.init_state(), {"observation": np.array([1.0])}),
        30,
        tol=1e-12,
    )
    result.require_converged()
    prior = conj.vague_mniw(1, 1)
    learnables = Learnables(
        engine,
        priors={"a_state": conj.vague_niw(1), "z_dynamics": prior},
    )
    checkpoint = learnables.state_dict()
    checkpoint["v2:z_dynamics:nu"] = np.asarray(prior.nu + 10.0)
    learnables.load_state_dict(checkpoint)
    assert learnables.params["z_dynamics"]["nu"] == prior.nu + 10.0

    learnables.replace(engine, result, only=("z_dynamics",))

    _assert_params(learnables.params["z_dynamics"], _mniw_params(prior))
    with pytest.raises(TypeError):
        learnables.initial_params["z_dynamics"] = {}
    with pytest.raises(ValueError):
        learnables.initial_params["z_dynamics"]["m"][0, 0] = 3.0


def test_updates_require_converged_status_and_validate_only_names():
    engine, result = _converged(1)
    learnables = Learnables(
        engine,
        priors={"a_state": conj.vague_niw(1), "z_dynamics": conj.vague_mniw(1, 1)},
    )
    before = learnables.state_dict()
    exhausted = dataclasses.replace(result, status=RunStatus.BUDGET_EXHAUSTED)
    for method in (learnables.accumulate, learnables.replace):
        with pytest.raises(InferenceError):
            method(engine, exhausted)
        with pytest.raises(TypeError, match="status-bearing RunResult"):
            method(engine, result.state)
        with pytest.raises(KeyError, match="does not declare requested"):
            method(engine, result, only=("missing",))
    for key, value in before.items():
        np.testing.assert_array_equal(learnables.state_dict()[key], value)


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (lambda row: row.update(proper=np.asarray(False)), "improper"),
        (lambda row: row.update(Syy=np.full_like(row["Syy"], np.nan)), "finite"),
        (lambda row: row.update(Syx=np.zeros((1, 1))), "statistic shapes"),
        (lambda row: row.update(Syy=np.eye(2) * -100.0), "positive definite"),
    ],
)
def test_selected_updates_are_atomic_on_hostile_mniw_statistics(
    monkeypatch, corrupt, match
):
    engine, result = _converged(2)
    learnables = Learnables(
        engine,
        priors={"a_state": conj.vague_niw(2), "z_dynamics": conj.vague_mniw(2, 2)},
    )
    before = learnables.state_dict()
    readout = engine.gaussian_node_stats(result.state)
    corrupt(readout["transition"])
    monkeypatch.setattr(engine, "gaussian_node_stats", lambda state: readout)

    with pytest.raises((ValueError, KeyError), match=match):
        learnables.replace(engine, result)

    for key, value in before.items():
        np.testing.assert_array_equal(learnables.state_dict()[key], value)


def test_checkpoint_round_trips_keyed_gaussian_hyperparameters_atomically():
    engine, result = _converged(2)
    priors = {
        "a_state": conj.vague_niw(2),
        "z_dynamics": conj.vague_mniw(2, 2),
    }
    source = Learnables(engine, priors=priors)
    source.replace(engine, result)
    checkpoint = source.state_dict()
    assert len(checkpoint) == 8
    restored = Learnables(engine, priors=priors)
    restored.load_state_dict(checkpoint)
    for name in source.params:
        _assert_params(restored.params[name], source.params[name])

    before = restored.state_dict()
    broken = dict(checkpoint)
    broken["v2:z_dynamics:psi"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="positive definite"):
        restored.load_state_dict(broken)
    for key, value in before.items():
        np.testing.assert_array_equal(restored.state_dict()[key], value)


def test_gaussian_family_declarations_serialize_and_addresses_stay_derived():
    template, _arrays = _learning_template(2)
    serialized = to_dict(template)
    assert serialized["learnables"]["a_state"]["family"] == "niw"
    assert serialized["learnables"]["z_dynamics"]["blobs"] == ["A", "b", "Q"]
    assert "member_of" not in serialized
    restored = from_json(to_json(template))
    assert restored.learnables == template.learnables
    assert structure_hash(restored) == structure_hash(template)
    changed = dataclasses.replace(
        template,
        learnables={
            **template.learnables,
            "z_dynamics": dataclasses.replace(
                template.learnables["z_dynamics"], sources=()
            ),
        },
    )
    assert structure_hash(changed) != structure_hash(template)


def test_float32_learning_and_write_stay_finite_and_preserve_destinations():
    with jax.enable_x64(False):
        template, arrays = _learning_template(2)
        arrays = {key: value.astype(np.float32) for key, value in arrays.items()}
        engine = Engine(compile_template(template, arrays=arrays))
        result = engine.run(
            engine.bind_data(
                engine.init_state(),
                {"observation": np.array([0.0, 1.0], dtype=np.float32)},
            ),
            30,
            tol=1e-6,
        )
        result.require_converged()
        learnables = Learnables(
            engine,
            priors={
                "a_state": conj.vague_niw(2),
                "z_dynamics": conj.vague_mniw(2, 2),
            },
        )
        learnables.replace(engine, result)
        written = learnables.write(engine, engine.init_state().params)
    for values in learnables.params.values():
        assert all(np.all(np.isfinite(value)) for value in values.values())
    assert all(value.dtype == np.float32 for value in written.values())
    assert all(np.all(np.isfinite(value)) for value in written.values())


def test_uncomfortable_tied_niw_scale_pools_one_vectorized_update():
    dimension, count = 16, 64
    template = Template(name="large_tied_niw")
    arrays = {"mean": np.zeros(dimension), "cov": np.eye(dimension)}
    template.blobs = {name: _blob(name) for name in arrays}
    for index in range(count):
        variable, factor = f"x{index}", f"prior{index}"
        template.variables[variable] = VariableSpec(variable, dim=dimension)
        template.factors[factor] = FactorSpec(
            factor,
            "gaussian",
            edges={"out": variable},
            param_blobs={"mean": "mean", "cov": "cov"},
        )
        template.ports[variable] = PortSpec(variable, variable, PortRole.GLUE)
    template.learnables["state"] = LearnableSpec(
        "state", "niw", ("mean", "cov")
    )
    engine = Engine(compile_template(template, arrays=arrays))
    result = engine.run(engine.init_state(), 5, tol=1e-6)
    result.require_converged()
    prior = conj.vague_niw(dimension)
    learnables = Learnables(engine, priors={"state": prior})

    learnables.replace(engine, result)
    written = learnables.write(engine, engine.init_state().params)

    assert learnables.params["state"]["kappa"] == prior.kappa + count
    assert learnables.params["state"]["nu"] == prior.nu + count
    address = engine.arena.member_of["prior0"].parameter("mean")
    assert written[address.key].shape == (count, dimension)
    assert np.all(np.isfinite(written[address.key]))


def test_unsupported_gaussian_family_pairs_fail_at_compile_time():
    template, arrays = _learning_template(1)
    template.learnables["bad"] = LearnableSpec("bad", "niw", ("A", "b", "Q"))
    del template.learnables["z_dynamics"]
    with pytest.raises(CompileError, match="learnable 'bad'.*niw/gaussian"):
        compile_template(template, arrays=arrays)

    template, arrays = _learning_template(1)
    template.learnables["bad"] = LearnableSpec("bad", "mniw", ("m0", "P0"))
    del template.learnables["a_state"]
    with pytest.raises(CompileError, match="learnable 'bad'.*mniw/gaussian_linear"):
        compile_template(template, arrays=arrays)

    mixture = Template(name="unsupported_mixture_learning")
    mixture.variables = {
        "out": VariableSpec("out", dim=1),
        "in": VariableSpec("in", dim=1),
        "selector": VariableSpec("selector", 2),
    }
    mixture.factors["switching"] = FactorSpec(
        "switching",
        "gaussian_mixture_linear",
        edges={"out": "out", "in": "in", "selector": "selector"},
        param_blobs={"matrix": "A", "offset": "b", "cov": "Q"},
    )
    mixture.ports = {
        name: PortSpec(name, name, PortRole.GLUE) for name in mixture.variables
    }
    mixture_arrays = {
        "A": np.ones((2, 1, 1)),
        "b": np.zeros((2, 1)),
        "Q": np.ones((2, 1, 1)),
    }
    mixture.blobs = {name: _blob(name) for name in mixture_arrays}
    mixture.learnables["bad"] = LearnableSpec(
        "bad", "mniw", ("A", "b", "Q")
    )
    with pytest.raises(CompileError, match="learnable 'bad'.*mniw/gaussian_linear"):
        compile_template(mixture, arrays=mixture_arrays)
