"""Predictive information about an unobserved port on a discrete tree."""

import jax
import numpy as np
import pytest

from cffg import Engine, RunStatus, compile_template
from cffg.compiled import NEG
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec, fan_out
from cffg.runtime_types import _replace


def information(joint):
    joint = np.asarray(joint, dtype=np.float64)
    joint = joint / joint.sum()
    independent = joint.sum(axis=0)[None, :] * joint.sum(axis=1)[:, None]
    supported = joint > 0
    log_ratio = np.log(joint[supported]) - np.log(independent[supported])
    return np.sum(joint[supported] * log_ratio)


def assert_information(actual, joint, dtype):
    assert np.isfinite(actual)
    expected = information(joint)
    tolerance = 128 * np.finfo(dtype).eps * joint.size
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
    return expected


def prior_model(prior):
    model = Template("predictive information")
    model.variables["query"] = VariableSpec("query", len(prior))
    model.factors["prior"] = FactorSpec(
        "prior", "categorical", {"out": "query"}, param_blobs={"statistic": "prior"},
    )
    return model, {"prior": np.asarray(prior)}


def add_channel(model, arrays, name, parent, conditional, *, port_role=None):
    model.variables[name] = VariableSpec(name, conditional.shape[0])
    model.factors[f"channel_{name}"] = FactorSpec(
        f"channel_{name}", "matrix_cat", {"out": name, "in": parent},
        param_blobs={"matrix": name},
    )
    arrays[name] = conditional
    if port_role is not None:
        model.ports[name] = PortSpec(name, name, port_role)


def make_engine(model, arrays, dtype, schedule="auto"):
    model.blobs = {name: BlobRef(name, name + ".npy") for name in arrays}
    arrays = {name: array.astype(dtype) for name, array in arrays.items()}
    return Engine(compile_template(model, arrays=arrays), schedule=schedule)


def solve(engine, state):
    return engine.run(state, 80, tol=1e-10).require_converged()


@pytest.mark.parametrize("x64", [False, True])
def test_binary_tree_matches_enumeration_after_observe_and_clear(x64):
    dtype = np.float64 if x64 else np.float32
    prior = np.array([.64, .36])
    motion = np.array([[.82, .25], [.18, .75]])
    sensor = np.array([[.87, .20], [.13, .80]])
    context = np.array([[.91, .41], [.09, .59]])
    with jax.enable_x64(x64):
        model, arrays = prior_model(prior)
        add_channel(model, arrays, "separator", "query", motion)
        model.blobs = {name: BlobRef(name, name + ".npy") for name in arrays}
        model.ports["attach"] = PortSpec("attach", "separator", PortRole.GLUE)
        model, (branch,) = fan_out(model, "separator", 1)
        add_channel(model, arrays, "sensor", model.ports.pop("attach").variable, sensor,
                    port_role=PortRole.DATA)
        add_channel(model, arrays, "context", model.ports.pop(branch).variable, context,
                    port_role=PortRole.DATA)

        for schedule in ("auto", "flood"):
            engine = make_engine(model, arrays, dtype, schedule=schedule)
            state = engine.init_state()
            for observed in (1, 0, None):
                state = engine.bind_data(state, {"context": observed, "sensor": None})
                state = solve(engine, state)
                before = [np.asarray(x).copy() for x in jax.tree.leaves(state)]
                weight = np.ones(2) if observed is None else context[observed]
                joint = np.einsum("q,mq,ym,m->qy", prior, motion, sensor, weight)
                returned = engine.eig(state, "query", ["sensor"])
                assert list(returned) == ["sensor"]
                assert_information(returned["sensor"], joint, dtype)
                assert engine.eig(state, "query", ["sensor"]) == returned
                for previous, current in zip(before, jax.tree.leaves(state), strict=True):
                    np.testing.assert_array_equal(previous, current)


def naive_bayes(prior, candidates, calibration):
    model, arrays = prior_model(prior)
    model.blobs["prior"] = BlobRef("prior", "prior.npy")
    model.ports["spare"] = PortSpec("spare", "query", PortRole.GLUE)
    model, branches = fan_out(model, "query", len(candidates) + 1)
    for branch, (name, channel) in zip(
        branches, [*candidates.items(), ("calibration", calibration)], strict=True,
    ):
        source = model.ports.pop(branch).variable
        add_channel(model, arrays, name, source, channel, port_role=PortRole.DATA)
    return model, arrays


@pytest.mark.parametrize("categories, ports", [(2, 3), (64, 64)])
def test_naive_bayes_matches_independent_posterior_and_ranking(categories, ports):
    dtype = np.float32
    prior = np.arange(1, categories + 1, dtype=float)
    if categories > 2:
        prior[::11] = 0.
    prior /= prior.sum()
    calibration_signal = np.linspace(.04, .91, categories)
    calibration = np.array([calibration_signal, 1. - calibration_signal])
    base = np.full((categories, categories), 1. / categories)
    candidates = {}
    for i, strength in enumerate(np.linspace(.07, .97, ports)):
        channel = np.roll(np.eye(categories), i, axis=0)
        candidates[f"sensor{i:02}"] = strength * channel + (1. - strength) * base
    order = list(candidates)[::2] + list(candidates)[1::2][::-1]
    with jax.enable_x64(False):
        model, arrays = naive_bayes(prior, candidates, calibration)
        engine = make_engine(model, arrays, dtype)
        state = solve(engine, engine.bind_data(engine.init_state(), {"calibration": 0}))
        actual = engine.eig(state, "query", order)
        posterior = prior * calibration[0]
        posterior /= posterior.sum()
        expected = {
            port: assert_information(actual[port], posterior[:, None] * candidates[port].T, dtype)
            for port in order
        }
        assert list(actual) == order
        assert sorted(actual, key=actual.get) == sorted(expected, key=expected.get)


def test_vacuous_missing_and_literal_zero_keep_their_probability_semantics():
    prior = np.array([.30, 0., .70])
    channels = {
        "perfect": np.eye(3),
        "irrelevant": np.array([[.14, .14, .14], [.86, .86, .86]]),
        "sparse": np.array([[1., .50, .17], [0., .50, .83]]),
    }
    calibration = np.full((2, 3), .5)
    with jax.enable_x64(False):
        model, arrays = naive_bayes(prior, channels, calibration)
        engine = make_engine(model, arrays, np.float32)
        state = solve(engine, engine.init_state())
        actual = engine.eig(state, "query", list(channels))
        for name, channel in channels.items():
            assert_information(actual[name], prior[:, None] * channel.T, np.float32)
        entropy = -(prior[[0, 2]] * np.log(prior[[0, 2]])).sum()
        np.testing.assert_allclose(actual["perfect"], entropy, atol=8 * np.finfo(np.float32).eps, rtol=0)
        np.testing.assert_allclose(actual["irrelevant"], 0., atol=8 * np.finfo(np.float32).eps, rtol=0)
        assert engine.marginals(state, ["query"])["query"][1] == 0.

        # A uniform prior and vacuous incoming messages already represent
        # the correct upstream cavity for a direct channel.
        model, arrays = prior_model([.5, .5])
        channel = np.array([[1., .22], [0., .78]])
        add_channel(model, arrays, "sensor", "query", channel, port_role=PortRole.DATA)
        engine = make_engine(model, arrays, np.float32)
        actual = engine.eig(engine.init_state(), "query", ["sensor"])
        assert_information(actual["sensor"], .5 * channel.T, np.float32)


def rejection_model():
    prior = np.array([.4, .6])
    model, arrays = naive_bayes(prior, {"sensor": np.eye(2)}, np.eye(2))
    model.variables["disjoint"] = VariableSpec("disjoint", 2)
    model.factors["disjoint_prior"] = FactorSpec(
        "disjoint_prior", "categorical", {"out": "disjoint"}, param_blobs={"statistic": "prior"},
    )
    model.ports["disjoint"] = PortSpec("disjoint", "disjoint", PortRole.DATA)
    return make_engine(model, arrays, np.float64)


def test_unknown_observed_glue_same_edge_and_disconnected_requests():
    engine = rejection_model()
    state = solve(engine, engine.init_state())
    with pytest.raises(KeyError, match="unknown_query"):
        engine.eig(state, "unknown_query", ["sensor"])
    with pytest.raises(KeyError, match="unknown_port"):
        engine.eig(state, "query", ["unknown_port"])
    with pytest.raises(ValueError, match="spare"):
        engine.eig(state, "query", ["spare"])
    with pytest.raises(ValueError, match="sensor"):
        engine.eig(state, "sensor", ["sensor"])
    with pytest.raises(ValueError, match=r"(?=.*query)(?=.*disjoint)"):
        engine.eig(state, "query", ["disjoint"])
    observed = solve(engine, engine.bind_data(state, {"sensor": 0}))
    with pytest.raises(ValueError, match="sensor"):
        engine.eig(observed, "query", ["sensor"])
    cleared = solve(engine, engine.bind_data(observed, {"sensor": None}))
    np.testing.assert_allclose(
        engine.eig(cleared, "query", ["sensor"])["sensor"],
        engine.eig(state, "query", ["sensor"])["sensor"], atol=128 * np.finfo(np.float64).eps,
    )


@pytest.mark.parametrize("replacement", [np.nan, NEG], ids=["nonfinite", "all-log-zero"])
def test_improper_incoming_path_message_is_rejected(replacement):
    engine = rejection_model()
    state = solve(engine, engine.init_state())
    edge = engine.arena.edges["query"]
    packed = engine.arena.message_indices((edge.slot0, edge.slot1), edge.card)
    invalid = _replace(state, messages=state.messages.at[packed].set(replacement))
    with pytest.raises(ValueError, match=r"query|sensor"):
        engine.eig(invalid, "query", ["sensor"])


def test_contradictory_literal_observations_leave_no_pair_support():
    channels = {"sensor": np.eye(2), "other": np.eye(2)}
    model, arrays = naive_bayes([.4, .6], channels, np.eye(2))
    engine = make_engine(model, arrays, np.float64)
    bounded = engine.bind_data(engine.init_state(), {"other": 1, "calibration": 0})
    result = engine.run(bounded, 80, tol=1e-10)
    assert result.status is RunStatus.INFEASIBLE
    with pytest.raises(ValueError, match=r"(?=.*query)(?=.*sensor)"):
        engine.eig(result.state, "query", ["sensor"])


@pytest.mark.parametrize("x64", [False, True])
def test_singleton_candidate_binding_survives_message_updates(x64):
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        model, arrays = prior_model([.4, .6])
        add_channel(model, arrays, "singleton", "query", np.ones((1, 2)), port_role=PortRole.DATA)
        engine = make_engine(model, arrays, dtype)
        state = engine.init_state()
        addresses = engine.arena.message_indices(engine.arena.data_slots["singleton"], 1)
        assert np.signbit(np.asarray(state.messages)[addresses]).item()
        for value in (None, 0, None):
            state = engine.bind_data(state, {"singleton": value})
            for candidate in (state, engine.step(state, damp=.25), solve(engine, state)):
                # The only numeric value is zero. Its sign distinguishes an
                # observed sole category from a missing categorical message.
                assert np.signbit(np.asarray(candidate.messages)[addresses]).item() is (value is None)
                if value is None:
                    information_nats = engine.eig(candidate, "query", ["singleton"])["singleton"]
                    assert information_nats == pytest.approx(0., abs=128 * np.finfo(dtype).eps)
                else:
                    with pytest.raises(ValueError, match="singleton"):
                        engine.eig(candidate, "query", ["singleton"])


def test_damped_five_way_port_distinguishes_missing_evidence_from_a_clamp():
    with jax.enable_x64(False):
        prior = np.array([.42, .58])
        channel = np.array([[.55, .05], [.20, .10], [.15, .20], [.05, .25], [.05, .40]])
        model, arrays = prior_model(prior)
        add_channel(model, arrays, "sensor", "query", channel, port_role=PortRole.DATA)
        engine = make_engine(model, arrays, np.float32, schedule="flood")
        state = engine.init_state()
        for value in (None, 0, None):
            state = engine.bind_data(state, {"sensor": value})
            state = engine.run(state, 80, tol=1e-10, damp=.1).require_converged()
            if value is None:
                actual = engine.eig(state, "query", ["sensor"])["sensor"]
                assert_information(actual, prior[:, None] * channel.T, np.float32)
            else:
                with pytest.raises(ValueError, match="sensor"):
                    engine.eig(state, "query", ["sensor"])
