"""Declared learning and persistence for a tied Gaussian-mixture component bank."""

from dataclasses import replace
import json

import jax
import numpy as np
import pytest

from cffg import conj, load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core import BlobRef, FactorSpec, LearnableSpec, PlateSpec, Template
from cffg.core.plates import expand_plates
from cffg.core.serialize import from_json, to_json
from cffg.engine import Engine
from cffg.learn import Learnables
from test_hybrid_readouts import _prior_mixture_template


OWNER = "objects/type"
SOURCES = ("bank/0/mixture", "bank/1/mixture")


def component_model(prior_dtype=np.float64):
    fragment, arrays = _prior_mixture_template(regimes=3, dimension=1)
    fragment.ports.clear()
    fragment.factors["sensor"] = FactorSpec(
        "sensor", "gaussian", {"out": "y"},
        param_blobs={"mean": "reading", "cov": "noise"},
    )
    fragment.blobs.update(reading=BlobRef("reading", "reading-{index}.npy"),
                          noise=BlobRef("noise", "noise.npy"))
    fragment.blobs["probability"] = replace(fragment.blobs["probability"], path="p-{index}.npy")
    plate = PlateSpec(fragment, (3,), per_copy_blobs=("reading", "probability"))
    template = expand_plates(Template("component-learning", plates={"bank": plate}))
    template.learnables[OWNER] = LearnableSpec(OWNER, "component_niw", ("mean", "covariance"), SOURCES)
    del arrays["probability"]
    arrays.update(mean=np.full((3, 1), 10000.), covariance=np.array([.25, .75, .125])[:, None, None],
                  noise=np.array([[.25]]))
    for i, p in enumerate(([.875, .125, 0.], [.25, .75, 0.], [0., 0., 1.])):
        arrays[f"bank/{i}/reading"] = np.array([10000.5 - i])
        arrays[f"bank/{i}/probability"] = np.array(p)
    prior = conj.NIW(np.array([9999.], dtype=prior_dtype), prior_dtype(2.),
                     np.array([[1.]], dtype=prior_dtype), prior_dtype(4.))
    priors = (prior,) * 3
    return template, arrays, {OWNER: priors}


@pytest.mark.parametrize("x64", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_component_niw_centered_tied_update_and_scoped_writes(x64):
    template, arrays, priors = component_model()
    with jax.enable_x64(x64):
        engine = Engine(compile_template(template, arrays=arrays))
        learner = Learnables(engine, priors)
        assert learner.plans[OWNER].sources == SOURCES
        result = engine.run(engine.init_state(), 50, tol=1e-6)
        learner.accumulate(engine, result)
        written = learner.write(engine, result.state.params)

    variance = arrays["covariance"].ravel()
    conditional_variance = variance * .25 / (variance + .25)
    weights, offsets = [], []
    for i in range(2):
        delta = arrays[f"bank/{i}/reading"][0] - 10000.
        likelihood = np.exp(-.5 * delta**2 / (variance + .25)) / np.sqrt(variance + .25)
        evidence = arrays[f"bank/{i}/probability"] * likelihood
        weights.append(evidence / evidence.sum())
        offsets.append(1 + delta * variance / (variance + .25))
    weights, offsets = np.stack(weights), np.stack(offsets)
    mass = weights.sum(axis=0)
    kappa, nu = 2 + mass, 4 + mass
    mean_shift = (weights * offsets).sum(axis=0) / kappa
    psi = 1 + (weights * (conditional_variance + (offsets - mean_shift)**2)).sum(axis=0) + 2 * mean_shift**2
    expected = {"mu": (9999. + mean_shift)[:, None], "kappa": kappa, "nu": nu,
                "psi": psi[:, None, None]}
    atol = 2e-3 if not x64 else 1e-10
    for key, value in expected.items():
        np.testing.assert_allclose(learner.params[OWNER][key], value, rtol=0, atol=atol)
        np.testing.assert_array_equal(learner.params[OWNER][key][-1], learner.initial_params[OWNER][key][-1])
    for i in range(3):
        member = engine.arena.member_of[f"bank/{i}/mixture"]
        for role, value in (("mean", expected["mu"]), ("cov", (nu / psi)[:, None, None])):
            target = member.parameter(role)
            np.testing.assert_allclose(written[target.key][target.index], value, rtol=0, atol=atol)
    np.testing.assert_allclose(learner.expected(OWNER, "cov")[:, 0, 0], psi / (nu - 2), rtol=0, atol=atol)


def test_component_niw_declaration_and_checkpoint_roundtrip(tmp_path):
    template, arrays, priors = component_model(prior_dtype=np.float32)
    serialized = to_json(template)
    legacy = json.loads(serialized)
    del legacy["learning_schema_version"], legacy["learnables"][OWNER]["semantic_identity"]
    assert to_json(from_json(json.dumps(legacy))) == serialized
    engine = Engine(compile_template(from_json(serialized), arrays=arrays))
    learner = Learnables(engine, priors)
    result = engine.run(engine.init_state(), 50, tol=1e-6)
    for updated in (False, True):
        if updated:
            learner.accumulate(engine, result)
        for key, arr in learner.params[OWNER].items():
            prior = learner.initial_params[OWNER][key]
            assert prior.dtype == np.float32
            assert arr.dtype == (np.float64 if updated else np.float32)
            np.testing.assert_array_equal(arr[-1], prior[-1])
        path = tmp_path / "learning.cffgz"
        save_bundle(path, template, arrays, learnables=learner)
        restored = load_bundle(path).learnables
        assert restored.persistent_descriptors() == learner.persistent_descriptors()
        resumed = Learnables(engine, priors)
        slash_keys = {f"{OWNER}/{key}": arr for key, arr in learner.params[OWNER].items()}
        resumed.load_state_dict(slash_keys, exact=True)
        for key, arr in learner.params[OWNER].items():
            np.testing.assert_array_equal(resumed.params[OWNER][key], arr)
        resumed.load_state_dict(learner.state_dict(), exact=True)
        assert resumed.persistent_descriptors() == learner.persistent_descriptors()
        restored.accumulate(engine, result)
        resumed.accumulate(engine, result)
        assert restored.persistent_descriptors() == resumed.persistent_descriptors()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_component_niw_rejects_unresolved_or_improper_priors(dtype):
    template, arrays = _prior_mixture_template(regimes=2, dimension=2)
    template.learnables[OWNER] = LearnableSpec(OWNER, "component_niw", ("mean", "covariance"))
    engine = Engine(compile_template(template, arrays=arrays))
    psi = np.broadcast_to([[1e-16, .4], [.4, 1e16]], (2, 2, 2)).astype(dtype)
    params = dict(mu=arrays["mean"].astype(dtype), kappa=np.ones(2, dtype=dtype),
                  psi=psi, nu=np.full(2, 4, dtype=dtype))
    learner = Learnables(engine, {OWNER: params})
    assert learner.state_dict()
    invalid = {
        "psi": np.array([psi[0], [[2, 2], [2, 2]]], dtype=dtype),
        "kappa": np.array([0., 1.]), "nu": np.array([1., 4.]),
        "mu": np.full((2, 2), np.nan),
    }
    for key, value in invalid.items():
        with pytest.raises(ValueError, match="component NIW"):
            Learnables(engine, {OWNER: {**params, key: value}})


def test_component_niw_failed_fold_is_atomic(monkeypatch):
    template, arrays, priors = component_model()
    engine = Engine(compile_template(template, arrays=arrays))
    learner = Learnables(engine, priors)
    result = engine.run(engine.init_state(), 50, tol=1e-6)
    rows = engine.gaussian_node_stats(result.state, centered_mixtures=True)
    rows[SOURCES[-1]]["component_covariance"] = np.array([[[1.]], [[-1.]], [[0.]]])
    monkeypatch.setattr(engine, "gaussian_node_stats", lambda state, *, centered_mixtures: rows)
    with pytest.raises(ValueError, match="positive semidefinite"):
        learner.accumulate(engine, result)
    for key, prior in learner.initial_params[OWNER].items():
        np.testing.assert_array_equal(learner.params[OWNER][key], prior)
