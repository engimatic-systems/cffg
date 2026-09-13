"""Persistence identity for structured recurrence posterior statistics."""

import dataclasses
import json

import numpy as np
import pytest

from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core.serialize import structure_hash, to_json
from cffg.engine import Engine, RunStatus
from cffg.learn import Learnables
from cffg.learning_schema import PERSISTENT_MANIFEST_KEY
from cffg.learning_checkpoint import (
    CHECKPOINT_MANIFEST_KEY,
    encode_manifest,
    RECURRENCE_CHECKPOINT_VERSION,
    RECURRENCE_PRIOR_IDENTITY,
    RECURRENCE_STATISTIC_IDENTITY,
)
from cffg.models.rslds import RSLDSLearning, rslds, vague_rslds_learning
from cffg.models.rslds_coordinate import legacy_snapshot
from cffg.models.rslds_fit import vague_rslds_fit_priors


def _model(dtype=np.float64):
    arrays = {
        "dynamics_matrix": np.asarray([[[0.7]], [[-0.35]]], dtype=dtype),
        "dynamics_offset": np.asarray([[0.1], [-0.15]], dtype=dtype),
        "dynamics_covariance": np.asarray([[[0.4]], [[0.6]]], dtype=dtype),
        "recurrence_weights": np.asarray([[[0.8]], [[-0.55]]], dtype=dtype),
        "recurrence_offsets": np.asarray([[0.2], [-0.3]], dtype=dtype),
        "emission_matrix": np.asarray([[1.0]], dtype=dtype),
        "emission_offset": np.asarray([0.0], dtype=dtype),
        "emission_covariance": np.asarray([[0.5]], dtype=dtype),
        "initial_state_mean": np.asarray([0.0], dtype=dtype),
        "initial_state_covariance": np.asarray([[1.0]], dtype=dtype),
        "initial_regime_probability": np.asarray([0.6, 0.4], dtype=dtype),
    }
    return rslds(3, **arrays)


def _recurrence_prior(dtype=np.float64):
    eta = np.arange(4, dtype=dtype).reshape(2, 1, 2) / np.asarray(
        10.0, dtype=dtype
    )
    precision = np.broadcast_to(
        np.eye(2, dtype=dtype), (2, 1, 2, 2)
    ).copy()
    return {"eta": eta, "precision": precision}


def _runtime(dtype=np.float64):
    template, arrays = _model(dtype)
    engine = Engine(compile_template(template, arrays=arrays))
    learning = Learnables(
        engine,
        priors={"recurrence": _recurrence_prior(dtype)},
        only=("recurrence",),
    )
    return template, arrays, engine, learning


def _legacy_checkpoint(learning):
    """The identified D9 format emitted before generic learning descriptors."""
    return {
        **learning.hyperparameter_snapshot(),
        CHECKPOINT_MANIFEST_KEY: encode_manifest(learning.plans, learning.params),
    }


def _copy(checkpoint):
    return {name: np.asarray(value).copy() for name, value in checkpoint.items()}


def _manifest(checkpoint):
    return json.loads(str(np.asarray(checkpoint[CHECKPOINT_MANIFEST_KEY]).item()))


def _with_manifest(checkpoint, document):
    changed = _copy(checkpoint)
    changed[CHECKPOINT_MANIFEST_KEY] = np.asarray(
        json.dumps(document, sort_keys=True, separators=(",", ":"))
    )
    return changed


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_legacy_d9_checkpoint_round_trips_identity_ownership_dtype_and_values(dtype):
    _template, _arrays, engine, source = _runtime(dtype)
    checkpoint = _legacy_checkpoint(source)
    document = _manifest(checkpoint)

    assert document["checkpoint_version"] == RECURRENCE_CHECKPOINT_VERSION
    assert document["statistic_identity"] == RECURRENCE_STATISTIC_IDENTITY
    assert len(document["learnables"]) == 1
    descriptor = document["learnables"][0]
    assert descriptor["name"] == "recurrence"
    assert descriptor["family"] == "pg_weights"
    assert descriptor["factor_type"] == "recurrence_link"
    assert descriptor["sources"] == ["switch1", "switch2"]
    assert {row["name"] for row in descriptor["hyperparameters"]} == {
        "eta",
        "precision",
    }
    assert {row["dtype"] for row in descriptor["hyperparameters"]} == {
        np.dtype(dtype).str
    }

    prior_dtype = np.float64 if dtype is np.float32 else np.float32
    restored_prior = _recurrence_prior(prior_dtype)
    restored = Learnables(
        engine,
        priors={
            "recurrence": {
                key: np.zeros_like(value)
                if key == "eta"
                else np.broadcast_to(
                    np.eye(value.shape[-1], dtype=prior_dtype), value.shape
                ).copy()
                for key, value in restored_prior.items()
            }
        },
        only=("recurrence",),
    )
    restored.load_state_dict(checkpoint)
    for key, value in source.params["recurrence"].items():
        np.testing.assert_array_equal(restored.params["recurrence"][key], value)
        assert restored.params["recurrence"][key].dtype == np.dtype(dtype)
    assert str(_legacy_checkpoint(restored)[CHECKPOINT_MANIFEST_KEY]) == str(
        checkpoint[CHECKPOINT_MANIFEST_KEY]
    )


def test_unidentified_v2_and_slash_recurrence_state_are_rejected():
    _template, _arrays, engine, source = _runtime()
    versioned_without_identity = source.hyperparameter_snapshot()
    legacy_slash = {
        "recurrence/eta": source.params["recurrence"]["eta"],
        "recurrence/precision": source.params["recurrence"]["precision"],
    }
    for checkpoint in (versioned_without_identity, legacy_slash):
        restored = Learnables(
            engine,
            priors={"recurrence": _recurrence_prior()},
            only=("recurrence",),
        )
        before = _copy(restored.hyperparameter_snapshot())
        with pytest.raises(ValueError, match="no statistic identity"):
            restored.load_state_dict(checkpoint)
        for key, value in before.items():
            np.testing.assert_array_equal(
                restored.hyperparameter_snapshot()[key], value
            )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda value: {**value, "checkpoint_version": "99.0.0"},
            "unknown recurrence checkpoint version",
        ),
        (
            lambda value: {
                **value,
                "statistic_identity": (
                    "cffg.recurrence.d8-singleton-product/1.0.0"
                ),
            },
            "statistic identity",
        ),
        (
            lambda value: {
                **value,
                "learnables": [
                    {**value["learnables"][0], "factor_type": "matrix_cat"}
                ],
            },
            "factor/source ownership",
        ),
        (
            lambda value: {
                **value,
                "learnables": [
                    {**value["learnables"][0], "sources": ["switch1"]}
                ],
            },
            "factor/source ownership",
        ),
    ),
)
def test_manifest_version_statistic_factor_and_source_tampering_fail(
    mutation, match
):
    _template, _arrays, engine, source = _runtime()
    checkpoint = _legacy_checkpoint(source)
    broken = _with_manifest(checkpoint, mutation(_manifest(checkpoint)))
    restored = Learnables(
        engine,
        priors={"recurrence": _recurrence_prior()},
        only=("recurrence",),
    )
    with pytest.raises(ValueError, match=match):
        restored.load_state_dict(broken)


def test_incomplete_unknown_shape_dtype_nonfinite_and_manifest_encoding_fail():
    _template, _arrays, engine, source = _runtime(np.float32)
    checkpoint = _legacy_checkpoint(source)
    eta_key = "v2:recurrence:eta"
    cases = []

    incomplete = _copy(checkpoint)
    incomplete.pop(eta_key)
    cases.append((incomplete, KeyError, "incomplete"))

    unknown = _copy(checkpoint)
    unknown["v2:recurrence:nope"] = np.zeros(1)
    cases.append((unknown, KeyError, "no hyperparameter"))

    wrong_shape = _copy(checkpoint)
    wrong_shape[eta_key] = wrong_shape[eta_key][..., :1]
    cases.append((wrong_shape, ValueError, "checkpoint shape"))

    wrong_dtype = _copy(checkpoint)
    wrong_dtype[eta_key] = wrong_dtype[eta_key].astype(np.float64)
    cases.append((wrong_dtype, ValueError, "checkpoint dtype"))

    tampered = _copy(checkpoint)
    tampered[eta_key][0, 0, 0] += np.float32(0.25)
    cases.append((tampered, ValueError, "value digest"))

    nonfinite = _copy(checkpoint)
    nonfinite[eta_key][0, 0, 0] = np.nan
    cases.append((nonfinite, ValueError, "finite"))

    malformed = _copy(checkpoint)
    malformed[CHECKPOINT_MANIFEST_KEY] = np.asarray("{")
    cases.append((malformed, ValueError, "valid JSON"))

    for broken, error, match in cases:
        restored = Learnables(
            engine,
            priors={"recurrence": _recurrence_prior(np.float32)},
            only=("recurrence",),
        )
        before = _copy(restored.hyperparameter_snapshot())
        with pytest.raises(error, match=match):
            restored.load_state_dict(broken)
        for key, value in before.items():
            np.testing.assert_array_equal(
                restored.hyperparameter_snapshot()[key], value
            )


def test_model_identity_does_not_authorize_a_d8_checkpoint():
    template, _arrays, engine, source = _runtime()
    template_identity = structure_hash(template)
    checkpoint = _legacy_checkpoint(source)
    legacy = _manifest(checkpoint)
    legacy["statistic_identity"] = (
        "cffg.recurrence.d8-singleton-product/1.0.0"
    )

    assert structure_hash(template) == template_identity
    assert RECURRENCE_STATISTIC_IDENTITY in to_json(template)
    with pytest.raises(ValueError, match="statistic identity"):
        Learnables(
            engine,
            priors={"recurrence": _recurrence_prior()},
            only=("recurrence",),
        ).load_state_dict(_with_manifest(checkpoint, legacy))


def test_model_only_bundle_stays_loadable_and_runs_current_pair_semantics(tmp_path):
    template, arrays = _model()
    path = tmp_path / "rslds.cffgz"
    save_bundle(path, template, arrays)
    restored, restored_arrays, manifest = load_bundle(path)

    assert structure_hash(restored) == structure_hash(template)
    assert CHECKPOINT_MANIFEST_KEY not in manifest
    engine = Engine(compile_template(restored, arrays=restored_arrays))
    result = engine.run(engine.init_state(), 200, tol=1e-8)
    assert result.status is RunStatus.CONVERGED
    pairs = engine.recurrence_pair_beliefs(result.state)
    assert set(pairs) == {"switch1", "switch2"}
    assert all(bool(pair["proper"]) for pair in pairs.values())


def test_rslds_facade_marks_priors_and_restores_only_structured_state():
    _template, _arrays, engine, _source = _runtime()
    prior = vague_rslds_learning(2, 1)
    assert prior.recurrence_statistic_identity == RECURRENCE_PRIOR_IDENTITY
    checkpoint = prior.state_dict(engine)
    restored = vague_rslds_learning(2, 1).load_state_dict(engine, checkpoint)
    assert (
        restored.recurrence_statistic_identity
        == RECURRENCE_STATISTIC_IDENTITY
    )
    for name, value in prior.parameter_arrays().items():
        np.testing.assert_array_equal(restored.parameter_arrays()[name], value)

    with pytest.raises(ValueError, match="unknown or legacy statistic"):
        dataclasses.replace(
            prior,
            recurrence_statistic_identity=(
                "cffg.recurrence.d8-singleton-product/1.0.0"
            ),
        )


def test_fit_legacy_numeric_snapshot_excludes_checkpoint_manifest():
    _template, _arrays, engine, _source = _runtime()
    priors = vague_rslds_fit_priors(2, 1, 1)
    learning = Learnables(engine, priors=priors.learning_priors())
    checkpoint = learning.state_dict()
    snapshot = legacy_snapshot(learning, {"probe": np.asarray([1.0])})

    assert PERSISTENT_MANIFEST_KEY in checkpoint
    assert all(np.issubdtype(value.dtype, np.number) for value in snapshot.values())
    assert not any(CHECKPOINT_MANIFEST_KEY in key for key in snapshot)
