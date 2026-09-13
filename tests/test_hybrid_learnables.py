"""Declared learning for hybrid Gaussian-mixture and recurrence factors."""

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
    PlateSpec,
    Template,
    VariableSpec,
)
from cffg.engine import Engine
from cffg.experiments.recurrence_smoothing import recurrence_statistics
from cffg.learn import Learnables
from cffg.core.serialize import from_json, structure_hash, to_json


def _blob(name: str) -> BlobRef:
    return BlobRef(name, f"blobs/{name}.npy")


def _mixture_template(
    regimes: int = 3,
    dimension: int = 2,
    *,
    dtype=np.float64,
):
    template = Template(name="neutral_component_learning")
    selector_probability = np.arange(regimes, 0, -1, dtype=dtype)
    selector_probability[-1] = 0.0
    selector_probability /= selector_probability.sum()
    arrays = {
        "matrix": np.broadcast_to(
            np.eye(dimension, dtype=dtype) * 0.7,
            (regimes, dimension, dimension),
        ).copy(),
        "offset": np.linspace(-0.2, 0.2, regimes, dtype=dtype)[:, None]
        * np.ones((1, dimension), dtype=dtype),
        "cov": np.broadcast_to(
            np.eye(dimension, dtype=dtype) * 0.5,
            (regimes, dimension, dimension),
        ).copy(),
        "input_mean": np.zeros(dimension, dtype=dtype),
        "input_cov": np.eye(dimension, dtype=dtype),
        "output_mean": np.ones(dimension, dtype=dtype) * 0.25,
        "output_cov": np.eye(dimension, dtype=dtype) * 1.2,
        "selector_probability": selector_probability,
    }
    template.blobs = {name: _blob(name) for name in arrays}
    sources = ("outer/source_a", "outer/source_b")
    for source in sources:
        input_name = f"{source}/input"
        output_name = f"{source}/output"
        selector_name = f"{source}/selector"
        template.variables[input_name] = VariableSpec(input_name, dim=dimension)
        template.variables[output_name] = VariableSpec(output_name, dim=dimension)
        template.variables[selector_name] = VariableSpec(
            selector_name, cardinality=regimes
        )
        template.factors[f"{source}/input_prior"] = FactorSpec(
            f"{source}/input_prior",
            "gaussian",
            edges={"out": input_name},
            param_blobs={"mean": "input_mean", "cov": "input_cov"},
        )
        template.factors[f"{source}/output_prior"] = FactorSpec(
            f"{source}/output_prior",
            "gaussian",
            edges={"out": output_name},
            param_blobs={"mean": "output_mean", "cov": "output_cov"},
        )
        template.factors[f"{source}/selector_prior"] = FactorSpec(
            f"{source}/selector_prior",
            "categorical",
            edges={"out": selector_name},
            param_blobs={"statistic": "selector_probability"},
        )
        template.factors[source] = FactorSpec(
            source,
            "gaussian_mixture_linear",
            edges={"out": output_name, "in": input_name, "selector": selector_name},
            param_blobs={"matrix": "matrix", "offset": "offset", "cov": "cov"},
        )
    template.learnables["neutral_dynamics"] = LearnableSpec(
        "neutral_dynamics",
        "component_mniw",
        ("matrix", "offset", "cov"),
        sources,
    )
    return template, arrays, sources


def _run_mixture(regimes=3, dimension=2, dtype=np.float64):
    template, arrays, sources = _mixture_template(regimes, dimension, dtype=dtype)
    engine = Engine(compile_template(template, arrays=arrays))
    result = engine.run(engine.init_state(), 100, tol=1e-7)
    result.require_converged()
    return template, arrays, sources, engine, result


def _stacked_params(posteriors):
    return {
        "m": np.stack([posterior.m for posterior in posteriors]),
        "lam": np.stack([posterior.lam for posterior in posteriors]),
        "psi": np.stack([posterior.psi for posterior in posteriors]),
        "nu": np.asarray([posterior.nu for posterior in posteriors]),
    }


def test_component_mniw_pools_tied_sources_and_matches_direct_oracle():
    _template, _arrays, sources, engine, result = _run_mixture()
    priors = tuple(conj.vague_mniw(2, 2) for _ in range(3))
    learnables = Learnables(engine, priors={"neutral_dynamics": priors})
    rows = engine.gaussian_node_stats(result.state)
    direct = tuple(
        prior.update(
            sum((rows[source]["Syy"][component] for source in sources), np.zeros((2, 2))),
            sum((rows[source]["Syx"][component] for source in sources), np.zeros((2, 3))),
            sum((rows[source]["Sxx"][component] for source in sources), np.zeros((3, 3))),
            sum(float(rows[source]["responsibility"][component]) for source in sources),
        )
        if any(rows[source]["responsibility"][component] > 0.0 for source in sources)
        else prior
        for component, prior in enumerate(priors)
    )

    learnables.replace(engine, result)

    for key, expected in _stacked_params(direct).items():
        np.testing.assert_allclose(learnables.params["neutral_dynamics"][key], expected)
    for role, index in (("matrix", 0), ("offset", 1), ("cov", 2)):
        expected = np.stack([posterior.rollover()[index] for posterior in direct])
        if role == "cov":
            expected = np.linalg.inv(expected)
        written = learnables.write(engine, dict(engine.init_state().params))
        for source in sources:
            address = engine.arena.member_of[source].parameter(role)
            np.testing.assert_allclose(
                written[address.key][address.index], expected
            )


def test_zero_responsibility_retains_component_prior_exactly():
    _template, _arrays, _sources, engine, result = _run_mixture()
    priors = tuple(conj.vague_mniw(2, 2) for _ in range(3))
    learnables = Learnables(engine, priors={"neutral_dynamics": priors})
    rows = engine.gaussian_node_stats(result.state)
    for row in rows.values():
        if "responsibility" not in row:
            continue
        assert row["responsibility"][-1] == 0.0
        np.testing.assert_array_equal(row["Syy"][-1], 0.0)
        np.testing.assert_array_equal(row["Syx"][-1], 0.0)
        np.testing.assert_array_equal(row["Sxx"][-1], 0.0)

    learnables.replace(engine, result)

    prior = _stacked_params(priors)
    for key in prior:
        np.testing.assert_array_equal(
            learnables.params["neutral_dynamics"][key][-1], prior[key][-1]
        )


def test_component_mniw_float32_k32_d16_updates_and_writes():
    with jax.enable_x64(False):
        _template, _arrays, _sources, engine, result = _run_mixture(
            regimes=32, dimension=16, dtype=np.float32
        )
        priors = tuple(conj.vague_mniw(16, 16) for _ in range(32))
        learnables = Learnables(engine, priors={"neutral_dynamics": priors})
        learnables.replace(engine, result)
        written = learnables.write(engine, dict(engine.init_state().params))
        for value in learnables.params["neutral_dynamics"].values():
            assert np.all(np.isfinite(value))
        for source in ("outer/source_a", "outer/source_b"):
            address = engine.arena.member_of[source]
            for role in ("matrix", "offset", "cov"):
                target = address.parameter(role)
                assert written[target.key].dtype == np.float32
                assert np.all(np.isfinite(written[target.key]))


def _recurrence_template(
    regimes: int = 3,
    dimension: int = 2,
    *,
    dtype=np.float64,
):
    template = Template(name="neutral_recurrence_learning")
    arrays = {
        "weights": (
            np.arange(regimes * (regimes - 1) * dimension, dtype=dtype)
            .reshape(regimes, regimes - 1, dimension)
            - 0.5 * (regimes * (regimes - 1) * dimension - 1)
        )
        * np.asarray(0.11, dtype=dtype),
        "offsets": np.broadcast_to(
            -np.log(np.arange(regimes - 1, 0, -1, dtype=dtype)),
            (regimes, regimes - 1),
        ).copy(),
        "in_probability": np.linspace(1.0, 2.0, regimes, dtype=dtype),
        "out_probability": np.linspace(2.0, 1.0, regimes, dtype=dtype),
        "drive_mean": np.zeros(dimension, dtype=dtype),
        "drive_cov": np.eye(dimension, dtype=dtype),
    }
    arrays["in_probability"] /= arrays["in_probability"].sum()
    arrays["out_probability"] /= arrays["out_probability"].sum()
    arrays["in_probability"][-1] = 0.0
    arrays["in_probability"] /= arrays["in_probability"].sum()
    template.blobs = {name: _blob(name) for name in arrays}
    sources = ("branch/alpha", "branch/beta")
    for source in sources:
        out_name = f"{source}/out"
        in_name = f"{source}/in"
        drive_name = f"{source}/drive"
        template.variables[out_name] = VariableSpec(out_name, cardinality=regimes)
        template.variables[in_name] = VariableSpec(in_name, cardinality=regimes)
        template.variables[drive_name] = VariableSpec(drive_name, dim=dimension)
        template.factors[f"{source}/out_prior"] = FactorSpec(
            f"{source}/out_prior",
            "categorical",
            edges={"out": out_name},
            param_blobs={"statistic": "out_probability"},
        )
        template.factors[f"{source}/in_prior"] = FactorSpec(
            f"{source}/in_prior",
            "categorical",
            edges={"out": in_name},
            param_blobs={"statistic": "in_probability"},
        )
        template.factors[f"{source}/drive_prior"] = FactorSpec(
            f"{source}/drive_prior",
            "gaussian",
            edges={"out": drive_name},
            param_blobs={"mean": "drive_mean", "cov": "drive_cov"},
        )
        template.factors[source] = FactorSpec(
            source,
            "recurrence_link",
            edges={"out": out_name, "in": in_name, "drive": drive_name},
            param_blobs={"weights": "weights", "offsets": "offsets"},
        )
    template.learnables["neutral_recurrence"] = LearnableSpec(
        "neutral_recurrence",
        "pg_weights",
        ("weights", "offsets"),
        sources,
    )
    return template, arrays, sources


def _run_recurrence(regimes=3, dimension=2, dtype=np.float64, schedule="auto"):
    template, arrays, sources = _recurrence_template(
        regimes, dimension, dtype=dtype
    )
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    result = engine.run(engine.init_state(), 100, tol=1e-7)
    result.require_converged()
    return template, arrays, sources, engine, result


def test_pg_weights_pool_tied_sources_and_match_direct_oracle():
    template, arrays, sources, engine, result = _run_recurrence()
    prior = conj.vague_pg_weights(3, 2)
    learnables = Learnables(engine, priors={"neutral_recurrence": prior})
    rows = engine.recurrence_weight_stats(result.state)
    beliefs = engine.recurrence_pair_beliefs(result.state)
    independent = []
    product_gap = []
    for source in sources:
        belief = beliefs[source]
        expected = recurrence_statistics(
            belief["pair_probability"],
            belief["drive_mean"],
            belief["drive_covariance"],
            arrays["weights"],
            arrays["offsets"],
        )
        product = recurrence_statistics(
            np.outer(
                belief["source_probability"],
                belief["destination_probability"],
            ),
            belief["drive_mean"],
            belief["drive_covariance"],
            arrays["weights"],
            arrays["offsets"],
        )
        for key in (
            "omega",
            "exposure",
            "signed_success",
            "linear",
            "quadratic",
        ):
            np.testing.assert_allclose(rows[source][key], expected[key])
        independent.append(expected)
        product_gap.append(
            np.max(np.abs(expected["linear"] - product["linear"]))
        )
        np.testing.assert_array_equal(
            belief["pair_probability"][-1], np.zeros(3)
        )
        np.testing.assert_array_equal(rows[source]["linear"][-1], 0.0)
        np.testing.assert_array_equal(rows[source]["quadratic"][-1], 0.0)
    assert max(product_gap) > 1e-4
    direct = prior.update(
        np.stack([row["linear"] for row in independent]),
        np.stack([row["quadratic"] for row in independent]),
    )

    learnables.replace(engine, result)

    np.testing.assert_allclose(learnables.params["neutral_recurrence"]["eta"], direct.eta)
    np.testing.assert_allclose(
        learnables.params["neutral_recurrence"]["precision"], direct.precision
    )
    np.testing.assert_array_equal(
        learnables.params["neutral_recurrence"]["eta"][-1], prior.eta[-1]
    )
    np.testing.assert_array_equal(
        learnables.params["neutral_recurrence"]["precision"][-1],
        prior.precision[-1],
    )
    np.testing.assert_allclose(
        learnables.expected("neutral_recurrence", "weights"), direct.rollover()[0]
    )
    written = learnables.write(engine, dict(engine.init_state().params))
    weights, offsets = direct.rollover()
    for source in sources:
        address = engine.arena.member_of[source]
        weight_target = address.parameter("weights")
        offset_target = address.parameter("offsets")
        np.testing.assert_allclose(
            written[weight_target.key][weight_target.index], weights
        )
        np.testing.assert_allclose(
            written[offset_target.key][offset_target.index], offsets
        )

    restricted = dataclasses.replace(
        template,
        learnables={
            "neutral_recurrence": dataclasses.replace(
                template.learnables["neutral_recurrence"],
                sources=(sources[1],),
            )
        },
    )
    restricted_engine = Engine(compile_template(restricted, arrays=arrays))
    restricted_result = restricted_engine.run(
        restricted_engine.init_state(), 100, tol=1e-7
    )
    restricted_result.require_converged()
    restricted_learning = Learnables(
        restricted_engine, priors={"neutral_recurrence": prior}
    )
    restricted_learning.replace(restricted_engine, restricted_result)
    restricted_belief = restricted_engine.recurrence_pair_beliefs(
        restricted_result.state
    )[sources[1]]
    restricted_stats = recurrence_statistics(
        restricted_belief["pair_probability"],
        restricted_belief["drive_mean"],
        restricted_belief["drive_covariance"],
        arrays["weights"],
        arrays["offsets"],
    )
    restricted_direct = prior.update(
        restricted_stats["linear"], restricted_stats["quadratic"]
    )
    np.testing.assert_allclose(
        restricted_learning.params["neutral_recurrence"]["eta"],
        restricted_direct.eta,
    )
    np.testing.assert_allclose(
        restricted_learning.params["neutral_recurrence"]["precision"],
        restricted_direct.precision,
    )


def test_pg_replace_is_idempotent_accumulate_adds_and_sources_are_explicit():
    template, arrays, _sources, engine, result = _run_recurrence()
    prior = conj.vague_pg_weights(3, 2)
    replacement = Learnables(engine, priors={"neutral_recurrence": prior})
    replacement.replace(engine, result, only=("neutral_recurrence",))
    first = replacement.state_dict()
    replacement.replace(engine, result)
    for key, value in first.items():
        np.testing.assert_array_equal(replacement.state_dict()[key], value)

    accumulated = Learnables(engine, priors={"neutral_recurrence": prior})
    accumulated.accumulate(engine, result)
    accumulated.accumulate(engine, result)
    np.testing.assert_allclose(
        accumulated.params["neutral_recurrence"]["eta"] - prior.eta,
        2.0 * (replacement.params["neutral_recurrence"]["eta"] - prior.eta),
    )
    unrestricted = dataclasses.replace(
        template,
        learnables={
            "neutral_recurrence": dataclasses.replace(
                template.learnables["neutral_recurrence"], sources=None
            )
        },
    )
    unrestricted_engine = Engine(compile_template(unrestricted, arrays=arrays))
    assert unrestricted_engine.arena.learn_plans[0].sources == (
        "branch/alpha",
        "branch/beta",
    )


def test_restricted_runtime_and_public_rollover_use_declared_contract():
    _template, _arrays, _sources, engine, result = _run_recurrence()
    prior = conj.vague_pg_weights(3, 2)
    learnables = Learnables(
        engine,
        priors={"neutral_recurrence": prior},
        only=("neutral_recurrence",),
    )
    learnables.replace(engine, result)
    weights, offsets = prior.update(
        np.stack(
            [
                row["linear"]
                for row in engine.recurrence_weight_stats(result.state).values()
            ]
        ),
        np.stack(
            [
                row["quadratic"]
                for row in engine.recurrence_weight_stats(result.state).values()
            ]
        ),
    ).rollover()
    np.testing.assert_allclose(
        learnables.rollover("neutral_recurrence", "weights"), weights
    )
    np.testing.assert_allclose(
        learnables.rollover("neutral_recurrence", "offsets"), offsets
    )


def test_pg_scheduled_and_flooded_statistics_produce_same_posterior():
    posteriors = []
    for schedule in ("auto", "flood"):
        _template, _arrays, _sources, engine, result = _run_recurrence(
            schedule=schedule
        )
        prior = conj.vague_pg_weights(3, 2)
        learnables = Learnables(engine, priors={"neutral_recurrence": prior})
        learnables.replace(engine, result)
        posteriors.append(learnables.hyperparameter_snapshot())
    for key, value in posteriors[0].items():
        np.testing.assert_allclose(value, posteriors[1][key], rtol=1e-7, atol=1e-7)


def test_pg_hostile_statistics_and_precision_fail_before_mutation(monkeypatch):
    _template, _arrays, _sources, engine, result = _run_recurrence()
    prior = conj.vague_pg_weights(3, 2)
    learnables = Learnables(engine, priors={"neutral_recurrence": prior})
    before = learnables.state_dict()
    rows = engine.recurrence_weight_stats(result.state)
    rows["branch/alpha"]["linear"] = np.full_like(
        rows["branch/alpha"]["linear"], np.nan
    )
    monkeypatch.setattr(engine, "recurrence_weight_stats", lambda state: rows)
    with pytest.raises(ValueError, match="finite"):
        learnables.replace(engine, result)
    for key, value in before.items():
        np.testing.assert_array_equal(learnables.state_dict()[key], value)

    broken = dict(before)
    broken["v2:neutral_recurrence:precision"] = -np.ones_like(prior.precision)
    with pytest.raises(ValueError, match="value digest"):
        learnables.load_state_dict(broken)
    for key, value in before.items():
        np.testing.assert_array_equal(learnables.state_dict()[key], value)


def test_pg_float32_k32_d16_updates_and_writes_without_dense_transition():
    with jax.enable_x64(False):
        _template, _arrays, sources, engine, result = _run_recurrence(
            regimes=32, dimension=16, dtype=np.float32
        )
        prior = conj.vague_pg_weights(32, 16)
        learnables = Learnables(engine, priors={"neutral_recurrence": prior})
        learnables.replace(engine, result)
        written = learnables.write(engine, dict(engine.init_state().params))
        for value in learnables.params["neutral_recurrence"].values():
            assert np.all(np.isfinite(value))
        for source in sources:
            address = engine.arena.member_of[source]
            for role in ("weights", "offsets"):
                target = address.parameter(role)
                assert written[target.key].dtype == np.float32
                assert np.all(np.isfinite(written[target.key]))
        group = engine.arena.recurrence_groups[0]
        assert group.d_gather_both.size + group.drive_gather_both.size < 2 * 32**2


@pytest.mark.parametrize(
    ("builder", "priors"),
    [
        (
            _mixture_template,
            lambda: {
                "neutral_dynamics": tuple(conj.vague_mniw(2, 2) for _ in range(3))
            },
        ),
        (
            _recurrence_template,
            lambda: {"neutral_recurrence": conj.vague_pg_weights(3, 2)},
        ),
    ],
)
def test_hybrid_declarations_serialize_and_checkpoints_round_trip(builder, priors):
    template, arrays, _sources = builder()
    restored_template = from_json(to_json(template))
    assert restored_template.learnables == template.learnables
    assert structure_hash(restored_template) == structure_hash(template)
    engine = Engine(compile_template(restored_template, arrays=arrays))
    result = engine.run(engine.init_state(), 100, tol=1e-7)
    result.require_converged()
    source = Learnables(engine, priors=priors())
    source.replace(engine, result)
    restored = Learnables(engine, priors=priors())
    restored.load_state_dict(source.state_dict())
    for name in source.params:
        for key in source.params[name]:
            np.testing.assert_array_equal(
                restored.params[name][key], source.params[name][key]
            )


@pytest.mark.parametrize(
    ("builder", "family", "match"),
    [
        (_mixture_template, "mniw", "mniw/gaussian_linear"),
        (_recurrence_template, "component_mniw", "gaussian_mixture_linear"),
    ],
)
def test_hybrid_learning_rejects_incompatible_family_pair(builder, family, match):
    template, arrays, _sources = builder()
    name, spec = next(iter(template.learnables.items()))
    template.learnables[name] = dataclasses.replace(spec, family=family, semantic_identity=None)
    with pytest.raises(CompileError, match=match):
        compile_template(template, arrays=arrays)


@pytest.mark.parametrize("builder", (_mixture_template, _recurrence_template))
def test_hybrid_learning_declaration_does_not_change_inference_or_ledger(builder):
    template, arrays, _sources = builder()
    undeclared = dataclasses.replace(template, learnables={})
    results = []
    for candidate in (undeclared, template):
        engine = Engine(compile_template(candidate, arrays=arrays))
        result = engine.run(engine.init_state(), 100, tol=1e-7)
        result.require_converged()
        results.append((engine, result))
    left_engine, left = results[0]
    right_engine, right = results[1]
    np.testing.assert_array_equal(left.state.messages, right.state.messages)
    np.testing.assert_array_equal(left.state.g_eta, right.state.g_eta)
    np.testing.assert_array_equal(left.state.g_lam, right.state.g_lam)
    assert left_engine.bfe(left.state) == right_engine.bfe(right.state)


def test_component_mniw_plate_copies_pool_into_one_tied_posterior():
    fragment, arrays, _sources = _mixture_template()
    fragment.learnables["neutral_dynamics"] = dataclasses.replace(
        fragment.learnables["neutral_dynamics"], sources=None
    )
    template = Template(
        name="component_plate",
        plates={"bank": PlateSpec(fragment, (2,), ())},
    )
    engine = Engine(compile_template(template, arrays=arrays))
    (plan,) = engine.arena.learn_plans
    assert plan.sources == (
        "bank/0/outer/source_a",
        "bank/0/outer/source_b",
        "bank/1/outer/source_a",
        "bank/1/outer/source_b",
    )
    result = engine.run(engine.init_state(), 100, tol=1e-7)
    result.require_converged()
    priors = tuple(conj.vague_mniw(2, 2) for _ in range(3))
    learnables = Learnables(engine, priors={"neutral_dynamics": priors})
    learnables.replace(engine, result)
    rows = engine.gaussian_node_stats(result.state)
    direct = tuple(
        prior.update(
            sum(
                (rows[source]["Syy"][component] for source in plan.sources),
                np.zeros((2, 2)),
            ),
            sum(
                (rows[source]["Syx"][component] for source in plan.sources),
                np.zeros((2, 3)),
            ),
            sum(
                (rows[source]["Sxx"][component] for source in plan.sources),
                np.zeros((3, 3)),
            ),
            sum(
                float(rows[source]["responsibility"][component])
                for source in plan.sources
            ),
        )
        if any(
            rows[source]["responsibility"][component] > 0.0
            for source in plan.sources
        )
        else prior
        for component, prior in enumerate(priors)
    )
    for key, expected in _stacked_params(direct).items():
        np.testing.assert_allclose(
            learnables.params["neutral_dynamics"][key], expected
        )


@pytest.mark.parametrize(
    ("builder", "priors", "legacy"),
    [
        (
            _mixture_template,
            lambda: {
                "neutral_dynamics": tuple(conj.vague_mniw(2, 2) for _ in range(3))
            },
            {
                "neutral_dynamics/m",
                "neutral_dynamics/lam",
                "neutral_dynamics/psi",
                "neutral_dynamics/nu",
            },
        ),
        (
            _recurrence_template,
            lambda: {"neutral_recurrence": conj.vague_pg_weights(3, 2)},
            {"neutral_recurrence/eta", "neutral_recurrence/precision"},
        ),
    ],
)
def test_hybrid_legacy_checkpoint_requires_complete_posterior(
    builder, priors, legacy
):
    template, arrays, _sources = builder()
    engine = Engine(compile_template(template, arrays=arrays))
    learnables = Learnables(engine, priors=priors())
    versioned = learnables.state_dict()
    legacy_checkpoint = {
        legacy_key: versioned[
            "v2:"
            + legacy_key.rsplit("/", 1)[0]
            + ":"
            + legacy_key.rsplit("/", 1)[1]
        ]
        for legacy_key in legacy
    }
    partial = dict(legacy_checkpoint)
    partial.pop(next(iter(partial)))
    if "neutral_recurrence" in learnables.params:
        for checkpoint in (partial, legacy_checkpoint):
            with pytest.raises(ValueError, match="no statistic identity"):
                learnables.load_state_dict(checkpoint)
        return
    with pytest.raises(KeyError, match="incompletely names"):
        learnables.load_state_dict(partial)
    learnables.load_state_dict(legacy_checkpoint)
