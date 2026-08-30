"""Declared learning for hybrid Gaussian-mixture and recurrence factors."""

import jax
import numpy as np

from cffg import conj
from cffg.compiler import compile_template
from cffg.core import BlobRef, FactorSpec, LearnableSpec, Template, VariableSpec
from cffg.engine import Engine
from cffg.learn import Learnables


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
