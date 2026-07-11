"""Truthful inference termination under hostile inputs (cff-j4mu)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.compiler import compile_template
from cffg.core import FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.ir import BlobRef
from cffg.engine import Engine, InferenceError, RunResult, RunStatus, _replace


def categorical_engine(dtype=np.float64) -> Engine:
    template = Template(
        "status_unit",
        variables={"x": VariableSpec("x", 3)},
        factors={
            "prior": FactorSpec(
                "prior", "categorical", edges={"out": "x"},
                param_blobs={"statistic": "p"},
            )
        },
        ports={"data": PortSpec("data", "x", PortRole.DATA)},
        blobs={"p": BlobRef("p", "p")},
    )
    return Engine(compile_template(template, arrays={"p": np.asarray([0.2, 0.3, 0.5], dtype=dtype)}))


def contradictory_exactly_zero() -> Engine:
    template = Template(
        "unsatisfiable",
        variables={"x0": VariableSpec("x0", 2), "x1": VariableSpec("x1", 2)},
        factors={
            "none_true": FactorSpec(
                "none_true", "exactly_k", edges={"x0": "x0", "x1": "x1"},
                param_blobs={"k": "k"},
            )
        },
        ports={
            "d0": PortSpec("d0", "x0", PortRole.DATA),
            "d1": PortSpec("d1", "x1", PortRole.DATA),
        },
        blobs={"k": BlobRef("k", "k")},
    )
    return Engine(compile_template(template, arrays={"k": np.asarray(0)}))


def test_run_result_fields_and_explicit_api_migration():
    engine = categorical_engine()
    result = engine.run(engine.init_state(), 8, tol=1e-13)
    assert isinstance(result, RunResult)
    assert result.status is RunStatus.CONVERGED
    assert result.iterations == len(result.message_deltas)
    assert len(result.free_energies) >= 2
    assert result.residual == abs(result.free_energies[-1] - result.free_energies[-2])
    assert result.delta_trace == result.message_deltas
    assert result.fe_trace == result.free_energies
    assert result.require_converged() is result.state
    with pytest.raises(TypeError):
        _, _ = result  # old ``state, deltas = run(...)`` migration is intentionally loud


def test_deliberate_budget_exhaustion_is_not_convergence():
    engine = categorical_engine()
    result = engine.run(engine.init_state(), 1, tol=1e-13)
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.iterations == 1
    with pytest.raises(InferenceError) as error:
        result.require_converged()
    assert error.value.result is result


def test_poisoned_message_diverges_before_a_sweep():
    engine = categorical_engine()
    state = engine.init_state()
    poisoned = _replace(state, messages=state.messages.at[0, 0].set(jnp.nan))
    result = engine.run(poisoned, 20, tol=1e-10)
    assert result.status is RunStatus.DIVERGED
    assert result.iterations == 0
    assert np.isinf(result.residual)


def test_contradictory_hard_constraints_are_infeasible_with_infinite_fe():
    engine = contradictory_exactly_zero()
    state = engine.bind_data(engine.init_state(), {"d0": 1, "d1": 1})
    result = engine.run(state, 20, tol=1e-12)
    assert result.status is RunStatus.INFEASIBLE
    assert np.isposinf(result.free_energies[-1])
    assert np.isposinf(engine.bfe(result.state))


def test_float32_tolerance_is_dtype_scaled():
    with jax.enable_x64(False):
        engine = categorical_engine(np.float32)
        result = engine.run(engine.init_state(), 8, tol=1e-30)
        assert result.state.messages.dtype == jnp.float32
        assert result.status is RunStatus.CONVERGED
        assert np.isfinite(result.residual)
