"""Local provenance survives JAX transforms without becoming a solver policy."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nonlinear_fixture import model

from cffg import compile_template
from cffg.compiled import NEG
from cffg.execution.contracts import ExecutionDelta
from cffg.execution.registered import execute_gaussian
from cffg.failure_facts import (
    FailureFact,
    FailurePhase,
    FailureReason,
    empty_failure,
    first_failure,
    in_phase,
    local_failure,
    nonfinite_rows,
)
from cffg.kernel_runtime import parameters


def assert_fact(fact, factor, reason, phase=FailurePhase.NONE):
    assert tuple(int(value) for value in fact) == (factor, reason, phase)
    assert all(value.shape == () and value.dtype == jnp.int32 for value in fact)


def test_simultaneous_failures_use_name_rank_and_structured_cause():
    fact = jax.jit(local_failure)(
        jnp.asarray([8, 3, 1]),
        status=jnp.asarray([2, 2, 0]),
        reasons=jnp.asarray([0, FailureReason.IMPROPER_CAVITY, 0]),
        nonfinite=jnp.asarray([True, True, False]),
    )
    assert_fact(fact, 3, FailureReason.IMPROPER_CAVITY)
    # Row order is not the model-local factor rank.
    reversed_fact = local_failure(
        jnp.asarray([1, 3, 8]), status=jnp.asarray([0, 2, 2]),
        reasons=jnp.asarray([0, FailureReason.IMPROPER_CAVITY, 0]),
        nonfinite=jnp.asarray([False, True, True]),
    )
    np.testing.assert_array_equal(fact, reversed_fact)


def test_chronology_and_origin_phase_survive_later_lower_rank_failure():
    first = local_failure(jnp.asarray([8]), status=jnp.asarray([1]))
    later = local_failure(jnp.asarray([1]), nonfinite=jnp.asarray([True]))
    chosen = jax.jit(first_failure)(in_phase(first, FailurePhase.PRIMING), later)
    assert_fact(chosen, 8, FailureReason.LOCAL_BUDGET, FailurePhase.PRIMING)
    assert_fact(in_phase(chosen, FailurePhase.SOFT), 8,
                FailureReason.LOCAL_BUDGET, FailurePhase.PRIMING)
    assert_fact(first_failure(empty_failure(), later), 1, FailureReason.NONFINITE_OUTPUT)
    assert_fact(in_phase(empty_failure(), FailurePhase.OUTER), -1, FailureReason.NONE)


def test_successful_informational_reasons_and_empty_groups_do_not_fail():
    informative = jnp.asarray([FailureReason.IMPROPER_CAVITY])
    for status in (None, jnp.asarray([0])):
        fact = jax.jit(local_failure)(jnp.asarray([0]), status=status, reasons=informative)
        assert_fact(fact, -1, FailureReason.NONE)
    assert_fact(jax.jit(local_failure)(jnp.zeros(0, jnp.int32)), -1, FailureReason.NONE)
    delta = ExecutionDelta(jnp.ones((1, 2)))
    assert delta.failure_reason is delta.status is delta.residual is None


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_sparse_log_support_and_gaussian_nonfinite_coordinates_are_distinct(dtype):
    values = jnp.asarray([[0., -jnp.inf], [NEG, NEG], [0., jnp.nan], [jnp.inf, 0.]], dtype)
    categorical = jax.jit(partial(nonfinite_rows, log_support=True))(values)
    gaussian = jax.jit(nonfinite_rows)(values)
    np.testing.assert_array_equal(categorical, [False, False, True, True])
    np.testing.assert_array_equal(gaussian, [True, False, True, True])
    assert_fact(local_failure(jnp.arange(2), nonfinite=categorical[:2]), -1, FailureReason.NONE)
    np.testing.assert_array_equal(nonfinite_rows(jnp.zeros((2, 0), dtype)), [False, False])
    np.testing.assert_array_equal(nonfinite_rows(values[:, 0]), [False, False, False, True])


def test_shape_errors_are_explicit_before_local_reduction():
    with pytest.raises(ValueError, match="leading member"):
        nonfinite_rows(jnp.asarray(0.))
    with pytest.raises(ValueError, match="one member"):
        local_failure(jnp.zeros((1, 1), jnp.int32))
    with pytest.raises(ValueError, match="one value per factor"):
        local_failure(jnp.arange(2), status=jnp.asarray([0]))


def test_fact_reduction_is_independent_per_vmapped_instance_at_scale():
    ids = jnp.arange(32, dtype=jnp.int32)
    status = jnp.zeros((40, 32), jnp.int32).at[::2, 9].set(2).at[::4, 3].set(1)
    results = jax.jit(jax.vmap(lambda row: local_failure(ids, status=row)))(status)
    assert isinstance(results, FailureFact)
    for index in range(40):
        fact = jax.tree.map(lambda value: value[index], results)
        if index % 4 == 0:
            assert_fact(fact, 3, FailureReason.LOCAL_BUDGET)
        elif index % 2 == 0:
            assert_fact(fact, 9, FailureReason.LOCAL_DIVERGENCE)
        else:
            assert_fact(fact, -1, FailureReason.NONE)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
def test_registered_nonlinear_reason_survives_jit_and_vmap(dtype, method):
    template, arrays = model(method)
    arena = compile_template(template, arrays=arrays)
    plan = next(group for group in arena.gauss_groups if group.type == "gaussian_nonlinear")
    values = {key: jnp.asarray(value, dtype) for key, value in arena.log_params.items()}
    params = parameters(plan, values)

    def update(input_precision):
        eta = jnp.zeros((1, 2, 1), dtype)
        precision = jnp.ones((1, 2, 1, 1), dtype).at[:, 1].set(input_precision)
        return execute_gaussian(plan, eta, precision, params)

    inputs = jnp.asarray([1., 0., -1.] * 10, dtype)
    actual = jax.jit(jax.vmap(update))(inputs)
    for index, input_precision in enumerate(inputs[:3]):
        direct = update(input_precision)
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(direct), strict=True):
            np.testing.assert_allclose(left[index], right, equal_nan=True)
    np.testing.assert_array_equal(actual.status[:, 0], [0, 0, 2] * 10)
    np.testing.assert_array_equal(actual.failure_reason[:, 0],
                                  [0, 0, FailureReason.IMPROPER_CAVITY] * 10)
    reduced = jax.jit(jax.vmap(lambda status, reasons: local_failure(
        jnp.asarray([7]), status=status, reasons=reasons,
    )))(actual.status, actual.failure_reason)
    np.testing.assert_array_equal(reduced.factor, [-1, -1, 7] * 10)
