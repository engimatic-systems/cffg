"""Batched chart geometry against the independent, original block operations."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.outer_chart import (
    OuterArrays, OuterChart, _CategoricalBlock, _PrecisionBlock, _VectorBlock,
)
from cffg.outer_coordinates import structural_signature
from cffg.outer_kernels import OuterKernelProvider


def reference_pack(chart, arrays):
    pieces = [arrays.messages[b.coordinates] - arrays.messages[b.reference]
              for b in chart.categorical if b.coordinates.size]
    pieces.extend(jnp.asarray(b.forward) @ getattr(arrays, b.field)[b.indices]
                  for b in chart.vectors)
    for block in chart.precisions:
        matrix = arrays.g_lam[block.indices]
        whitening = jnp.asarray(block.whitening)
        reduced = jnp.linalg.solve(whitening, jnp.linalg.solve(whitening, matrix).T).T
        pieces.append(reduced[block.lower[:, 0], block.lower[:, 1]])
    return jnp.concatenate(pieces) if pieces else jnp.zeros(0, chart.base.messages.dtype)


def reference_unpack(chart, coordinates):
    arrays, offset = chart.base, 0
    for block in chart.categorical:
        messages = arrays.messages.at[block.support].set(0.)
        width = block.coordinates.size
        messages = messages.at[block.coordinates].set(coordinates[offset:offset + width])
        arrays = arrays._replace(messages=messages)
        offset += width
    for block in chart.vectors:
        width = block.indices.size
        value = jnp.asarray(block.inverse) @ coordinates[offset:offset + width]
        targets = block.indices if block.aliases is None else block.aliases
        field = getattr(arrays, block.field).at[targets].set(value)
        arrays = arrays._replace(**{block.field: field})
        offset += width
    for block in chart.precisions:
        width = len(block.lower)
        matrix = jnp.zeros(block.indices.shape, arrays.g_lam.dtype)
        rows, columns = block.lower.T
        matrix = matrix.at[rows, columns].set(coordinates[offset:offset + width])
        matrix = matrix.at[columns, rows].set(coordinates[offset:offset + width])
        whitening = jnp.asarray(block.whitening)
        matrix = whitening @ matrix @ whitening.T
        arrays = arrays._replace(g_lam=arrays.g_lam.at[block.indices.reshape(-1)].set(matrix.reshape(-1)))
        offset += width
    return arrays


def mixed_chart(dtype, members):
    arrays = OuterArrays(
        jnp.array([-.4, -jnp.inf, -1.3, -jnp.inf, -.7], dtype),
        jnp.arange(2 * members + 1, dtype=dtype) / 7,
        jnp.tile(jnp.array([[2., .3], [.3, 1.]], dtype).reshape(-1), members),
        jnp.array([.2, .8, .2, .8], dtype),
    )
    whitening = np.array([[1.2, 0.], [.4, .8]], dtype)
    vectors = [
        _VectorBlock("g_eta", np.arange(2 * i, 2 * i + 2), whitening, np.linalg.inv(whitening))
        for i in range(members)
    ]
    vectors.extend([
        _VectorBlock("g_eta", np.array([2 * members]), np.array([[2.]], dtype), np.array([[.5]], dtype)),
        _VectorBlock("ef_eta", np.array([0, 1]), np.eye(2, dtype=dtype), np.eye(2, dtype=dtype),
                     np.array([[0, 1], [2, 3]])),
    ])
    precision = tuple(_PrecisionBlock(np.arange(4 * i, 4 * i + 4).reshape(2, 2),
                                     np.array([[0, 0], [1, 0], [1, 1]]), whitening)
                      for i in range(members))
    categorical = (_CategoricalBlock(np.arange(4), np.array([0, 2]), 0, np.array([2])),
                   _CategoricalBlock(np.array([4]), np.array([4]), 4, np.array([], dtype=int)))
    widths = [1, *([2] * members), 1, 2, *([3] * members)]
    ends = np.cumsum([0, *widths])
    return OuterChart(arrays, categorical, tuple(vectors), precision,
                      tuple(zip(ends[:-1], ends[1:], strict=True)), True, (2, 2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("members", [2, 200])
def test_batched_geometry_preserves_primal_support_aliases_and_jvp(dtype, members):
    with jax.enable_x64(dtype is np.float64):
        chart = mixed_chart(dtype, members)
        # First access happens under tracing: cached metrics must never retain tracers.
        coordinates = jax.jit(chart.pack)(chart.base)
        eps = np.finfo(dtype).eps
        np.testing.assert_allclose(coordinates, reference_pack(chart, chart.base), rtol=32 * eps, atol=32 * eps)
        np.testing.assert_allclose(chart.pack(chart.base), coordinates, rtol=32 * eps, atol=32 * eps)
        actual, expected = chart.unpack(coordinates), reference_unpack(chart, coordinates)
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, reference, rtol=32 * eps, atol=32 * eps)
        assert np.isneginf(np.asarray(actual.messages)[[1, 3]]).all()
        assert actual.messages[4] == 0.
        np.testing.assert_array_equal(actual.ef_eta[:2], actual.ef_eta[2:])
        direction = jnp.ones_like(coordinates)
        actual_jvp = jax.jvp(chart.unpack, (coordinates,), (direction,))[1]
        expected_jvp = jax.jvp(lambda x: reference_unpack(chart, x), (coordinates,), (direction,))[1]
        for value, reference in zip(actual_jvp, expected_jvp, strict=True):
            np.testing.assert_allclose(value, reference, rtol=32 * eps, atol=32 * eps)


def test_active_blocks_preserve_dtype_promotion_and_dynamic_metrics():
    chart = mixed_chart(np.float64, 2)
    mixed = replace(chart, base=chart.base._replace(messages=chart.base.messages.astype(jnp.float32)), categorical=())
    assert mixed.pack(mixed.base).dtype == reference_pack(mixed, mixed.base).dtype == jnp.float64
    vectors = tuple(replace(b, forward=b.forward * 3, inverse=b.inverse / 3) for b in chart.vectors)
    changed = replace(chart, vectors=vectors)
    assert structural_signature(chart) == structural_signature(changed)
    assert chart.program is changed.program
    np.testing.assert_allclose(chart.pack(chart.base, changed.metrics), reference_pack(changed, changed.base))
    replacement = replace(chart.categorical[0], support=np.array([1, 2]), reference=1)
    different_support = replace(chart, categorical=(replacement, chart.categorical[1]))
    assert structural_signature(chart) != structural_signature(different_support)


def test_reused_kernel_binds_new_metrics_and_matches_independent_derivatives():
    chart = mixed_chart(np.float64, 2)
    chart = replace(chart, base=chart.base._replace(messages=jnp.zeros(0)), categorical=())
    provider = OuterKernelProvider()

    def step(messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params, damp, harden):
        return messages, eta * .5 + obs[0], lam * .7, ef_eta * .8, 0., 0

    for index, scale in enumerate((1., 3.)):
        changed = replace(chart, vectors=tuple(replace(b, forward=b.forward * scale,
                                                      inverse=b.inverse / scale) for b in chart.vectors))
        state = SimpleNamespace(g_obs=jnp.array([scale]), g_mask=jnp.zeros(0),
                                ef_obs=jnp.zeros(0), ef_mask=jnp.zeros(0), params={})
        plan = provider.plan(changed, state, [], iter_step=step, reconstruct=None)
        coordinates = reference_pack(changed, changed.base)

        def merit(value):
            arrays = reference_unpack(changed, value)
            mapped = OuterArrays(arrays.messages, arrays.g_eta * .5 + scale,
                                 arrays.g_lam * .7, arrays.ef_eta * .8)
            residual = reference_pack(changed, mapped) - value
            return .5 * jnp.vdot(residual, residual).real

        actual, gradient = plan.root_diagnostics(coordinates, jnp.ones_like(coordinates), plan.context)
        np.testing.assert_allclose(actual, merit(coordinates), rtol=1e-12)
        np.testing.assert_allclose(gradient, jnp.linalg.norm(jax.grad(merit)(coordinates)), rtol=1e-12)
        assert plan.compiled is (index == 0)
    assert len(provider.cache) == 1
