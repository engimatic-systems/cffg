import jax
import numpy as np
import pytest

from cffg import recurrence
from cffg.hybrid_learning import PGWeightsFamily


def _row(stats, index=0):
    return {
        name: np.asarray(getattr(stats, name)[index])
        for name in stats._fields
    }


def test_switch447_uses_direct_covariance_without_roundoff_rejection():
    q_in = np.array(
        [
            0.9997978465089572,
            1.562243022193262e-7,
            0.00020199726673797268,
            2.514293071161196e-15,
        ]
    )
    q_out = np.array(
        [
            4.637976741017066e-5,
            0.9999536202301598,
            6.984843611201459e-16,
            2.429309517397572e-12,
        ]
    )
    mean = np.array([2.044281105819261, -0.40074051172844327])
    covariance = np.array(
        [
            [0.0055827036792674015, 0.00026013456494927695],
            [0.000260134564949277, 0.005557219631225886],
        ]
    )
    archived_second = np.array(
        [
            [4.184667943288888, -0.8189661218978493],
            [-0.8189661218978492, 0.16615017737160045],
        ]
    )
    weights = np.array(
        [
            [
                [-0.13164682236302921, 0.03772991965771724],
                [0.4323209731938957, 0.19098220535226798],
                [-0.46377395415157585, 0.3625933851626348],
            ],
            [
                [0.028412006698972623, 0.2235589556660718],
                [0.46365101943050846, -0.019294508296292236],
                [-0.2517485658334102, 0.11295270749899616],
            ],
            [
                [0.1431362720492547, 0.09952367948673185],
                [0.4977452354359765, 0.014093164313205772],
                [-0.28255027178821435, 0.2669505910724383],
            ],
            [
                [0.07411083529744496, 0.2686750524596794],
                [0.44432123889003505, 0.1515602174064471],
                [-0.5072888264097051, 0.026947166618742118],
            ],
        ]
    )
    offsets = np.array(
        [
            [0.4661519094211754, -0.703477777211779, 0.2019984742861999],
            [-4.066156400450575, -2.4945476879587716, 2.242313688861843],
            [-2.5314371045079027, 0.196467008983759, 1.9711208583331834],
            [-2.5881475415888398, -2.795910333281714, -1.1625349856090008],
        ]
    )

    reconstructed = archived_second - np.outer(mean, mean)
    symmetry_error = np.max(np.abs(reconstructed - reconstructed.T))
    tolerance = (
        16.0
        * mean.size
        * np.finfo(reconstructed.dtype).eps
        * np.max(np.abs(reconstructed))
    )
    assert symmetry_error > tolerance

    stats = recurrence.weight_stats(
        q_out[None],
        q_in[None],
        mean[None],
        covariance[None],
        weights[None],
        offsets[None],
    )

    assert bool(stats.proper[0])
    assert bool(stats.drive_ok[0])
    assert np.all(np.isfinite(stats.linear[0]))
    assert np.all(np.isfinite(stats.quadratic[0]))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_large_mean_small_covariance_remains_proper(dtype):
    with jax.enable_x64(dtype is np.float64):
        mean = np.array([[1.0e8, 3.0e7]], dtype=dtype)
        covariance = np.array(
            [[[0.25, 0.5], [0.5, 1.5]]], dtype=dtype
        )
        stats = recurrence.weight_stats(
            np.full((1, 4), 0.25, dtype=dtype),
            np.full((1, 4), 0.25, dtype=dtype),
            mean,
            covariance,
            np.zeros((1, 4, 3, 2), dtype=dtype),
            np.zeros((1, 4, 3), dtype=dtype),
        )

    assert bool(stats.proper[0])
    assert stats.linear.dtype == dtype
    assert stats.quadratic.dtype == dtype
    assert np.all(np.isfinite(stats.linear[0]))
    assert np.all(np.isfinite(stats.quadratic[0]))


def test_1279_concentrated_rows_preserve_exact_zero_source_support():
    count = 1279
    q_out = np.broadcast_to([0.4, 0.3, 0.2, 0.1], (count, 4))
    q_in = np.broadcast_to([0.3, 0.3, 0.4, 0.0], (count, 4))
    mean = np.broadcast_to([1.0e8, 3.0e7], (count, 2))
    covariance = np.broadcast_to(
        [[0.25, 0.5], [0.5, 1.5]], (count, 2, 2)
    )
    stats = recurrence.weight_stats(
        q_out,
        q_in,
        mean,
        covariance,
        np.zeros((count, 4, 3, 2)),
        np.zeros((count, 4, 3)),
    )

    assert np.all(stats.proper)
    assert stats.linear.shape == (count, 4, 3, 3)
    assert stats.quadratic.shape == (count, 4, 3, 3, 3)
    np.testing.assert_array_equal(stats.exposure[:, 3], 0.0)
    np.testing.assert_array_equal(stats.linear[:, 3], 0.0)
    np.testing.assert_array_equal(stats.quadratic[:, 3], 0.0)


@pytest.mark.parametrize(
    ("q_out", "mean", "covariance", "weights", "failed_predicate"),
    [
        (
            [-0.1, 0.5, 0.4, 0.2],
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            np.zeros((4, 3, 2)),
            "probability_ok",
        ),
        (
            [0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            np.zeros((4, 3, 2)),
            "probability_ok",
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            [np.nan, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            np.zeros((4, 3, 2)),
            "drive_ok",
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0],
            [[1.0, 2.0], [2.0, 1.0]],
            np.zeros((4, 3, 2)),
            "drive_ok",
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            np.full((4, 3, 2), np.nan),
            "parameter_ok",
        ),
    ],
)
def test_invalid_recurrence_statistics_remain_loud(
    q_out, mean, covariance, weights, failed_predicate
):
    stats = recurrence.weight_stats(
        np.asarray(q_out)[None],
        np.full((1, 4), 0.25),
        np.asarray(mean)[None],
        np.asarray(covariance)[None],
        np.asarray(weights)[None],
        np.zeros((1, 4, 3)),
    )
    readout = _row(stats)

    assert not bool(readout["proper"])
    assert not bool(readout[failed_predicate])
    with pytest.raises(ValueError, match=failed_predicate):
        PGWeightsFamily().stats("recurrence_link", readout)


def test_material_covariance_asymmetry_reports_scale_relative_values():
    stats = recurrence.weight_stats(
        np.full((1, 4), 0.25),
        np.full((1, 4), 0.25),
        np.zeros((1, 2)),
        np.array([[[1.0, 0.5], [0.0, 1.0]]]),
        np.zeros((1, 4, 3, 2)),
        np.zeros((1, 4, 3)),
    )
    readout = _row(stats)

    assert not bool(readout["drive_ok"])
    assert readout["drive_symmetry_error"] > readout["drive_tolerance"]
    with pytest.raises(
        ValueError,
        match=r"drive_ok=false .*symmetry_error=.*tolerance=",
    ):
        PGWeightsFamily().stats("recurrence_link", readout)
