import numpy as np
import pytest

from cffg.models.coordinate_fit import (
    coordinate_decision,
    coordinate_identity,
    coordinate_residual,
)
from cffg.models.rslds_coordinate import (
    initial_runtime_blocks,
    transition_probabilities,
    transition_residual,
)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_coordinate_residual_reports_named_array_and_family_evidence(dtype):
    before = {
        "a": {
            "location": np.array([0.0, 2.0], dtype=dtype),
            "probability": np.array([0.25, 0.75], dtype=dtype),
        },
        "b": {"scale": np.array([4.0], dtype=dtype)},
    }
    after = {
        "a": {
            "location": np.array([0.5, 1.0], dtype=dtype),
            "probability": np.array([0.5, 0.5], dtype=dtype),
        },
        "b": {"scale": np.array([2.0], dtype=dtype)},
    }

    residual = coordinate_residual(before, after)

    assert residual.maximum == pytest.approx(0.5)
    assert residual.per_family == pytest.approx({"a": 0.5, "b": 0.5})
    assert residual.per_array == pytest.approx(
        {
            "a/location": 0.5,
            "a/probability": 0.25,
            "b/scale": 0.5,
        }
    )
    assert residual.vector == pytest.approx((0.25, -0.5, 0.25, -0.25, -0.5))
    assert residual.root_mean_square == pytest.approx(
        np.sqrt(np.mean(np.square(residual.vector)))
    )


def test_coordinate_residual_is_zero_for_an_exact_fixed_point():
    view = {"model": {"x": np.array([1.0, -2.0])}}

    residual = coordinate_residual(view, view)

    assert residual.maximum == 0.0
    assert residual.root_mean_square == 0.0
    assert residual.per_family == {"model": 0.0}


@pytest.mark.parametrize(
    ("before", "after", "message"),
    (
        (
            {"a": {"x": np.ones(1)}},
            {"b": {"x": np.ones(1)}},
            "parameter families",
        ),
        (
            {"a": {"x": np.ones(1)}},
            {"a": {"y": np.ones(1)}},
            "parameter arrays",
        ),
        (
            {"a": {"x": np.ones(1)}},
            {"a": {"x": np.ones(2)}},
            "changed shape",
        ),
        (
            {"a": {"x": np.ones(1)}},
            {"a": {"x": np.array([np.nan])}},
            "non-finite",
        ),
    ),
)
def test_coordinate_residual_rejects_incomparable_views(before, after, message):
    with pytest.raises((TypeError, ValueError), match=message):
        coordinate_residual(before, after)


def test_coordinate_residual_rejects_integer_arrays():
    with pytest.raises(TypeError, match="inexact dtype"):
        coordinate_residual(
            {"a": {"x": np.array([1], dtype=np.int32)}},
            {"a": {"x": np.array([2], dtype=np.int32)}},
        )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_second_runtime_map_hit_certifies_the_evaluated_coordinate(dtype):
    first = coordinate_decision(
        evaluated_index=0,
        residual=dtype(0.1),
        tolerance=0.1,
        previous_hits=0,
    )
    second = coordinate_decision(
        evaluated_index=1,
        residual=dtype(0.05),
        tolerance=0.1,
        previous_hits=first.consecutive_hits,
    )

    assert first.installs_proposal
    assert first.installed == "coordinate/1"
    assert first.certified is None
    assert not second.installs_proposal
    assert second.evaluated == "coordinate/1"
    assert second.proposed == "coordinate/2"
    assert second.installed == second.evaluated
    assert second.certified == second.evaluated


def test_rebound_resets_hits_across_a_long_coordinate_sequence():
    residuals = [value for _ in range(10) for value in (0.05, 0.2)]
    residuals.extend((0.05, 0.05))
    coordinate_index = 0
    hits = 0

    for residual in residuals:
        decision = coordinate_decision(
            evaluated_index=coordinate_index,
            residual=residual,
            tolerance=0.1,
            previous_hits=hits,
        )
        assert decision.evaluated == coordinate_identity(coordinate_index)
        hits = decision.consecutive_hits
        if decision.installs_proposal:
            coordinate_index += 1

    assert len(residuals) == 22
    assert decision.certified == "coordinate/21"
    assert decision.installed == decision.evaluated


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            dict(
                evaluated_index=-1,
                residual=0.0,
                tolerance=0.1,
                previous_hits=0,
            ),
            "index",
        ),
        (
            dict(
                evaluated_index=0,
                residual=np.nan,
                tolerance=0.1,
                previous_hits=0,
            ),
            "residual",
        ),
        (
            dict(
                evaluated_index=0,
                residual=0.0,
                tolerance=np.inf,
                previous_hits=0,
            ),
            "tolerance",
        ),
        (
            dict(
                evaluated_index=0,
                residual=0.0,
                tolerance=0.1,
                previous_hits=-1,
            ),
            "hits",
        ),
        (
            dict(
                evaluated_index=0,
                residual=0.0,
                tolerance=0.1,
                previous_hits=2,
            ),
            "hits",
        ),
    ),
)
def test_coordinate_decision_rejects_malformed_evidence(kwargs, message):
    with pytest.raises(ValueError, match=message):
        coordinate_decision(**kwargs)


def _rslds_points(dtype=np.float64):
    return {
        "initial_regime_probability": np.array([2.0, 0.0], dtype=dtype),
        "dynamics_matrix": np.ones((2, 1, 1), dtype=dtype),
        "dynamics_offset": np.zeros((2, 1), dtype=dtype),
        "dynamics_covariance": np.ones((2, 1, 1), dtype=dtype),
        "recurrence_weights": np.zeros((2, 1, 1), dtype=dtype),
        "recurrence_offsets": np.zeros((2, 1), dtype=dtype),
        "emission_matrix": np.ones((2, 1), dtype=dtype),
        "emission_offset": np.zeros(2, dtype=dtype),
        "emission_covariance": np.eye(2, dtype=dtype),
    }


def test_initial_runtime_blocks_normalize_without_filling_zero_support():
    blocks = initial_runtime_blocks(_rslds_points())

    np.testing.assert_array_equal(
        blocks["initial"]["regime_probability"], np.array([1.0, 0.0])
    )
    assert set(blocks) == {"initial", "dynamics", "recurrence", "emission"}


def test_transition_probe_matches_a_hand_computed_three_state_decision_list():
    states = np.array([[-3.0], [0.0], [2.0]])
    weights = np.zeros((3, 2, 1))
    offsets = np.zeros((3, 2))

    probabilities = transition_probabilities(states, weights, offsets)

    expected = np.broadcast_to(np.array([0.5, 0.25, 0.25]), probabilities.shape)
    np.testing.assert_allclose(probabilities, expected)
    np.testing.assert_allclose(np.sum(probabilities, axis=-1), 1.0)


def test_transition_residual_is_probability_space_change_on_a_fixed_probe():
    before = _rslds_points()
    after = _rslds_points()
    after["recurrence_offsets"] = np.ones((2, 1)) * np.log(3.0)

    residual = transition_residual(before, after, np.array([[0.0], [1.0]]))

    assert residual.maximum == pytest.approx(0.25)
    assert residual.per_family["transition"] == pytest.approx(0.25)
