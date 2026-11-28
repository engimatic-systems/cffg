"""Independent short-horizon checks for the partial-goal research comparison."""

import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from partial_goal_comparison import (
    dense_reference,
    exact_terminal_pair,
    gaussian_future,
)

from cffg import Engine, compile_template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from hybrid_planning_fixture import declared_rule_reference, planning_fixture


@pytest.mark.parametrize("habit", [None, [0.4, 0.6, 0.0], [0.0, 1.0, 0.0]])
def test_path_sums_match_joint_observation_density(habit):
    for horizon in (1, 2, 3):
        _, data, _ = planning_fixture(horizon, habit=habit)
        goals = np.zeros(horizon)
        exact = exact_terminal_pair(data, goals, 2.0)
        for record, final in zip(exact, (0.0, 2.0), strict=True):
            goals[-1] = final
            dense = dense_reference(data, goals)
            approx = gaussian_future(data, goals)
            np.testing.assert_allclose(
                record["first"], dense["first"], atol=2e-13, rtol=0.0
            )
            np.testing.assert_allclose(
                record["log_evidence"], dense["log_evidence"], atol=2e-12, rtol=0.0
            )
            assert np.all(np.asarray(approx["first"])[data["habit"] == 0.0] == 0.0)
            if horizon == 1 or np.count_nonzero(data["habit"]) == 1:
                for key in record:
                    np.testing.assert_allclose(
                        approx[key], record[key], atol=2e-12, rtol=0.0
                    )


def test_informative_direction_and_limits_of_common_dynamics():
    _, data, _ = planning_fixture(2)
    row = data["C"][0] @ data["A"][0]
    null = np.array([-row[1], row[0]])
    np.testing.assert_allclose(row @ null, 0.0, atol=2 * np.finfo(float).eps)
    assert np.linalg.matrix_rank(np.stack([data["C"][0], row])) == 2
    data["A"][1, 0] = null
    with pytest.raises(ValueError, match="common dynamics"):
        gaussian_future(data, [0.0, 2.0])
    # An impossible action cannot rotate the common informative direction.
    data["habit"] = np.array([0.5, 0.0, 0.5])
    exact = exact_terminal_pair(data, [0.0, 0.0], 2.0)[1]
    np.testing.assert_allclose(
        exact["first"], dense_reference(data, [0.0, 2.0])["first"], atol=2e-13
    )


def test_float32_public_solver_and_exact_response():
    with jax.enable_x64(False):
        template, data, observations = planning_fixture(2, np.float32)
        engine = Engine(compile_template(template, arrays=data))
        # Evaluate the represented f32 model in double for the reference.
        reference = {k: v.astype(float) for k, v in data.items()}
        # The categorical prior is normalized by public parameter packing.
        reference["habit"] /= reference["habit"].sum()
        exact = exact_terminal_pair(reference, [0.0, 0.0], 2.0)
        assert [np.argmax(item["first"]) for item in exact] == [0, 2]
        for final in (0.0, 2.0):
            changed = {**observations, "g2": np.array([final], dtype=np.float32)}
            run = engine.run(
                engine.bind_data(engine.init_state(), changed), 100, tol=1e-5
            )
            actual = engine.marginals(run.require_converged(), edges=["z0"])["z0"]
            assert actual.dtype == np.float32
            assert np.argmax(actual) == 0
            declared = declared_rule_reference(reference, changed)
            np.testing.assert_allclose(
                actual,
                declared.regimes[0],
                rtol=0.0,
                atol=64 * np.finfo(np.float32).eps,
            )
