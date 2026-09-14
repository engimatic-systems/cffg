"""Independent complete tiny D12 EC+xi checkpoint and returned-coordinate checks."""

import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from hybrid_joint_reference import (
    ARITHMETIC, EPS, TAU, Natural, ReferenceFailure, evaluate, fixed_drive,
    load_case, omega, specification,
)
from hybrid_joint_solver_reference import (
    Coordinate, jacobian, output_arrays, solve, stopping_allowance,
)
from hybrid_separator_reference import enumerate_chain, recurrence_increments, reflected_mixture_gap


VALID = [name for name in specification()["cases"]
         if name not in ("empty_source_t2", "underflow_source_t2")]


@lru_cache(None)
def solution(name, start=0.):
    result = solve(load_case(name), start)
    assert result.status == "converged", result
    return result


def path_reference(name, xi):
    f = specification()["cases"][name]
    args = {k: np.asarray(f[k], dtype=float) for k in
            ("site_constant", "site_linear", "site_precision", "weights", "offsets", "matrix", "bias", "noise")}
    return enumerate_chain(**args, c=np.sqrt(xi[..., 0]))


def assert_arithmetic(a, b):
    np.testing.assert_allclose(a, b, atol=ARITHMETIC, rtol=ARITHMETIC)


@pytest.mark.parametrize("name", VALID)
@pytest.mark.parametrize("start", [0., 1.])
def test_complete_reference_certifies_every_returned_coordinate(name, start):
    model = load_case(name)
    result = solution(name, start)
    held = evaluate(model, *Coordinate(model).unpack(result.coordinate))
    assert result.evaluation["proper_active_beliefs"]
    assert max(result.evaluation["certificates"].values()) <= TAU
    for key, value in held["certificates"].items():
        assert value == result.evaluation["certificates"][key]
    for name, expected in output_arrays(held).items():
        np.testing.assert_array_equal(output_arrays(result.evaluation)[name], expected)
    assert np.all(held["xi"][~model.auxiliary_support] == 0)
    if model.weights.any():
        assert np.all(held["xi"][model.auxiliary_support] > 0)
    for belief in held["regions"] + held["separators"]:
        assert np.all(np.linalg.eigvalsh(belief.covariance[belief.active]) > 0)
        assert np.all(belief.probability[~belief.active] == 0)


@pytest.mark.parametrize("name", ["coupled_t2", "source_masked_t2", "mean_forward", "mean_reverse", "variance_forward", "variance_reverse", "independent_t3"])
def test_exact_reductions_at_returned_xi_match_boundaries_energy_and_learning(name):
    model = load_case(name)
    held = solution(name).evaluation
    exact = path_reference(name, held["xi"])
    for t, region in enumerate(held["regions"]):
        assert_arithmetic(region.probability, exact["pair"][t])
        assert_arithmetic(region.probability[..., None] * region.mean, exact["first"][t])
        assert_arithmetic(region.probability[..., None, None] * region.second, exact["second"][t])
        expected = recurrence_increments(exact["pair"][t], exact["first"][t], exact["second"][t], np.sqrt(held["xi"][t, :, 0]))
        for key, value in zip(("linear", "precision"), expected):
            assert_arithmetic(held["learning"][t][key][:, 0], value)
        if len(held["regions"]) == 1:
            for side in range(2):
                mass = exact["pair"][t].sum(axis=1 - side)
                active = mass > 0
                first = exact["first"][t][..., side].sum(axis=1 - side)[active] / mass[active]
                second = exact["second"][t][..., side, side].sum(axis=1 - side)[active] / mass[active]
                variance = second - first**2
                site = model.sites[side]
                precision = 1 / variance - site.precision[active, 0, 0]
                linear = first / variance - site.linear[active, 0]
                constant = np.log(mass[active]) - (np.log(2 * np.pi * variance) + first**2 / variance) / 2 - site.constant[active]
                constant -= constant[0]
                for actual, expected in zip(held["outgoing"][t][side].arrays(), (constant, linear[:, None], precision[:, None, None])):
                    assert_arithmetic(actual[active], expected)
    energy_atol = 512 * max(4, 5 * model.shape[0]) * EPS
    assert abs(held["free_energy"] - exact["free_energy"]) <= energy_atol
    assert abs(held["dual_energy"] - exact["free_energy"]) <= energy_atol
    if name == "independent_t3":
        assert_arithmetic(held["xi"], model.offsets**2)
        assert min(np.max(abs(b.covariance[..., 0, 1])) for b in held["regions"]) > .05


@pytest.mark.parametrize("name", ["mean_forward", "mean_reverse", "variance_forward", "variance_reverse"])
def test_d11_twins_preserve_unweighted_conditionals_and_source_learning(name):
    f = specification()["cases"][name]
    means, variances = np.array(f["known_source_mean"]), np.array(f["known_source_variance"])
    held = solution(name).evaluation
    source = held["projections"][0][0]
    assert_arithmetic(source.mean[:, 0], means)
    assert_arithmetic(source.covariance[:, 0, 0], variances)
    learning = held["learning"][0]
    assert_arithmetic(learning["linear"][:, 0, 0], means / 4)
    assert_arithmetic(learning["precision"][:, 0, 0, 0], (variances + means**2) / 8)
    np.testing.assert_array_equal(held["xi"], 0)
    np.testing.assert_array_equal(held["regions"][0].probability[:, 1], 0)


def test_inactive_source_has_exact_zero_auxiliary_and_learning():
    held = solution("source_masked_t2", 1.).evaluation
    for key in ("target", "exposure", "linear", "precision", "centered_linear", "centered_precision"):
        np.testing.assert_array_equal(held["learning"][0][key][1], 0)
    np.testing.assert_array_equal(held["xi"][0, 1], 0)
    np.testing.assert_array_equal(held["regions"][0].probability[1], 0)


def test_fixed_observed_drive_is_conditioned_categorical_multistick_algebra():
    f = {k: np.asarray(v, dtype=float) for k, v in specification()["fixed_drive"].items()}
    actual = fixed_drive(**f)
    w, o, x = f["weights"], f["offsets"], f["drive"]
    psi = w @ x + o
    assert_arithmetic(actual["xi"], psi**2)
    assert_arithmetic(actual["bound"], actual["log_factor"])
    source, destination = f["source_log_site"], f["destination_log_site"]
    pair = actual["pair"]
    active = pair > 0
    direct_primal = np.sum(pair[active] * (np.log(pair[active]) - (source[:, None] + destination[None, :] + actual["log_factor"])[active]))
    assert_arithmetic(actual["free_energy"], direct_primal)
    assert_arithmetic(np.exp(source + actual["left_message"] + actual["free_energy"]), pair.sum(axis=1))
    assert_arithmetic(np.exp(destination + actual["right_message"] + actual["free_energy"]), pair.sum(axis=0))
    feature = np.append(x, 1.)
    for i in range(3):
        for l in range(2):
            first, second = np.zeros(3), np.zeros((3, 3))
            for j in range(l, 3):
                first += (.5 if j == l else -.5) * pair[i, j] * feature
                second += pair[i, j] * np.outer(feature, feature)
            assert_arithmetic(actual["linear"][i, l], first)
            assert_arithmetic(actual["precision"][i, l], omega(psi[i, l]**2) * second)
    np.testing.assert_array_equal(pair[:, 1], 0)


def test_fixed_origins_preserve_physical_same_coordinate_arrays():
    centered = load_case("centered_t3")
    ordinary = load_case("coupled_t3")
    result = solution("centered_t3")
    messages, xi = Coordinate(centered).unpack(result.coordinate)
    for t, bank in enumerate(messages):
        for side, message in enumerate(bank):
            c = centered.origins[t + side]
            a, h, p = message.arrays()
            new_a = a - np.einsum("ki,ki->k", h, c) - np.einsum("ki,kij,kj->k", c, p, c) / 2
            new_a -= new_a[0]
            bank[side] = Natural(new_a, h + np.einsum("kij,kj->ki", p, c), p.copy())
    same = evaluate(ordinary, messages, xi)
    expected = result.evaluation
    for key in ("regions", "separators"):
        for a, b in zip(same[key], expected[key]):
            for field in ("probability", "mean", "covariance", "second"):
                assert_arithmetic(getattr(a, field), getattr(b, field))
            assert_arithmetic(b.displacement + b.origin, b.mean)
    for a, b in zip(same["learning"], expected["learning"]):
        for key in ("target", "exposure", "linear", "precision"):
            assert_arithmetic(a[key], b[key])
    assert abs(same["free_energy"] - expected["free_energy"]) <= 512 * 15 * EPS


def test_held_coordinate_residuals_and_dual_gradient_are_independent_of_solving():
    model = load_case("coupled_t3")
    coordinate = Coordinate(model)
    vector = solution("coupled_t3").coordinate.copy()
    vector += .01 * np.sin(np.arange(len(vector)) + 1)
    e = coordinate.evaluate(vector)
    assert max(e["certificates"].values()) > TAU
    gradient = jacobian(lambda x: np.array([coordinate.evaluate(x)["dual_energy"]]), vector)[0]
    for index, (time, side, kind, location) in enumerate(coordinate.entries):
        if time + side != 1:
            assert abs(gradient[index]) < 64 * EPS**(2 / 3)
            continue
        opposite = e["regions"][1 - time]
        other_side = 0 if time == 0 else 1
        q = e["separators"][1]
        label = location[0]
        pair_mass = opposite.probability[label] if other_side == 0 else opposite.probability[:, label]
        if kind == 0:
            expected = q.probability[label] - pair_mass.sum()
        elif kind == 1:
            pair_mean = opposite.mean[label, :, other_side] if other_side == 0 else opposite.mean[:, label, other_side]
            expected = q.probability[label] * q.mean[label, 0] - pair_mass @ pair_mean
        else:
            pair_second = opposite.second[label, :, other_side, other_side] if other_side == 0 else opposite.second[:, label, other_side, other_side]
            expected = -.5 * (q.probability[label] * q.second[label, 0, 0] - pair_mass @ pair_second)
        assert gradient[index] == pytest.approx(expected, abs=64 * EPS**(2 / 3))


def test_start_sensitivity_uses_prospective_stopping_propagation():
    model = load_case("coupled_t3")
    first, second = solution("coupled_t3", 0.), solution("coupled_t3", 1.)
    propagation = stopping_allowance(model, first.coordinate)
    assert np.isfinite(propagation["coordinate_inverse_norm"])
    for key, a in output_arrays(first.evaluation).items():
        b = output_arrays(second.evaluation)[key]
        scaled = np.max(abs(a - b) / (1 + abs(a)))
        assert scaled <= propagation["groups"][key]["scaled_allowance"]


@pytest.mark.parametrize("name,reason", [("empty_source_t2", "empty support"), ("underflow_source_t2", "underflow")])
def test_invalid_support_never_reports_success(name, reason):
    result = solve(load_case(name))
    assert result.status == "diverged"
    assert reason in result.reason
    assert result.evaluation is None


def test_tiny_active_mass_does_not_hide_conditional_residuals():
    model = load_case("coupled_t2")
    model.sites[0].constant[1] = -600.
    coordinate = Coordinate(model)
    messages, xi = coordinate.unpack(solution("coupled_t2").coordinate)
    messages[0][0].linear[1, 0] += .01
    held = evaluate(model, messages, xi)
    assert held["separators"][0].probability[1] < np.sqrt(np.finfo(float).tiny)
    assert held["certificates"]["conditional_mean_residual"] > .001
    assert held["certificates"]["natural_residual"] > .001
    assert np.max(abs(held["label_mass_residuals"][0])) < TAU


def test_signed_cavity_is_allowed_when_complete_active_beliefs_are_proper():
    model = load_case("coupled_t2")
    model.sites[0].precision[0, 0, 0] = -.01
    held = Coordinate(model).evaluate(solution("coupled_t2").coordinate)
    assert held["proper_active_beliefs"]
    assert model.sites[0].precision[0, 0, 0] == -.01


def test_positive_auxiliary_target_underflow_is_not_stationarity():
    model = load_case("coupled_t2")
    model.weights[:] = 1e-200
    model.offsets[:] = 0
    with pytest.raises(ReferenceFailure, match="auxiliary target underflow"):
        Coordinate(model).evaluate(solution("coupled_t2").coordinate)


def test_conditioned_fixed_drive_inactive_source_has_zero_statistics():
    f = {k: np.asarray(v, dtype=float) for k, v in specification()["fixed_drive"].items()}
    f["source_log_site"][1] = -np.inf
    result = fixed_drive(**f)
    for key in ("xi", "pair", "linear", "precision"):
        np.testing.assert_array_equal(result[key][1], 0)


@pytest.mark.parametrize("failure", ["nan", "plus_inf", "negative_noise", "improper", "shape", "negative_xi"])
def test_malformed_and_improper_initialization_is_explicit(failure):
    model = load_case("coupled_t2")
    start = 0.
    if failure == "nan":
        model.weights[0, 0, 0, 0] = np.nan
    elif failure == "plus_inf":
        model.sites[0].constant[0] = np.inf
    elif failure == "negative_noise":
        model.noise[:] = -1
    elif failure == "improper":
        for site in model.sites:
            site.precision[:] = -100
    elif failure == "shape":
        model.weights = model.weights[..., :0]
    else:
        start = -1.
    result = solve(model, start)
    assert result.status == "diverged"
    assert result.reason
    assert result.attempts


def test_budget_exhaustion_and_inactive_coordinate_are_not_convergence():
    model = load_case("coupled_t3")
    result = solve(model, 0., max_iterations=0)
    assert result.status == "exhausted"
    assert result.evaluation["certificates"]["auxiliary_residual"] > TAU
    masked = load_case("source_masked_t2")
    messages, xi = Coordinate(masked).unpack(solution("source_masked_t2").coordinate)
    xi[0, 1, 0] = 1
    with pytest.raises(ReferenceFailure, match="inactive"):
        evaluate(masked, messages, xi)


def test_reference_imports_and_approximation_contract_stay_separate():
    for name in ("hybrid_joint_reference.py", "hybrid_joint_solver_reference.py", "hybrid_separator_reference.py"):
        tree = ast.parse(Path(__file__).with_name(name).read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(module and (module.startswith("cffg") or module.startswith("test_")) for module in imports)
    assert specification()["comparison"]["fixed_auxiliary_approximation_ceilings"] == {
        "pair": .002, "first_and_second_moments": .01, "free_energy": .002, "learning": .0001}
    # Existing independent falsifier is reported as temporal approximation loss.
    mean, destination = reflected_mixture_gap()
    assert abs(mean) > ARITHMETIC
    assert np.max(abs(destination - .5)) > ARITHMETIC
