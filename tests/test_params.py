"""Role-bearing runtime addresses and atomic named parameter writes."""

import dataclasses

import jax
import numpy as np
import pytest

from cffg import set_params
from cffg.compiler import NEG, MemberAddress, ParameterAddress, compile_template
from cffg.core import (
    BlobRef,
    Expose,
    FactorSpec,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.engine import Engine


def _blob(name: str) -> BlobRef:
    return BlobRef(name, f"blobs/{name}.npy")


def _address_template() -> tuple[Template, dict[str, np.ndarray]]:
    k, u, d = 3, 2, 2
    template = Template(name="parameter_addresses")

    def discrete(name: str, card: int = k) -> str:
        template.variables[name] = VariableSpec(name, card)
        return name

    def gaussian(name: str, dim: int = d) -> str:
        template.variables[name] = VariableSpec(name, dim=dim)
        return name

    template.factors["cat"] = FactorSpec(
        "cat",
        "categorical",
        edges={"out": discrete("cat_out")},
        param_blobs={"statistic": "cat_p"},
    )
    template.factors["matrix"] = FactorSpec(
        "matrix",
        "matrix_cat",
        edges={"out": discrete("matrix_out"), "in": discrete("matrix_in")},
        param_blobs={"matrix": "matrix_p"},
    )
    template.factors["transition"] = FactorSpec(
        "transition",
        "transition_mixture",
        edges={
            "out": discrete("transition_out"),
            "in": discrete("transition_in"),
            "selector": discrete("transition_selector", u),
        },
        param_blobs={"matrices": "transition_p"},
    )
    template.factors["gaussian"] = FactorSpec(
        "gaussian",
        "gaussian",
        edges={"out": gaussian("gaussian_out")},
        param_blobs={"mean": "gaussian_mean", "cov": "gaussian_cov"},
    )
    template.factors["linear"] = FactorSpec(
        "linear",
        "gaussian_linear",
        edges={"out": gaussian("linear_out"), "in": gaussian("linear_in")},
        param_blobs={
            "matrix": "linear_matrix",
            "offset": "linear_offset",
            "cov": "linear_cov",
        },
    )
    template.factors["linear_obs"] = FactorSpec(
        "linear_obs",
        "gaussian_linear",
        edges={
            "out": gaussian("linear_obs_out"),
            "in": gaussian("linear_obs_in"),
        },
        param_blobs={
            "matrix": "linear_obs_matrix",
            "offset": "linear_obs_offset",
            "cov": "linear_obs_cov",
        },
    )
    template.ports["linear_data"] = PortSpec(
        "linear_data", "linear_obs_out", PortRole.DATA
    )
    template.factors["mixture"] = FactorSpec(
        "mixture",
        "gaussian_mixture",
        edges={
            "out": gaussian("mixture_out"),
            "selector": discrete("mixture_selector"),
        },
        param_blobs={"mean": "mixture_mean", "cov": "mixture_cov"},
    )
    template.factors["mixture_linear"] = FactorSpec(
        "mixture_linear",
        "gaussian_mixture_linear",
        edges={
            "out": gaussian("mixture_linear_out"),
            "in": gaussian("mixture_linear_in"),
            "selector": discrete("mixture_linear_selector"),
        },
        param_blobs={
            "matrix": "mixture_linear_matrix",
            "offset": "mixture_linear_offset",
            "cov": "mixture_linear_cov",
        },
    )
    template.factors["recurrence"] = FactorSpec(
        "recurrence",
        "recurrence_link",
        edges={
            "out": discrete("recurrence_out"),
            "in": discrete("recurrence_in"),
            "drive": gaussian("recurrence_drive"),
        },
        param_blobs={"weights": "recurrence_weights", "offsets": "recurrence_offsets"},
    )
    template.variables["precision"] = VariableSpec("precision", ef="gamma")
    template.factors["precision_gaussian"] = FactorSpec(
        "precision_gaussian",
        "gaussian_prec",
        edges={"out": gaussian("precision_out", 1), "prec": "precision"},
        param_blobs={"mean": "precision_mean"},
    )
    ported = {port.variable for port in template.ports.values()}
    for variable in template.variables:
        if variable not in ported:
            template.ports[f"glue_{variable}"] = PortSpec(
                f"glue_{variable}", variable, PortRole.GLUE
            )

    arrays = {
        "cat_p": np.full(k, 1.0 / k),
        "matrix_p": np.full((k, k), 1.0 / k),
        "transition_p": np.full((u, k, k), 1.0 / k),
        "gaussian_mean": np.zeros(d),
        "gaussian_cov": np.eye(d),
        "linear_matrix": np.eye(d),
        "linear_offset": np.zeros(d),
        "linear_cov": np.eye(d),
        "linear_obs_matrix": np.eye(d),
        "linear_obs_offset": np.zeros(d),
        "linear_obs_cov": np.eye(d),
        "mixture_mean": np.zeros((k, d)),
        "mixture_cov": np.broadcast_to(np.eye(d), (k, d, d)).copy(),
        "mixture_linear_matrix": np.broadcast_to(np.eye(d), (k, d, d)).copy(),
        "mixture_linear_offset": np.zeros((k, d)),
        "mixture_linear_cov": np.broadcast_to(np.eye(d), (k, d, d)).copy(),
        "recurrence_weights": np.zeros((k, k - 1, d)),
        "recurrence_offsets": np.zeros((k, k - 1)),
        "precision_mean": np.zeros(1),
    }
    template.blobs = {name: _blob(name) for name in arrays}
    return template, arrays


def test_member_addresses_cover_every_named_write_family():
    template, arrays = _address_template()
    arena = compile_template(template, arrays=arrays)
    expected = {
        "cat": {"statistic": "probability_to_log"},
        "matrix": {"matrix": "probability_to_log"},
        "transition": {"matrices": "probability_to_log"},
        "gaussian": {"mean": "identity", "cov": "covariance_to_precision"},
        "linear": {
            "matrix": "identity",
            "offset": "identity",
            "cov": "covariance_to_precision",
        },
        "linear_obs": {
            "matrix": "identity",
            "offset": "identity",
            "cov": "covariance_to_precision",
        },
        "mixture": {"mean": "identity", "cov": "covariance_to_precision"},
        "mixture_linear": {
            "matrix": "identity",
            "offset": "identity",
            "cov": "covariance_to_precision",
        },
        "recurrence": {"weights": "identity", "offsets": "identity"},
        "precision_gaussian": {"mean": "identity"},
    }
    assert arena.writable_members == frozenset(expected)
    for factor, roles in expected.items():
        member = arena.member_of[factor]
        assert isinstance(member, MemberAddress)
        assert member.factor_type == template.factors[factor].type
        assert {address.role: address.transform for address in member.parameters} == roles
        assert all(isinstance(address, ParameterAddress) for address in member.parameters)
        assert all(address.blob == template.factors[factor].param_blobs[address.role] for address in member.parameters)
    with pytest.raises(dataclasses.FrozenInstanceError):
        arena.member_of["linear"].factor_type = "other"


def test_set_params_transforms_roles_and_preserves_inputs():
    template, arrays = _address_template()
    engine = Engine(compile_template(template, arrays=arrays))
    params = engine.init_state().params
    before = {key: np.asarray(value).copy() for key, value in params.items()}
    matrix = np.array([[1.2, -0.3], [0.4, 0.8]])
    offset = np.array([0.25, -0.5])
    covariance = np.array([[2.0, 0.3], [0.3, 0.7]])

    written = set_params(
        engine,
        params,
        {"linear_obs": {"matrix": matrix, "offset": offset, "cov": covariance}},
    )

    assert written is not params
    address = engine.arena.member_of["linear_obs"]
    np.testing.assert_array_equal(
        written[address.parameter("matrix").key][address.parameter("matrix").index],
        matrix,
    )
    np.testing.assert_array_equal(
        written[address.parameter("offset").key][address.parameter("offset").index],
        offset,
    )
    np.testing.assert_allclose(
        written[address.parameter("cov").key][address.parameter("cov").index],
        np.linalg.inv(covariance),
    )
    for key, value in params.items():
        np.testing.assert_array_equal(value, before[key])


def test_probability_writes_preserve_zero_and_transition_layout():
    template, arrays = _address_template()
    engine = Engine(compile_template(template, arrays=arrays))
    params = engine.init_state().params
    categorical = np.array([1.0, 0.0, 0.0])
    matrices = np.zeros((2, 3, 3))
    matrices[:, 0, :] = 1.0

    written = set_params(
        engine,
        params,
        {
            "cat": {"statistic": categorical},
            "transition": {"matrices": matrices},
        },
    )
    cat = engine.arena.member_of["cat"].parameter("statistic")
    transition = engine.arena.member_of["transition"].parameter("matrices")
    np.testing.assert_array_equal(
        written[cat.key][cat.index], np.array([0.0, NEG, NEG])
    )
    np.testing.assert_array_equal(
        written[transition.key][transition.index],
        np.where(matrices.transpose(1, 2, 0) > 0.0, 0.0, NEG),
    )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"missing": {"mean": np.zeros(2)}}, "unknown or parameterless"),
        ({"linear": {"unknown": np.zeros(2)}}, "has roles"),
        ({"linear": {"matrix": np.zeros((3, 3))}}, "blob shape"),
        ({"linear": {"offset": np.array([0.0, np.nan])}}, "finite"),
        ({"linear": {"matrix": [["x", "y"], ["z", "w"]]}}, "real numeric"),
        ({"cat": {"statistic": np.array([0.2, 0.2, 0.2])}}, "sum to one"),
        ({"cat": {"statistic": np.array([1.0, 0.1, -0.1])}}, "nonnegative"),
        ({"linear": {"cov": np.array([[1.0, 0.2], [0.1, 1.0]])}}, "symmetric"),
        ({"linear": {"cov": np.array([[1.0, 2.0], [2.0, 1.0]])}}, "positive definite"),
    ],
)
def test_set_params_rejects_invalid_updates_atomically(updates, match):
    template, arrays = _address_template()
    engine = Engine(compile_template(template, arrays=arrays))
    params = engine.init_state().params
    before = {key: np.asarray(value).copy() for key, value in params.items()}
    updates = {"gaussian": {"mean": np.ones(2)}, **updates}
    with pytest.raises((KeyError, TypeError, ValueError), match=match):
        set_params(engine, params, updates)
    for key, value in params.items():
        np.testing.assert_array_equal(value, before[key])


def _tied_gaussians(count: int = 64, dimension: int = 16):
    template = Template(name="tied_gaussians")
    arrays = {
        "shared_mean": np.zeros(dimension),
        "shared_cov": np.eye(dimension),
        "private_mean": np.full(dimension, 7.0),
        "private_cov": np.eye(dimension) * 3.0,
    }
    template.blobs = {name: _blob(name) for name in arrays}
    for index in range(count):
        variable = f"x{index:02d}"
        factor = f"prior{index:02d}"
        template.variables[variable] = VariableSpec(variable, dim=dimension)
        template.factors[factor] = FactorSpec(
            factor,
            "gaussian",
            edges={"out": variable},
            param_blobs={"mean": "shared_mean", "cov": "shared_cov"},
        )
        template.ports[variable] = PortSpec(variable, variable, PortRole.GLUE)
    template.variables["unrelated_x"] = VariableSpec("unrelated_x", dim=dimension)
    template.factors["unrelated"] = FactorSpec(
        "unrelated",
        "gaussian",
        edges={"out": "unrelated_x"},
        param_blobs={"mean": "private_mean", "cov": "private_cov"},
    )
    template.ports["unrelated_x"] = PortSpec(
        "unrelated_x", "unrelated_x", PortRole.GLUE
    )
    return template, arrays


def test_tied_updates_fan_out_in_one_group_without_coupling_other_members():
    template, arrays = _tied_gaussians()
    engine = Engine(compile_template(template, arrays=arrays))
    params = engine.init_state().params
    unrelated = engine.arena.member_of["unrelated"]
    unrelated_before = {
        role: np.asarray(params[unrelated.parameter(role).key][unrelated.parameter(role).index]).copy()
        for role in ("mean", "cov")
    }
    mean = np.linspace(-1.0, 1.0, 16)
    covariance = np.eye(16) * 2.5

    written = set_params(
        engine,
        params,
        {"prior00": {"mean": mean, "cov": covariance}},
    )

    for index in range(64):
        address = engine.arena.member_of[f"prior{index:02d}"]
        np.testing.assert_array_equal(
            written[address.parameter("mean").key][address.parameter("mean").index],
            mean,
        )
        np.testing.assert_array_equal(
            written[address.parameter("cov").key][address.parameter("cov").index],
            np.eye(16) / 2.5,
        )
    for role in ("mean", "cov"):
        address = unrelated.parameter(role)
        np.testing.assert_array_equal(
            written[address.key][address.index], unrelated_before[role]
        )

    with pytest.raises(ValueError, match="tied blob.*unequal"):
        set_params(
            engine,
            params,
            {
                "prior00": {"mean": np.zeros(16)},
                "prior01": {"mean": np.ones(16)},
            },
        )


def test_float32_and_plate_scoped_writes_preserve_destination_dtype():
    dimension = 2
    fragment = Template(
        name="sensor",
        variables={
            "out": VariableSpec("out", dim=dimension),
            "in": VariableSpec("in", dim=dimension),
        },
        factors={
            "emit": FactorSpec(
                "emit",
                "gaussian_linear",
                edges={"out": "out", "in": "in"},
                param_blobs={"matrix": "A", "offset": "b", "cov": "cov"},
            )
        },
        ports={
            "out": PortSpec("out", "out", PortRole.DATA),
            "in": PortSpec("in", "in", PortRole.GLUE),
        },
        blobs={name: _blob(name) for name in ("A", "b", "cov")},
    )
    template = Template(
        name="plate",
        plates={"sensor": PlateSpec(fragment, (3,), (Expose(),))},
    )
    arrays = {
        "A": np.eye(dimension, dtype=np.float32),
        "b": np.zeros(dimension, dtype=np.float32),
        "cov": np.eye(dimension, dtype=np.float32),
    }
    with jax.enable_x64(False):
        engine = Engine(compile_template(template, arrays=arrays))
        written = set_params(
            engine,
            engine.init_state().params,
            {
                "sensor/0/emit": {
                    "matrix": np.eye(dimension, dtype=np.float32) * 2.0,
                    "offset": np.ones(dimension, dtype=np.float32),
                    "cov": np.eye(dimension, dtype=np.float32) * 4.0,
                }
            },
        )
    for index in range(3):
        member = engine.arena.member_of[f"sensor/{index}/emit"]
        for role in ("matrix", "offset", "cov"):
            address = member.parameter(role)
            assert written[address.key].dtype == np.float32
        np.testing.assert_array_equal(
            written[member.parameter("offset").key][member.parameter("offset").index],
            np.ones(dimension, dtype=np.float32),
        )
