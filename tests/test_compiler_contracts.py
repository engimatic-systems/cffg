"""Observable compiler contracts at bf6bbe9527b5e1eea9172a30b8c7c214a08ed9ee.

These small existing models pin structure before compiler extraction, not new
mathematics. Literal addresses are paired with readable role/member assertions;
hashes seal serialized identity and the schedule that consumes those addresses.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import jax
import numpy as np
import pytest
from conftest import make_hmm_slice
from test_csp import priors_plus_constraints
from test_dirichlet_param_edges import _make_template as parameter_edge_model
from test_discrete_kernels import CUSTOM, declaration
from test_discrete_kernels import model as external_model
from test_ef_engine import gamma_poisson
from test_gfe import make_model
from test_hybrid_engine import mixlin_template
from test_hybrid_gfe import _probe_template
from test_kalman import M0, P0, F, H, Q, R
from test_learnables import two_chain_template
from test_mu3 import _arrays as mu3_arrays
from test_mu3 import _model as mu3_model
from test_plates import pixel_fragment
from test_recurrence_link import _template as recurrence_model
from test_schedule import hmm_arrays, hmm_template
from test_structured_recurrence_engine import _loop_template

from cffg import kernels
from cffg._immutable import FrozenArray
from cffg.compiler import CompileError, compile_template
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    Expose,
    FactorSpec,
    LearnableSpec,
    ModelError,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.serialize import from_json, structure_hash, to_dict, to_json
from cffg.engine import Engine
from cffg.models.kalman import lgssm
from cffg.models.rollout import build_rollout
from cffg.schedule import build_schedule

# Full hashes pin identity; the tests below explain the corresponding layouts.
IDENTITIES = {
    "hmm": (
        "eb22fa7a661a22c9be4f439d43acab4ebecff61767bd16c6341fab67cd62114e",
        "two_pass",
        "6b4688aedafb0cf86fa70a297f53495149cb52e95da9700c21ef6ac3bba587e9",
    ),
    "gaussian": (
        "9c4e9686801b4784b9f2e3cbc48f94e7c3571d6f43d5ed2978eb4804b144e9f4",
        "two_pass",
        "718e88e4b6d4da6ff05075694bbe69182c63842a62881a563e4e887668113711",
    ),
    "ef": (
        "1dfadb6af1f42e1573f76b43a97fc459de6bcc762335bac1930b540903d35d02",
        "two_pass",
        "40271dfdc059685ea6dc794b8bea6b9ff3d3493f19f7c54421d93598294a6d3b",
    ),
    "gfe": (
        "bc3613824fcef5de449f58866ac44f3ecad7a7b0e3d04e9fb7a9f625a95952d4",
        "mixed",
        "cd5323696b300b5b6ec48216e3570461f25e7300837697365f76b7ae5aca320c",
    ),
    "mixture": (
        "ce68e96418b43f46a927c6270403a7654b7097b661fe826bf834df81662ee771",
        "mixed",
        "045179ce0decd8fe8a792c04605253b54186f9813496b31849d61b38f76eca85",
    ),
    "hybrid_gfe": (
        "42468b5561d7d2dbff3c2c7b0bd104e43b66d4e1cb8fe70d8b5bccd3ce153320",
        "mixed",
        "63d7530caf57d326c30f64e59a8416cc3e1a32d8de83fbb94d0043e27f700eb2",
    ),
    "recurrence": (
        "2b34a6a06d116d335db9fd1926a737720eeeccf4deb97bfe23f7a575a0a539c6",
        "mixed",
        "93fd9ef32ace50b1620c0058297c08d394145d3164a6fe1eca29b92d662efb99",
    ),
    "recurrence_loop": (
        "e5f04735e7042fffb109a1cde7d139eda8e50aff2b90d3ed6a7e3611e24ff7c4",
        "mixed",
        "81d94a6ed87815a29eefea3bbeea645bec54cafc0381c0fef06cdce0f4710154",
    ),
    "mu3": (
        "d16d00bab50b8429f299dc1b9fa9ec2668f7e43e88e174190de2968868f429cf",
        "mixed",
        "c44f455d3bc71f68dad130ff7b0d14344bdc1818793867936b386c5c35ccfb54",
    ),
    "learning": (
        "9d7b679129879a07c310122a6aa831e07c7a723d877260520969bd3a30f7b07d",
        "two_pass",
        "c0b3f1a9c468ca043a5735f5970cb83f1b916fc4895e57fa7a92da945e75ed5c",
    ),
    "agent": (
        "afde0a8e53be36afd58662c39ef36c27b9740412a8133a243018b157ae94f161",
        "mixed",
        "df58038adff8df1ee92ba318b91e0b0303c04d2846cdecb63ba676bd38386c5c",
    ),
    "plate": (
        "f72db8da29e562fd2aba785ad6230d3d6488bc90f94f7d990f0180314c02bb76",
        "two_pass",
        "84e20e24d2ffd96cf428911c0233214dd44a1663a498339019e1c20fb22c29f3",
    ),
    "constraint": (
        "e28b82447258aa89da399b5b7ca69022de16a67621f3811ba158a3a8c261c8fc",
        "mixed",
        "cdc30356603aa9d05a0c6b06a7e300dfe6b8d5f760ee631d955687e155f1f665",
    ),
    "parameter_edge": (
        "9ab84a54ddd105eafc1c7b032d7ccf8b9105f714d20405e9e9b479c727f1edcf",
        "mixed",
        "94103ea465ebd0803c11e1e44c2ab3e9fc8f903f82eac43c1bb5abb6530cebfc",
    ),
}


def model(name):
    if name == "parameter_edge":
        return parameter_edge_model("matrix_cat", (3, 2), latent_parents=True), {
            "alpha0": np.arange(1, 7, dtype=float).reshape(3, 2),
            "d_in": np.array([0.4, 0.6]),
        }
    if name == "hmm":
        return hmm_template(1), hmm_arrays(1)
    if name == "gaussian":
        return lgssm(2, F, Q, H, R, M0, P0)
    if name == "ef":
        return gamma_poisson(2)
    if name == "gfe":
        arrays = {
            "A": np.array([[0.8, 0.2], [0.2, 0.8]]),
            "c": np.array([0.7, 0.3]),
            "d": np.array([0.4, 0.6]),
        }
        return make_model(**arrays), arrays
    if name == "mixture":
        return mixlin_template(K=3, do=2, di=1)
    if name == "hybrid_gfe":
        _, template, arrays = _probe_template()
        return template, arrays
    if name == "recurrence":
        return recurrence_model(weights=np.zeros((3, 2, 3)), offsets=np.zeros((3, 2)))
    if name == "recurrence_loop":
        return _loop_template()
    if name == "mu3":
        return mu3_model(slices=2), mu3_arrays(slices=2)
    if name == "learning":
        return two_chain_template()
    if name == "agent":
        return build_rollout(
            "bound",
            steps=2,
            d=np.array([0.7, 0.3]),
            A=np.eye(2),
            c=np.array([0.4, 0.6]),
            e=np.array([0.5, 0.5]),
            Bs=np.stack([np.eye(2), np.eye(2)]),
            learn=("d",),
        )
    if name == "plate":
        return Template(
            "grid", plates={"pix": PlateSpec(pixel_fragment(), (2,), (Expose(),))}
        ), None
    if name == "constraint":
        template, arrays = priors_plus_constraints(
            3,
            2,
            [np.array([0.4, 0.6])] * 3,
            FactorSpec(
                "c",
                "exactly_k",
                {f"x{i}": f"x{i}" for i in range(3)},
                param_blobs={"k": "kk"},
            ),
            extra_blobs={"kk": BlobRef("kk", "kk.npy")},
        )
        return template, {**arrays, "kk": np.array(1)}
    raise AssertionError(f"unknown fixture {name!r}")


def assert_fields(value, **expected):
    for name, wanted in expected.items():
        actual = getattr(value, name)
        if isinstance(actual, (np.ndarray, FrozenArray)):
            np.testing.assert_array_equal(actual, wanted, err_msg=name)
        else:
            assert actual == wanted, name


def assert_same(left, right):
    """Compare live values field by field; no snapshot serialization format."""
    if isinstance(left, (np.ndarray, FrozenArray)):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, kernels.FactorKernel):
        assert left is right
    elif is_dataclass(left):
        assert type(left) is type(right)
        for field in fields(left):
            assert_same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, Mapping):
        assert tuple(left) == tuple(right)
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    else:
        assert left == right


def compiled(name):
    template, arrays = model(name)
    return compile_template(template, arrays=arrays)


@pytest.mark.parametrize("name", IDENTITIES)
def test_baseline_identity_and_complete_reconstructed_plans(name):
    template, arrays = model(name)
    serialized = to_json(template)
    arena = compile_template(template, arrays=arrays)
    restored = from_json(serialized)
    rebuilt = compile_template(restored, arrays=arrays)
    assert to_json(template) == serialized
    expected_hash, kind, fingerprint = IDENTITIES[name]
    assert structure_hash(template) == structure_hash(restored) == expected_hash
    schedule = build_schedule(arena)
    assert (schedule.kind, schedule.fingerprint) == (kind, fingerprint)
    assert_same(schedule, build_schedule(rebuilt))
    assert build_schedule(arena, "flood").kind == "flood"
    assert_same(arena, rebuilt)


@pytest.mark.parametrize("name", ["hmm", "gaussian", "ef", "mixture", "learning"])
def test_source_dtype_and_jax_precision_do_not_change_compiler_structure(name):
    template, arrays = model(name)
    with jax.enable_x64(True):
        baseline = compile_template(template, arrays=arrays)
    with jax.enable_x64(False):
        narrow = compile_template(
            template, arrays={k: v.astype(np.float32) for k, v in arrays.items()}
        )
    # Resolution widens source values; tiny rounding differences are not layout.
    assert_same(replace(baseline, log_params={}), replace(narrow, log_params={}))
    assert {k: v.shape for k, v in baseline.log_params.items()} == {
        k: v.shape for k, v in narrow.log_params.items()
    }
    assert_same(build_schedule(baseline), build_schedule(narrow))
    assert structure_hash(template) == IDENTITIES[name][0]


def test_hmm_group_role_order_and_discrete_prefix_offsets():
    arena = compiled("hmm")
    expected = [
        ("categorical", ("out",), ("prior/init",), [[8]], [[9]]),
        ("equality", ("a", "b", "c"), ("chain/t0/eq",), [[3, 5, 7]], [[2, 4, 6]]),
        ("matrix_cat", ("out", "in"), ("chain/t0/obs",), [[1, 4]], [[0, 5]]),
        ("matrix_cat", ("out", "in"), ("chain/t0/trans",), [[2, 9]], [[3, 8]]),
    ]
    for group, (kind, roles, members, gather, scatter) in zip(
        arena.groups, expected, strict=True
    ):
        assert type(group).__name__ == "GroupPlan"
        assert_fields(
            group,
            type=kind,
            roles=roles,
            members=members,
            gather=gather,
            scatter=scatter,
        )
        np.testing.assert_array_equal(
            group.gather_both, np.stack([scatter, gather], axis=-1)
        )
    assert_fields(
        arena,
        n_slots=10,
        message_starts=[0, 2, 4, 7, 10, 13, 16, 19, 22, 25, 28],
        data_slots={"chain/t0/sense": 1},
    )


def test_gaussian_conditioning_and_parameter_rows():
    arena = compiled("gaussian")
    expected = [
        ("equality", ("eq0",), [[1, 3, 4]], [[0, 2, 5]]),
        ("gaussian", ("prior",), [[0]], [[1]]),
        ("gaussian_linear", ("trans1",), [[6, 2]], [[7, 3]]),
        (
            "gaussian_linear_obs",
            ("emis0", "emis1"),
            [[9, 5], [11, 7]],
            [[8, 4], [10, 6]],
        ),
    ]
    for group, (kind, members, gather, scatter) in zip(
        arena.gauss_groups, expected, strict=True
    ):
        assert type(group).__name__ == "GaussGroupPlan"
        assert_fields(group, type=kind, members=members, gather=gather, scatter=scatter)
    assert_fields(
        arena.gauss_groups[-1],
        roles=("out", "in"),
        dims=(1, 2),
        observed_role=0,
        obs_rows=[0, 1],
    )
    assert_fields(
        arena,
        n_gslots=12,
        n_obs_rows=2,
        gauss_data={"y0": (0, 1), "y1": (1, 1)},
        g_eta_starts=[0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20],
        g_lam_starts=[0, 4, 8, 12, 16, 20, 24, 28, 32, 33, 34, 35, 36],
        g_obs_starts=[0, 1, 2],
    )
    for index in range(2):
        address = arena.member_of[f"emis{index}"]
        assert address.factor_type == "gaussian_linear"
        for parameter, (role, blob, key, transform) in zip(
            address.parameters,
            (
                ("cov", "R", "W", "covariance_to_precision"),
                ("matrix", "H", "A", "identity"),
                ("offset", "zm", "b", "identity"),
            ),
            strict=True,
        ):
            assert_fields(
                parameter,
                role=role,
                blob=blob,
                key=f"gauss:gaussian_linear_obs:1x2:{key}",
                index=index,
                transform=transform,
                probability_axis=None,
                permutation=(),
            )


def test_ef_observation_rows_and_independent_slots():
    arena = compiled("ef")
    expected = [
        ("equality", ("a", "b", "c"), ("eq0",), [[1, 5, 3]], [[0, 4, 2]]),
        ("gamma", ("out",), ("prior",), [[0]], [[1]]),
        ("poisson_obs", ("rate",), ("obs0", "obs1"), [[4], [2]], [[5], [3]]),
    ]
    for group, (kind, roles, members, gather, scatter) in zip(
        arena.ef_groups, expected, strict=True
    ):
        assert type(group).__name__ == "EfGroupPlan"
        assert_fields(
            group,
            type=kind,
            family="gamma",
            roles=roles,
            members=members,
            gather=gather,
            scatter=scatter,
        )
    assert_fields(arena.ef_groups[-1], obs_rows=[0, 1])
    assert_fields(
        arena,
        n_efslots=10,
        n_ef_obs=2,
        n_slots=0,
        n_gslots=0,
        ef_data={"count0": (0, "poisson"), "count1": (1, "poisson")},
    )


def test_learning_sources_targets_and_serialized_axis_permutation():
    arena = compiled("learning")
    (plan,) = arena.learn_plans
    assert_fields(
        plan,
        name="B",
        family="dirichlet",
        ftype="transition_mixture",
        shapes={"Bs": (2, 3, 3)},
        sources=("tma", "tmb"),
    )
    for index, (member, target) in enumerate(
        zip(plan.sources, plan.targets, strict=True)
    ):
        key = "bp:transition_mixture:3x3x2"
        assert_fields(
            target,
            key=key,
            index=index,
            form="expected",
            layout="tm",
            blob="Bs",
            role="matrices",
            transform="probability_to_log",
        )
        (parameter,) = arena.member_of[member].parameters
        assert_fields(
            parameter,
            key=key,
            index=index,
            blob="Bs",
            role="matrices",
            transform="probability_to_log",
            probability_axis=1,
            permutation=(1, 2, 0),
        )
    template, arrays = model("learning")
    template.learnables["B"] = replace(template.learnables["B"], sources=("tmb",))
    (subset,) = compile_template(template, arrays=arrays).learn_plans
    assert subset.sources == ("tmb",)
    assert subset.targets == plan.targets


def test_agent_spec_remains_serialized_structure_not_compiled_addresses():
    template, arrays = model("agent")
    encoded = to_dict(template)
    assert encoded["agent"] == {
        "action": "u1",
        "beliefs": {
            "state": {
                "blocks": ["state"],
                "prior": "prior",
                "posterior": "zm1",
                "factorization": "joint",
                "episode_learnable": "d",
            }
        },
    }
    (plan,) = compile_template(from_json(to_json(template)), arrays=arrays).learn_plans
    assert_fields(plan, sources=("prior",), shapes={"d": (2,)})
    assert_fields(
        plan.targets[0],
        key="bp:categorical:2",
        index=2,
        form="expected",
        layout="as_is",
        blob="d",
        role="statistic",
        transform="probability_to_log",
    )
    assert to_dict(template) == encoded
    assert "member_of" not in encoded


@pytest.mark.parametrize("functional", [False, True])
def test_learned_gfe_count_targets_never_become_statistic_sources(functional):
    template = mu3_model(slices=2, parameter_message=functional)
    template.learnables["A"] = LearnableSpec("A", "dirichlet", ("Ac",))
    arena = compile_template(template, arrays=mu3_arrays(slices=2))
    (plan,) = arena.learn_plans
    assert_fields(
        plan,
        name="A",
        family="dirichlet",
        ftype="matrix_cat",
        shapes={"Ac": (2, 2)},
        sources=(),
    )
    assert len(plan.targets) == 2
    key = f"{'gfeLM' if functional else 'gfeL'}:2x2:counts"
    for index, target in enumerate(plan.targets):
        assert_fields(
            target,
            key=key,
            index=index,
            form="counts",
            layout="as_is",
            blob="Ac",
            role="matrix",
            transform="identity",
        )


def test_compact_plate_expansion_preserves_member_and_parameter_order():
    arena = compiled("plate")
    prior, observation = arena.groups
    assert_fields(
        prior,
        members=("pix/0/prior", "pix/1/prior"),
        gather=[[2], [6]],
        scatter=[[3], [7]],
    )
    assert_fields(
        observation,
        members=("pix/0/obs", "pix/1/obs"),
        gather=[[1, 3], [5, 7]],
        scatter=[[0, 2], [4, 6]],
    )
    assert_fields(
        arena,
        data_slots={"pix/0/x": 1, "pix/1/x": 5},
        message_starts=[0, 2, 4, 7, 10, 12, 14, 17, 20],
    )
    for index, member in enumerate(observation.members):
        assert_fields(
            arena.member_of[member].parameters[0],
            blob="A",
            index=index,
            key="bp:matrix_cat:2x3",
            role="matrix",
        )


def test_hard_constraint_threshold_and_endpoint_order():
    arena = compiled("constraint")
    (plan,) = arena.constraint_groups
    assert type(plan).__name__ == "ConstraintPlan"
    assert_fields(
        plan,
        name="c",
        type="exactly_k",
        members=("c",),
        cards=(2, 2, 2),
        gather=[1, 3, 5],
        scatter=[0, 2, 4],
        parameter_key="constraint:c:k",
    )
    assert not hasattr(plan, "k") and not hasattr(plan, "table")
    np.testing.assert_array_equal(arena.log_params[plan.parameter_key], 1)


def test_parameter_edges_have_count_slots_not_categorical_entropy():
    arena = compiled("parameter_edge")
    (plan,) = arena.dirichlet_groups
    assert type(plan).__name__ == "DirichletGroupPlan"
    assert_fields(
        plan,
        name="dirvmp:matrix_cat:3x2",
        type="matrix_cat",
        roles=("out", "in"),
        cards=(3, 2),
        members=("lik",),
        priors=("theta_prior",),
        gather_both=[[[2, 3], [0, 1]]],
        scatter=[[2, 0]],
        prior_scatter=[5],
        count_scatter=[4],
        param_shape=(3, 2),
        output_axis=0,
    )
    assert_fields(
        arena.edges["theta"],
        slot0=4,
        slot1=5,
        interior=True,
        card=6,
        param_shape=(3, 2),
        param_axis=0,
    )
    assert_fields(
        arena,
        message_starts=[0, 2, 4, 7, 10, 16, 22],
        data_slots={"out": 3},
        interior_slot_pairs=[[0, 1]],
    )
    assert tuple(arena.log_params) == (
        "dirvmp:matrix_cat:3x2:alpha0",
        "bp:categorical:2",
    )


def test_fixed_a_gfe_owns_substituted_edge_and_goal():
    arena = compiled("gfe")
    assert len(arena.gfe_groups) == 1
    assert_fields(
        arena.gfe_groups[0],
        name="gfe:2x2",
        cards=(2, 2),
        members=("obs",),
        param_blobs=("A",),
        learned=False,
        mu3=False,
        goal_learned=False,
        gather_z=[3],
        scatter_z=[2],
        scatter_x=[1],
    )
    assert [g.members for g in arena.groups] == [("prior",)]
    assert_fields(arena, message_starts=[0, 2, 4, 6, 8], interior_slot_pairs=[[2, 3]])
    assert arena.gfe_x_edges == {"x"}
    assert arena.edges["x"].interior


def test_mixture_and_hybrid_composite_cross_arena_addresses():
    arena = compiled("mixture")
    assert_fields(
        arena.mix_groups[0],
        name="mix:gaussian_mixture_linear:2x1:3",
        type="gaussian_mixture_linear",
        dims=(2, 1),
        k_sel=3,
        members=("mix",),
        g_gather=[[3, 1]],
        g_scatter=[[2, 0]],
        s_gather=[1],
        s_scatter=[0],
    )
    assert_fields(arena, g_eta_starts=[0, 1, 2, 4, 6], g_lam_starts=[0, 1, 2, 6, 10])
    arena = compiled("hybrid_gfe")
    assert_fields(
        arena.hybrid_gfe_groups[0],
        name="gfeH:gaussian_mixture_linear:1x1:2",
        type="gaussian_mixture_linear",
        dims=(1, 1),
        k_sel=2,
        members=("emission",),
        goal_members=("goal",),
        g_gather=[[1, 3]],
        g_scatter=[[0, 2]],
        s_gather=[1],
        s_scatter=[0],
    )
    assert not arena.mix_groups
    assert [g.members for g in arena.gauss_groups] == [("state",)]
    assert_same(arena.gauss_interior, [(1, np.array([[2, 3]], dtype=np.int64))])
    assert arena.gfe_x_edges == {"x"}
    assert tuple(
        k.rsplit(":", 1)[1] for k in arena.log_params if k.startswith("gfeH:")
    ) == (
        "A",
        "b",
        "W",
        "goal_m",
        "goal_W",
    )


def test_recurrence_roles_and_component_rows():
    arena = compiled("recurrence")
    assert_fields(
        arena.recurrence_groups[0],
        name="recurrence:3:3",
        cards=(3, 3),
        dim=3,
        members=("recurrence",),
        d_gather_both=[[[3, 2], [1, 0]]],
        d_scatter=[[3, 1]],
        drive_gather_both=[[1, 0]],
        drive_scatter=[1],
        component_ids=[0],
    )
    assert_fields(
        arena.recurrence_components[0],
        members=("recurrence",),
        categorical_factors=("in_prior", "out_prior", "recurrence"),
        acyclic=True,
    )
    arena = compiled("recurrence_loop")
    assert_fields(
        arena.recurrence_components[0],
        members=("switch0", "switch1"),
        categorical_factors=("switch0", "switch1"),
        acyclic=False,
    )
    assert_fields(
        arena.recurrence_groups[0],
        members=("switch0", "switch1"),
        component_ids=[0, 0],
        d_gather_both=[[[2, 3], [0, 1]], [[1, 0], [3, 2]]],
        d_scatter=[[2, 0], [1, 3]],
        drive_gather_both=[[1, 0], [3, 2]],
        drive_scatter=[1, 3],
    )


def test_functional_mu3_links_rows_to_one_count_array():
    arena = compiled("mu3")
    assert_fields(
        arena.gfe_groups[0],
        name="gfeLM:2x2",
        members=("obs0", "obs1"),
        param_blobs=("Ac", "Ac"),
        learned=True,
        mu3=True,
        gather_z=[5, 7],
        scatter_z=[4, 6],
        scatter_x=[1, 3],
    )
    assert len(arena.mu3_groups) == 1
    assert_fields(
        arena.mu3_groups[0],
        name="mu3:gfeLM:2x2:Ac",
        evaluator="cffg.mu3_is.v1",
        gfe_name="gfeLM:2x2",
        rows=[0, 1],
        members=("obs0", "obs1"),
        param_blob="Ac",
        count_key="gfeLM:2x2:counts",
        cards=(2, 2),
    )
    assert tuple(arena.log_params) == (
        "gfeLM:2x2:counts",
        "gfeLM:2x2:c",
        "bp:categorical:2",
    )


@pytest.fixture
def external_kernel():
    kernel = kernels.register_kernel(declaration())
    yield kernel
    kernels.unregister_kernel(CUSTOM)


def test_external_kernel_role_and_parameter_order(external_kernel):
    template, arrays = external_model()
    arena = compile_template(template, arrays=arrays)
    group = next(g for g in arena.groups if g.kernel is external_kernel)
    assert_fields(
        group,
        name="bp:bilinear_spin_color:2x3",
        roles=("spin", "color"),
        cards=(2, 3),
        members=("interaction",),
        gather=[[3, 0]],
        scatter=[[2, 1]],
    )
    assert group.parameter_paths == tuple(
        (key, f"bp:bilinear_spin_color:2x3:{key}")
        for key in ("strength", "spin_tilt", "color_tilt", "allowed")
    )
    assert (
        structure_hash(template)
        == "70a1048302b2c26382773e2df62a0dc11b4b8cf799669bfe3d10c72004d3d5b5"
    )
    assert (
        build_schedule(arena).fingerprint
        == "0fe936d929e64b3c0d69323a1ca01b6b46dff301c2073696316acacf04544193"
    )
    assert_same(arena, compile_template(from_json(to_json(template)), arrays=arrays))


def test_32_distinct_groups_have_deterministic_unpadded_addresses():
    template = Template("many-cardinalities")
    arrays = {}
    for index in reversed(range(32)):
        name, card = f"v{index:02}", index + 2
        template.variables[name] = VariableSpec(name, card)
        template.factors[name] = FactorSpec(
            name, "categorical", {"out": name}, param_blobs={"statistic": name}
        )
        template.blobs[name] = BlobRef(name, f"{name}.npy")
        template.ports[name] = PortSpec(name, name, PortRole.GLUE)
        arrays[name] = np.full(card, 1 / card)
    arena = compile_template(template, arrays=arrays)
    assert len(arena.groups) == 32
    lengths = np.repeat(np.arange(2, 34), 2)
    np.testing.assert_array_equal(arena.message_starts, np.r_[0, np.cumsum(lengths)])
    assert arena.n_slots == 64
    for index, group in enumerate(arena.groups):
        name = f"v{index:02}"
        assert_fields(
            group,
            name=f"bp:categorical:{index + 2}",
            members=(name,),
            cards=(index + 2,),
            gather=[[2 * index + 1]],
            scatter=[[2 * index]],
        )
        assert_fields(
            arena.member_of[name].parameters[0], blob=name, index=0, key=group.name
        )
    reordered = from_json(to_json(template))
    assert_same(
        arena, compile_template(reordered, arrays=dict(reversed(tuple(arrays.items()))))
    )
    assert (
        build_schedule(arena).fingerprint
        == build_schedule(compile_template(reordered, arrays=arrays)).fingerprint
    )


@pytest.mark.parametrize(
    ("change", "problems"),
    [
        ({"type": "quantum"}, ["factor 'trans': unknown type 'quantum'"]),
        (
            {"edges": {"in": "z_in", "sideways": "z_mid"}},
            [
                "factor 'trans': edge roles ['in', 'sideways'] != required ['in', 'out']",
            ],
        ),
        (
            {"param_blobs": {"wrong": "B"}},
            [
                "factor 'trans': param role 'matrix' must be bound by exactly one of edge/blob, got 0",
                "factor 'trans': unknown param role 'wrong'",
            ],
        ),
    ],
)
def test_structural_diagnostics_preserve_problem_order(change, problems):
    template = make_hmm_slice()
    template.factors["trans"] = replace(template.factors["trans"], **change)
    with pytest.raises(ModelError) as caught:
        compile_template(template)
    assert caught.value.problems == problems


def test_legacy_moment_matching_round_trips_but_still_cannot_compile():
    template = make_hmm_slice()
    template.constraints.append(ConstraintSpec(ConstraintKind.MOMENT_MATCHING, "trans"))
    restored = from_json(to_json(template))
    assert restored.constraints == template.constraints
    with pytest.raises(CompileError) as caught:
        compile_template(restored)
    assert str(caught.value) == (
        "cannot compile 'hmm_slice':\n  - factor 'trans': legacy moment-matching syntax "
        "does not identify an incident edge endpoint or projection"
    )


def test_incompatible_learning_sources_fail_at_compile_time():
    template, arrays = model("learning")
    template.learnables["B"] = LearnableSpec("B", "dirichlet", ("Bs", "A"))
    with pytest.raises(CompileError) as caught:
        compile_template(template, arrays=arrays)
    assert str(caught.value) == (
        "learnable 'B': blobs ['A', 'Bs'] are read by mixed factor types "
        "['matrix_cat', 'transition_mixture']; one posterior needs one sufficient statistic"
    )


def test_unknown_functional_evaluator_is_a_compile_error():
    template = mu3_model()
    template.constraints[0] = ConstraintSpec(
        ConstraintKind.P_SUBSTITUTION,
        "obs0",
        (
            ("parameter_message", "unknown_eval"),
            ("substituted_role", "out"),
        ),
    )
    with pytest.raises(CompileError) as caught:
        compile_template(template, arrays=mu3_arrays())
    assert str(caught.value) == (
        "cannot compile 'mu3_1':\n  - p_substitution on 'obs0': "
        "unknown parameter_message 'unknown_eval' (known: 'mu3_is_v1')"
    )


def test_incompatible_conjugate_family_factor_pair_is_rejected():
    template, arrays = model("gaussian")
    template.learnables["bad"] = LearnableSpec("bad", "niw", ("F", "zn", "Q"))
    with pytest.raises(CompileError) as caught:
        compile_template(template, arrays=arrays)
    assert str(caught.value) == (
        "learnable 'bad': NIW does not support factor type 'gaussian_linear'; "
        "supported pair is niw/gaussian"
    )


def test_missing_external_implementation_fails_before_compilation(external_kernel):
    serialized = to_json(external_model()[0])
    kernels.unregister_kernel(CUSTOM)
    try:
        with pytest.raises(ValueError) as caught:
            from_json(serialized)
        assert str(caught.value) == (
            "factor 'interaction': unregistered factor type 'bilinear_spin_color' "
            "(template requires implementation 'cffgtests.bilinear_spin_color' version 1.2.0); "
            "register its FactorKernel before loading"
        )
    finally:
        kernels.register_kernel(external_kernel)


def test_observation_shape_is_checked_at_binding_not_compilation():
    engine = Engine(compiled("gaussian"))
    with pytest.raises(ValueError) as caught:
        engine.bind_data(engine.init_state(), {"y0": np.ones(2)})
    assert str(caught.value) == "port 'y0': observation has dim 2, edge has 1"
