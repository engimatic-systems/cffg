"""Factored rollout construction and the two-block corridor (cff-b47g)."""

from __future__ import annotations

import functools

import jax
import numpy as np

from cffg.agent import RollingAgent
from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core.ir import BoundaryFactorization
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.core.validate import validate
from cffg.engine import Engine
from cffg.learn import Learnables
from cffg.models.factored import (
    CouplingSpec,
    ObservationSpec,
    StateBlockSpec,
    TransitionNoveltySpec,
    build_factored_corridor,
    build_factored_rollout,
)
from cffg.models.rollout import build_rollout


def _converged(
    template,
    arrays,
    *,
    data: dict[str, int] | None = None,
    damp: float = 0.0,
    sweeps: int = 300,
):
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()
    if data:
        state = engine.bind_data(state, data)
    result = engine.run(state, sweeps, tol=1e-10, damp=damp)
    return engine, result, result.require_converged()


def test_builder_lowers_products_fanout_and_serialized_agent_bindings(tmp_path):
    plan, plan_arrays = build_factored_corridor(
        steps=2, length=3, modes=2, gfe=True
    )
    filt, filt_arrays = build_factored_corridor(
        past=1, steps=1, length=3, modes=2, gfe=False
    )
    for template in (plan, filt):
        validate(template)
        restored = from_json(to_json(template))
        assert restored.agent == template.agent
        assert structure_hash(restored) == structure_hash(template)
        assert set(f.type for f in template.factors.values()) <= {
            "categorical",
            "matrix_cat",
            "transition_mixture",
            "equality",
        }
        assert any(name.startswith("product/selector/position/") for name in template.factors)
        assert any(name.startswith("branch/") for name in template.factors)
        assert all(
            binding.factorization is BoundaryFactorization.MEAN_FIELD
            and len(binding.constraints) == 1
            for binding in template.agent.beliefs
        )

    position_tm = plan.factors["transition/position/1"]
    mode_tm = plan.factors["transition/mode/1"]
    assert plan.variables[position_tm.edges["selector"]].cardinality == 2 * 3
    assert plan.variables[mode_tm.edges["selector"]].cardinality == 3

    plan_engine, plan_result, _ = _converged(
        plan, plan_arrays, damp=0.5
    )
    assert np.isfinite(plan_engine.bfe(plan_result.state))
    filter_engine, filter_result, _ = _converged(
        filt,
        filt_arrays,
        data={"action1": 2, "observation1": 2},
    )
    assert np.isfinite(filter_engine.bfe(filter_result.state))

    bundle = tmp_path / "factored.cffgz"
    save_bundle(bundle, plan, plan_arrays)
    bundled, bundled_arrays, _manifest = load_bundle(bundle)
    assert bundled.agent == plan.agent
    assert structure_hash(bundled) == structure_hash(plan)
    assert set(bundled_arrays) == set(plan_arrays)

    agent = RollingAgent(
        functools.partial(
            build_factored_corridor, length=3, modes=2
        ),
        horizon=2,
        n_sweeps=200,
    )
    action = agent.act()
    observed_position = 1 if action == 2 else 0
    beliefs = agent.observe(action, observed_position * 2)
    assert set(beliefs) == {"position", "mode"}


def _flatten_corridor(
    factored_arrays: dict[str, np.ndarray],
    *,
    length: int,
    modes: int,
) -> np.ndarray:
    """Independent dense P(z',m' | z,m,u), position-major."""
    n_actions = factored_arrays["action/prior"].shape[0]
    position = factored_arrays["block/position/transition"].reshape(
        modes, n_actions, length, length
    )
    mode = factored_arrays["block/mode/transition"]
    joint = np.zeros(
        (n_actions, length * modes, length * modes), dtype=position.dtype
    )
    for action in range(n_actions):
        for old_position in range(length):
            for old_mode in range(modes):
                old = old_position * modes + old_mode
                for new_position in range(length):
                    for new_mode in range(modes):
                        new = new_position * modes + new_mode
                        joint[action, new, old] = (
                            position[
                                old_mode, action, new_position, old_position
                            ]
                            * mode[action, new_mode, old_mode]
                        )
    return joint


def test_two_block_filter_matches_independently_flattened_joint_and_bfe():
    length, modes = 3, 2
    likelihood = np.array(
        [
            [0.7, 0.2, 0.6, 0.3, 0.5, 0.4],
            [0.3, 0.8, 0.4, 0.7, 0.5, 0.6],
        ]
    )
    factored, factored_arrays = build_factored_corridor(
        past=1,
        steps=1,
        length=length,
        modes=modes,
        position_prior=np.array([0.6, 0.4, 0.0]),
        mode_prior=np.array([1.0, 0.0]),
        observation_matrix=likelihood,
        preference=np.array([0.5, 0.5]),
        gfe=False,
    )
    dense_transition = _flatten_corridor(
        factored_arrays, length=length, modes=modes
    )
    flat, flat_arrays = build_rollout(
        "flat-corridor-oracle",
        past=1,
        steps=1,
        d=np.kron(
            factored_arrays["block/position/initial"],
            factored_arrays["block/mode/initial"],
        ),
        A=likelihood,
        c=np.array([0.5, 0.5]),
        e=factored_arrays["action/prior"],
        Bs=dense_transition,
        gfe=False,
    )

    factored_engine, factored_result, factored_state = _converged(
        factored,
        factored_arrays,
        data={"action1": 1, "observation1": 0},
    )
    flat_engine, flat_result, flat_state = _converged(
        flat,
        flat_arrays,
        data={"act1": 1, "obs1": 0},
    )
    product_edge = factored.factors["observation/sensor/1"].edges["in"]
    factored_joint = factored_engine.marginals(factored_state)[product_edge]
    flat_joint = flat_engine.marginals(flat_state)["zm1"]
    np.testing.assert_allclose(factored_joint, flat_joint, atol=1e-10)
    np.testing.assert_allclose(
        factored_engine.bfe(factored_result.state),
        flat_engine.bfe(flat_result.state),
        atol=1e-10,
    )
    assert 0 < factored_joint[0] < 1
    assert 0 < factored_joint[2] < 1


def _identity_selected(
    selector_shape: tuple[int, ...], card: int
) -> np.ndarray:
    out = np.zeros((*selector_shape, card, card))
    out[...] = np.eye(card)
    return out


def test_three_block_public_api_compiles_runs_and_round_trips():
    template, arrays = build_factored_rollout(
        "three-block",
        steps=1,
        blocks=(
            StateBlockSpec(
                "a",
                np.array([1.0, 0.0]),
                _identity_selected((3, 2), 2),
                selector="b_parent",
            ),
            StateBlockSpec(
                "b",
                np.array([0.0, 1.0, 0.0]),
                _identity_selected((2,), 3),
            ),
            StateBlockSpec(
                "c",
                np.array([1.0, 0.0]),
                _identity_selected((2, 3, 2), 2),
                selector="ab_parent",
            ),
        ),
        couplings=(
            CouplingSpec("b_parent", ("b",)),
            CouplingSpec("ab_parent", ("a", "b")),
            CouplingSpec("all_state", ("a", "b", "c")),
        ),
        actions=np.array([0.5, 0.5]),
        observation=ObservationSpec(
            "sensor",
            "all_state",
            np.eye(12),
            np.full(12, 1 / 12),
        ),
        gfe=False,
    )
    validate(template)
    assert len(template.agent.beliefs) == 3
    restored = from_json(to_json(template))
    assert restored.agent == template.agent
    assert structure_hash(restored) == structure_hash(template)
    engine, result, _ = _converged(template, arrays, damp=0.5)
    assert np.isfinite(engine.bfe(result.state))


def test_tied_transition_statistic_updates_every_time_member():
    template, arrays = build_factored_corridor(
        past=1,
        steps=2,
        length=3,
        modes=2,
        observation_source="position",
        gfe=False,
        learn=("position_transition",),
    )
    engine, _result, state = _converged(
        template,
        arrays,
        data={"action1": 2, "observation1": 1},
    )
    counts = np.ones_like(arrays["block/position/transition"])
    learnables = Learnables(
        engine, priors={"position_transition": counts}
    )
    before = learnables.params["position_transition"]["alpha"].copy()
    learnables.accumulate(engine, state)
    after = learnables.params["position_transition"]["alpha"]
    np.testing.assert_allclose((after - before).sum(), 1.0, atol=1e-10)

    # mode=0/action=right is selector 2 in C-order.  The observed move 0→1
    # raises that prediction above its uniform 1/3 prior.
    mean = learnables.expected("position_transition")
    assert mean[2, 1, 0] > 1 / 3

    written = learnables.write(engine, dict(state.params))
    key1, member1 = engine.arena.member_of["transition/position/1"]
    key2, member2 = engine.arena.member_of["transition/position/2"]
    assert key1 == key2
    np.testing.assert_array_equal(
        written[key1][member1], written[key2][member2]
    )


def test_declared_novelty_schedule_routes_tied_counts_and_half_life():
    length, modes = 3, 2
    active_public = np.zeros((modes, 3, length), dtype=bool)
    active_public[0, 2, :2] = True
    active_public[1, 0, :2] = True
    template, arrays = build_factored_corridor(
        steps=2,
        length=length,
        modes=modes,
        gfe=True,
        learn=("position_transition",),
        epistemic_b=True,
        transition_novelty=(
            TransitionNoveltySpec(
                "position_transition",
                half_life=4.0,
                active_columns=active_public,
                saturation_weight=1 / 16,
            ),
        ),
    )
    validate(template)
    restored = from_json(to_json(template))
    assert set(restored.constraints) == set(template.constraints)
    assert structure_hash(restored) == structure_hash(template)

    engine = Engine(compile_template(template, arrays=arrays))
    prior = np.full_like(
        arrays["block/position/transition_counts"], 1 / length
    )
    learnables = Learnables(
        engine, priors={"position_transition": prior}
    )
    initial = learnables.write(engine, dict(engine.init_state().params))
    (group,) = engine.arena.gfe_tm_groups
    assert group.learnables == ("position_transition",) * 2
    assert group.novelty_key is not None
    np.testing.assert_array_equal(initial[group.novelty_key], 1.0)

    active = active_public.reshape(modes * 3, length)
    learnables.params["position_transition"]["alpha"] += (
        active[:, None, :] * (4.0 / length)
    )
    written = learnables.write(engine, initial)
    weights = np.asarray(written[group.novelty_key])
    np.testing.assert_allclose(weights[:, active], 0.5, atol=1e-12)
    np.testing.assert_allclose(weights[:, ~active], 1.0, atol=1e-12)

    # Both future slices read one tied posterior; neither planning instance
    # creates its own count mass or schedule.
    count_key = group.name + ":counts"
    np.testing.assert_array_equal(written[count_key][0], written[count_key][1])


def test_saturated_active_columns_dispatch_plain_and_keep_learning():
    length, modes = 3, 2
    active_public = np.zeros((modes, 3, length), dtype=bool)
    active_public[0, 2, :2] = True
    active_public[1, 0, :2] = True
    schedule = TransitionNoveltySpec(
        "position_transition",
        half_life=4.0,
        active_columns=active_public,
        saturation_weight=1 / 16,
    )
    builder = functools.partial(
        build_factored_corridor,
        length=length,
        modes=modes,
        transition_novelty=(schedule,),
    )
    prior = np.full((modes * 3, length, length), 1 / length)
    agent = RollingAgent(
        builder,
        horizon=2,
        learn=("position_transition",),
        priors={"position_transition": prior},
        epistemic_b=True,
        n_sweeps=200,
        select="sample",
        seed=7,
    )
    assert agent.plain_plan is not None
    assert not agent.novelty_saturated()
    assert agent.plan.arena.gfe_tm_groups
    assert not agent.plain_plan.arena.gfe_tm_groups

    active = active_public.reshape(modes * 3, length)
    agent.learnables.params["position_transition"]["alpha"] += (
        active[:, None, :] * (16.0 / length)
    )
    assert agent.novelty_saturated()
    latched = agent.plan_action(use_plain=False)
    assert latched.planner == "epistemic_b"
    planned = agent.plan_action()
    assert planned.planner == "plain"
    assert planned.run.status.value == "converged"
    assert np.isfinite(planned.free_energy)

    before = agent.learnables.params["position_transition"]["alpha"].sum()
    # mode 0, action right: true next state is (position=1, mode=0), whose
    # full-state observation index is 1*2+0.
    agent.observe(2, 2)
    after = agent.learnables.params["position_transition"]["alpha"].sum()
    np.testing.assert_allclose(after - before, 1.0, atol=1e-10)
    assert agent.novelty_saturated()


def test_exact_support_factored_gfe_certifies_roundoff_stationarity():
    """Regression for cff-qr92's gauge-fixed Newton roundoff plateau.

    The hard corridor transitions and sensor put the 24-logit future-state
    solve exactly on sparse support. Its recomputed inner residual can settle
    just above 512 float64 ULPs even though outer messages become stationary.
    """
    builder = functools.partial(
        build_factored_corridor,
        length=8,
        modes=2,
        dtype=np.float64,
    )
    agent = RollingAgent(
        builder,
        horizon=8,
        damp=0.88,
        n_sweeps=2_000,
        select="sample",
        seed=0,
    )
    agent.beliefs = {
        "position": np.r_[1.0, np.zeros(7)],
        "mode": np.array([1.0, 0.0]),
    }

    planned = agent.plan_action()

    assert planned.run.status.value == "converged"
    assert planned.run.inner_status.value == "converged"
    assert planned.run.iterations < agent.n_sweeps
    assert np.isfinite(planned.free_energy)
    assert np.isfinite(np.asarray(planned.run.state.messages)).all()


def test_multiple_arbitrarily_named_learnables_ignore_inactive_columns():
    length, modes = 3, 2
    _canonical, base = build_factored_corridor(
        steps=1, length=length, modes=modes
    )
    motion = base["block/position/transition"].reshape(
        modes, 3, length, length
    )
    regime = base["block/mode/transition"]
    motion_active = np.zeros((modes, 3, length), dtype=bool)
    motion_active[0, 2, :2] = True
    regime_active = np.zeros((3, modes), dtype=bool)
    regime_active[1, 0] = True
    schedules = (
        TransitionNoveltySpec(
            "motion_kernel", 4.0, motion_active, 1 / 16
        ),
        TransitionNoveltySpec(
            "regime_kernel", 4.0, regime_active, 1 / 16
        ),
    )

    def builder(*, past, steps, **kwargs):
        return build_factored_rollout(
            "arbitrary-novelty-names",
            past=past,
            steps=steps,
            blocks=(
                StateBlockSpec(
                    "where",
                    np.array([1.0, 0.0, 0.0]),
                    motion,
                    selector="regime_parent",
                    transition_learnable="motion_kernel",
                ),
                StateBlockSpec(
                    "regime",
                    np.array([1.0, 0.0]),
                    regime,
                    transition_learnable="regime_kernel",
                ),
            ),
            couplings=(
                CouplingSpec("regime_parent", ("regime",)),
                CouplingSpec("joint", ("where", "regime")),
            ),
            actions=np.full(3, 1 / 3),
            observation=ObservationSpec(
                "sensor", "joint", np.eye(length * modes),
                np.full(length * modes, 1 / (length * modes)),
            ),
            transition_novelty=schedules,
            **kwargs,
        )

    priors = {
        "motion_kernel": np.full((modes * 3, length, length), 1 / length),
        "regime_kernel": np.full((3, modes, modes), 1 / modes),
    }
    agent = RollingAgent(
        builder,
        horizon=1,
        learn=tuple(priors),
        priors=priors,
        epistemic_b=True,
        n_sweeps=100,
    )
    assert not agent.novelty_saturated()
    for name, mask, card in (
        ("motion_kernel", motion_active.reshape(modes * 3, length), length),
        ("regime_kernel", regime_active, modes),
    ):
        agent.learnables.params[name]["alpha"] += (
            mask[:, None, :] * (16.0 / card)
        )
    assert agent.novelty_saturated()

    # Unselected columns retain λ=1 and still cannot hold the plain path open.
    np.testing.assert_array_equal(
        agent.learnables.novelty_weights("motion_kernel", 4.0)[
            ~motion_active.reshape(modes * 3, length)
        ],
        1.0,
    )
    np.testing.assert_array_equal(
        agent.learnables.novelty_weights("regime_kernel", 4.0)[
            ~regime_active
        ],
        1.0,
    )


def test_sparse_float32_and_32_by_4_keep_learned_tables_factored():
    length, modes, n_actions = 32, 4, 3
    with jax.enable_x64(False):
        template, arrays = build_factored_corridor(
            past=1,
            steps=1,
            length=length,
            modes=modes,
            observation_source="position",
            dtype=np.float32,
            gfe=False,
            learn=("position_transition", "mode_transition"),
        )
        engine, result, state = _converged(
            template,
            arrays,
            data={"action1": 1, "observation1": 0},
        )
        assert np.isfinite(engine.bfe(result.state))
        marginals = engine.marginals(state)
        bindings = {binding.name: binding for binding in template.agent.beliefs}
        position = marginals[bindings["position"].posterior]
        mode = marginals[bindings["mode"].posterior]
        assert position.shape == (length,) and mode.shape == (modes,)
        assert position.dtype == mode.dtype == np.float32
        assert np.all(np.isfinite(position)) and np.all(np.isfinite(mode))

        learned_size = (
            arrays["block/position/transition"].size
            + arrays["block/mode/transition"].size
        )
        flat_size = n_actions * (length * modes) ** 2
        assert learned_size < flat_size
        assert arrays["block/position/transition"].shape == (
            modes * n_actions,
            length,
            length,
        )
        product_tables = [
            value
            for name, value in arrays.items()
            if name.startswith("deterministic/product/")
        ]
        assert product_tables
        assert all(np.count_nonzero(table == 0) > 0 for table in product_tables)
