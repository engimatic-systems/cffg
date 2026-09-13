"""Declared learnables (cff-x0sv): the generic conjugate-update API.

Three claims are load-bearing and each gets a test:

1. Addressing is by declaration, not by node name — a model with two
   transition nodes on separate chains learns both, and the agent code that
   drives it never mentions either.
2. Accumulation follows parameter tying — several factor instances sharing
   one blob pay into one posterior, which is what makes learning transfer.
3. The instances that are exercised are the ones that improve, and the
   recovered parameters match the generating process (d and e alongside the
   A and B paths that already had coverage).
"""

import jax
import numpy as np
import pytest

from cffg.agent import RollingAgent
from cffg.compiler import NEG, CompileError, compile_template
from cffg.core.ir import (
    BlobRef,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.compose import merge, scoped as scoped_template
from cffg.core.validate import ModelError, validate
from cffg.engine import Engine, RunStatus, _replace
from cffg.learn import Learnables
from cffg.learning_schema import PERSISTENT_MANIFEST_KEY
from cffg.models.rollout import build_rollout

RNG = np.random.default_rng(17)

KC, U = 4, 2
ACC = 0.9
SLIP = 0.15


def obs_matrix(k: int = KC, acc: float = ACC) -> np.ndarray:
    a = np.full((k, k), (1 - acc) / (k - 1))
    np.fill_diagonal(a, acc)
    return a


def corridor_bs() -> np.ndarray:
    """left/right with slip: move as intended w.p. 1−SLIP, else stay."""
    bs = np.zeros((U, KC, KC))
    for z in range(KC):
        left, right = max(z - 1, 0), min(z + 1, KC - 1)
        bs[0, left, z] += 1 - SLIP
        bs[0, z, z] += SLIP
        bs[1, right, z] += 1 - SLIP
        bs[1, z, z] += SLIP
    return bs


def corridor_builder(*, d=None, Bs=None, A=None, gfe=False, **kw):
    return build_rollout(
        "corridor",
        d=np.full(KC, 1 / KC) if d is None else d,
        A=obs_matrix() if A is None else A,
        c=np.full(KC, 1 / KC),
        e=np.full(U, 0.5),
        Bs=np.full((U, KC, KC), 1 / KC) if Bs is None else Bs,
        gfe=gfe,
        **kw,
    )


# --- declaration and resolution ---------------------------------------------


def test_plan_resolves_sources_and_targets_across_slices():
    """One `Bs` blob, three slices, one posterior: every instance is a write
    target and (in the perception graph) a stat source."""
    t, arrays = corridor_builder(past=3, steps=3, learn=("B",))
    arena = compile_template(t, arrays=arrays)
    (plan,) = arena.learn_plans
    assert plan.name == "B" and plan.family == "dirichlet"
    assert plan.ftype == "transition_mixture"
    assert plan.sources == ("tm1", "tm2", "tm3")
    assert len(plan.targets) == 3
    # The BP group holds transition_mixture as [K', K, U]; the blob is
    # [U, K', K], so the write transposes.
    assert {tg.layout for tg in plan.targets} == {"tm"}
    assert {tg.form for tg in plan.targets} == {"expected"}


def test_gfe_composites_are_targets_but_never_sources():
    """A planning window with parameter uncertainty reads A only through the
    composite: nothing there produces sufficient statistics, and the raw
    counts are what the novelty term needs written back."""
    t, arrays = corridor_builder(
        past=0, steps=2, gfe=True, learned_a=True, learn=("A",)
    )
    arena = compile_template(t, arrays=arrays)
    (plan,) = arena.learn_plans
    assert plan.sources == ()
    assert {tg.form for tg in plan.targets} == {"counts"}


def test_perception_window_supplies_the_statistics():
    t, arrays = corridor_builder(past=1, steps=1, learn=("A",))
    arena = compile_template(t, arrays=arrays)
    (plan,) = arena.learn_plans
    assert plan.sources == ("obs1",)
    assert {tg.form for tg in plan.targets} == {"expected"}


def test_planning_reads_the_habit_prior_but_never_learns_from_it():
    """Considering an action is not evidence that it was taken. Every
    planning slice reads e, because the prior applies at every step of the
    horizon, and none of them supplies a statistic."""
    t, arrays = corridor_builder(past=0, steps=3, learn=("e",))
    arena = compile_template(t, arrays=arrays)
    (plan,) = arena.learn_plans
    assert plan.sources == ()
    assert len(plan.targets) == 3


def test_perception_supplies_the_habit_statistic():
    """The perception slice branches its control edge through an equality
    node so the habit prior sits beside the clamped action."""
    t, arrays = corridor_builder(past=1, steps=1, learn=("e",))
    arena = compile_template(t, arrays=arrays)
    (plan,) = arena.learn_plans
    assert plan.sources == ("e1",)
    assert "ueq1" in t.factors  # the control equality branch
    assert t.ports["act1"].variable == "ud1"


def test_unknown_unversioned_family_is_a_validation_error():
    t, arrays = corridor_builder(past=1, steps=1)
    t.learnables["B"] = LearnableSpec("B", "beta", ("Bs",))
    with pytest.raises(ModelError, match="unknown conjugate family"):
        compile_template(t, arrays=arrays)


def test_learnable_over_mixed_factor_types_is_refused():
    t, arrays = corridor_builder(past=1, steps=1)
    t.learnables["mixed"] = LearnableSpec("mixed", "dirichlet", ("d", "A"))
    with pytest.raises(CompileError, match="mixed factor types"):
        compile_template(t, arrays=arrays)


def test_two_learnables_cannot_own_one_blob():
    t, _ = corridor_builder(past=1, steps=1)
    t.learnables["B1"] = LearnableSpec("B1", "dirichlet", ("Bs",))
    t.learnables["B2"] = LearnableSpec("B2", "dirichlet", ("Bs",))
    with pytest.raises(ModelError, match="already learned by"):
        validate(t)


def test_source_must_read_one_of_the_blobs():
    t, _ = corridor_builder(past=1, steps=1)
    t.learnables["B"] = LearnableSpec("B", "dirichlet", ("Bs",), sources=("obs1",))
    with pytest.raises(ModelError, match="does not read any of its blobs"):
        validate(t)


# --- tying: several instances, one posterior --------------------------------


K3 = 3


def two_chain_template(start_a: int = 0, start_b: int = 2):
    """Two independent chains whose transitions name the SAME `Bs` blob. The
    tie is the point: statistics from either chain pay into one posterior,
    which both chains then read."""
    t = Template(name="twochain")
    t.blobs = {b: BlobRef(b, f"{b}.npy") for b in ("da", "db", "Bs", "A")}

    def pinned(z: int) -> np.ndarray:
        d = np.full(K3, 1e-6)
        d[z] = 1.0
        return d / d.sum()

    arrays = {
        "da": pinned(start_a),
        "db": pinned(start_b),
        "Bs": np.full((U, K3, K3), 1 / K3),
        "A": obs_matrix(K3, 0.98),
    }
    for c in ("a", "b"):
        t.variables |= {
            f"z0{c}": VariableSpec(f"z0{c}", K3),
            f"u{c}": VariableSpec(f"u{c}", U),
            f"z1{c}": VariableSpec(f"z1{c}", K3),
            f"x{c}": VariableSpec(f"x{c}", K3),
        }
        t.factors |= {
            f"prior{c}": FactorSpec(
                f"prior{c}", "categorical", edges={"out": f"z0{c}"},
                param_blobs={"statistic": f"d{c}"},
            ),
            f"tm{c}": FactorSpec(
                f"tm{c}", "transition_mixture",
                edges={"out": f"z1{c}", "in": f"z0{c}", "selector": f"u{c}"},
                param_blobs={"matrices": "Bs"},
            ),
            f"obs{c}": FactorSpec(
                f"obs{c}", "matrix_cat", edges={"out": f"x{c}", "in": f"z1{c}"},
                param_blobs={"matrix": "A"},
            ),
        }
        t.ports |= {
            f"act{c}": PortSpec(f"act{c}", f"u{c}", PortRole.DATA),
            f"see{c}": PortSpec(f"see{c}", f"x{c}", PortRole.DATA),
        }
    t.learnables["B"] = LearnableSpec("B", "dirichlet", ("Bs",))
    return t, arrays


def test_tied_blob_accumulates_from_every_chain():
    t, arrays = two_chain_template(start_a=0, start_b=1)
    engine = Engine(compile_template(t, arrays=arrays))
    (plan,) = engine.arena.learn_plans
    assert plan.sources == ("tma", "tmb")  # both chains, one posterior

    learn = Learnables(engine, priors={"B": np.ones((U, K3, K3))})
    st = engine.bind_data(
        engine.init_state(), {"acta": 0, "seea": 1, "actb": 1, "seeb": 2}
    )
    result = engine.run(st, 40, tol=1e-12)
    result.require_converged()
    learn.accumulate(engine, result)
    added = learn.params["B"]["alpha"] - np.ones((U, K3, K3))
    np.testing.assert_allclose(added.sum(), 2.0, atol=1e-9)  # one count per chain
    # Chain a moved under u=0 out of state 0, chain b under u=1 out of state
    # 1. Each chain's evidence lands in its own column of the shared array.
    assert added[0, :, 0].sum() > 0.95
    assert added[1, :, 1].sum() > 0.95


def test_learning_at_one_member_transfers_to_another():
    """The transfer property. Both chains start in state 0 and take u=0, but
    only chain a is ever observed to land in state 1. Because the blob is
    tied, chain b's belief about where it went sharpens too — it read a
    parameter it never supplied evidence for."""
    t, arrays = two_chain_template(start_a=0, start_b=0)
    engine = Engine(compile_template(t, arrays=arrays))
    learn = Learnables(engine, priors={"B": np.ones((U, K3, K3))})

    def run(data: dict) -> object:
        st = engine.init_state()
        st = _replace(st, params=learn.write(engine, dict(st.params)))
        return engine.run(engine.bind_data(st, data), 40, tol=1e-12)

    naive = np.asarray(
        engine.marginals(run({"acta": 0, "actb": 0}).require_converged())["z1b"]
    )
    for _ in range(40):
        result = run({"acta": 0, "seea": 1, "actb": 0})
        result.require_converged()
        learn.accumulate(engine, result, only=("B",))
    taught = np.asarray(
        engine.marginals(run({"acta": 0, "actb": 0}).require_converged())["z1b"]
    )

    post = learn.expected("B")
    assert post[0, 1, 0] > 0.85  # u=0 out of state 0 is now sharply 0 -> 1
    assert naive[1] < 0.4 < taught[1]  # and chain b predicts it without data
    np.testing.assert_allclose(post[0, :, 0].sum(), 1.0)


# --- recovery: d and e ------------------------------------------------------


def test_d_recovers_the_episode_start_distribution():
    """Episodes start at cell 0 or cell 3 with equal probability; the
    initial-state prior should learn that, one statistic per episode."""
    true_d = np.array([0.5, 0.0, 0.0, 0.5])
    true_bs = corridor_bs()
    a = obs_matrix(acc=0.98)
    # The agent knows the dynamics, so the smoothed q(z0) is informative;
    # with a flat B, no observation says anything about where it started.
    # The sensor is sharp because a soft count converges to E[q(z0)], which
    # keeps whatever spread the smoothing leaves — that floor is a property
    # of the sensor, not of the update.
    def builder(**kw):
        return corridor_builder(Bs=true_bs, A=a, **kw)

    agent = RollingAgent(
        builder, horizon=2, learn=("d",),
        priors={"d": np.full(KC, 0.5)}, n_sweeps=60, seed=4,
    )
    for _ in range(160):
        agent.reset_episode()
        z = int(RNG.choice(KC, p=true_d))
        for _ in range(3):
            u = int(RNG.integers(U))
            x = int(RNG.choice(KC, p=a[:, z]))
            agent.observe(u, x)
            z = int(RNG.choice(KC, p=true_bs[u, :, z]))

    learned = agent.learnables.expected("d")
    assert 0.5 * np.abs(learned - true_d).sum() < 0.12, np.round(learned, 3)


def habit_builder(**kw):
    d = np.zeros(KC)
    d[0] = 1.0
    goal = np.full(KC, 0.02)
    goal[KC - 1] = 1.0
    goal /= goal.sum()
    return build_rollout(
        "habit", d=d, A=obs_matrix(), c=goal, e=np.full(U, 0.5),
        Bs=corridor_bs(), **kw,
    )


def test_habit_increment_is_the_executed_control():
    """The statistic is what the agent did, not what it considered. One
    observation of action u must add exactly one count to u and nothing to
    any other — a soft policy posterior would spread mass across both."""
    agent = RollingAgent(
        habit_builder, horizon=3, learn=("e",), priors={"e": np.full(U, 1.0)},
        n_sweeps=80, seed=6,
    )
    before = agent.learnables.params["e"]["alpha"].copy()
    agent.observe(1, 1)
    added = agent.learnables.params["e"]["alpha"] - before
    np.testing.assert_allclose(added, [0.0, 1.0], atol=1e-9)


@pytest.mark.slow
def test_planning_alone_does_not_move_the_habit():
    """act() is read-only: a caller may plan, discard the result and plan
    again without teaching the agent anything."""
    agent = RollingAgent(
        habit_builder, horizon=3, learn=("e",), priors={"e": np.full(U, 1.0)},
        n_sweeps=80, seed=6,
    )
    before = agent.learnables.params["e"]["alpha"].copy()
    for _ in range(3):
        agent.act()
    np.testing.assert_array_equal(agent.learnables.params["e"]["alpha"], before)


@pytest.mark.slow
def test_e_recovers_the_agent_habit():
    """On a corridor whose goal is to the right, the executed controls are
    mostly RIGHT, so that is the habit the agent acquires."""
    d = np.zeros(KC)
    d[0] = 1.0
    goal = np.full(KC, 0.02)
    goal[KC - 1] = 1.0
    goal /= goal.sum()

    agent = RollingAgent(
        habit_builder, horizon=3, learn=("e",), priors={"e": np.full(U, 1.0)},
        n_sweeps=80, seed=6,
    )
    z, taken = 0, []
    for _ in range(24):
        u = agent.act()
        taken.append(u)
        z = int(RNG.choice(KC, p=corridor_bs()[u, :, z]))
        agent.observe(u, int(RNG.choice(KC, p=obs_matrix()[:, z])))

    # The posterior is the prior plus a tally of what was executed, exactly.
    tally = np.array([1.0 + taken.count(u) for u in range(U)])
    np.testing.assert_allclose(agent.learnables.params["e"]["alpha"], tally, atol=1e-6)
    habit = agent.learnables.expected("e")
    assert habit[1] > 0.7, habit  # RIGHT is the habit it acquired
    np.testing.assert_allclose(habit.sum(), 1.0, atol=1e-12)


def test_d_is_not_clobbered_by_the_running_belief():
    """Within an episode the prior slot carries the current belief; the
    learned d only re-anchors it at a reset."""
    agent = RollingAgent(
        corridor_builder, horizon=2, learn=("d",),
        priors={"d": np.array([10.0, 1.0, 1.0, 1.0])}, n_sweeps=60, seed=8,
    )
    for _ in range(3):
        agent.observe(1, 3)
    drifted = agent.belief.copy()
    assert np.argmax(drifted) != 0  # the walk moved it off the prior's mode
    reset = agent.reset_episode()
    assert np.argmax(reset) == 0
    np.testing.assert_allclose(reset, agent.learnables.expected("d"))


# --- persistence and back-compatibility -------------------------------------


def test_checkpoint_carries_every_learnable(tmp_path):
    def make():
        return RollingAgent(
            corridor_builder, horizon=2, learn=("A", "B", "d"),
            priors={
                "A": np.ones((KC, KC)),
                "B": np.ones((U, KC, KC)),
                "d": np.ones(KC),
            },
            n_sweeps=40, seed=5,
        )

    agent = make()
    agent.observe(1, 2)
    agent.observe(0, 1)
    path = tmp_path / "agent.npz"
    agent.checkpoint(path)

    other = make()
    other.restore(path)
    for name in ("A", "B", "d"):
        np.testing.assert_array_equal(
            other.learnables.params[name]["alpha"], agent.learnables.params[name]["alpha"]
        )
    np.testing.assert_array_equal(other.belief, agent.belief)


def test_legacy_learn_flags_map_onto_the_api():
    old = RollingAgent(
        corridor_builder, horizon=2, learn_a=True, initial_counts=np.ones((KC, KC)),
        learn_b=True, initial_b_counts=np.ones((U, KC, KC)), n_sweeps=40, seed=5,
    )
    new = RollingAgent(
        corridor_builder, horizon=2, learn=("A", "B"),
        priors={"A": np.ones((KC, KC)), "B": np.ones((U, KC, KC))},
        n_sweeps=40, seed=5,
    )
    for agent in (old, new):
        agent.observe(1, 2)
    np.testing.assert_allclose(old.counts, new.learnables.params["A"]["alpha"])
    np.testing.assert_allclose(old.b_counts, new.learnables.params["B"]["alpha"])


def test_restore_preserves_episode_cadence(tmp_path):
    """The episode-start statistic must fire once per episode. If a restore
    forgot where it was in the episode, the tick after a checkpoint would be
    read as a second episode start and fold q(z0) twice from one episode."""
    def make():
        return RollingAgent(
            corridor_builder, horizon=2, learn=("d",),
            priors={"d": np.ones(KC)}, n_sweeps=40, seed=5,
        )

    straight = make()
    straight.observe(1, 2)
    before = straight.learnables.params["d"]["alpha"].copy()
    straight.observe(0, 1)
    uninterrupted = straight.learnables.params["d"]["alpha"] - before

    resumed = make()
    resumed.observe(1, 2)
    path = tmp_path / "agent.npz"
    resumed.checkpoint(path)
    other = make()
    other.restore(path)
    mid = other.learnables.params["d"]["alpha"].copy()
    other.observe(0, 1)
    after_restore = other.learnables.params["d"]["alpha"] - mid

    np.testing.assert_allclose(after_restore, uninterrupted, atol=1e-9)
    np.testing.assert_allclose(after_restore, np.zeros(KC), atol=1e-9)


def test_scoped_learnable_checkpoints_round_trip():
    """Composing scopes a template's names with '/', so a scoped learnable is
    called `scope/B` and its checkpoint key `scope/B/alpha`. Splitting that
    on the FIRST separator reads the scope as the learnable name and loses
    the checkpoint."""
    t, arrays = two_chain_template()
    scoped = scoped_template(t, "outer")
    engine = Engine(compile_template(scoped, arrays={
        f"outer/{k}": v for k, v in arrays.items()
    }))
    (plan,) = engine.arena.learn_plans
    assert plan.name == "outer/B"

    learn = Learnables(engine, priors={"outer/B": np.full((U, K3, K3), 2.0)})
    keys = list(learn.state_dict())
    assert keys == ["v2:outer%2FB:alpha", PERSISTENT_MANIFEST_KEY], keys

    other = Learnables(engine, priors={"outer/B": np.ones((U, K3, K3))})
    other.load_state_dict(learn.state_dict())
    np.testing.assert_array_equal(
        other.params["outer/B"]["alpha"], learn.params["outer/B"]["alpha"]
    )


def test_nested_scopes_cannot_collide_in_checkpoint_keys():
    """A delimiter heuristic is insufficient even when splitting on the last
    slash: one learnable may be named ``scope/B`` while another is named
    ``scope/B/alpha``, exactly the apparent old checkpoint key of the first.
    Versioned component encoding must preserve both, while the ambiguous old
    spelling fails loudly instead of restoring into the wrong posterior."""
    left_t, left_arrays = two_chain_template()
    left = scoped_template(left_t, "scope")

    right_t, right_arrays = two_chain_template()
    right_t.learnables = {
        "alpha": LearnableSpec("alpha", "dirichlet", ("Bs",))
    }
    right = scoped_template(right_t, "scope/B")

    model = merge(left, right, "checkpoint_collision")
    arrays = {
        **{f"scope/{name}": value for name, value in left_arrays.items()},
        **{f"scope/B/{name}": value for name, value in right_arrays.items()},
    }
    engine = Engine(compile_template(model, arrays=arrays))
    source = Learnables(
        engine,
        priors={
            "scope/B": np.full((U, K3, K3), 2.0),
            "scope/B/alpha": np.full((U, K3, K3), 3.0),
        },
    )
    checkpoint = source.state_dict()
    assert len(checkpoint) == 3

    restored = Learnables(
        engine,
        priors={
            "scope/B": np.ones((U, K3, K3)),
            "scope/B/alpha": np.ones((U, K3, K3)),
        },
    )
    restored.load_state_dict(checkpoint)
    for name in source.params:
        np.testing.assert_array_equal(
            restored.params[name]["alpha"], source.params[name]["alpha"]
        )

    with pytest.raises(KeyError, match="ambiguous legacy"):
        restored.load_state_dict(
            {"scope/B/alpha": np.ones((U, K3, K3))}
        )


def test_unknown_checkpoint_key_is_refused():
    t, arrays = two_chain_template()
    engine = Engine(compile_template(t, arrays=arrays))
    learn = Learnables(engine, priors={"B": np.ones((U, K3, K3))})
    with pytest.raises(KeyError, match="no learnable for checkpoint key"):
        learn.load_state_dict({"nope/alpha": np.ones((U, K3, K3))})


def test_legacy_checkpoint_keys_are_migrated(tmp_path):
    """Checkpoints written before the learnables API named their two arrays
    `counts` and `b_counts`. Restoring one must reload them, not silently
    keep the constructor priors."""
    path = tmp_path / "legacy.npz"
    counts = np.full((KC, KC), 3.0)
    b_counts = np.full((U, KC, KC), 7.0)
    np.savez(
        path, belief=np.full(KC, 1 / KC), tick=np.asarray(4),
        counts=counts, b_counts=b_counts,
    )
    agent = RollingAgent(
        corridor_builder, horizon=2, learn=("A", "B"),
        priors={"A": np.ones((KC, KC)), "B": np.ones((U, KC, KC))},
        n_sweeps=20, seed=5,
    )
    agent.restore(path)
    np.testing.assert_array_equal(agent.counts, counts)
    np.testing.assert_array_equal(agent.b_counts, b_counts)
    assert agent.episode_tick == 4  # no episode concept then: ticks are one episode


def test_legacy_checkpoint_for_an_undeclared_learnable_is_refused(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(
        path, belief=np.full(KC, 1 / KC), tick=np.asarray(1),
        counts=np.full((KC, KC), 3.0),
    )
    agent = RollingAgent(
        corridor_builder, horizon=2, learn=("B",),
        priors={"B": np.ones((U, KC, KC))}, n_sweeps=20, seed=5,
    )
    with pytest.raises(KeyError, match="legacy 'counts'"):
        agent.restore(path)


# --- numerical robustness ---------------------------------------------------


def deterministic_bs() -> np.ndarray:
    """left/right with no slip: exact zeros everywhere off the moved-to cell."""
    bs = np.zeros((U, KC, KC))
    for z in range(KC):
        bs[0, max(z - 1, 0), z] = 1.0
        bs[1, min(z + 1, KC - 1), z] = 1.0
    return bs


def test_exact_zero_statistic_leaves_counts_positive_and_writes_finite():
    """A deterministic transition puts exact zeros through the node belief,
    so most cells accumulate nothing and stay at their prior. E[log θ] must
    stay finite everywhere: ψ(α) is undefined at α = 0, so a cell that
    reached zero would be a NaN in the planner, not a small number."""
    def builder(**kw):
        return corridor_builder(Bs=deterministic_bs(), A=obs_matrix(acc=0.99), **kw)

    agent = RollingAgent(
        builder, horizon=2, learn=("A",), priors={"A": np.ones((KC, KC))},
        n_sweeps=40, seed=5,
    )
    agent.belief = np.zeros(KC)
    agent.belief[0] = 1.0  # exact zeros in three of four cells
    with np.errstate(divide="raise", invalid="raise"):
        agent.observe(1, 1)  # RIGHT from cell 0 lands in cell 1, exactly

    alpha = agent.learnables.params["A"]["alpha"]
    assert np.all(alpha > 0), alpha
    # The statistic was a one-hot: one column moved, the rest are untouched.
    np.testing.assert_allclose(alpha[:, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(alpha[1, 1], 2.0, atol=1e-3)

    params = agent.learnables.write(agent.filter, dict(agent.filter.init_state().params))
    for arr in params.values():
        assert np.all(np.isfinite(np.asarray(arr))), "expected parameters went non-finite"


def test_float32_engine_keeps_the_learning_tick_finite():
    """Consumers run the engine at float32. Accumulation is numpy float64
    throughout, but the beliefs it reads and the parameters it writes cross
    that boundary, and neither may produce a NaN or an infinity — including
    when the belief carries exact zeros."""
    with jax.enable_x64(False):
        agent = RollingAgent(
            corridor_builder, horizon=2, learn=("A", "B", "d"),
            priors={
                "A": np.ones((KC, KC)),
                "B": np.ones((U, KC, KC)),
                "d": np.ones(KC),
            },
            n_sweeps=40, seed=5,
        )
        agent.belief = np.zeros(KC, dtype=np.float32)
        agent.belief[0] = 1.0
        with np.errstate(divide="raise", invalid="raise"):
            for u, x in ((1, 1), (1, 2), (0, 1)):
                agent.observe(u, x)
        for name in ("A", "B", "d"):
            alpha = agent.learnables.params[name]["alpha"]
            assert np.all(np.isfinite(alpha)) and np.all(alpha > 0), name
        assert np.all(np.isfinite(agent.belief))


@pytest.mark.parametrize("schedule", ("auto", "flood"))
@pytest.mark.parametrize("alpha", (1e-3, 1e-31), ids=("exp-underflow", "below-neg"))
def test_sparse_dirichlet_prior_is_not_infeasible(schedule, alpha):
    """Every positive Dirichlet concentration is proper. α = 1e-3 puts
    E[log θ] near -750, where exp() underflows; α = 1e-31 puts it below the
    finite NEG sentinel itself. Neither magnitude means exact zero.

    The auto schedule exercises the in-JIT exact-support check and flooding
    exercises its NumPy counterpart. Both must distinguish the one reserved
    NEG value from every other finite log-potential without a threshold."""
    t, arrays = corridor_builder(past=1, steps=1, learn=("A",))
    engine = Engine(compile_template(t, arrays=arrays), schedule=schedule)
    learn = Learnables(engine, priors={"A": np.full((KC, KC), alpha)})

    st = engine.init_state()
    st = _replace(st, params=learn.write(engine, dict(st.params)))
    st = engine.bind_data(st, {"act1": 0, "obs1": 2})
    result = engine.run(st, 60, tol=1e-12)
    assert result.status is RunStatus.CONVERGED, (result.status, result.residual)

    # E[log A] really is at the magnitude that underflows exp().
    address = engine.arena.member_of["obs1"].parameter("matrix")
    elog = np.asarray(result.state.params[address.key])
    assert elog.min() < -700, elog.min()
    assert np.exp(elog).max() == 0.0  # exactly why the old test was wrong
    if alpha < 1e-30:
        assert elog.min() < NEG  # a magnitude threshold would still be wrong


@pytest.mark.parametrize("schedule", ("auto", "flood"))
def test_float32_dirichlet_rollover_escapes_the_reserved_neg_value(schedule):
    """The finite NEG bit pattern is reserved for exact zero, but a proper
    positive Dirichlet can round to that exact E[log theta] in float32.
    Rollover moves only that collision by one ULP, retaining support without a
    magnitude floor."""
    with jax.enable_x64(False):
        t, arrays = corridor_builder(past=1, steps=1, learn=("A",))
        engine = Engine(compile_template(t, arrays=arrays), schedule=schedule)
        learn = Learnables(engine, priors={"A": np.full((KC, KC), 7.5e-31)})

        st = engine.init_state()
        st = _replace(st, params=learn.write(engine, dict(st.params)))
        st = engine.bind_data(st, {"act1": 0, "obs1": 2})
        address = engine.arena.member_of["obs1"].parameter("matrix")
        elog = np.asarray(st.params[address.key])
        assert elog.dtype == np.float32
        assert np.all(elog != np.asarray(NEG, dtype=np.float32))

        result = engine.run(st, 60, tol=1e-6)
        assert result.status is RunStatus.CONVERGED, (
            result.status,
            result.residual,
        )


def test_uncomfortable_count_scale_stays_finite():
    """Counts accumulate without bound across a long run. ψ(α) − ψ(α₀) is a
    difference of two large numbers at scale, and it must not lose its sign
    or go non-finite when one column has seen far more evidence than another."""
    t, arrays = corridor_builder(past=1, steps=1, learn=("B",))
    engine = Engine(compile_template(t, arrays=arrays))
    lopsided = np.full((U, KC, KC), 1e-4)
    lopsided[0, 0, 0] = 1e9  # nine orders of magnitude apart in one column
    learn = Learnables(engine, priors={"B": lopsided})
    params = learn.write(engine, dict(engine.init_state().params))
    for arr in params.values():
        arr = np.asarray(arr)
        assert np.all(np.isfinite(arr)), "E[log theta] went non-finite at scale"
    # The dominant cell is the near-certain outcome of its column.
    post = learn.expected("B")
    np.testing.assert_allclose(post[0, 0, 0], 1.0, atol=1e-9)


def test_prior_for_an_undeclared_learnable_is_refused():
    with pytest.raises(KeyError, match="does not declare"):
        RollingAgent(
            corridor_builder, horizon=2, learn=("B",),
            priors={"B": np.ones((U, KC, KC)), "A": np.ones((KC, KC))},
            n_sweeps=20, seed=5,
        )
