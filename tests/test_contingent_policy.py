"""Independent complete-policy sums score the compact native action-prefix trees."""
from itertools import product
from math import log

import jax
import numpy as np
import pytest

from cffg.bundle import load_bundle, save_bundle
from cffg.contingent import ContingentPlanner
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.experiments import behavioral_epistemics as cue
from cffg.models.contingent import build_contingent, build_contingent_rollout
from cffg.runtime_types import InferenceError


def source(depth, dtype):
    rng = np.random.default_rng(76431)
    goals, obs = [], []
    for t in range(depth + 1):
        shape = (2,) * (2*t + 1)
        goals.append(rng.uniform(.1, .95, shape).astype(dtype))
        if t < depth:
            p = rng.uniform(.1, .9, shape)
            obs.append(np.stack((p, 1-p), -1).astype(dtype))
    return goals, obs


def enumerate_policies(observations, preferences, fixed=False):
    """Enumerate full action maps and recursively visit their physical worlds."""
    entries = [(h, a) for t, table in enumerate(preferences)
               for h in product(*(range(observations[j].shape[-1]) for j in range(t)))
               for a in [table.shape[-1]]]
    keys = {len(h) if fixed else h: n for h, n in entries}
    records = []
    for commands in product(*(range(n) for n in keys.values())):
        rules = dict(zip(keys, commands))
        policy = {h: rules[len(h) if fixed else h] for h, _ in entries}
        reach = {}

        def visit(h, past, mass):
            reach[h] = mass
            if not mass:
                return 0.
            actions = past + (policy[h],)
            score = -mass * log(float(preferences[len(h)][h + actions]))
            if len(h) < len(observations):
                for reading, p in enumerate(observations[len(h)][h + actions]):
                    score += visit(h + (reading,), actions, mass * float(p))
            return score

        records.append((visit((), (), 1.), policy, reach))
    return records


@pytest.mark.parametrize("dtype,depth", [
    pytest.param(np.float32, 1, marks=pytest.mark.slow), (np.float32, 2),
    (np.float64, 1), pytest.param(np.float64, 2, marks=pytest.mark.slow),
])
def test_native_policy_matches_full_enumeration(depth, dtype):
    with jax.enable_x64(dtype == np.float64):
        pref, obs = source(depth, dtype)
        plan = ContingentPlanner(*build_contingent("tree", observations=obs, preferences=pref, dtype=dtype))
        result = plan.solve()
        records = enumerate_policies(obs, pref)
        best, actions, reach = min(records, key=lambda row: row[0])
        weights = np.exp(-np.array([x[0] for x in records]))
        tolerance = 256 * np.finfo(dtype).eps * len(actions)
        assert plan._base.messages.dtype == dtype
        assert np.isclose(result.expected_loss, best, rtol=0, atol=tolerance)
        assert np.isclose(result.free_energy, -log(weights.mean()), rtol=0, atol=tolerance)
        assert np.isclose(result.committed_free_energy, best + log(len(records)), rtol=0, atol=tolerance)
        for h, branch in result.branches.items():
            assert branch.action == actions[h]
            assert np.isclose(branch.reach, reach[h], rtol=0, atol=tolerance)
            conditional = np.zeros(2)
            for (_, rule, visits), weight in zip(records, weights):
                if all(rule[h[:t]] == actions[h[:t]] for t in range(len(h))):
                    conditional[rule[h]] += weight * visits[h]
            np.testing.assert_allclose(branch.probabilities, conditional / conditional.sum(),
                                       rtol=0, atol=tolerance)


def test_json_bundle_resume_and_causal_branch_lookup(tmp_path, monkeypatch):
    pref, obs = source(2, np.float32)
    obs[0][:] = [1., 0.]
    obs[1][:] = [.5, .5]
    pref[1][1] = 0  # An impossible branch must have no likelihood penalty.
    t, a = build_contingent("resume", observations=obs, preferences=pref)
    clone = from_json(to_json(t))
    assert to_json(clone) == to_json(t)
    assert structure_hash(clone) == structure_hash(t)
    original = ContingentPlanner(t, a).solve()
    save_bundle(tmp_path/"tree.cffgz", clone, a)
    bundle = load_bundle(tmp_path/"tree.cffgz")
    assert structure_hash(bundle.template) == structure_hash(t)
    planner = ContingentPlanner(bundle.template, bundle.arrays)
    restored = planner.solve()
    assert original == restored
    assert set(restored.branches) == {(), (0,), (0,0), (0,1)}
    # There is no further inference between the root command and its final response.
    monkeypatch.setattr(planner.engine, "run", lambda *args: pytest.fail("inference at execution"))
    assert restored.action([0,1]) in (0, 1)
    with pytest.raises(KeyError):
        restored.action((1,))


def test_sparse_vacuous_and_infeasible_preferences():
    with jax.enable_x64(False):
        obs = [np.array([[1., 0.], [.25, .75]])]
        pref = [np.array([1., 0.]), np.ones((2, 2, 2))]
        result = ContingentPlanner(*build_contingent("neutral", observations=obs, preferences=pref)).solve()
        assert set(result.branches) == {(), (0,)}
        assert result.action() == 0
        np.testing.assert_allclose(result.branches[(0,)].probabilities, [.5, .5],
                                   atol=16*np.finfo(np.float32).eps)
        pref[0][:] = 0
        planner = ContingentPlanner(*build_contingent("empty", observations=obs, preferences=pref))
        with pytest.raises(InferenceError) as failure:
            planner.solve()
        assert failure.value.result.status.value != "converged"


@pytest.mark.parametrize("open_loop", [False, True])
def test_paid_cue_uses_physical_outcomes_and_a_precommitted_response(open_loop):
    task = cue.CueTask(cue_quality=.9, reward_utility=4., cue_cost=1.)
    c = cue.goal_statistic(task)
    t, a = build_contingent_rollout("paid", depth=1, d=cue.initial_state(), Bs=cue.transition_stack(),
                                    A=cue.observation_matrix(task), c=c, open_loop=open_loop)
    with jax.enable_x64(False):
        policy = ContingentPlanner(t, a).solve()
    # Per-step log c = physical utility minus log Z; neutral has utility zero.
    realized = -2 * log(c[cue.NEUTRAL]) - policy.expected_loss
    if open_loop:
        assert policy.action() in (0, 1)
        assert realized == pytest.approx(0., abs=2e-5)
    else:
        assert policy.action() == 2
        assert policy.action((cue.CUE_LEFT,)) == 0
        assert policy.action((cue.CUE_RIGHT,)) == 1
        assert set(policy.branches) == {(), (cue.CUE_LEFT,), (cue.CUE_RIGHT,)}
        assert realized == pytest.approx(.9*4 - .1*4 - 1, abs=2e-5)


def test_support_shape_depth_and_size_admission():
    good = np.ones
    with pytest.raises(ValueError, match="combinatorial"):
        build_contingent("deep", observations=[None]*3, preferences=[None]*4)
    with pytest.raises(ValueError, match="sum to one"):
        build_contingent("likelihood", observations=[good((2, 2))], preferences=[good(2), good((2,2,2))])
    with pytest.raises(ValueError, match="observation history"):
        build_contingent("layout", observations=[good((2, 1))], preferences=[good(2), good((2,1,2))])
    with pytest.raises(ValueError, match="finite"):
        build_contingent("nan", observations=[good((2, 1))], preferences=[[1., np.nan], good((1,2,2))])
    with pytest.raises(ValueError, match="limit"):
        build_contingent("large", observations=[good((3,1))], preferences=[good(3),good((1,3,3))],max_entries=10)
    with pytest.raises(ValueError, match="finite"):
        build_contingent_rollout("prior", depth=1, d=[-.1, 1.1], A=np.eye(2),
                                 Bs=np.eye(2)[None], c=[.5, .5])


@pytest.mark.parametrize("open_loop", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_two_readings_require_two_distinct_contingent_moves(open_loop):
    # Buy both readings (-.25) or stay home (0). The first response must match
    # the first bit, and the second must name their parity to earn +1 vs -1.
    kernel = np.full((2, 2), .5)
    second = np.full((2, 2, 2, 2), .5)
    final = np.ones((2, 2, 2, 2, 2))
    for bit, next_bit, first, middle, last in np.ndindex(final.shape):
        payoff = 0 if first else (1 if middle == bit and last == bit ^ next_bit else -1)
        final[bit, next_bit, first, middle, last] = np.exp(payoff - 1)
    pref = [np.exp([-.25, 0]), np.ones((2, 2, 2)), final]
    t, a = build_contingent("two bits", observations=[kernel, second], preferences=pref,
                            open_loop=open_loop)
    result = ContingentPlanner(t, a).solve()
    assert result.action() == int(open_loop)
    assert 1 - result.expected_loss == pytest.approx(0 if open_loop else .75, abs=2e-5)
    if not open_loop:
        for bit, next_bit in np.ndindex(2, 2):
            assert result.action((bit,)) == bit
            assert result.action((bit, next_bit)) == bit ^ next_bit


def test_small_positive_cost_cannot_become_exactly_neutral():
    p = np.array([[1e-9, 1.], [1e-9, 1.]], dtype=np.float32)
    with pytest.raises(ValueError, match="rounds to neutral"):
        build_contingent("rare", observations=[p], preferences=[np.ones(2), np.full((2, 2, 2), .2)])
    with pytest.raises(ValueError, match="histories"):
        build_contingent("wide", observations=[np.full((1, 10), .1)],
                         preferences=[np.ones(1), np.ones((10, 1, 1))], max_histories=5)


def test_committed_tiny_loss_and_modified_bundle_admission():
    below_one = np.nextafter(np.float32(1), np.float32(0))
    t, a = build_contingent("near-neutral", observations=[np.full((2, 2), .5)],
                            preferences=[[below_one]*2, np.ones((2, 2, 2))])
    with jax.enable_x64(False):
        policy = ContingentPlanner(t, a).solve()
    # This term is smaller than one ulp of the prior cost at log(8).
    assert policy.expected_loss == -log(float(below_one))
    bad = dict(a, prior=np.array([.25, .75], dtype=np.float32))
    with pytest.raises(ValueError, match="uniform"):
        ContingentPlanner(t, bad)
    bad = dict(a)
    bad["response/0"] = a["response/0"].copy()
    bad["response/0"][:2] *= .5
    bad["response/0"][-1] = .75
    with pytest.raises(ValueError, match="masses"):
        ContingentPlanner(t, bad)
    with pytest.raises(ValueError, match="CPT entries"):
        ContingentPlanner(t, a, max_entries=1)


def test_large_rollout_refused_before_expansion(monkeypatch):
    sensor = np.full((32, 2), 1/32)
    Bs = np.broadcast_to(np.eye(2), (16, 2, 2))
    with monkeypatch.context() as check:
        check.setattr(np, "empty", lambda *args, **kwargs: pytest.fail("allocated before resource guard"))
        with pytest.raises(ValueError, match="limit"):
            build_contingent_rollout("oversize", depth=2, d=[.5, .5], A=sensor, Bs=Bs,
                                     c=np.full(32, 1/32))


def test_causal_forward_underflow_cannot_hide_a_zero_preference():
    tiny = 1e-200
    # The unseen second state has positive prior and can emit the forbidden
    # outcome. Their product is positive mathematically and below float64.
    with pytest.raises(ValueError, match="possible sensor"):
        build_contingent_rollout("underflow", depth=1, d=[1, tiny],
                                 A=[[1, 1], [0, tiny]], Bs=np.eye(2)[None],
                                 c=[1, 0], dtype=np.float64)
