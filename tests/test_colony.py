"""Colony: structure-hash batching, vmapped ticks, host port routing."""

import numpy as np

from cffg.colony import Colony
from cffg.compiler import compile_template
from cffg.engine import Engine
from cffg.models.gridworld import build_gridworld
from cffg.models.tmaze import build_tmaze

N_ITERS = 60


def spawn_gridworlds(colony, starts, seed_obs=None):
    ids = []
    for s in starts:
        t, arrays = build_gridworld(k=6, start=s)
        ids.append(colony.spawn(t, arrays))
    return ids


def test_batched_equals_sequential():
    colony = Colony()
    starts = [0, 1, 2, 3, 4]
    ids = spawn_gridworlds(colony, starts)
    colony.run(N_ITERS, damp=0.5)

    for s, iid in zip(starts, ids):
        t, arrays = build_gridworld(k=6, start=s)
        engine = Engine(compile_template(t, arrays=arrays))
        solo, _ = engine.run(engine.init_state(), N_ITERS, damp=0.5)
        ref = engine.marginals(solo)
        got = colony.marginals(iid)
        for var in ("u1", "zm1"):
            np.testing.assert_allclose(got[var], ref[var], atol=1e-10, err_msg=f"start={s} {var}")


def test_one_engine_per_structure():
    colony = Colony()
    spawn_gridworlds(colony, [0, 1, 2, 3])  # same structure, different d
    t, arrays = build_tmaze()
    colony.spawn(t, arrays)  # different structure
    assert len(colony) == 5
    assert len(colony.engines) == 2  # 5 instances, 2 compilations


def test_mixed_structures_run_and_agree():
    colony = Colony()
    gw_ids = spawn_gridworlds(colony, [0, 2])
    t, arrays = build_tmaze()
    tm_id = colony.spawn(t, arrays)
    colony.run(400, damp=0.5)

    # T-maze member reproduces the golden cue-first posture inside the colony.
    u1 = colony.marginals(tm_id)["u1"]
    assert int(np.argmax(u1)) == 3
    # Gridworld members still plan rightward.
    for iid in gw_ids:
        assert int(np.argmax(colony.marginals(iid)["u1"])) == 2


def test_kill_shrinks_batch_without_corruption():
    colony = Colony()
    ids = spawn_gridworlds(colony, [0, 1, 2])
    colony.kill(ids[1])
    assert len(colony) == 2
    colony.run(N_ITERS, damp=0.5)
    t, arrays = build_gridworld(k=6, start=2)
    engine = Engine(compile_template(t, arrays=arrays))
    solo, _ = engine.run(engine.init_state(), N_ITERS, damp=0.5)
    np.testing.assert_allclose(
        colony.marginals(ids[2])["u1"], engine.marginals(solo)["u1"], atol=1e-10
    )


def test_host_port_routing():
    """The host clamps one instance's data port; only that instance moves."""
    colony = Colony()
    t, arrays = build_gridworld(past=1, steps=1, k=6)  # perception model
    arrays = {**arrays, "d": np.full(6, 1 / 6)}  # uniform prior: evidence decides
    a = colony.spawn(t, arrays, observations={"act1": 2, "obs1": 1})
    b = colony.spawn(t, arrays, observations={"act1": 2, "obs1": 4})
    colony.run(N_ITERS)
    qa, qb = colony.marginals(a)["zm1"], colony.marginals(b)["zm1"]
    assert int(np.argmax(qa)) == 1
    assert int(np.argmax(qb)) == 4  # same graph, different evidence, different belief


def test_bind_unknown_port_raises():
    import pytest

    colony = Colony()
    (iid,) = spawn_gridworlds(colony, [0])
    with pytest.raises(KeyError, match="no DATA port"):
        colony.bind(iid, {"nonexistent": 1})
