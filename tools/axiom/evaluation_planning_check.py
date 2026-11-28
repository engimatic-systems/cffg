"""Compare every sampled class against the direct planner on a qualified Explode development tape."""

import argparse
import hashlib
import json
import tempfile
from functools import partial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from cffg.experiments.axiom_observation import observe_scene
from cffg.experiments.axiom_planning import SearchSettings, _candidates, refit
from cffg.experiments.axiom_rollout import freeze_world, predict_scene
from cffg.rollout_planning import sample_action_sequences
from evaluation_actor import Actor, _chunk, _context, _labels, _plan
from evaluation_planning import cached_plan


@partial(jax.jit, static_argnames=("config", "settings"))
def oracle(model, scene, proposal, key, *, config, settings):
    pkey, sim = jax.random.split(key)
    policies, valid = _candidates(proposal, jnp.ones(config.actions, bool), settings, pkey)
    focus = model.focused_slots
    take = jnp.minimum(focus, scene.shape[0] - 1)
    def step(m, s, a, k):
        out = predict_scene(m, s, a, k, config=config)
        return out.state, (out.reward, out.novelty, jnp.where(focus < len(s), out.component[take], -1),
                           jnp.where(focus < len(s), out.motion[take], -1))
    _, (r, b, component, motion) = sample_action_sequences(step, model, scene, policies, sim,
        replicates=settings.replicates, batch_size=settings.batch_size)
    return refit(policies, r.mean(axis=1), b.mean(axis=1), proposal, settings, valid=valid), component, motion


def duplicate_live(bank, n):
    """Exercise 100+ live rules: repeat actual trained prototypes with identical conditional heads."""
    live = np.flatnonzero(np.asarray(bank.active))
    if not len(live) or n > bank.active.size:
        raise ValueError("cannot build the high-occupancy control")
    source = np.resize(live, n)
    return jax.tree.map(lambda a: a.at[:n].set(a[source]), bank)._replace(
        active=jnp.arange(bank.active.size)<n)


def compare(actor, synthetic_count=None):
    s = actor.scene
    identity = _context(s.identity.posterior, s.identity.generations, actor.virgin_identity)
    labels, ok = _labels(identity, s.tracks.scene)
    assert ok
    bank = _context(s.interaction.posterior, s.interaction.generations, actor.virgin_interaction)
    if synthetic_count:
        bank = duplicate_live(bank, synthetic_count)
    m = freeze_world(bank, s.motion, labels, s.tracks.tracked, s.absent_distance, config=actor.config)
    cfg = actor.config
    key = jax.random.fold_in(actor.key, actor.tick)
    chunk = _chunk(m.component_rows.size, m.focused_slots.size)
    settings = SearchSettings(batch_size=chunk)
    expected, components, motions = jax.block_until_ready(oracle(m, s.tracks.scene, actor.proposal, key,
                                                        config=cfg, settings=settings))
    assert expected.valid
    trace = [np.asarray(a).reshape(512*3, 32, -1) for a in (components, motions)]
    mismatch = [0,0]
    def inspect(t, comp, mode):
        for i, actual in enumerate((comp, mode)):
            mismatch[i] += int(np.count_nonzero(actual != trace[i][:, t]))
    actual, info = cached_plan(m, s.tracks.scene, actor.proposal, key, config=cfg, settings=settings, visit=inspect)
    assert actual.valid
    drift = {name:float(np.max(np.abs(np.asarray(getattr(expected,name))-np.asarray(getattr(actual,name)))))
             for name in ("reward","novelty","discounted_return","total_return")}
    same_actions = bool(np.array_equal(expected.policies, actual.policies) and expected.action==actual.action
                        and np.array_equal(expected.proposal.best,actual.proposal.best))
    support = lambda result: np.asarray(result.proposal.probabilities) > 0
    same_support = np.array_equal(support(expected), support(actual))
    probs = float(np.max(np.abs(np.asarray(expected.proposal.probabilities)-np.asarray(actual.proposal.probabilities))))
    args = (m,s.tracks.scene,actor.proposal,key)
    direct = _plan.lower(*args,config=cfg,batch_size=chunk).compile()
    jax.block_until_ready(direct(*args))
    start = perf_counter()
    jax.block_until_ready(direct(*args))
    direct_s = perf_counter()-start
    start = perf_counter()
    jax.block_until_ready(cached_plan(*args,config=cfg,settings=settings)[0])
    cached_s = perf_counter()-start
    return dict(tick=actor.tick, synthetic_repeated_prototypes=bool(synthetic_count), active=int(bank.active.sum()),
                tracked=int(s.tracks.tracked.sum()), outcome_evaluations=512*3*32*int(s.tracks.tracked.sum()),
                component_mismatches=mismatch[0],motion_mismatches=mismatch[1],same_winner_and_action=same_actions,
                same_policy_support=bool(same_support), proposal_max_abs=probs,score_max_abs=drift,
                direct_s=direct_s,cached_s=cached_s,direct_chunk=chunk,
                direct_temp_bytes=direct.memory_analysis().temp_size_in_bytes,**info)


def run(reference, output):
    tape_path = reference/"tapes/Explode/base.npz"
    manifest = json.loads((reference/"tapes/manifest.json").read_text())
    entry = next(r for r in manifest["runs"] if r["game"]=="Explode" and r["condition"]=="base")
    assert hashlib.sha256(tape_path.read_bytes()).hexdigest()==entry["npz_sha256"]
    rows = []
    with np.load(tape_path,allow_pickle=False) as tape:
        actor = Actor(tape["pixels"][0],"Explode",3,87037)
        observe = jax.jit(partial(observe_scene,settings=actor.settings))
        # Real recorded actions train the model here; hypothetical qualification
        # queries leave the development tape and the posterior unchanged.
        for t in range(15):
            if t in (3,8,14):
                rows.append(compare(actor))
                if t==14:
                    rows.append(compare(actor,128))
                print(rows[-1],flush=True)
                output.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
            action = tape["action"][t]
            result = observe(actor.scene,tape["pixels"][tape["post_frame"][t]],jnp.int32(action),
                             jnp.float32(tape["reward"][t]),jnp.int32(t))
            assert result.valid and not tape["terminated"][t] and not tape["truncated"][t]
            actor.scene = result.state
    for r in rows:
        assert r["component_mismatches"]==r["motion_mismatches"]==0
        assert r["same_winner_and_action"] and r["same_policy_support"]
        assert r["proposal_max_abs"]<=1e-6 and max(r["score_max_abs"].values())<=2e-6


def acted(reference, output):
    """Check the full actor through real transitions and one populated disk restart."""
    from acted_smoke import GAME_SEED, MODEL_SEED, simulator, tree_digest
    from evaluation_checkpoint import save, restore

    rows = []
    with simulator(reference) as (env, _), tempfile.TemporaryDirectory() as directory:
        initial, _ = env("reset")
        actor = Actor(initial, "Explode", 3, MODEL_SEED, audit_plan=True)
        for tick in range(8):
            resumed = None
            if tick == 5:
                path = Path(directory) / "checkpoint.npz"
                sources = hashlib.sha256()
                from cffg import axiom_agent
                from cffg.experiments import axiom_cached_plan
                for module in (axiom_agent, axiom_cached_plan):
                    sources.update(Path(module.__file__).read_bytes())
                metadata = dict(source_sha256=sources.hexdigest(), stage="development",
                                game="Explode", model_seed=MODEL_SEED, game_seed=GAME_SEED,
                                goal_steps=8, tick=tick, trace_sha256=tree_digest(rows))
                save(path, actor, metadata)
                resumed = Actor(initial, "Explode", 3, MODEL_SEED, audit_plan=True)
                restore(path, resumed, metadata)
            action = actor.choose()
            decision = actor.last_decision.copy()
            if resumed is not None:
                assert decision["active"] > 0 and decision["tracked"] > 0
                assert resumed.choose() == action and resumed.last_decision["plan_sha256"] == decision["plan_sha256"]
                decision["matched_reloaded_plan_sha256"] = resumed.last_decision["plan_sha256"]
            frame, outcome = env({"tick": tick, "action": action})
            receipt = actor.observe(frame, outcome["applied"], outcome["reward"])
            assert receipt["valid"] and actor.tick == tick + 1
            rows.append(dict(decision=decision, observation=receipt, rgb_sha256=outcome["rgb_sha256"]))
            if outcome["terminated"] or outcome["truncated"]:
                frame, _ = env("reset")
                assert actor.observe(frame, 0, 0, reset=True)["valid"]
    output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path.home() / "agent-share/axiom-gameworld-reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acted", action="store_true", help="run eight real development decisions and test restart")
    args = parser.parse_args()
    (acted if args.acted else run)(args.reference, args.output)
