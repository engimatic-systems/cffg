"""Run 24 closed-loop development actions on Explode with the qualified game overlay."""

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

GAME_SEED = 270037
MODEL_SEED = 87037
ACTIONS = 24
RELOAD_AT = 12
HERE = Path(__file__).resolve()
DEFAULT_REFERENCE = Path.home() / "agent-share/axiom-gameworld-reference"
DEFAULT_OUTPUT = Path.home() / "agent-share/axiom-online-precision/acted-smoke.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def game_worker(root):
    """Only this Python 3.11 process imports Gymnasium and the MIT simulator."""
    import gymnasium as gym
    from gameworld_replay import ACTIONS as LABELS, GameRandom, pixels, source_identity
    from reference_protocol import EVIDENCE

    for name in ("overlay/uv.lock", "tapes/manifest.json"):
        if sha((root / name).read_bytes()) != EVIDENCE[name]:
            raise ValueError(f"qualified reference file changed: {name}")
    runtime = source_identity(root / "gameworld")
    if runtime != json.loads((root / "tapes/manifest.json").read_text())["runtime"]:
        raise ValueError("simulator runtime differs from the qualified reference")
    random = GameRandom(GAME_SEED)
    env = random.call(gym.make, "Gameworld-Explode-v0", perturb=None, perturb_step=None)
    tick, episode, needs_reset = 0, -1, True
    try:
        assert env.action_space.n == 3 and env.action_space.start == 0
        print(json.dumps({"runtime": runtime, "actions": LABELS["Explode"]}), flush=True)
        for line in sys.stdin:
            command = json.loads(line)
            if command == "reset" and needs_reset:
                frame, _ = random.call(env.reset)
                needs_reset = False
                episode += 1
                row = {"event": "reset", "episode": episode, "tick": tick}
            elif isinstance(command, dict) and not needs_reset:
                action = command["action"]
                if command["tick"] != tick or type(action) is not int or not env.action_space.contains(action):
                    raise ValueError("out-of-order or invalid native command")
                frame, reward, term, trunc, _ = random.call(env.step, action)
                row = {"event": "step", "applied": action, "reward": float(reward),
                       "terminated": bool(term), "truncated": bool(trunc), "episode": episode, "tick": tick}
                tick += 1
                needs_reset = term or trunc
            else:
                raise ValueError("unexpected command at this simulator clock")
            raw = pixels(frame).tobytes()
            row.update(rgb=base64.b64encode(raw).decode("ascii"), rgb_sha256=sha(raw))
            print(json.dumps(row, allow_nan=False), flush=True)
    finally:
        random.call(env.close)


@contextmanager
def simulator(reference):
    cmd = ["uv", "run", "--frozen", "--no-sync", "python", "-u", str(HERE),
           "--game-worker", "--reference", str(reference)]
    env = os.environ.copy()
    for variable in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(variable, None)
    with subprocess.Popen(cmd, env=env, cwd=reference / "overlay", stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True) as worker:
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError(f"game worker exited {worker.poll()}")
            return json.loads(line)

        def request(command):
            import numpy as np
            worker.stdin.write(json.dumps(command) + "\n")
            worker.stdin.flush()
            row = receive()
            raw = base64.b64decode(row.pop("rgb"), validate=True)
            if len(raw) != 210 * 160 * 3 or sha(raw) != row["rgb_sha256"]:
                raise ValueError("simulator RGB transport failed")
            return np.frombuffer(raw, np.uint8).reshape(210, 160, 3), row

        try:
            yield request, receive()
        finally:
            worker.stdin.close()
            code = worker.wait(timeout=10)
            if code:
                raise RuntimeError(f"simulator worker exited {code}")


def arrays(state):
    import jax
    import numpy as np
    leaves, template = jax.tree.flatten_with_path(state)
    return {jax.tree_util.keystr(key): np.asarray(a) for key, a in leaves}, template


def tree_digest(state):
    leaves, _ = arrays(state)
    rows = [(key, a.dtype.str, a.shape, sha(a.tobytes())) for key, a in leaves.items()]
    return sha(json.dumps(rows).encode())


def reload_checkpoint(payload, template, result):
    """Restore numeric leaves from an actual file into an independently supplied tree layout."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    leaves, _ = arrays(payload)
    layout, tree = arrays(template)
    with tempfile.TemporaryDirectory(prefix="cffg-acted-") as directory:
        path = Path(directory) / "scene-and-planner.npz"
        start = perf_counter()
        np.savez(path, **leaves)
        result["write_s"] = perf_counter() - start
        result["bytes"] = path.stat().st_size
        result["archive_sha256"] = sha(path.read_bytes())
        start = perf_counter()
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(layout):
                raise ValueError("scene checkpoint inventory differs")
            decoded = []
            for key, expected in layout.items():
                a = archive[key]
                if a.dtype != expected.dtype or a.shape != expected.shape:
                    raise ValueError(f"checkpoint leaf layout differs: {key}")
                decoded.append(jnp.asarray(a))
        restored = jax.block_until_ready(tree.unflatten(decoded))
        result["read_s"] = perf_counter() - start
    result["leaf_count"] = len(leaves)
    result["tree_sha256"] = tree_digest(restored)
    if tree_digest(payload) != result["tree_sha256"]:
        raise ValueError("exact model/proposal/random key reload mismatch")
    return restored


def describe(samples):
    if not samples:
        return {"n": 0}
    return dict(n=len(samples), minimum=min(samples), median=statistics.median(samples), maximum=max(samples))


def source_files():
    paths = {HERE, HERE.with_name("gameworld_replay.py"), HERE.with_name("reference_protocol.py"),
             HERE.parents[2] / "uv.lock"}
    for name, module in sys.modules.copy().items():
        if name == "cffg" or name.startswith("cffg."):
            path = getattr(module, "__file__", None)
            if path and path.endswith(".py"):
                paths.add(Path(path).resolve())
    root = HERE.parents[2]
    tree = subprocess.check_output(["git", "ls-tree", "-rz", "HEAD"], cwd=root)
    tracked = {}
    for entry in tree.split(b"\0"):
        if entry:
            info, name = entry.split(b"\t", 1)
            tracked[os.fsdecode(name)] = info.split()[-1].decode()
    hashes = {}
    for p in sorted(paths):
        name = p.relative_to(root).as_posix()
        data = p.read_bytes()
        git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if tracked.get(name) != git_blob:
            raise ValueError(f"loaded source differs from the sealed Git revision: {name}")
        hashes[name] = sha(data)
    return hashes


def run(reference, report):
    from functools import partial
    import jax
    import jax.numpy as jnp
    import numpy as np
    from cffg.experiments.axiom_banks import identity_labels, identity_records, seeded_dormant
    from cffg.experiments.axiom_evidence import initialized_view
    from cffg.experiments.axiom_observation import SceneSettings, initialize_scene, observe_scene, scene_keys
    from cffg.experiments.axiom_planning import SearchSettings, initial_proposal, plan
    from cffg.experiments.axiom_rollout import WorldStatus, freeze_world, rollout_step

    cfg = SceneSettings(3)
    features = cfg.interaction_config()
    search = SearchSettings(candidates=16, replicates=3)
    proposal = initial_proposal(3, horizon=8)
    model_key = jax.random.PRNGKey(MODEL_SEED)
    key = jax.random.fold_in(model_key, 0x504C414E)
    observe = jax.jit(partial(observe_scene, settings=cfg))
    context = jax.jit(initialized_view)
    labels = jax.jit(lambda bank, x: identity_labels(bank, identity_records(x)))
    decide = jax.jit(lambda model, scene, prev, k: plan(partial(rollout_step, config=features), model,
                                                       scene, prev, k, settings=search))
    trace, resets, seen_shapes = report["trace"], report["reset_observations"], set()
    sources = source_files()
    report.update(runtime=dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
                               jax_x64=jax.config.x64_enabled, model_dtype="float32", device=str(jax.devices()[0])),
                  source_sha256=sources,
                  head_at_start=subprocess.check_output(["git","rev-parse","HEAD"],text=True,
                                                        cwd=HERE.parents[2]).strip())
    with simulator(reference) as (request, game):
        report["simulator"] = game
        image, receipt = request("reset")
        report["initial_reset"] = receipt
        start = perf_counter()
        state, valid = jax.block_until_ready(initialize_scene(image, model_key, cfg))
        report["initialize_scene_s"] = perf_counter() - start
        if not bool(valid) or int(state.tick) != 0:
            raise ArithmeticError("invalid initial scene or clock")
        template = (state, proposal, key)
        keys = scene_keys(model_key)
        start = perf_counter()
        virgin_identity, virgin_relation = jax.block_until_ready((
            seeded_dormant(state.identity.prior, keys.identity),
            seeded_dormant(state.interaction.prior, keys.interaction)))
        report["seeded_dormant_init_s"] = perf_counter() - start

        def choose(current, prev, stream):
            next_key, subkey = jax.random.split(stream)
            start = perf_counter()
            identity = context(current.identity.posterior, current.identity.generations, virgin_identity)
            current_ids, valid = labels(identity, current.tracks.scene)
            if not bool(valid):
                raise ArithmeticError("invalid identity readout")
            relation = context(current.interaction.posterior, current.interaction.generations, virgin_relation)
            model = jax.block_until_ready(freeze_world(relation, current.motion, current_ids,
                         current.tracks.tracked, current.absent_distance, config=features))
            if not bool(model.valid):
                raise ArithmeticError(f"invalid planner snapshot: {WorldStatus(int(model.status)).name}")
            middle = perf_counter()
            out = jax.block_until_ready(decide(model, current.tracks.scene, prev, subkey))
            end = perf_counter()
            if not bool(out.valid):
                raise ArithmeticError("invalid sampled decision")
            shape = (int(model.component_rows.size), int(model.focused_slots.size))
            return out, next_key, shape, middle - start, end - middle

        def assimilate(current, frame, action, reward, *, reset):
            before = int(current.tick)
            start = perf_counter()
            out = jax.block_until_ready(observe(current, jnp.asarray(frame), jnp.int32(action),
                jnp.float32(reward), jnp.int32(before), update=jnp.bool_(not reset), reset=jnp.bool_(reset)))
            dt = perf_counter() - start
            row = dict(observe_s=dt, observed_action=int(action), observed_reward=float(reward),
                       update=not reset, reset=reset, tick_before=before, tick_after=int(out.state.tick),
                       spatial_status=int(out.status), valid=bool(out.valid), spatial_births=int(out.spatial_births),
                       identity_writes=int(out.identity_writes.sum()), interaction_writes=int(out.interaction_writes.sum()),
                       identity_births=int((out.identity_born>=0).sum()), motion_births=int((out.motion_born>=0).sum()),
                       interaction_births=int((out.interaction_born>=0).sum()),
                       dynamics_eligible=int(out.dynamics_eligible.sum()),
                       tracked_after=int(out.state.tracks.tracked.sum()),
                       active_identity=int(out.state.identity.posterior.active.sum()),
                       active_motion=int(out.state.motion.occupied.sum()),
                       active_interaction=int(out.state.interaction.posterior.active.sum()))
            return out.state, row

        for t in range(ACTIONS):
            entry = {"tick": t, "pre_rgb_sha256": receipt["rgb_sha256"]}
            trace.append(entry)
            report["phase"] = f"plan at native action {t}"
            if int(state.tick) != t:
                raise RuntimeError("actual and model decision clocks disagree")
            before = tree_digest(state)
            out, next_key, shape, prep, solve = choose(state, proposal, key)
            after = tree_digest(state)
            if before != after:
                raise ArithmeticError("planning changed an observed model leaf")
            action = int(out.action)
            entry.update(planned=action, snapshot_s=prep, plan_s=solve, decision_s=prep+solve,
                         shape=list(shape), shape_seen=shape in seen_shapes,
                         observed_model_before=before, observed_model_after_plan=after,
                         total_score_range=float(out.total_return.max()-out.total_return.min()),
                         predicted_reward_range=float(out.reward.max()-out.reward.min()),
                         novelty_range=float(out.novelty.max()-out.novelty.min()))
            seen_shapes.add(shape)
            if t == RELOAD_AT:
                report["phase"] = "exact disk reload and next decision comparison"
                evidence = report["checkpoint"] = dict(at_decision=t)
                restored, old_proposal, old_key = reload_checkpoint((state, proposal, key), template, evidence)
                check, replay_key, _, _, _ = choose(restored, old_proposal, old_key)
                evidence.update(original_action=action, reloaded_action=int(check.action),
                                original_decision_sha256=tree_digest((out,next_key)),
                                reloaded_decision_sha256=tree_digest((check,replay_key)))
                if evidence["original_decision_sha256"] != evidence["reloaded_decision_sha256"]:
                    raise ArithmeticError("disk restart changed the next decision")
                if tree_digest(restored) != before:
                    raise ArithmeticError("replanned restored scene changed")
                state, out, next_key, action = restored, check, replay_key, int(check.action)
            proposal, key = out.proposal, next_key
            report["phase"] = f"step and observe native action {t}"
            image, receipt = request({"action":action,"tick":t})
            if receipt["tick"] != t or receipt["applied"] != action:
                raise RuntimeError("game and model applied commands differ")
            entry["game"] = receipt
            state, observed = assimilate(state,image,receipt["applied"],receipt["reward"],reset=False)
            entry["observation"] = observed
            if not observed["valid"] or observed["tick_after"] != t + 1:
                raise ArithmeticError(f"failed real observation: {observed}")
            if receipt["terminated"] or receipt["truncated"]:
                image, receipt = request("reset")
                state, row = assimilate(state,image,0,0,reset=True)
                resets.append(dict(game=receipt, observation=row))
                if (not row["valid"] or row["tick_after"] != t+1 or row["identity_writes"]
                        or row["interaction_writes"] or row["motion_births"]):
                    raise ArithmeticError("reset updated a learned transition or advanced its clock")
            print(f"action {t+1}/{ACTIONS}, tracked={observed['tracked_after']}, rMM={observed['active_interaction']}",
                  file=sys.stderr, flush=True)
        report["final_scene_sha256"] = tree_digest(state)
    current = source_files()
    changed = [path for path, digest in sources.items() if current.get(path) != digest]
    report["source_sha256"] = current
    if changed:
        raise RuntimeError(f"source files changed on disk during the run: {changed}")
    warm = [row for row in trace if row["shape_seen"]]
    report["summary"] = dict(real_actions=len(trace), resets=len(resets), model_hash_checks=len(trace),
        native_reward_sum=sum(row["game"]["reward"] for row in trace),
        applied_histogram=np.bincount([row["game"]["applied"] for row in trace], minlength=3).tolist(),
        decisions_with_score_spread=sum(row["total_score_range"]>0 for row in trace),
        warm_decision_s=describe([row["decision_s"] for row in warm]),
        warm_snapshot_s=describe([row["snapshot_s"] for row in warm]),
        warm_plan_s=describe([row["plan_s"] for row in warm]),
        first_by_shape_decision_s=describe([row["decision_s"] for row in trace if not row["shape_seen"]]),
        first_observe_s=trace[0]["observation"]["observe_s"],
        warm_observe_s=describe([row["observation"]["observe_s"] for row in trace[1:]]),
        total_identity_writes=sum(row["observation"]["identity_writes"] for row in trace),
        total_interaction_writes=sum(row["observation"]["interaction_writes"] for row in trace),
        final_active={k:trace[-1]["observation"]["active_"+k] for k in ("identity","motion","interaction")})
    report["source_reverified"] = True
    report["passed"] = True
    report["phase"] = "complete"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference",type=Path, default=DEFAULT_REFERENCE)
    cli.add_argument("--output",type=Path, default=DEFAULT_OUTPUT)
    cli.add_argument("--game-worker",action="store_true",help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.game_worker:
        game_worker(args.reference.resolve())
        return
    report = dict(passed=False, phase="initialization", game="Explode", model_seed=MODEL_SEED,
        game_seed=GAME_SEED, real_budget=ACTIONS, checkpoint_at=RELOAD_AT, slots=32, identities=32,
        motions=500, interactions=5000, horizon=8,candidates=16,replicates=3, trace=[],reset_observations=[],
        scope="Single development trajectory. Warm decisions reuse both compiled bucket shapes. Timings block on JAX outputs; model hashes, checkpoint, game IPC and reset are measured outside per-decision timing.")
    try:
        run(args.reference.resolve(),report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w",dir=args.output.parent,delete=False) as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        Path(stream.name).replace(args.output)


if __name__ == "__main__":
    main()
