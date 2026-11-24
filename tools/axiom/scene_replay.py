"""Follow recorded Gameworld transitions with the untuned AXIOM observation model."""

import argparse
import hashlib
import io
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import cffg

from cffg.experiments.axiom_observation import SceneSettings, initialize_scene, observe_scene


def reload_tree(state):
    """A development-only structural replay control with named NPZ arrays and no pickle."""
    entries, tree = jax.tree.flatten_with_path(state)
    params = {jax.tree_util.keystr(key): np.asarray(value) for key, value in entries}
    with io.BytesIO() as stream:
        np.savez_compressed(stream, **params)
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as data:
            result = tree.unflatten(
                [jnp.asarray(data[jax.tree_util.keystr(key)]) for key, _ in entries]
            )
    return result


def state_hashes(state):
    return {
        jax.tree_util.keystr(key): hashlib.sha256(np.asarray(val).tobytes()).hexdigest()
        for key, val in jax.tree.flatten_with_path(state)[0]
    }


def run(tapes, game, steps, dtype, seed, cfg=None):
    manifest = json.loads((tapes / "manifest.json").read_text())
    row = next(r for r in manifest["runs"] if r["game"] == game and r["condition"] == "base")
    archive = tapes / row["file"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != row["npz_sha256"]:
        raise ValueError("recording digest mismatch")
    settings = cfg or SceneSettings(len(row["actions"]))
    tick = 0
    with np.load(archive, allow_pickle=False) as data:
        if not 1 <= steps <= len(data["action"]):
            raise ValueError("requested real transition count exceeds recording")
        pixels = data["pixels"]
        pos = data["pre_frame"]
        post = data["post_frame"]
        actions = data["action"]
        rewards = data["reward"]
        initial, ok = initialize_scene(
            jnp.asarray(pixels[pos[0]]), jax.random.key(seed), settings, dtype=dtype
        )
        if not ok:
            raise ArithmeticError("invalid initial frame")
        state = initial
        step = jax.jit(partial(observe_scene, settings=settings))
        rows = []
        continuation = None
        restored_at = None

        def advance(image, action, reward, update):
            nonlocal tick, state, continuation, restored_at
            args = (
                jnp.asarray(image),
                jnp.int32(action),
                jnp.asarray(reward, dtype),
                jnp.int32(tick),
            )
            start = perf_counter()
            out = jax.block_until_ready(
                step(state, *args, update=jnp.bool_(update), reset=jnp.bool_(not update))
            )
            duration = perf_counter() - start
            if not bool(out.valid):
                raise ArithmeticError(
                    f"{game} invalid observation tick {tick}, update={update}, spatial status {int(out.status)}"
                )
            if continuation is not None:
                shadow = step(
                    continuation, *args, update=jnp.bool_(update), reset=jnp.bool_(not update)
                )
                if not bool(shadow.valid):
                    raise ArithmeticError("resumed update failed")
                continuation = shadow.state
            state = out.state
            history = dict(
                tick=tick,
                update=update,
                action=int(action),
                reward=float(reward),
                spatial_status=int(out.status),
                spatial_births=int(out.spatial_births),
                visible_slots=np.where(np.asarray(state.tracks.scene[:, 2]) == 0)[0].tolist(),
                identity_eligible=np.flatnonzero(out.identity_eligible).tolist(),
                tracked=np.flatnonzero(state.tracks.tracked).tolist(),
                identity_born=np.asarray(out.identity_born)[
                    np.asarray(out.identity_born) >= 0
                ].tolist(),
                motion_born=np.asarray(out.motion_born)[np.asarray(out.motion_born) >= 0].tolist(),
                interaction_born=np.asarray(out.interaction_born)[
                    np.asarray(out.interaction_born) >= 0
                ].tolist(),
                interaction_writes=int(out.interaction_writes.sum()),
                identity_writes=int(out.identity_writes.sum()),
                active_identities=int(state.identity.posterior.active.sum()),
                active_interactions=int(state.interaction.posterior.active.sum()),
                active_motion=int(state.motion.occupied.sum()),
                observed_step_s=duration,
            )
            rows.append(history)
            tick += int(update)
            if update and tick == steps // 2:
                continuation = reload_tree(state)
                restored_at = tick

        for i in range(steps):
            if i and pos[i] != post[i - 1]:
                advance(pixels[pos[i]], 0, 0, False)
            advance(pixels[post[i]], actions[i], rewards[i], True)
    exact = continuation is not None and state_hashes(continuation) == state_hashes(state)
    if not exact:
        raise ArithmeticError("restart failed to reproduce final state")
    return dict(
        source_sha256={
            str(p.relative_to(Path(cffg.__file__).parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(cffg.__file__).parent.rglob("*.py"))
        },
        replay_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        backend=str(jax.devices()[0]),
        jax=jax.__version__,
        game=game,
        model_seed=seed,
        recording_sha256=row["npz_sha256"],
        dtype=np.dtype(dtype).name,
        config=dict(
            slots=settings.slots,
            identities=settings.banks.identities,
            motions=settings.banks.motions,
            interactions=settings.banks.motions * settings.banks.interactions_per_motion,
        ),
        real_actions=steps,
        reset_observations=len(rows) - steps,
        exact_reloaded_at_tick=restored_at,
        final_all_arrays_equal=exact,
        final_state_sha256=state_hashes(state),
        trace=rows,
        scope="Fixed-action development replay. Tracks, identity, motion and interaction learning use pixels, actual actions and rewards. Timing includes synchronized observation model calls; no planner, game return comparison or scheduled reduction.",
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("tapes", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--games", nargs="+", default=["Explode", "Bounce", "Aviate"])
    cli.add_argument("--steps", type=int, default=48)
    cli.add_argument("--seed", type=int, default=17)
    cli.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = cli.parse_args()
    if args.dtype == "float64" and not jax.config.x64_enabled:
        raise SystemExit("enable JAX_ENABLE_X64 for the untuned study")
    results = []
    for game in args.games:
        result = run(args.tapes, game, args.steps, getattr(jnp, args.dtype), args.seed)
        results.append(result)
        print(game, result["real_actions"], result["trace"][-1]["active_interactions"], flush=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
