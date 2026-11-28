"""Exact restart at the action, observation and scheduled reduction boundaries."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.component_bank_device import birth
from cffg.experiments.axiom_banks import BankSettings
from cffg.experiments.axiom_observation import SceneSettings
from cffg.neural_artifacts import write_artifact


@contextmanager
def evaluation():
    # Both interpreters reconstruct the same fixed, reduced-capacity RGB actor.
    with pytest.MonkeyPatch.context() as mp, jax.enable_x64(False):
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools/axiom"))
        import evaluation_actor
        import evaluation_checkpoint

        def settings(actions):
            return SceneSettings(actions, slots=2,
                                 banks=BankSettings(identities=2, motions=4, interactions_per_motion=1))

        with patch.object(evaluation_actor, "SceneSettings", settings):
            yield evaluation_actor.Actor, evaluation_checkpoint


def frame():
    rgb = np.zeros((210, 160, 3), np.uint8)
    rgb[45:80, 20:40, 0] = 200
    return rgb


def meta(tick):
    return dict(stage="development", game="fixture", model_seed=71, game_seed=19, goal_steps=501,
                tick=tick, trace_sha256=hashlib.sha256(str(tick).encode()).hexdigest(),
                source_sha256="a" * 64)


def fresh(factory):
    return factory(frame(), "fixture", 3, 71, audit_plan=True)


def at_first_reduction(actor):
    # A live component with positive mixing evidence permits a real scheduled draw.
    bank = birth(actor.scene.interaction).bank
    b = bank.posterior
    alpha = b.alpha.at[0].set(1.1)
    low = b.carry.alpha.at[0].set(np.spacing(np.float32(1.1)) / 4)
    b = b._replace(alpha=alpha, carry=b.carry._replace(alpha=low))
    actor.scene = actor.scene._replace(interaction=bank._replace(posterior=b), tick=jnp.int32(499))


def advance(actor, tick):
    action = actor.choose()
    plan_digest = actor.last_decision["plan_sha256"]
    receipt = actor.observe(np.roll(frame(), tick % 3, axis=0), action, 1)
    assert actor.tick == tick and actor.pending is None
    if tick == 500:
        assert receipt["reduction"]["records_total"] == 2_000
        assert np.any(actor.reduction.records.features != 0)
    return action, plan_digest


def restarted(directory):
    with evaluation() as (factory, codec):
        actor = fresh(factory)
        codec.restore(directory / "pending.npz", actor, meta(499))
        assert actor.pending is not None
        first = advance(actor, 500)
        boundary = directory / "resumed-500.npz"
        codec.save(boundary, actor, meta(500))
        actor = fresh(factory)
        codec.restore(boundary, actor, meta(500))
        second = advance(actor, 501)
        codec.save(directory / "resumed-501.npz", actor, meta(501))
        return [first, second]


@pytest.mark.slow
def test_fresh_process_pending_and_observed_restart_across_reduction(tmp_path):
    with evaluation() as (factory, codec):
        actor = fresh(factory)
        at_first_reduction(actor)
        actor.choose()
        codec.save(tmp_path / "pending.npz", actor, meta(499))
        expected = []
        for tick in (500, 501):
            expected.append(list(advance(actor, tick)))
            codec.save(tmp_path / f"original-{tick}.npz", actor, meta(tick))
    replay = subprocess.check_output([sys.executable, __file__, str(tmp_path)], text=True)
    assert json.loads(replay) == expected  # Includes every candidate, its scores and the next key.
    for tick in (500, 501):
        with (np.load(tmp_path / f"original-{tick}.npz") as original,
              np.load(tmp_path / f"resumed-{tick}.npz") as resumed):
            assert set(original.files) == set(resumed.files)
            for field in original.files:
                a, b = original[field], resumed[field]
                assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes()), field


def test_rejects_wrong_run_corruption_and_invalid_resigned_state_atomically(tmp_path):
    path = tmp_path / "actor.npz"
    with evaluation() as (factory, codec):
        actor = fresh(factory)
        codec.save(path, actor, meta(0))
        with np.load(path) as saved:
            fields = {k: saved[k] for k in saved.files if k != "metadata"}
            header = json.loads(saved["metadata"].item())
        for key, value in (("stage", "evaluation"), ("goal_steps", 10000), ("game", "other"),
                           ("model_seed", 72), ("game_seed", 7), ("tick", 1),
                           ("source_sha256", "b" * 64), ("trace_sha256", "0" * 64)):
            with pytest.raises(ValueError):
                codec.restore(path, actor, {**meta(0), key: value})

        before = (actor.scene, actor.proposal, actor.key, actor.reduction, actor.pending)

        def reject(payload, info, *, resign=False):
            info = dict(info)
            if resign:
                info.pop("sha256")
                info["sha256"] = codec._digest(info, payload)
            write_artifact(path, info, payload)
            with pytest.raises(ValueError):
                codec.restore(path, actor, meta(0))
            after = (actor.scene, actor.proposal, actor.key, actor.reduction, actor.pending)
            assert all(a is b for a, b in zip(before, after, strict=True))

        reject(fields, {**header, "version": 2})
        reject({**fields, "scene[2]": fields["scene[2]"] ^ np.uint32(1)}, header)
        reject({**fields, "scene[0].absent_distance": np.full(2, np.nan, np.float32)}, header)
        for name, value in (
            ("scene[0].tick", np.int32(1)),
            ("scene[2]", fields["scene[2]"].astype(np.uint64)),
            ("scene[0].absent_distance", fields["scene[0].absent_distance"] + 1),
            ("scene[0].identity.next_id", np.array([0, 42], np.uint32)),
            ("scene[1].best", np.full_like(fields["scene[1].best"], actor.actions)),
            ("identity", np.asarray("different reset image")),
            ("reduction/categories", fields["reduction/categories"].astype(np.uint16)),
            ("reduction/key_impl", np.asarray("rbg")),
            ("reduction/last_step", np.int64(500)),
        ):
            reject({**fields, name: value}, header, resign=True)
        reject({k: v for k, v in fields.items() if k != "scene[0].tracks.use"}, header, resign=True)


def small_public():
    from cffg.axiom_agent import AxiomAgent

    settings = SceneSettings(3, slots=2, banks=BankSettings(identities=2, motions=4, interactions_per_motion=1))
    return AxiomAgent(frame(), "fixture", 3, 71, settings=settings, audit_plan=True)


def test_public_pending_clock_applied_override_and_atomic_reset():
    with jax.enable_x64(False):
        actor = small_public()
        old = actor.propose()
        assert old == actor.propose()
        before = actor.scene
        actor.abandon(old)
        assert before is actor.scene
        plan = actor.propose()
        assert plan.key != old.key
        with pytest.raises(ValueError, match="stale"):
            actor.observe_applied(old, frame(), old.action, 1)
        applied = (plan.action + 1) % 3
        with pytest.raises(ValueError, match="frames"):
            actor.observe_applied(plan, frame(), applied, 1, reset_rgb=np.zeros((2, 2, 3), np.uint8))
        assert actor.scene is before and actor.propose() == plan
        receipt = actor.observe_applied(plan, frame(), applied, -2, reset_rgb=frame())
        assert receipt["applied"] == applied and receipt["reward"] == -2
        assert receipt["reset_observation"]["update"] is False
        assert receipt["tick_after"] == receipt["reset_observation"]["tick_after"] == actor.tick == 1
        with pytest.raises(ValueError, match="stale"):
            actor.observe_applied(plan, frame(), applied, -2)


def test_public_schema_checked_before_installing(tmp_path):
    from cffg import axiom_checkpoint as codec

    with jax.enable_x64(False):
        actor = small_public()
        at_first_reduction(actor)
        path = tmp_path / "actor.npz"
        actor.checkpoint(path, meta(499), exact=False)
        with np.load(path, allow_pickle=False) as file:
            header = json.loads(file["metadata"].item())
            payload = {k: file[k] for k in file.files if k != "metadata"}
        assert header["mode"] == "compact" and header["model"]["scene"]["slots"] == 2
        assert header["model"]["reduction"]["period"] == 500
        assert not any(".posterior.carry." in k or ".identity.prior." in k for k in payload)
        source = small_public()
        raw = actor.state_dict()
        raw["scene[0].tracks.use"] = np.full_like(raw["scene[0].tracks.use"], np.nan)
        with pytest.raises(ValueError, match="nonfinite"):
            source.load_state_dict(raw)
        before = source.scene, source.key, source.proposal, source.reduction
        bad_models = [dict(header["model"], plan_key_impl="rbg"),
                      dict(header["model"], scene=dict(header["model"]["scene"], slots=32))]
        for h, fields in [
            *((dict(header, model=m), payload) for m in bad_models),
            (dict(header, fixed_sha256="0"*64), payload),
            (dict(header, mode="exact"), payload),
            (header, {k:v for k,v in payload.items() if k != "scene[0].tracks.use"}),
            (header, dict(payload, **{"scene[0].identity.posterior.carry.mu": np.zeros(1)})),
        ]:
            h = dict(h)
            h.pop("sha256")
            h["sha256"] = codec._digest(h, fields)
            write_artifact(path, h, fields)
            with pytest.raises(ValueError):
                source.restore(path, meta(499))
            assert all(a is b for a, b in zip(before, (source.scene, source.key, source.proposal, source.reduction)))


@pytest.mark.slow
@pytest.mark.parametrize("exact", (True, False), ids=("exact", "compact"))
def test_installed_actor_restart_matches_lifecycle_oracle_in_fresh_python(tmp_path, exact):
    from cffg.component_bank import ComponentBank
    from cffg.component_bank_device import from_host, to_host

    with jax.enable_x64(False):
        original = small_public()
        at_first_reduction(original)
        pending = original.propose()
        original.checkpoint(tmp_path / "pending.npz", meta(499), exact=exact)
        assert original.propose() == pending
        assert np.any(original.scene.interaction.posterior.carry.alpha)
        if not exact:
            def rounded(device):
                record = to_host(device).state_dict(exact=False)
                return from_host(ComponentBank.from_state_dict(record))
            original.scene = original.scene._replace(identity=rounded(original.scene.identity),
                                                     interaction=rounded(original.scene.interaction))
        expected = []
        for tick in (500, 501):
            expected.append(list(advance(original, tick)))
            original.checkpoint(tmp_path / f"original-{tick}.npz", meta(tick))
    out = subprocess.check_output([sys.executable, __file__, str(tmp_path), "public"], text=True)
    assert json.loads(out) == expected
    for tick in (500, 501):
        with np.load(tmp_path / f"original-{tick}.npz") as a, np.load(tmp_path / f"resumed-{tick}.npz") as b:
            assert set(a.files) == set(b.files)
            for k in a.files:
                assert (a[k].shape, a[k].dtype, a[k].tobytes()) == (b[k].shape, b[k].dtype, b[k].tobytes()), k


def public_restarted(directory):
    actions = []
    with jax.enable_x64(False):
        path = directory / "pending.npz"
        for tick in (500, 501):
            actor = small_public()
            actor.restore(path, meta(tick-1))
            actions.append(advance(actor, tick))
            path = directory / f"resumed-{tick}.npz"
            actor.checkpoint(path, meta(tick))
    return actions


if __name__ == "__main__":
    run = public_restarted if sys.argv[2:] == ["public"] else restarted
    print(json.dumps(run(Path(sys.argv[1]))))
