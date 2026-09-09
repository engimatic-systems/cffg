"""Executable authority for the supported newcomer facade and checkpoint."""

from __future__ import annotations

import ast
import io
import json
import os
from functools import partial
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import jax
import numpy as np
import pytest

import cffg
from cffg import RollingAgent, build_gridworld
from cffg.agent import RollingAgent as CanonicalRollingAgent
from cffg.bundle import load_bundle as canonical_load_bundle
from cffg.bundle import save_bundle as canonical_save_bundle
from cffg.compiler import compile_template as canonical_compile_template
from cffg.core import BlobRef, FactorSpec, PlateSpec, Template, VariableSpec
from cffg.engine import Engine as CanonicalEngine
from cffg.engine import RunStatus as CanonicalRunStatus
from cffg.models.gridworld import CorridorEnv as CanonicalCorridorEnv
from cffg.models.gridworld import build_gridworld as canonical_build_gridworld
from cffg.params import set_params as canonical_set_params


REPO = Path(__file__).resolve().parents[1]
QUICKSTART_START = "<!-- public-quickstart:start -->"
QUICKSTART_END = "<!-- public-quickstart:end -->"


def _quickstart_source() -> str:
    readme = (REPO / "README.md").read_text()
    assert readme.count(QUICKSTART_START) == readme.count(QUICKSTART_END) == 1
    marked = readme.split(QUICKSTART_START, 1)[1].split(QUICKSTART_END, 1)[0]
    assert marked.count("```python") == 1
    assert marked.count("```") == 2
    return marked.split("```python", 1)[1].split("```", 1)[0].strip()


def _make_agent(
    *,
    k: int = 2,
    acc: float = 1.0,
    seed: int = 7,
    damp: float = 0.5,
    n_sweeps: int = 100,
    select: str = "sample",
    learn_goal: bool = False,
) -> RollingAgent:
    return RollingAgent(
        partial(build_gridworld, k=k, acc=acc),
        horizon=1,
        learn_b=True,
        initial_b_counts=np.ones((3, k, k)),
        learn_goal=learn_goal,
        initial_goal_counts=np.arange(1, k + 1, dtype=float),
        damp=damp,
        n_sweeps=n_sweeps,
        select=select,
        seed=seed,
    )


def _make_tied_parameter_agent(*, start: int) -> RollingAgent:
    def builder(**kwargs):
        template, arrays = build_gridworld(
            k=3,
            acc=1.0,
            start=start,
            **kwargs,
        )
        if kwargs["past"] == 0:
            template.factors["e1"].param_blobs["statistic"] = "d"
        return template, arrays

    return RollingAgent(builder, horizon=1, n_sweeps=100)


def _make_plated_parameter_agent(*, first_mass: float) -> RollingAgent:
    fragment = Template(
        name="static-fragment",
        variables={"z": VariableSpec("z", 2)},
        factors={
            "left": FactorSpec(
                "left",
                "categorical",
                edges={"out": "z"},
                param_blobs={"statistic": "plate_w"},
            ),
            "right": FactorSpec(
                "right",
                "categorical",
                edges={"out": "z"},
                param_blobs={"statistic": "plate_w"},
            ),
        },
        blobs={"plate_w": BlobRef("plate_w", "plate_w.npy")},
    )

    def builder(**kwargs):
        template, arrays = build_gridworld(k=2, acc=1.0, **kwargs)
        template.plates["static"] = PlateSpec(fragment, (2,))
        arrays["plate_w"] = np.asarray([first_mass, 1.0 - first_mass])
        return template, arrays

    return RollingAgent(builder, horizon=1, n_sweeps=100)


def _checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _write_checkpoint(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)


def _agent_snapshot(agent: RollingAgent):
    return (
        {name: value.copy() for name, value in agent.beliefs.items()},
        agent.learnables.state_dict(),
        {name: value.copy() for name, value in agent.goal_priors.items()},
        None
        if agent._pending_goal_priors is None
        else {
            name: value.copy()
            for name, value in agent._pending_goal_priors.items()
        },
        agent.tick,
        agent.episode_tick,
        json.dumps(agent.rng.bit_generator.state, sort_keys=True),
    )


def _assert_agent_snapshot(agent: RollingAgent, expected) -> None:
    beliefs, learnables, goal_priors, pending, tick, episode_tick, rng = expected
    assert set(agent.beliefs) == set(beliefs)
    for name, value in beliefs.items():
        np.testing.assert_array_equal(agent.beliefs[name], value)
    assert set(agent.learnables.state_dict()) == set(learnables)
    for name, value in learnables.items():
        np.testing.assert_array_equal(agent.learnables.state_dict()[name], value)
    assert set(agent.goal_priors) == set(goal_priors)
    for name, value in goal_priors.items():
        np.testing.assert_array_equal(agent.goal_priors[name], value)
    if pending is None:
        assert agent._pending_goal_priors is None
    else:
        assert set(agent._pending_goal_priors or {}) == set(pending)
        for name, value in pending.items():
            np.testing.assert_array_equal(agent._pending_goal_priors[name], value)
    assert agent.tick == tick
    assert agent.episode_tick == episode_tick
    assert json.dumps(agent.rng.bit_generator.state, sort_keys=True) == rng


def test_top_level_facade_reexports_only_canonical_owners():
    expected = {
        "CorridorEnv": CanonicalCorridorEnv,
        "Engine": CanonicalEngine,
        "RollingAgent": CanonicalRollingAgent,
        "RunStatus": CanonicalRunStatus,
        "build_gridworld": canonical_build_gridworld,
        "compile_template": canonical_compile_template,
        "load_bundle": canonical_load_bundle,
        "save_bundle": canonical_save_bundle,
        "set_params": canonical_set_params,
    }
    assert cffg.__all__ == list(expected)
    for name, owner in expected.items():
        assert getattr(cffg, name) is owner
    assert not hasattr(cffg, "hello")


@pytest.mark.parametrize(
    ("jax_enable_x64", "expected_dtype"),
    (("0", "np.float32"), ("1", "np.float64")),
)
def test_readme_quickstart_uses_only_the_facade_and_runs_in_configured_precision(
    tmp_path,
    jax_enable_x64,
    expected_dtype,
):
    source = _quickstart_source()
    syntax = ast.parse(source)
    cffg_imports = [
        node
        for node in ast.walk(syntax)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name == "cffg" or alias.name.startswith("cffg.") for alias in node.names)
            if isinstance(node, ast.Import)
            else node.module == "cffg" or (node.module or "").startswith("cffg.")
        )
    ]
    assert len(cffg_imports) == 1
    assert isinstance(cffg_imports[0], ast.ImportFrom)
    assert cffg_imports[0].module == "cffg"
    assert {alias.name for alias in cffg_imports[0].names} <= set(cffg.__all__)

    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = jax_enable_x64
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            source + f"\nassert state.messages.dtype == np.dtype({expected_dtype})",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "corridor.cffgz").is_file()
    assert (tmp_path / "agent.npz").is_file()


def test_checkpoint_envelope_round_trips_and_records_exact_fields(tmp_path):
    source = _make_agent()
    source.observe(1, 0)
    path = tmp_path / "agent.npz"
    source.checkpoint(path)

    arrays = _checkpoint_arrays(path)
    assert int(arrays["checkpoint_version"]) == 1
    manifest = json.loads(str(arrays["checkpoint_fields"]))
    assert manifest == sorted(arrays)
    assert len(arrays["plan_structure_hash"].item()) == 64
    assert len(arrays["filter_structure_hash"].item()) == 64
    assert len(arrays["plan_parameter_hash"].item()) == 64
    assert len(arrays["filter_parameter_hash"].item()) == 64
    assert int(arrays["horizon"]) == source.horizon
    assert float(arrays["damp"]) == source.damp
    assert int(arrays["n_sweeps"]) == source.n_sweeps
    assert str(arrays["select"]) == source.select
    assert bool(arrays["jax_enable_x64"]) is source._jax_enable_x64

    restored = _make_agent(seed=99)
    restored.restore(path)
    np.testing.assert_array_equal(restored.belief, source.belief)
    np.testing.assert_array_equal(restored.b_counts, source.b_counts)
    assert restored.tick == source.tick
    assert restored.episode_tick == source.episode_tick
    assert restored.rng.bit_generator.state == source.rng.bit_generator.state


def test_checkpoint_rejects_version_structure_shape_and_fields_atomically(tmp_path):
    source = _make_agent()
    source.observe(1, 0)
    good = tmp_path / "good.npz"
    source.checkpoint(good)
    original = _checkpoint_arrays(good)

    wrong_structure = _make_agent(k=3)
    with pytest.raises(ValueError, match="plan structure mismatch"):
        wrong_structure.restore(good)

    wrong_parameters = _make_agent(acc=0.65)
    expected_parameters = _agent_snapshot(wrong_parameters)
    with pytest.raises(ValueError, match="plan parameter mismatch"):
        wrong_parameters.restore(good)
    _assert_agent_snapshot(wrong_parameters, expected_parameters)

    tied_source = _make_tied_parameter_agent(start=0)
    tied_path = tmp_path / "tied-static-parameter.npz"
    tied_source.checkpoint(tied_path)
    tied_target = _make_tied_parameter_agent(start=2)
    tied_expected = _agent_snapshot(tied_target)
    with pytest.raises(ValueError, match="plan parameter mismatch"):
        tied_target.restore(tied_path)
    _assert_agent_snapshot(tied_target, tied_expected)

    plated_source = _make_plated_parameter_agent(first_mass=0.8)
    plated_path = tmp_path / "plated-static-parameter.npz"
    plated_source.checkpoint(plated_path)
    plated_target = _make_plated_parameter_agent(first_mass=0.2)
    plated_expected = _agent_snapshot(plated_target)
    with pytest.raises(ValueError, match="plan parameter mismatch"):
        plated_target.restore(plated_path)
    _assert_agent_snapshot(plated_target, plated_expected)

    for mismatch in (
        _make_agent(select="map"),
        _make_agent(n_sweeps=101),
        _make_agent(damp=0.25),
    ):
        expected_mismatch = _agent_snapshot(mismatch)
        with pytest.raises(ValueError, match="runtime configuration mismatch"):
            mismatch.restore(good)
        _assert_agent_snapshot(mismatch, expected_mismatch)

    target = _make_agent(seed=31)
    target.observe(1, 0)
    expected = _agent_snapshot(target)

    corruptions = []

    version = dict(original)
    version["checkpoint_version"] = np.asarray(99)
    corruptions.append((version, ValueError, "checkpoint version"))

    belief_shape = dict(original)
    belief_shape["belief"] = np.full(3, 1 / 3)
    corruptions.append((belief_shape, ValueError, "belief.*shape"))

    belief_mass = dict(original)
    belief_mass["belief"] = np.asarray([2.0, 1.0])
    corruptions.append((belief_mass, ValueError, "belief.*sum to one"))

    learn_shape = dict(original)
    learn_key = next(key for key in learn_shape if key.startswith("learn:"))
    learn_shape[learn_key] = np.ones((3, 3, 3))
    corruptions.append((learn_shape, ValueError, "checkpoint shape"))

    complex_learnable = dict(original)
    complex_learnable[learn_key] = complex_learnable[learn_key].astype(complex) + 1j
    corruptions.append((complex_learnable, ValueError, "must be real"))

    unknown = dict(original)
    unknown["future_state"] = np.asarray(1)
    unknown_manifest = json.loads(str(unknown["checkpoint_fields"]))
    unknown_manifest.append("future_state")
    unknown["checkpoint_fields"] = np.asarray(json.dumps(sorted(unknown_manifest)))
    corruptions.append((unknown, KeyError, "unknown checkpoint fields"))

    missing = dict(original)
    del missing[learn_key]
    missing_manifest = json.loads(str(missing["checkpoint_fields"]))
    missing_manifest.remove(learn_key)
    missing["checkpoint_fields"] = np.asarray(json.dumps(missing_manifest))
    corruptions.append((missing, KeyError, "missing required fields"))

    malformed_rng = dict(original)
    malformed_rng["rng_state"] = np.asarray("not-json")
    corruptions.append((malformed_rng, ValueError, "rng_state.*valid JSON"))

    invalid_rng = dict(original)
    invalid_rng["rng_state"] = np.asarray(json.dumps({"not": "a state"}))
    corruptions.append((invalid_rng, ValueError, "rng_state.*bit-generator state"))

    overflowing_rng = dict(original)
    overflowing_rng_state = json.loads(str(overflowing_rng["rng_state"]))
    overflowing_rng_state["uinteger"] = 2**40
    overflowing_rng["rng_state"] = np.asarray(json.dumps(overflowing_rng_state))
    corruptions.append(
        (overflowing_rng, ValueError, "rng_state.*bit-generator state")
    )

    invalid_cadence = dict(original)
    invalid_cadence["episode_tick"] = np.asarray(int(original["tick"]) + 1)
    corruptions.append((invalid_cadence, ValueError, "cadence is invalid"))

    execution_precision = dict(original)
    execution_precision["jax_enable_x64"] = np.asarray(
        not jax.config.jax_enable_x64
    )
    corruptions.append(
        (execution_precision, ValueError, "runtime configuration mismatch")
    )

    for index, (arrays, error_type, message) in enumerate(corruptions):
        path = tmp_path / f"corrupt-{index}.npz"
        _write_checkpoint(path, arrays)
        with pytest.raises(error_type, match=message):
            target.restore(path)
        _assert_agent_snapshot(target, expected)

    duplicate = tmp_path / "duplicate.npz"
    shutil.copyfile(good, duplicate)
    duplicate_tick = io.BytesIO()
    np.save(duplicate_tick, np.asarray(99))
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "a") as archive:
            archive.writestr("tick.npy", duplicate_tick.getvalue())
    with pytest.raises(KeyError, match="duplicate array names"):
        target.restore(duplicate)
    _assert_agent_snapshot(target, expected)

    boundary = next(iter(source.beliefs))
    legacy_ambiguous = {
        "belief": source.belief,
        f"belief:{boundary}": source.belief,
        "tick": np.asarray(source.tick),
        "episode_tick": np.asarray(source.episode_tick),
        "rng_state": original["rng_state"],
        "b_counts": source.b_counts,
    }
    ambiguous_path = tmp_path / "legacy-ambiguous.npz"
    _write_checkpoint(ambiguous_path, legacy_ambiguous)
    with pytest.raises(KeyError, match="mixes legacy 'belief'"):
        target.restore(ambiguous_path)
    _assert_agent_snapshot(target, expected)

    legacy_alias_collision = dict(legacy_ambiguous)
    del legacy_alias_collision[f"belief:{boundary}"]
    legacy_alias_collision["learn:B"] = source.b_counts
    collision_path = tmp_path / "legacy-learnable-collision.npz"
    _write_checkpoint(collision_path, legacy_alias_collision)
    with pytest.raises(KeyError, match="both 'learn:B' and legacy 'b_counts'"):
        target.restore(collision_path)
    _assert_agent_snapshot(target, expected)

    legacy_partial = dict(legacy_ambiguous)
    del legacy_partial[f"belief:{boundary}"]
    del legacy_partial["b_counts"]
    partial_path = tmp_path / "legacy-partial.npz"
    _write_checkpoint(partial_path, legacy_partial)
    with pytest.raises(KeyError, match="missing learnable hyperparameters"):
        target.restore(partial_path)
    _assert_agent_snapshot(target, expected)

    legacy_without_rng = dict(legacy_partial)
    legacy_without_rng["b_counts"] = source.b_counts
    del legacy_without_rng["rng_state"]
    rng_path = tmp_path / "legacy-without-rng.npz"
    _write_checkpoint(rng_path, legacy_without_rng)
    with pytest.raises(KeyError, match="no RNG state"):
        target.restore(rng_path)
    _assert_agent_snapshot(target, expected)


def test_checkpoint_rejects_goal_corruption_atomically(tmp_path):
    source = _make_agent(learn_goal=True)
    action = source.act()
    source.observe(action, 0)
    good = tmp_path / "goal-good.npz"
    source.checkpoint(good)
    original = _checkpoint_arrays(good)

    target = _make_agent(seed=31, learn_goal=True)
    target_action = target.act()
    target.observe(target_action, 1)
    expected = _agent_snapshot(target)

    goal_prior_key = next(key for key in original if key.startswith("goal:prior:"))
    pending_key = next(key for key in original if key.startswith("goal:pending:"))
    corruptions = []

    invalid_prior = dict(original)
    invalid_prior[goal_prior_key] = np.zeros_like(invalid_prior[goal_prior_key])
    corruptions.append((invalid_prior, "goal prior.*finite positive"))

    invalid_pending = dict(original)
    invalid_pending[pending_key] = np.ones(3)
    corruptions.append((invalid_pending, "pending goal.*shape"))

    for index, (arrays, message) in enumerate(corruptions):
        path = tmp_path / f"goal-corrupt-{index}.npz"
        _write_checkpoint(path, arrays)
        with pytest.raises(ValueError, match=message):
            target.restore(path)
        _assert_agent_snapshot(target, expected)


def test_checkpoint_binds_precision_at_agent_construction(tmp_path):
    original_precision = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        source = _make_agent()
        jax.config.update("jax_enable_x64", True)
        path = tmp_path / "float32-agent.npz"
        source.checkpoint(path)
        target = _make_agent()
        expected = _agent_snapshot(target)

        with pytest.raises(ValueError, match="runtime configuration mismatch"):
            target.restore(path)
        _assert_agent_snapshot(target, expected)
    finally:
        jax.config.update("jax_enable_x64", original_precision)
