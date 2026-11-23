"""Replay the first coolant command in PCGym to isolate the effect of a deadline hold.

This post-hoc calculation compares one plant step on the archived initial condition.
It does not change a timed episode or extrapolate the intervention to later steps.
"""
import argparse
import json
from pathlib import Path

import cstr_reactor_pilot as pilot


def read(path):
    return json.loads(path.read_text())


def first_step(make_env, params, seed, observation, command):
    np = pilot.np
    env = make_env(params)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    np.testing.assert_array_equal(obs, observation)
    next_observation, reward, _, _, _ = env.step(np.array([command], dtype=float))
    return dict(reward=float(np.asarray(reward).item()), true_state=env.state.copy(),
                observation=next_observation.copy(), command_kelvin=298.5 + 3.5 * command)


def run(study, output, experiment):
    make_env, params, info = pilot.load_runtime(experiment)
    np = pilot.np
    rows = []
    for seed in pilot.PROTOCOL["held_out_paired_seeds"]:
        folder = study / f"seed-{seed:02d}"
        records = {a: read(folder / a / "decision-00.json") for a in ("directed", "host")}
        np.testing.assert_array_equal(records["directed"]["observation"], records["host"]["observation"])
        outcomes = {}
        for arm, record in records.items():
            manifest = read(folder / arm / "manifest.json")
            assert manifest["runtime"]["library_tree"] == info["library_tree"]
            assert manifest["parameters"] == pilot.serial(params)
            step = first_step(make_env, params, seed, record["observation"], record["applied_normalized"])
            saved = read(folder / arm / "step-00.json")
            np.testing.assert_allclose(step["reward"], saved["reward"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(step["true_state"], saved["next_true_state"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(step["observation"], saved["next_observation"], rtol=1e-12, atol=1e-12)
            outcomes[arm] = step
        host = records["host"]
        if host["checks"] is not None and host["error"] is None:
            command = host["clipped_normalized"]
            outcomes["host_checked_proposal"] = first_step(make_env, params, seed, host["observation"], command)
        rows.append(dict(seed=seed, host_was_late=host["deadline_miss"], host_held=host["fallback"],
                         outcomes=outcomes))
    pilot.save(output, dict(study=str(study), seeds=rows, library_tree=info["library_tree"],
                            pcgym_source=info["pcgym_source"], controller="recorded checked and clipped proposal",
                            description="Only the first PCGym input is changed. Original timed trajectories are retained."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="timed run directory")
    parser.add_argument("output", type=Path, help="new JSON file")
    parser.add_argument("--experiment", type=Path, default=pilot.EXPERIMENT)
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve(), args.experiment.resolve())
