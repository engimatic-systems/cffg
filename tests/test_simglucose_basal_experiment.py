from cffg.experiments.simglucose_basal import (
    COHORT,
    Protocol,
    randomized_trial_specs,
    run_cohort,
    run_patient,
    run_trials,
    write_artifacts,
)


def test_basal_experiment_is_basal_only_and_eight_hours():
    trace = run_patient("adult#001")

    assert len(trace.bg) == 160
    assert trace.time_min[-1] == 480.0
    assert max(trace.bolus) == 0.0
    assert trace.tir >= 0.95


def test_basal_experiment_writes_review_artifacts(tmp_path):
    result = run_cohort(["adult#001"])

    write_artifacts(
        result,
        tmp_path,
        preset="clock-prior-aggressive",
        protocol=Protocol(),
        note="Experiment test note.\n",
    )

    assert (tmp_path / "note.txt").read_text() == "Experiment test note.\n"
    assert "cohort TIR" in (tmp_path / "summary.txt").read_text()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "traces.csv").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "plots" / "adult#001.png").exists()


def test_randomized_trials_are_balanced_and_variable():
    specs = randomized_trial_specs(100, seed=123, patients=COHORT)

    counts = {patient: 0 for patient in COHORT}
    for spec in specs:
        counts[spec.patient] += 1

    assert set(counts.values()) == {5}
    assert len({spec.protocol.meals for spec in specs}) > 90


def test_randomized_trial_runner_records_hidden_meals():
    result = run_trials("randomized-hazard-safe-v1", n_trials=2, seed=123, patients=["adult#001"])

    assert len(result.patients) == 2
    assert result.patients[0].trial == 1
    assert result.patients[0].patient == "adult#001"
    assert result.patients[0].meals != result.patients[1].meals
    assert max(max(trace.bolus) for trace in result.patients) == 0.0
