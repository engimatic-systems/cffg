from cffg.experiments.simglucose_basal import (
    Protocol,
    run_cohort,
    run_patient,
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
