from pathlib import Path
import importlib.util

from cffg.diagram import render_template_svg, save_template_svg
from cffg.experiments.simglucose_basal import policy_template_for_preset
from cffg.models.tmaze import build_tmaze


def test_template_svg_diagram_is_deterministic_and_labeled(tmp_path):
    template, _ = build_tmaze(steps=2, gfe=True)

    first = render_template_svg(template)
    second = render_template_svg(template)

    assert first == second
    assert "<svg" in first
    assert "prior" in first
    assert "z0" in first
    assert "p_substitution" in first

    out = tmp_path / "model_diagram.svg"
    assert save_template_svg(template, out)
    assert out.read_text() == first
    assert save_template_svg(template, out, overwrite=False) is False


def test_simglucose_bolus_policy_template_draws_therapy_profile():
    template = policy_template_for_preset("patient-specific-hard-mask-bolus-safe-v1")
    svg = render_template_svg(template)

    assert "bolus_policy" in svg
    assert "therapy_profile" in svg
    assert "hard_safety_mask" in svg


def test_experiment_diagram_script_skips_existing_outputs(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "add_experiment_model_diagrams.py"
    spec = importlib.util.spec_from_file_location("add_experiment_model_diagrams", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exp = tmp_path / "experiments" / "001"
    exp.mkdir(parents=True)
    (exp / "config.json").write_text('{"preset": "clock-prior-aggressive"}\n')

    written, skipped = module.add_diagrams(tmp_path / "experiments")
    assert (written, skipped) == (1, 0)
    assert (exp / module.OUTPUT_NAME).exists()

    written, skipped = module.add_diagrams(tmp_path / "experiments")
    assert (written, skipped) == (0, 1)

    # Keep the test honest about the expected artifact location.
    assert Path(exp / module.OUTPUT_NAME).read_text().startswith("<?xml")
