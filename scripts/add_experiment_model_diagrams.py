"""Add deterministic model diagrams to experiment artifact folders.

The script skips folders that already contain ``model_diagram.svg``. It can
render canonical serialized CFFG templates when present, and it knows how to
draw the simglucose controller policy graph from that experiment's config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cffg.core.serialize import from_dict, load
from cffg.diagram import save_template_svg
from cffg.experiments.simglucose_basal import policy_template_for_preset


OUTPUT_NAME = "model_diagram.svg"
TEMPLATE_NAMES = ("template.json", "model.json", "cffg.json")


def add_diagrams(root: Path, *, overwrite: bool = False) -> tuple[int, int]:
    written = 0
    skipped = 0
    for config_path in sorted(root.rglob("config.json")):
        exp_dir = config_path.parent
        out = exp_dir / OUTPUT_NAME
        if out.exists() and not overwrite:
            skipped += 1
            continue
        template = _template_for_experiment(exp_dir, config_path)
        if template is None:
            skipped += 1
            continue
        if save_template_svg(template, out, overwrite=overwrite):
            written += 1
        else:
            skipped += 1
    return written, skipped


def _template_for_experiment(exp_dir: Path, config_path: Path):
    for name in TEMPLATE_NAMES:
        path = exp_dir / name
        if path.exists():
            return load(path)

    config = json.loads(config_path.read_text())
    if {"variables", "factors", "schema_version"} <= set(config):
        return from_dict(config)
    preset = config.get("preset")
    if isinstance(preset, str):
        template = policy_template_for_preset(preset)
        template.name = f"{exp_dir.name}: {template.name}"
        return template
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="experiments", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    written, skipped = add_diagrams(args.root, overwrite=args.overwrite)
    print(f"model diagrams: {written} written, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
