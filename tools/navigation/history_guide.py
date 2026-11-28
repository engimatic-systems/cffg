"""Render a closed movement-history archive as a single offline replay page."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def verified(root, name, expected):
    payload = (root / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{name} differs from its archived digest")
    return json.loads(payload)


def render(root):
    summary = json.loads((root / "summary.json").read_bytes())
    if summary["status"] != "complete":
        raise ValueError("a completed evaluation is required")
    freeze = verified(root, "freeze.json", summary["freeze_sha256"])
    if freeze["phase"] != "frozen" or set(freeze["files"]) != {"fit-tapes.json", "models.json"}:
        raise ValueError("incomplete fit receipt")
    models = verified(root, "models.json", freeze["files"]["models.json"])
    verified(root, "fit-tapes.json", freeze["files"]["fit-tapes.json"])
    cases = verified(root, "evaluation.json", summary["evaluation_sha256"])
    protocol = freeze["protocol"]
    if set(cases) != set(protocol["layout_order"]) or set(cases) != set(summary["layouts"]):
        raise ValueError("incomplete layout inventory")

    # The viewer reconstructs only the display of the state at a cursor. All
    # decisions, movement, forecast probabilities and scores come from the archive.
    study = dict(summary=summary, cases=cases, order=protocol["layout_order"],
                 training_layout=protocol["training"]["layout"], tile=models["tile"],
                 history_keys=models["history"]["keys"], locations=models["history"]["locations"])
    payload = json.dumps(study, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("history_guide.html").read_text()
    token = "__MOVEMENT_STUDY_JSON__"
    if template.count(token) != 1:
        raise ValueError("missing or duplicated guide data placeholder")
    destination = root / "index.html"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as out:
        out.write(template.replace(token, payload))
    Path(out.name).replace(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(render(parser.parse_args().directory))
