"""Run the two learned-place comparisons and produce a self-contained replay guide."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile

import jax
import jaxlib

from evidence import fit_counts, score, select
from fixture import build_cases, navigate


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CANDIDATES = ("appearance", "split")
SOURCES = [
    {"title": "AIMAPP · Place proposals and map expansion",
     "url": "https://arxiv.org/html/2510.09574v2#S3.SS5"},
    {"title": "AIMAPP · Goal-reaching protocol and results",
     "url": "https://arxiv.org/html/2510.09574v2#S4.SS4"},
    {"title": "From pixels to planning · Fast structure learning, Algorithm 1",
     "url": "https://arxiv.org/pdf/2407.20292#page=26"},
    {"title": "From pixels to planning · The Pong training sequences",
     "url": "https://arxiv.org/pdf/2407.20292#page=50"},
]


def score_partition(case: dict, counts: dict, phase: str) -> dict:
    scores = {}
    for name in CANDIDATES:
        events = case["candidates"][name][phase]
        scores[name] = score(counts[name], events, dtype="float64")
    if scores["appearance"]["outcomes"] != scores["split"]["outcomes"]:
        raise ValueError(f"Candidate models received different {phase} observations")
    return scores


def evaluate(case: dict) -> dict:
    counts = {}
    for name in CANDIDATES:
        candidate = case["candidates"][name]
        counts[name] = fit_counts(candidate["fit"], n_rows=candidate["n_rows"],
                                  n_outcomes=candidate["n_outcomes"])
    selection_scores = score_partition(case, counts, "selection")
    decision = select(selection_scores["appearance"], selection_scores["split"])
    final_scores = score_partition(case, counts, "final")
    scores = {name: {"selection": selection_scores[name], "final": final_scores[name]}
              for name in CANDIDATES}

    case["routes"] = {name: navigate(case, name) for name in (*CANDIDATES, "tile")}
    routes = {}
    for name, tapes in case["routes"].items():
        routes[name] = {
            "completed": sum(1 for tape in tapes if tape["success"]),
            "attempted": len(tapes),
            "actions": sum(len(tape["steps"]) for tape in tapes),
        }
    return {"id": case["id"], "scores": scores, "selection": decision, "routes": routes}


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def build_study() -> dict:
    if not jax.config.x64_enabled:
        raise ValueError("Run with JAX_ENABLE_X64=1 for the float64 reference study.")
    protocol_path = HERE / "protocol.json"
    sources = [protocol_path, HERE / "evidence.py", HERE / "fixture.py", Path(__file__).resolve()]
    cases = build_cases()
    return {
        "schema_version": 1,
        "protocol": json.loads(protocol_path.read_text()),
        "git_commit": git_value("rev-parse", "HEAD"),
        "cffg_tree": git_value("rev-parse", "HEAD:src/cffg"),
        "source_sha256": {str(p.relative_to(REPO)): sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "runtime": {"jax": jax.__version__, "jaxlib": jaxlib.__version__,
                    "backend": jax.default_backend(), "precision": "float64"},
        "method": "Separate fit, model selection, final prediction and closed-loop navigation.",
        "findings": [evaluate(case) for case in cases],
        "cases": cases,
        "sources": SOURCES,
    }


def render_guide(study_json: str, template: str) -> str:
    marker = "__NAVIGATION_STUDY_JSON__"
    if template.count(marker) != 1:
        raise ValueError("Guide template needs exactly one study-data placeholder.")
    # Keep the embedded data safe inside a script element, including quoted map symbols.
    escaped = study_json.replace("&", r"\u0026").replace("<", r"\u003c").replace(">", r"\u003e")
    return template.replace(marker, escaped)


def write_study(destination: Path, scale_path: Path | None = None) -> None:
    if destination.exists():
        raise FileExistsError(f"Choose a new directory to preserve the existing study at {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / "study"
        stage.mkdir()
        study = build_study()
        if scale_path is not None:
            raw = scale_path.read_bytes()
            study["scale"] = json.loads(raw)
            study["scale_sha256"] = sha256(raw).hexdigest()
        data = json.dumps(study, allow_nan=False, separators=(",", ":"))
        (stage / "study.json").write_text(data + "\n", encoding="utf-8")
        html = render_guide(data, (HERE / "guide.html").read_text(encoding="utf-8"))
        (stage / "index.html").write_text(html, encoding="utf-8")
        stage.rename(destination)
    for finding in study["findings"]:
        print(finding["id"], finding["selection"], finding["routes"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New output directory for the guide and raw data")
    parser.add_argument("--scale", type=Path, help="Include a separate map-size benchmark receipt")
    args = parser.parse_args()
    write_study(args.directory.resolve(), args.scale)


if __name__ == "__main__":
    main()
