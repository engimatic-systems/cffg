"""Bounded CPU cost of the native navigation evidence scorer on synthetic loop traversals."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TILES = (24, 96, 240)
ASSIGNMENTS = ("appearance", "split")
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
WARM = 5
ENV = {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true", "JAX_ENABLE_COMPILATION_CACHE": "false",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
       "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
PROTOCOL = dict(
    tiles=TILES, assignments=ASSIGNMENTS, dtype="float64", warm_calls=WARM, environment=ENV,
    policy="Synthetic square perimeter. Two identical junction cues on its north side; every other tile has a unique label. "
           "Fit traverses one clockwise and one counterclockwise lap. Validation has two 12-step probes, one at each cue. "
           "Each probe localizes at the unique adjacent eastern tile and moves west to the cue before six scored E/W pairs. "
           "The two positioning moves are outside the scored probes. All actions are legal cardinal moves.",
    comparison="Appearance assigns one place per observed label. Split assigns one place per fitted cue coordinate and "
               "keeps other labels unchanged. Both predict the same public destination symbols on the same scored actions.",
    timing="Serial fresh process per cell. Clock one first and five warm complete calls to evidence.score on the same "
           "fixed table and 24 events. Includes CPT preparation, binding, exact inference, synchronized scalar BFE "
           "and the scorer's built-in count oracle. First call includes first use of the compiler and executable. "
           "Excludes imports, geometry, fitting and the separate observed-event oracle. No persistent JAX compilation cache.",
    bytes="CPT bytes count one dense float64 probability table. Counts bytes count one int64 fit table. Peak RSS is "
          "the lifetime process high-water mark, reported in MiB (2^20 bytes), including runtime, compiler and temporary arrays.",
    scope="Exploratory CPU wall times on one host. Browser QA may overlap. The observations are synthetic. "
          "These measurements cover dense next-landmark scoring on 24 fixed validation moves.",
)


def traversal(tiles):
    side = tiles // 4
    perimeter = ([(0, j) for j in range(side)] + [(i, side) for i in range(side)] +
                 [(side, j) for j in range(side, 0, -1)] + [(i, 0) for i in range(side, 0, -1)])
    cues = (1, side - 1)
    labels = {p: ("junction" if i in cues else f"tile-{i:03d}") for i, p in enumerate(perimeter)}
    assert len(labels) == tiles

    def walk(start, destinations):
        p = start
        steps = []
        for q in destinations:
            action = MOVES.index((q[0] - p[0], q[1] - p[1]))
            steps.append((p, labels[p], action, q, labels[q]))
            p = q
        return steps

    clockwise = perimeter[1:] + perimeter[:1]
    anticlockwise = list(reversed(perimeter[1:])) + perimeter[:1]
    fit = walk(perimeter[0], clockwise + anticlockwise)
    probes = []
    for i in cues:
        cue, east = perimeter[i], perimeter[i + 1]
        positioning = walk(east, [cue])
        assert labels[east] != "junction" and positioning[0][2] == MOVES.index((0, -1))
        probes.extend(walk(positioning[-1][3], [east, cue] * 6))
    return fit, probes


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest_rows(rows):
    return sha(json.dumps(rows, separators=(",", ":")).encode())


def fixture(tiles, assignment):
    # The fitted table and place identifiers see only the fit tape.
    from evidence import fit_counts

    fit, probes = traversal(tiles)

    def place(p, label):
        return (label, p if assignment == "split" and label == "junction" else ())

    source_keys = sorted({place(p, label) for p, label, _, _, _ in fit})
    source_id = {key: i for i, key in enumerate(source_keys)}
    symbols = sorted({dst for _, _, _, _, dst in fit})
    outcome_id = {key: i for i, key in enumerate(symbols)}

    def event(step):
        p, label, action, _, dst = step
        return 4 * source_id[place(p, label)] + action, outcome_id[dst]

    counts = fit_counts(map(event, fit), n_rows=4 * len(source_keys), n_outcomes=len(symbols))
    freq = Counter((place(p, label), action, dst) for p, label, action, _, dst in fit)
    totals = Counter((place(p, label), action) for p, label, action, _, _ in fit)
    oracle = []
    for p, label, action, _, dst in probes:
        key = place(p, label)
        oracle.append(math.log(totals[key, action] + len(symbols)) - math.log(freq[key, action, dst] + 1))
    transitions = len({(p, q) for p, _, _, q, _ in fit})
    assert transitions == 2 * tiles and len(fit) == counts.sum() == 2 * tiles
    info = dict(tiles=tiles, assignment=assignment, n_places=len(source_keys), n_rows=counts.shape[0],
                n_outcomes=len(symbols), fit_steps=len(fit), n_valid=len(probes),
                fit_tape_sha256=digest_rows(fit), validation_tape_sha256=digest_rows(probes),
                validation_outcome_ids_sha256=digest_rows([event(step)[1] for step in probes]),
                observed_directed_transitions=transitions, nonzero_cells=len(freq), dense_entries=counts.size,
                cpt_bytes=counts.size * 8, counts_bytes=counts.nbytes)
    return counts, [event(step) for step in probes], oracle, info


def worker(tiles, assignment):
    import jax
    import numpy as np
    from evidence import score

    counts, probe, oracle, row = fixture(tiles, assignment)
    assert row["n_valid"] == 24 and row["nonzero_cells"] == np.count_nonzero(counts)
    assert row["n_rows"] == 4 * (tiles - (assignment == "appearance")) and row["n_outcomes"] == tiles - 1
    times, max_error = [], 0.0
    total = None
    for _ in range(1 + WARM):
        start = time.perf_counter()
        result = score(counts, probe, dtype="float64")
        times.append(time.perf_counter() - start)
        assert result["status"] == "ok" and result["schedule"] == "two_pass"
        assert result["n_events"] == len(probe) and result["outcomes"] == [y for _, y in probe]
        for observed, expected in zip(result["energies"], oracle, strict=True):
            error = abs(observed - expected)
            if not math.isfinite(error) or error > 64 * np.finfo(float).eps * max(1.0, abs(expected)):
                raise ArithmeticError(f"native BFE {observed}, independently counted {expected}")
            max_error = max(max_error, error)
        if total is not None and total != result["total_bfe"]:
            raise ArithmeticError("non-deterministic total on repeated score")
        total = result["total_bfe"]
    warm = times[1:]
    assert jax.default_backend() == "cpu"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss_bytes = rss
    elif sys.platform.startswith("linux"):
        rss_bytes = rss * 1024
    else:
        raise OSError("unknown native ru_maxrss unit")
    row.update(cold_s=times[0], warm_s=dict(median=statistics.median(warm), min=min(warm), max=max(warm), blocks=warm),
               peak_rss_mib=rss_bytes / (1 << 20), max_abs_error=max_error, total_bfe=total)
    return row


def metadata():
    import importlib.metadata
    import importlib.util

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    source = REPO / "src/cffg"
    if Path(importlib.util.find_spec("cffg").origin).resolve().parent != source:
        raise ValueError("CFFG import path must point to this checkout")
    rel = ["tools/navigation/evidence.py", "tools/navigation/scale.py", "uv.lock"]
    if git("status", "--porcelain", "--", "src/cffg", *rel):
        raise ValueError("scorer, runtime and environment lock must match the pinned Git tree")
    return dict(git_commit=git("rev-parse", "HEAD"), cffg_tree=git("rev-parse", "HEAD:src/cffg"),
                source_sha256={name:sha((REPO / name).read_bytes()) for name in rel},
                python=sys.version, packages={name:importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy", "equinox")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=int, choices=TILES)
    parser.add_argument("--assignment", choices=ASSIGNMENTS)
    args = parser.parse_args()
    if args.worker:
        if not args.assignment:
            parser.error("--worker requires --assignment")
        print(json.dumps(worker(args.worker, args.assignment), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output required")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = dict(status="running", protocol=PROTOCOL, source=metadata(),
                  hardware=dict(cpu=cpu, machine=platform.machine(), platform=platform.platform(),
                                logical_cpus=os.cpu_count()), rows=[])

    def publish():
        partial = output.with_suffix(".partial.json")
        partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        partial.replace(output)

    publish()
    try:
        for i, n in enumerate(TILES):
            # Alternate the first assignment to limit systematic timing-order effects.
            rows = []
            for assignment in (ASSIGNMENTS if i % 2 == 0 else reversed(ASSIGNMENTS)):
                raw = subprocess.check_output([sys.executable, __file__, "--worker", str(n), "--assignment", assignment],
                                              cwd=REPO, env={**os.environ, **ENV}, text=True)
                row = json.loads(raw)
                rows.append(row)
                report["rows"].append(row)
                publish()
            for k in ("validation_tape_sha256", "validation_outcome_ids_sha256", "fit_tape_sha256"):
                if rows[0][k] != rows[1][k]:
                    raise ValueError(f"candidate mismatch in {k}")
        final = metadata()
        for k in ("source_sha256", "cffg_tree", "packages", "python"):
            if final[k] != report["source"][k]:
                raise RuntimeError(f"benchmark source or environment changed during the run: {k}")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
