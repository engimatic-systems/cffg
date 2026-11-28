"""Run the frozen fit and evaluation stages of the movement-history experiment."""

import argparse
from collections import Counter, deque
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
import time

import jax
import numpy as np

import evidence
from fixture import (ACTIONS, TileWorld, Traversal, _fit_map, _landmark, _layout,
                     _survey, _tile_policy, _turn, load_protocol)
from movement_history import ObservedMap, Places, context, fit, origins


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_NAMES = ("history_protocol.json", "history_study.py", "movement_history.py",
                "fixture.py", "protocol.json", "evidence.py")
ARMS = ("appearance", "history", "tile")
LEADS = (1, 3)


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    payload = packed(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(payload)
    try:
        os.link(f.name, path)
    finally:
        os.unlink(f.name)


def read(path):
    return json.loads(path.read_bytes())


def sources():
    return {name: digest(HERE / name) for name in SOURCE_NAMES}


def runtime():
    paths = ["src/cffg", "uv.lock", "pyproject.toml"]
    status = subprocess.check_output(["git", "status", "--porcelain", "--", *paths], cwd=REPO)
    if status.strip():
        raise ValueError("runtime or dependency inputs differ from the committed tree")
    return {
        "git_objects": {path: subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{path}"], cwd=REPO, text=True).strip() for path in paths},
        "packages": {p: version(p) for p in ("jax", "jaxlib", "numpy", "equinox")},
    }


def layout(config, name):
    geometry = _layout(load_protocol(), "same-exits", 0)
    tiles = {tuple(row["position"]): row["symbol"] for row in geometry["tiles"]}
    for r, c in config["layouts"][name]["remove"]:
        del tiles[r, c]
    for r, c, symbol in config["layouts"][name]["add"]:
        if (r, c) in tiles:
            raise ValueError("duplicate layout coordinate")
        tiles[r, c] = symbol
    return {"tiles": [{"position": list(p), "symbol": y} for p, y in sorted(tiles.items())]}


def walk(grid, name, phase, heading="N", start="home", position=(0, 0)):
    return Traversal(TileWorld(grid), name, phase, start, position, heading)


def first_moves(survey, start):
    first = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for action in ACTIONS:
            nxt = survey.links[current, action]
            if nxt not in first:
                first[nxt] = first[current] or action
                queue.append(nxt)
    return first


def training(config):
    grid = layout(config, config["training"]["layout"])
    tapes = []
    settings = config["training"]
    for heading in settings["survey_headings"]:
        trail = walk(grid, "survey-" + heading, "fit", heading)
        _survey(trail)
        tapes.append(trail.tape)
    observed = _fit_map(tapes)
    survey = ObservedMap(observed)
    places = sorted(survey.tiles)
    directions = {p: first_moves(survey, p) for p in places}
    for seed in settings["seeds"]:
        rng = random.Random(seed)
        trail = walk(grid, f"fit-{seed}", "fit")
        target = trail.position
        for _ in range(settings["steps"]):
            if target == trail.position:
                target = rng.choice([p for p in places if p != trail.position])
            if rng.random() < settings["probe_probability"]:
                action = rng.choice(ACTIONS)
            else:
                action = directions[trail.position][target]
            trail.apply(action)
        tapes.append(trail.tape)
    return tapes, observed


def fit_stage(root):
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fit requires an empty output directory")
    config = read(HERE / "history_protocol.json")
    original = sources()
    tapes, observed = training(config)
    alphabet = sorted({tile["symbol"] for tile in observed["tiles"]})
    begin = time.perf_counter()
    models = {arm: fit(tapes, alphabet, histories=(arm == "history")).record
              for arm in ARMS[:2]}
    fit_s = time.perf_counter() - begin
    write(root / "fit-tapes.json", tapes)
    write(root / "models.json", {**models, "tile": observed})
    if original != sources():
        raise RuntimeError("source changed during fit")
    freeze = {
        "phase": "frozen", "sources": original, "runtime": runtime(), "fit_s": fit_s,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "files": {p: digest(root / p) for p in ("fit-tapes.json", "models.json")},
        "fit_steps": sum(len(t["steps"]) for t in tapes), "fit_episodes": len(tapes),
        "protocol": config,
    }
    write(root / "freeze.json", freeze)
    return freeze


def frozen(root):
    receipt = read(root / "freeze.json")
    if (receipt["phase"] != "frozen" or receipt["sources"] != sources()
            or receipt["runtime"] != runtime()
            or receipt["protocol"] != read(HERE / "history_protocol.json")
            or set(receipt["files"]) != {"models.json", "fit-tapes.json"}
            or any(digest(root / p) != h for p, h in receipt["files"].items())):
        raise ValueError("fit data, model, protocol, or scoring source differs from freeze")
    return receipt


class NativeScores:
    """Reuse each identical conditional probability vector and observed endpoint."""
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.bases = {}
        self.energies = {}
        self.maximum_error = 0.
        begin = time.perf_counter()
        self.engine = evidence._tree(2, len(alphabet), np.dtype("float64"))
        self.build_s = time.perf_counter() - begin
        # Both conditioned rows carry the same forecast. Compilation includes lazy JIT.
        self.score(np.full(len(alphabet), 1. / len(alphabet)), 0)
        self.cold_s = time.perf_counter() - begin

    def score(self, probabilities, outcome):
        if (probabilities.dtype != np.float64 or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0) or abs(probabilities.sum() - 1) > 64*np.finfo(float).eps):
            raise ValueError("invalid probability vector")
        key = probabilities.tobytes()
        if key not in self.bases:
            matrix = np.stack([probabilities, probabilities], axis=1)
            self.bases[key] = evidence._condition(self.engine, matrix)
        stamp = key, outcome
        if stamp not in self.energies:
            energy = evidence._event_energy(self.engine, self.bases[key], 0, outcome)
            direct = -math.log(probabilities[outcome])
            error = abs(energy - direct)
            if error > 64 * np.finfo(float).eps * max(1., direct):
                raise ArithmeticError(f"tree BFE {energy} differs from log score {direct}")
            self.energies[stamp] = energy
            self.maximum_error = max(self.maximum_error, error)
        return self.energies[stamp]


def commands(config):
    settings = config["evaluation"]
    for i, script in enumerate(settings["scripts"]):
        yield f"script-{i}", script.replace(" ", "")
    for seed in settings["random_seeds"]:
        rng = random.Random(seed)
        yield f"random-{seed}", "".join(rng.choice(ACTIONS) for _ in range(settings["random_steps"]))


def prediction_tapes(config, grid):
    tapes = []
    for name, script in commands(config):
        trail = walk(grid, name, "evaluation", config["evaluation"]["heading"])
        for action in script:
            trail.apply(action)
        tapes.append(trail.tape)
    return tapes


def forecasts(tapes, models, tile, native):
    predictions = []
    history = models["history"]
    for tape in tapes:
        steps = tape["steps"]
        for t, (symbol, key, position) in enumerate(origins(tape)):
            state = history.classify(symbol, key)
            for lead in LEADS:
                if t + lead > len(steps):
                    continue
                window = steps[t:t + lead]
                moves = "".join(row["action"] for row in window)
                final = window[-1]
                outcome = history.symbols[final["next_symbol"]]
                event = dict(tape=tape["id"], t=t, horizon=lead, commands=moves,
                             endpoint=final["next_symbol"], blocked=final["displacement"] == [0, 0],
                             context_available=bool(key), context_seen=key in history.seen,
                             retained=key in history.ids, alias=symbol in history.aliases,
                             place_entropy_nats=history.place_entropy(state), arms={})
                for arm in ARMS:
                    if arm == "tile":
                        p, source = tile.predict(position, moves, history.alphabet)
                    else:
                        p, source = models[arm].predict(symbol, key, moves)
                    event["arms"][arm] = dict(source=source, probability=float(p[outcome]),
                                              bfe=native.score(p, outcome))
                predictions.append(event)
    return predictions


def routes(config, grid, observed, models):
    budget = config["routing"]["budget"]
    tile = ObservedMap(observed)
    result = []
    for start, goal, heading in config["routing"]["tasks"]:
        origin = _landmark(tile.tiles, start)
        for arm in ARMS:
            policy = (_tile_policy(observed, goal) if arm == "tile"
                      else models[arm].route_policy(goal, budget))
            trail = walk(grid, f"{start}-{goal}", "route", heading, start, origin)
            for remaining in range(budget, 0, -1):
                if trail.symbol == goal:
                    break
                key = context(trail.tape["steps"][-2:])
                costs = (policy(trail.position, trail.symbol, remaining) if arm == "tile"
                         else policy(trail.symbol, key, remaining))
                order = [_turn(trail.heading, k) for k in range(4)]
                action = next(a for a in order if costs[ACTIONS.index(a)] <= min(costs) + 1e-12)
                trail.apply(action)
            steps = trail.tape["steps"]
            trail.tape.update(arm=arm, goal=goal, success=trail.symbol == goal, budget=budget,
                              blocked=sum(row["displacement"] == [0, 0] for row in steps))
            result.append(trail.tape)
    return result


def prediction_summary(rows):
    if not rows:
        return {"n": 0, "mean_nll": {arm: None for arm in ARMS}}
    past = [row for row in rows if row["context_available"]]
    return {"n": len(rows),
            "mean_nll": {arm: math.fsum(r["arms"][arm]["bfe"] for r in rows)/len(rows) for arm in ARMS},
            "complete_history": len(past), "novel_history": sum(not r["context_seen"] for r in past),
            "retained_history": sum(r["retained"] for r in past),
            "mean_empirical_fit_place_entropy_nats": math.fsum(r["place_entropy_nats"] for r in rows)/len(rows),
            "history_forecast_sources": dict(Counter(r["arms"]["history"]["source"] for r in rows))}


def evaluate(root):
    seal = frozen(root)
    config = seal["protocol"]
    if any((root / name).exists() for name in ("evaluation.json", "summary.json")):
        raise FileExistsError("completed or partial output already present")
    saved = read(root / "models.json")
    models = {arm: Places(saved[arm]) for arm in ARMS[:2]}
    tile = ObservedMap(saved["tile"])
    measurements = {}
    archive = {}
    unseen = []
    with jax.enable_x64(True):
        native = NativeScores(models["history"].alphabet)
        begin = time.perf_counter()
        for name in config["layout_order"]:
            grid = layout(config, name)
            tapes = prediction_tapes(config, grid)
            events = forecasts(tapes, models, tile, native)
            actual = routes(config, grid, saved["tile"], models)
            archive[name] = dict(layout=grid, tapes=tapes, predictions=events, routes=actual)
            scores = {}
            for lead in LEADS:
                subset = [r for r in events if r["horizon"] == lead]
                scores[str(lead)] = {
                    "all": prediction_summary(subset),
                    **{label: prediction_summary([r for r in subset if r["blocked"] == blocked])
                       for label, blocked in (("blocked", True), ("moved", False))}
                }
                if lead == 3 and name != config["training"]["layout"]:
                    unseen.extend(subset)
            measurements[name] = {"scores": scores, "routes": {}}
            for arm in ARMS:
                selected = [r for r in actual if r["arm"] == arm]
                measurements[name]["routes"][arm] = {
                    "successes": sum(r["success"] for r in selected), "total": len(selected),
                    "actions": sum(len(r["steps"]) for r in selected),
                    "blocked": sum(r["blocked"] for r in selected)}
        seconds = time.perf_counter() - begin
    if frozen(root) != seal:
        raise ValueError("fit receipt changed during evaluation")
    pooled = prediction_summary(unseen)
    primary = pooled["mean_nll"]["history"] - pooled["mean_nll"]["appearance"]
    write(root / "evaluation.json", archive)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = dict(status="complete", freeze_sha256=digest(root/"freeze.json"),
                   evaluation_sha256=digest(root/"evaluation.json"), primary_delta_nats=primary,
                   primary_unseen_h3=pooled, layouts=measurements, fit_s=seal["fit_s"],
                   fit_steps=seal["fit_steps"], serialized_bytes={a: len(packed(saved[a])) for a in ARMS},
                   dictionary={a: len(models[a].keys) for a in ARMS[:2]},
                   distinct_fit_histories=len(models["history"].seen), tile_cells=len(tile.tiles),
                   native_cold_s=native.cold_s, native_template_and_engine_s=native.build_s,
                   native_first_event_s=native.cold_s - native.build_s, evaluation_s=seconds,
                   native_distinct_vectors=len(native.bases), native_distinct_events=len(native.energies),
                   oracle_max_abs_nats=native.maximum_error,
                   process_peak_rss_bytes=int(rss * (1 if sys.platform == "darwin" else 1024)),
                   backend=jax.default_backend(), jax_version=jax.__version__)
    write(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = (fit_stage if args.stage == "fit" else evaluate)(args.output)
    print(json.dumps({k: v for k, v in result.items() if k in
                      ("phase", "fit_steps", "status", "primary_delta_nats", "native_cold_s")}))
