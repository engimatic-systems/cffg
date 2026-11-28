"""Reconstruct stored histories, movements and predictive scores from the raw tapes."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def coords(grid):
    return {tuple(r["position"]): r["symbol"] for r in grid["tiles"]}


def history(rows, t):
    if t < 2:
        return None
    return tuple((r["symbol"], r["action"], tuple(r["displacement"]), r["next_symbol"])
                 for r in (rows[t-2], rows[t-1]))


def replay(tape, cells):
    p = tuple(tape["start"]["position"])
    require(cells[p] == tape["start"]["symbol"], "reset sensor mismatch")
    for i, row in enumerate(tape["steps"], start=1):
        delta = DELTA[row["action"]]
        target = tuple(x + dx for x, dx in zip(p, delta))
        if target not in cells:
            delta, target = (0, 0), p
        require(row["t"] == i and tuple(row["position"]) == p and row["symbol"] == cells[p]
                and tuple(row["displacement"]) == delta and tuple(row["next_position"]) == target
                and row["next_symbol"] == cells[target], "replay diverged from primitive step")
        p = target
    return cells[p]


def projected(tapes):
    visits, occurrences = defaultdict(set), Counter()
    by_symbol, by_context = defaultdict(Counter), defaultdict(Counter)
    links = {}
    for tape in tapes:
        rows = tape["steps"]
        require(tape["phase"] == "fit", "non-fit tape")
        for t, step in enumerate(rows):
            position, dest = tuple(step["position"]), tuple(step["next_position"])
            visits[step["symbol"]].add(position)
            visits[step["next_symbol"]].add(dest)
            links[position, step["action"]] = dest, step["next_symbol"]
            previous = history(rows, t)
            if previous is not None:
                occurrences[previous] += 1
            for lead in (1, 3):
                if t + lead <= len(rows):
                    script = tuple(r["action"] for r in rows[t:t+lead])
                    endpoint = rows[t+lead-1]["next_symbol"]
                    by_symbol[step["symbol"], script][endpoint] += 1
                    if previous is not None:
                        by_context[previous, script][endpoint] += 1
    retained = {k for k, c in occurrences.items() if c >= 2 and len(visits[k[-1][-1]]) > 1}
    return visits, links, retained, by_symbol, by_context


def scripts(config):
    settings = config["evaluation"]
    result = {f"script-{i}": s.replace(" ", "") for i, s in enumerate(settings["scripts"])}
    for seed in settings["random_seeds"]:
        rng = random.Random(seed)
        result[f"random-{seed}"] = "".join(rng.choice("NESW") for _ in range(settings["random_steps"]))
    return result


def audit(root):
    def load(name):
        return json.loads((root/name).read_text())

    def sha(name):
        return hashlib.sha256((root/name).read_bytes()).hexdigest()

    fit_tapes, freeze = load("fit-tapes.json"), load("freeze.json")
    archive, summary = load("evaluation.json"), load("summary.json")
    require(summary["status"] == "complete" and summary["freeze_sha256"] == sha("freeze.json")
            and summary["evaluation_sha256"] == sha("evaluation.json"), "result digest mismatch")
    require(all(sha(name) == expected for name, expected in freeze["files"].items()),
            "fit input digest mismatch")
    protocol = freeze["protocol"]
    ordered = protocol["layout_order"]
    require(set(archive) == set(ordered), "layout inventory mismatch")
    training_name = protocol["training"]["layout"]
    fit_cells = coords(archive[training_name]["layout"])
    for tape in fit_tapes:
        replay(tape, fit_cells)
    visits, links, retained, by_symbol, by_context = projected(fit_tapes)
    n = len(visits)
    scripts_by_id = scripts(protocol)
    n_score = n_routes = n_actions = 0
    primary = []
    worst = 0.
    for name in ordered:
        case = archive[name]
        expected_geometry = dict(fit_cells)
        patch = protocol["layouts"][name]
        for p in patch["remove"]:
            del expected_geometry[tuple(p)]
        expected_geometry.update({(r, c): label for r, c, label in patch["add"]})
        cells = coords(case["layout"])
        require(cells == expected_geometry, "geometry mismatch")
        events = {(r["tape"], r["t"], r["horizon"]): r for r in case["predictions"]}
        require(len(events) == len(case["predictions"]), "duplicated forecast")
        require([t["id"] for t in case["tapes"]] == list(scripts_by_id), "evaluation tape inventory")
        for tape in case["tapes"]:
            replay(tape, cells)
            rows = tape["steps"]
            require("".join(r["action"] for r in rows) == scripts_by_id[tape["id"]], "changed commands")
            for t, row in enumerate(rows):
                before = history(rows, t)
                for lead in (1, 3):
                    if t + lead > len(rows):
                        continue
                    event = events.pop((tape["id"], t, lead))
                    suffix = rows[t:t+lead]
                    end = suffix[-1]["next_symbol"]
                    actions = tuple(step["action"] for step in suffix)
                    require(event["endpoint"] == end and event["commands"] == "".join(actions)
                            and event["blocked"] == (suffix[-1]["displacement"] == [0, 0]),
                            "forecast endpoint, command or movement mismatch")
                    base = by_symbol.get((row["symbol"], actions), {})
                    rich = by_context.get((before, actions), {}) if before in retained else {}
                    pos = tuple(row["position"])
                    pred = {}
                    for command in actions:
                        link = links.get((pos, command))
                        if link is None:
                            break
                        pos, tile_symbol = link
                    else:
                        pred[tile_symbol] = 1
                    energies = {}
                    for arm, counts in (("appearance",base), ("history",rich or base), ("tile",pred)):
                        total = sum(counts.values())
                        p = .99 * counts.get(end, 0) / total + .01/n if total else 1/n
                        score = -math.log(p)
                        actual = event["arms"][arm]
                        require(abs(actual["probability"] - p) < 1e-14
                                and abs(actual["bfe"] - score) < 1e-12, "independent NLL mismatch")
                        energies[arm] = score
                        worst = max(worst, abs(actual["bfe"] - score))
                        n_score += 1
                    if lead == 3 and name != training_name:
                        primary.append(energies["history"] - energies["appearance"])
        require(not events, "unexpected predictions")
        tasks = protocol["routing"]["tasks"]
        require(len(case["routes"]) == len(tasks)*3, "route inventory length")
        seen = set()
        for tape in case["routes"]:
            stop = replay(tape, cells)
            key = (tape["start"]["symbol"], tape["goal"], tape["heading"], tape["arm"])
            require(key not in seen, "duplicate route")
            seen.add(key)
            steps = tape["steps"]
            require(tape["success"] == (stop == tape["goal"])
                    and all(r["symbol"] != tape["goal"] for r in steps)
                    and tape["budget"] == protocol["routing"]["budget"]
                    and (len(steps) <= tape["budget"] if tape["success"] else len(steps) == tape["budget"])
                    and sum(r["displacement"] == [0, 0] for r in steps) == tape["blocked"],
                    "invalid route termination or movement")
            n_routes += 1
            n_actions += len(steps)
        require(seen == {(*task, arm) for task in tasks for arm in ("appearance","history","tile")},
                "unknown route")
    delta = math.fsum(primary)/len(primary)
    require(abs(delta-summary["primary_delta_nats"]) < 1e-12, "wrong primary result")
    return dict(status="passed", fit_episodes=len(fit_tapes), arm_scores=n_score, routes=n_routes,
                route_actions=n_actions, primary_delta_nats=delta, maximum_nll_error=worst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    print(json.dumps(audit(parser.parse_args().output), indent=2))
