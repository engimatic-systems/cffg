"""Check the navigation pilot by replaying its saved observations and applied actions."""

import argparse
from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
import math
import re
import subprocess
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def integer(value, label):
    require(type(value) is int, f"{label}: expected an integer")
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: expected a finite number")
    return value


def same(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: values differ ({str(actual)[:100]} vs {str(expected)[:100]})")


def unique_keys(items):
    result = {}
    for k, v in items:
        require(k not in result, f"duplicate JSON key {k!r}")
        result[k] = v
    return result


def read_json(root, name, hashes):
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), f"invalid artifact filename {name!r}")
    raw = path.read_bytes()
    hashes[name] = sha256(raw).hexdigest()
    return json.loads(raw, object_pairs_hook=unique_keys, parse_constant=lambda s: finite(s, "JSON number"))


def save_receipt(path, result):
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        tmp = Path(f.name)
        try:
            json.dump(result, f, allow_nan=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


ACTIONS = ("N", "E", "S", "W")
MOVES = dict(zip(ACTIONS, ((-1, 0), (0, 1), (1, 0), (0, -1)), strict=True))
PHASES = ("fit", "selection", "final")


def point(raw):
    require(isinstance(raw, list) and len(raw) == 2 and all(type(v) is int for v in raw),
            f"invalid row/column coordinate: {raw!r}")
    return tuple(raw)


def tile_dict(tiles):
    result = {}
    for record in tiles:
        pos = point(record["position"])
        symbol = record["symbol"]
        require(isinstance(symbol, str) and symbol and pos not in result, "duplicate or malformed tile")
        result[pos] = symbol
    require(result, "empty map")
    return result


def neighbors(pos):
    for action, (dy, dx) in MOVES.items():
        yield action, (pos[0] + dy, pos[1] + dx)


def topology(layout):
    adj = {pos: [dst for _, dst in neighbors(pos) if dst in layout] for pos in layout}
    pending = deque([next(iter(layout))])
    seen = set(pending)
    while pending:
        for dst in adj[pending.popleft()]:
            if dst not in seen:
                seen.add(dst)
                pending.append(dst)
    same(len(seen), len(layout), "connected layout")
    edges = sum(map(len, adj.values())) // 2
    return {"tiles": len(layout), "undirected_edges": edges, "independent_cycles": edges - len(layout) + 1}


def replay(tape, layout):
    start = tape["start"]
    pos = point(start["position"])
    require(pos in layout, "tape starts outside layout")
    same(start["symbol"], layout[pos], "start sensor")
    require(tape["heading"] in ACTIONS, "invalid initial heading")
    events = []
    for t, step in enumerate(tape["steps"], 1):
        same(integer(step["t"], "t"), t, "step clock")
        same(point(step["position"]), pos, "pre-action odometry")
        same(step["symbol"], layout[pos], "pre-action sensor")
        act = step["action"]
        require(act in MOVES, "unknown action")
        dy, dx = MOVES[act]
        dst = (pos[0]+dy, pos[1]+dx)
        if dst not in layout:
            dst = pos
        same(point(step["next_position"]), dst, "post-action odometry")
        same(point(step["displacement"]), (dst[0]-pos[0], dst[1]-pos[1]), "applied displacement")
        same(step["next_symbol"], layout[dst], "post-action sensor")
        events.append((pos, act, dst))
        pos = dst
    return events


def observed_places(tapes, layout):
    """Only the starting tiles and executed movements are visible in fit data."""
    tiles = {}
    links = set()
    for tape in tapes:
        pos = point(tape["start"]["position"])
        tiles[pos] = layout[pos]
        for pos, action, dst in replay(tape, layout):
            tiles[dst] = layout[dst]
            links.add((pos, action, dst))
    return tiles, links


def near(actual, expected, tolerance, label):
    error = abs(finite(actual, label) - expected)
    require(error <= tolerance, f"{label}: error {error:g} exceeds {tolerance:g}")
    return error


def predictive_checks(counts, heldout, score):
    """Evaluate each held-out event from raw fit counts without fitting on a scored tape."""
    n_rows, alphabet_size = len(counts), len(counts[0])
    for name, value in (("dtype", "float64"), ("status", "ok"), ("schedule", "two_pass"),
                        ("n_rows", n_rows), ("n_outcomes", alphabet_size), ("n_events", len(heldout))):
        same(score[name], value, f"score {name}")
    same(score["outcomes"], [symbol for _, symbol in heldout], "ordered held-out observations")
    require(heldout, "empty held-out tape")
    for key in ("energies", "oracle_energies"):
        same(len(score[key]), len(heldout), key)
    totals = [sum(row) + alphabet_size for row in counts]
    expected = [-math.log((counts[r][s]+1)/totals[r]) for r, s in heldout]
    bounds = [64 * sys.float_info.epsilon * max(1., nats) for nats in expected]
    worst = 0.
    for i, (nats, atol) in enumerate(zip(expected, bounds, strict=True)):
        worst = max(worst, near(score["energies"][i], nats, atol, f"native BFE event {i}"))
        near(score["oracle_energies"][i], nats, atol, f"archived oracle event {i}")
    total = math.fsum(expected)
    total_bound = math.fsum(bounds)
    near(score["total_bfe"], total, total_bound, "held-out total")
    near(score["mean_bfe"], total/len(heldout), total_bound/len(heldout), "held-out mean")
    same(score["total_bfe"], math.fsum(score["energies"]), "sum of native event scores")
    same(score["mean_bfe"], score["total_bfe"]/len(heldout), "mean native score")
    same(score["max_abs_error"], max(abs(a-b) for a, b in zip(score["energies"], score["oracle_energies"],
                                                             strict=True)), "reported native/oracle error")
    near(score["total_error_bound"], total_bound, 4*math.ulp(total_bound), "selection error bound")
    return {"events": len(heldout), "independent_total_nats": total, "max_abs_energy_error_nats": worst,
            "independent_error_bound": total_bound}


def source_receipt(study, development):
    repo = Path(__file__).resolve().parents[2]
    rev = study["git_commit"]
    require(isinstance(rev, str) and re.fullmatch("[0-9a-f]{40}", rev), "invalid source commit")
    tree = subprocess.check_output(["git", "rev-parse", f"{rev}:src/cffg"], cwd=repo, text=True).strip()
    same(study["cffg_tree"], tree, "CFFG source tree")
    sources = study["source_sha256"]
    expected = {"tools/navigation/" + name for name in ("protocol.json", "fixture.py", "evidence.py", "study.py")}
    same(set(sources), expected, "study source inventory")
    same(sources["tools/navigation/protocol.json"],
         "ee15143294d95d8c7b014b5b3b4da7973f44696bd47ebf0078c08b9623ff2429",
         "pre-scoring protocol SHA-256")
    for name, h in sources.items():
        if development:
            raw = (repo / name).read_bytes()
        else:
            raw = subprocess.check_output(["git", "show", f"{rev}:{name}"], cwd=repo)
        same(sha256(raw).hexdigest(), h, f"{name} source bytes")
        if name.endswith("protocol.json"):
            same(json.loads(raw, object_pairs_hook=unique_keys), study["protocol"], "registered protocol")
    return {"commit": rev, "cffg_tree": tree, "verified_against": "working tree" if development else "Git objects",
            "files": sources}


def check_guide(root, study, hashes):
    raw = (root/"index.html").read_bytes()
    hashes["index.html"] = sha256(raw).hexdigest()
    matches = re.findall(r'<script id="study" type="application/json">(.*?)</script>', raw.decode(), re.DOTALL)
    same(len(matches), 1, "embedded study count")
    same(json.loads(matches[0], object_pairs_hook=unique_keys), study, "guide and study.json")


def select_model(actions, finding):
    base, split = (actions[a]["selection"] for a in ("appearance", "split"))
    threshold = math.fsum(s["independent_error_bound"] for s in (base, split))
    diff = base["independent_total_nats"] - split["independent_total_nats"]
    selected = "split" if diff > threshold else "appearance"
    evidence = finding["scores"]
    reported = finding["selection"]
    raw_gain = evidence["appearance"]["selection"]["total_bfe"] - evidence["split"]["selection"]["total_bfe"]
    same(reported["gain"], raw_gain, "published selection gain")
    near(reported["gain"], diff, threshold, "independent selection gain")
    near(reported["tie_tolerance"], threshold, 4*math.ulp(threshold), "selection threshold")
    same(reported["selected"], selected, "model selected by reserved selection tapes")
    same(reported["reason"], "precision_tie" if abs(raw_gain) <= reported["tie_tolerance"] else "lower_bfe",
         "selection reason")
    return {"selected": selected, "independent_gain_nats": diff, "tie_tolerance_nats": threshold}


def event_tables(case, candidate_name, phases, layout, fit_tiles):
    cand = case["candidates"][candidate_name]
    alphabet = case["alphabet"]
    require(set(fit_tiles.values()) <= set(alphabet), "observed symbol outside the registered alphabet")
    symbol_id = {symbol: i for i, symbol in enumerate(alphabet)}
    proposal = case["proposal"]
    alias = proposal["symbol"]
    centers = {pos for pos, symbol in fit_tiles.items() if symbol == alias}
    same({point(pos) for pos in proposal["positions"]}, centers, "proposed split sites derived from fit")
    same(len(centers), 2, "number of aliased junctions")
    places = {}
    for place in cand["places"]:
        symbol = place["symbol"]
        pos = point(place["position"]) if place["position"] is not None else None
        key = (symbol, pos)
        require(key not in places, "duplicate candidate place definition")
        places[key] = integer(place["id"], "place ID")
    expected = {(s, None) for s in fit_tiles.values()}
    if candidate_name == "split":
        expected.remove((alias, None))
        expected.update((alias, pos) for pos in centers)
    same(set(places), expected, "candidate places")
    same([places[k] for k in sorted(expected)], list(range(len(expected))), "registered candidate ID order")
    same(integer(cand["n_rows"], "n_rows"), len(places)*len(ACTIONS), "prediction rows")
    same(integer(cand["n_outcomes"], "n_outcomes"), len(alphabet), "prediction alphabet")
    def place_id(pos):
        s = layout[pos]
        return places[s, pos if candidate_name == "split" and s == alias else None]
    counts = [[0]*len(alphabet) for _ in range(cand["n_rows"])]
    links, events = Counter(), {}
    for phase, steps in phases.items():
        rows = []
        for pos, action, nxt in steps:
            src, dst = place_id(pos), place_id(nxt)
            r, y = src*len(ACTIONS) + ACTIONS.index(action), symbol_id[layout[nxt]]
            rows.append([r, y])
            if phase == "fit":
                counts[r][y] += 1
                links[src, action, dst] += 1
        require(all(isinstance(row, list) and len(row) == 2 and all(type(v) is int for v in row)
                    for row in cand[phase]), f"invalid {candidate_name}/{phase} row IDs")
        same(cand[phase], rows, f"{candidate_name}/{phase}: recorded place/action/observation rows")
        events[phase] = rows
    recorded = Counter()
    for item in cand["transitions"]:
        key = (integer(item["source"], "source"), item["action"], integer(item["target"], "target"))
        require(key not in recorded, "duplicate transition row")
        n = integer(item["count"], "transition count")
        require(n > 0, "nonpositive transition count")
        recorded[key] = n
    same(recorded, links, f"{candidate_name}: fit transition counts")
    return counts, events, links, place_id


def rotation_check(original, rotated):
    """Compare adjacency, public labels and commanded paths under the quarter turn."""
    a, b = tile_dict(original["layout"]["tiles"]), tile_dict(rotated["layout"]["tiles"])
    rot = lambda p: (p[1], -p[0])
    same({rot(p): symbol for p, symbol in a.items()}, b, "90-degree rotated topology and labels")
    for phase in PHASES:
        tapes = original["traversals"][phase]
        rotated_tapes = rotated["traversals"][phase]
        same(len(tapes), len(rotated_tapes), "paired rotated tape count")
        for one, two in zip(tapes, rotated_tapes, strict=True):
            same(rot(point(one["start"]["position"])), point(two["start"]["position"]), "rotation of tape start")
            same([rot(point(s["next_position"])) for s in one["steps"]],
                 [point(s["next_position"]) for s in two["steps"]], "rotation of executed movement")
            same([ACTIONS[(ACTIONS.index(s["action"])+1)%4] for s in one["steps"]],
                 [s["action"] for s in two["steps"]], "rotation of intended action")


def audit(root, *, development=False):
    root = root.expanduser().resolve(strict=True)
    hashes = {}
    study = read_json(root, "study.json", hashes)
    same(integer(study["schema_version"], "schema version"), 1, "schema version")
    same(study["runtime"]["precision"], "float64", "reference precision")
    sources = source_receipt(study, development)
    check_guide(root, study, hashes)
    cases = {case["id"]: case for case in study["cases"]}
    findings = {f["id"]: f for f in study["findings"]}
    expected = {"aliased", "aliased-r90", "same-exits", "same-exits-r90"}
    same(set(cases), expected, "case identities")
    same(set(findings), expected, "finding identities")
    same(len(study["cases"]), len(cases), "case count")
    same(len(study["findings"]), len(findings), "finding count")
    results = [audit_case(case, findings[case["id"]], study["protocol"]) for case in study["cases"]]
    for condition in ("aliased", "same-exits"):
        rotation_check(cases[condition], cases[condition+"-r90"])
    same(hashes["study.json"], sha256((root/"study.json").read_bytes()).hexdigest(), "stable study")
    same(hashes["index.html"], sha256((root/"index.html").read_bytes()).hexdigest(), "stable guide")
    return dict(status="passed", artifacts=hashes, verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                source_seal=sources, cases=results,
                scope="Four designed loop cases, their held-out scores and navigation tapes. "
                      "The rotations reuse the two layouts. Any separate scale benchmark has its own receipt.")


def compass(action, rotation):
    return ACTIONS[(ACTIONS.index(action)+rotation//90) % 4]


def check_survey(events, heading):
    """Read the archived moves in order, with a stack of surveyed coordinates."""
    cursor = 0
    seen = set()
    ordered = ACTIONS[ACTIONS.index(heading):] + ACTIONS[:ACTIONS.index(heading)]

    def take(pos, action):
        nonlocal cursor
        require(cursor < len(events), "fit tape ends during its survey")
        start, applied, dst = events[cursor]
        cursor += 1
        same((start, applied), (pos, action), "depth-first survey policy")
        return dst

    def explore(pos):
        seen.add(pos)
        for action in ordered:
            dst = take(pos, action)
            if dst != pos:
                if dst not in seen:
                    explore(dst)
                same(take(dst, compass(action, 180)), pos, "return to survey origin")

    explore((0, 0))
    same(cursor, len(events), "extra steps after complete survey")


def check_archived_paths(case, layout, protocol):
    rotation = case["rotation"]
    phases = {}
    signatures = set()
    require(set(case["traversals"]) == set(PHASES), "traversal phases differ")
    for phase in PHASES:
        inputs = case["traversals"][phase]
        specs = protocol["fit"]["headings"] if phase == "fit" else protocol[phase]
        same(len(inputs), len(specs), f"{phase} tape count")
        events = []
        for tape, spec in zip(inputs, specs, strict=True):
            same(tape["phase"], phase, "declared tape phase")
            same(point(tape["start"]["position"]), (0, 0), "traversal starts at home")
            start_heading = spec if phase == "fit" else spec["heading"]
            same(tape["heading"], compass(start_heading, rotation), "initial heading")
            tape_events = replay(tape, layout)
            moves = "".join(a for _, a, _ in tape_events)
            require(moves not in signatures, "entire action tape reused within/across partitions")
            signatures.add(moves)
            if phase == "fit":
                same(tape["id"], f"survey-{spec}", "fit traversal identity")
                check_survey(tape_events, tape["heading"])
            else:
                same(tape["id"], spec["id"], "reserved traversal identity")
                planned = "".join(compass(a, rotation) for a in "".join(spec["legs"]))
                same(moves, planned, f"{phase} precommitted action tape")
            events.extend(tape_events)
        phases[phase] = events
    return phases


def route_costs(links, goal, count, budget):
    """Finite-horizon expected step counts, with the registered terminal loss."""
    dist = defaultdict(Counter)
    for (src, action, dst), n in links.items():
        dist[src, action][dst] += n
    value = [0. if i == goal else float(budget+1) for i in range(count)]
    layers = [None]
    for _ in range(budget):
        costs = {}
        for src in range(count):
            for action in ACTIONS:
                edges = dist[src, action]
                future = (math.fsum(n*value[dst] for dst, n in edges.items()) / edges.total()
                          if edges else value[src])
                costs[src, action] = 1. + future
        layers.append(costs)
        value = [0. if src == goal else min(costs[src, a] for a in ACTIONS) for src in range(count)]
    return layers


def tile_distances(known_links, goal):
    parents = defaultdict(set)
    for src, _, dst in known_links:
        parents[dst].add(src)
    pending, cost = deque([goal]), {goal: 0}
    while pending:
        dst = pending.popleft()
        for src in parents[dst]:
            if src not in cost:
                cost[src] = cost[dst] + 1
                pending.append(src)
    return cost


def navigation_checks(case, finding, layouts, policy, known_links, learned):
    layout, fit_tiles = layouts
    budget = policy["budget"]
    tasks = policy["tasks"]
    arms = ("appearance", "split", "tile")
    same(set(case["routes"]), set(arms), "controller set")
    same(set(finding["routes"]), set(arms), "reported controller set")
    result = {}
    for arm in arms:
        routes = case["routes"][arm]
        same(len(routes), len(tasks), "registered route count")
        records = []
        for tape, task in zip(routes, tasks, strict=True):
            same(tape["id"], task["id"], "navigation task")
            same(tape["goal"], task["goal"], "navigation destination")
            same(tape["start"]["symbol"], task["start"], "route starting landmark")
            same(tape["arm"], arm, "route controller label")
            same(tape["phase"], "navigation", "route phase")
            same(tape["budget"], budget, "route budget")
            facing = compass(task["heading"], case["rotation"])
            same(tape["heading"], facing, "route heading")
            start, goal = [], []
            for p, symbol in fit_tiles.items():
                if symbol == task["start"]:
                    start.append(p)
                if symbol == task["goal"]:
                    goal.append(p)
            require(len(start) == len(goal) == 1, "ambiguous route endpoint")
            same(point(tape["start"]["position"]), start[0], "fit-only landmark localization")
            if arm == "tile":
                distance = tile_distances(known_links, goal[0])
                dest = {(src, act): dst for src, act, dst in known_links}
            else:
                links, place_id, n = learned[arm]
                layers = route_costs(links, place_id(goal[0]), n, budget)
            events = replay(tape, layout)
            require(0 < len(events) <= budget, "empty or overlong navigation tape")
            for t, (p, move, _) in enumerate(events):
                require(p != goal[0], "action after arriving at goal")
                if arm == "tile":
                    scores = {a: distance.get(dest.get((p, a)), math.inf) for a in ACTIONS}
                else:
                    scores = {a: layers[budget-t][place_id(p), a] for a in ACTIONS}
                order = [compass(facing, quarter*90) for quarter in range(4)]
                best = min(scores.values())
                require(math.isfinite(best), "no reachable action under recorded model")
                expected = next(a for a in order if scores[a] <= best+1.e-12)
                same(move, expected, f"{arm}/{task['id']} action {t+1}")
                facing = move
            solved = events[-1][2] == goal[0]
            require(tape["success"] is solved, "incorrect navigation success")
            same(tape["reason"], "goal" if solved else "budget", "route stop reason")
            require(solved or len(events) == budget, "route stopped before exhaustion or first arrival")
            records.append({"id": task["id"], "success": solved, "actions": len(events)})
        summary = dict(attempted=len(records), completed=sum(r["success"] for r in records),
                       actions=sum(r["actions"] for r in records))
        same(finding["routes"][arm], summary, f"navigation summary for {arm}")
        result[arm] = dict(**summary, tasks=records)
    return result


def audit_case(case, finding, protocol):
    condition, degrees = case["condition"], integer(case["rotation"], "rotation")
    require(condition in protocol["conditions"] and degrees in (0, 90), "unknown case conditions")
    same(case["id"], condition + ("-r90" if degrees else ""), "case condition and rotation")
    same(case["actions"], list(ACTIONS), "compass action IDs")
    geometry = {(y, x): symbol for y, x, symbol in protocol["geometry"]}
    same(case["alphabet"], sorted(set(geometry.values())), "shared fixed observation alphabet")
    if condition == "same-exits":
        replacements = protocol["same_exit_replacements"]
        geometry = {p: replacements.get(s, s) for p, s in geometry.items()}
    if degrees:
        geometry = {(x, -y): s for (y, x), s in geometry.items()}
    layout = tile_dict(case["layout"]["tiles"])
    same(layout, geometry, "declared topology and sensors")
    ys, xs = zip(*layout, strict=True)
    same(case["layout"]["bounds"], [min(ys), min(xs), max(ys), max(xs)], "layout bounds")
    topo = topology(layout)
    same(topo["independent_cycles"], 1, "connected loop count")
    same(case["proposal"]["symbol"], "junction", "registered split eligibility")
    exits = [tuple(layout.get(nxt) for _, nxt in neighbors(pos)) for pos, s in layout.items() if s == "junction"]
    require(len(exits) == 2 and all(None not in row for row in exits), "junctions lack four legal exits")
    same(exits[0] == exits[1], condition == "same-exits", "matched-exit negative control")
    phases = check_archived_paths(case, layout, protocol)
    fit_tiles, known_links = observed_places(case["traversals"]["fit"], layout)
    same(tile_dict(case["observed_map"]["tiles"]), fit_tiles, "fit-only tile map")
    links = [(point(record["source"]), record["action"], point(record["target"]))
             for record in case["observed_map"]["links"]]
    same(len(links), len(set(links)), "duplicate observed link count")
    same(set(links), known_links, "fit-only directed tile links")
    same(set(case["candidates"]), {"appearance", "split"}, "candidate set")
    same(set(finding["scores"]), set(case["candidates"]), "scored candidate set")
    scores, learned = {}, {}
    for name, candidate in case["candidates"].items():
        counts, events, transitions, place_id = event_tables(case, name, phases, layout, fit_tiles)
        scores[name] = {p: predictive_checks(counts, events[p], finding["scores"][name][p])
                        for p in ("selection", "final")}
        learned[name] = transitions, place_id, len(candidate["places"])
    selection = select_model(scores, finding)
    routes = navigation_checks(case, finding, (layout, fit_tiles), protocol["navigation"], known_links, learned)
    return dict(id=case["id"], topology=topo, fit_observed_tiles=len(fit_tiles),
                fit_directed_links=len(known_links), traversal_events={k: len(v) for k, v in phases.items()},
                candidates=scores, selection=selection, navigation=routes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="directory containing study.json and replay guide")
    parser.add_argument("--output", type=Path, help="path for an audit.json receipt (default stdout)")
    parser.add_argument("--development", action="store_true", help="compare source hashes with local development files")
    args = parser.parse_args()
    if args.output and args.output.name != "audit.json":
        parser.error("receipt filename must be audit.json")
    try:
        result = audit(args.root, development=args.development)
    except (ValueError, KeyError, IndexError, TypeError, OSError, ArithmeticError, subprocess.CalledProcessError) as exc:
        result = dict(status="failed", error=str(exc))
    if args.output:
        save_receipt(args.output.expanduser(), result)
        print(f"{args.output}: {result['status']}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] == "failed":
        print(result["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
