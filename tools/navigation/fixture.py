"""Frozen tile-navigation fixture. The simulator exposes one symbol and achieved motion."""

from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np


ACTIONS = ("N", "E", "S", "W")
_DELTA = dict(zip(ACTIONS, ((-1, 0), (0, 1), (1, 0), (0, -1)), strict=True))


def load_protocol():
    return json.loads(Path(__file__).with_name("protocol.json").read_text())


def _turn(action, quarters):
    return ACTIONS[(ACTIONS.index(action) + quarters) % 4]


def _add(point, delta):
    return point[0] + delta[0], point[1] + delta[1]


def _layout(protocol, condition, rotation):
    if condition not in protocol["conditions"] or rotation not in protocol["rotations"]:
        raise ValueError("unknown navigation case")
    replacements = protocol["same_exit_replacements"] if condition == "same-exits" else {}
    tiles = []
    for r, c, symbol in protocol["geometry"]:
        if rotation == 90:
            r, c = c, -r
        tiles.append(dict(position=[r, c], symbol=replacements.get(symbol, symbol)))
    rows, cols = zip(*(tile["position"] for tile in tiles), strict=True)
    return dict(tiles=tiles, bounds=[min(rows), min(cols), max(rows), max(cols)])


def _landmark(tiles, symbol):
    positions = [p for p, label in tiles.items() if label == symbol]
    if len(positions) != 1:
        raise ValueError(f"expected one landmark named {symbol}, found {len(positions)}")
    return positions[0]


class TileWorld:
    """Only reset and step read the physical layout or simulator position."""

    def __init__(self, layout):
        self._tiles = {tuple(t["position"]): t["symbol"] for t in layout["tiles"]}
        self._position = None

    def reset(self, start):
        self._position = _landmark(self._tiles, start)
        return self._tiles[self._position]

    def step(self, action):
        if self._position is None:
            raise ValueError("reset before stepping the maze")
        delta = _DELTA[action]
        proposed = _add(self._position, delta)
        if proposed in self._tiles:
            self._position = proposed
        else:
            delta = (0, 0)
        return self._tiles[self._position], delta


class Traversal:
    """Accumulate exact odometry from observations and record every attempted move."""

    def __init__(self, world, tape_id, phase, start, position, heading):
        self._world = world
        self.position = position
        self.symbol = world.reset(start)
        self.heading = heading
        self.tape = dict(id=tape_id, phase=phase, start=dict(symbol=self.symbol, position=list(position)),
                         heading=heading, steps=[])

    def apply(self, action):
        symbol, delta = self._world.step(action)
        dest = _add(self.position, delta)
        steps = self.tape["steps"]
        steps.append(dict(t=len(steps) + 1, position=list(self.position), symbol=self.symbol,
                          action=action, displacement=list(delta), next_position=list(dest),
                          next_symbol=symbol))
        self.position, self.symbol, self.heading = dest, symbol, action
        return delta


def _survey(walk):
    order = tuple(_turn(walk.heading, k) for k in range(4))
    seen = set()

    def probe():
        start = walk.position
        seen.add(start)
        for action in order:
            if walk.apply(action) == (0, 0):
                continue
            if walk.position not in seen:
                probe()
            walk.apply(_turn(action, 2))
            if walk.position != start:
                raise ValueError("survey backtracking failed to return to the observed tile")

    probe()


def _traversals(protocol, layout, rotation):
    tapes = {}
    turns = rotation // 90
    for phase in ("fit", "selection", "final"):
        records = []
        if phase == "fit":
            plans = [dict(id=f"survey-{h}", heading=h) for h in protocol["fit"]["headings"]]
        else:
            plans = protocol[phase]
        for plan in plans:
            heading = _turn(plan["heading"], turns)
            walk = Traversal(TileWorld(layout), plan["id"], phase, protocol["origin"], (0, 0), heading)
            if phase == "fit":
                _survey(walk)
            else:
                for leg in plan["legs"]:
                    for command in leg:
                        walk.apply(_turn(command, turns))
            records.append(walk.tape)
        tapes[phase] = records
    return tapes


def _fit_map(fit_tapes):
    cells = {}
    links = set()
    for tape in fit_tapes:
        cells[tuple(tape["start"]["position"])] = tape["start"]["symbol"]
        for row in tape["steps"]:
            dst = tuple(row["next_position"])
            cells[dst] = row["next_symbol"]
            links.add((tuple(row["position"]), row["action"], dst))
    return dict(tiles=[dict(position=list(p), symbol=y) for p, y in sorted(cells.items())],
                links=[dict(source=list(p), action=a, target=list(q)) for p, a, q in sorted(links)])


def fit_places(fit_tapes):
    """Propose the one eligible split using only distinct observed fit coordinates."""
    observed = _fit_map(fit_tapes)
    positions = [tile["position"] for tile in observed["tiles"] if tile["symbol"] == "junction"]
    if len(positions) < 2:
        return dict(symbol=None, positions=[])
    return dict(symbol="junction", positions=positions)


def _place_lookup(places):
    return {(p["symbol"], None if p["position"] is None else tuple(p["position"])): p["id"]
            for p in places}


def _encode(lookup, symbol, pos):
    if (symbol, None) in lookup:
        return lookup[symbol, None]
    return lookup[symbol, tuple(pos)]


def _candidate(name, tapes, observed, proposal, alphabet):
    places = []
    for symbol in sorted({tile["symbol"] for tile in observed["tiles"]}):
        split = name == "split" and symbol == proposal["symbol"]
        for pos in proposal["positions"] if split else [None]:
            places.append(dict(id=len(places), symbol=symbol, position=pos))
    lookup = _place_lookup(places)
    symbol_id = {symbol: i for i, symbol in enumerate(alphabet)}
    candidate = dict(places=places, n_rows=len(places) * len(ACTIONS), n_outcomes=len(alphabet))
    transitions = Counter()
    for phase, records in tapes.items():
        events = []
        for tape in records:
            for step in tape["steps"]:
                source = _encode(lookup, step["symbol"], step["position"])
                action = step["action"]
                events.append([source * len(ACTIONS) + ACTIONS.index(action), symbol_id[step["next_symbol"]]])
                if phase == "fit":
                    target = _encode(lookup, step["next_symbol"], step["next_position"])
                    transitions[source, action, target] += 1
        candidate[phase] = events
    candidate["transitions"] = [dict(source=s, action=a, target=t, count=n)
                                for (s, a, t), n in sorted(transitions.items())]
    return candidate


def build_cases():
    """Generate disjoint traversal schedules and fit-only encodings, without scoring."""
    protocol = load_protocol()
    alphabet = sorted({symbol for _, _, symbol in protocol["geometry"]})
    cases = []
    for condition in protocol["conditions"]:
        for rotation in protocol["rotations"]:
            layout = _layout(protocol, condition, rotation)
            tapes = _traversals(protocol, layout, rotation)
            observed = _fit_map(tapes["fit"])
            proposal = fit_places(tapes["fit"])
            candidates = {name: _candidate(name, tapes, observed, proposal, alphabet)
                          for name in ("appearance", "split")}
            case_id = condition if rotation == 0 else f"{condition}-r{rotation}"
            cases.append(dict(id=case_id, condition=condition, rotation=rotation, layout=layout,
                              alphabet=alphabet, actions=list(ACTIONS), proposal=proposal,
                              traversals=tapes, observed_map=observed, candidates=candidates))
    return cases


def _place_policy(candidate, goal, budget):
    """Finite-horizon travel cost from exploration counts; terminal failure costs budget + 1."""
    places = candidate["places"]
    lookup = _place_lookup(places)
    goal_id = lookup[goal, None]
    n = len(places)
    counts = np.zeros((n, len(ACTIONS), n))
    for edge in candidate["transitions"]:
        counts[edge["source"], ACTIONS.index(edge["action"]), edge["target"]] = edge["count"]
    for source, action in np.argwhere(counts.sum(axis=2) == 0):
        counts[source, action, source] = 1
    probabilities = counts / counts.sum(axis=2, keepdims=True)
    value = np.full(n, float(budget + 1))
    value[goal_id] = 0.0
    costs = []
    for _ in range(budget):
        trial = 1.0 + probabilities @ value
        costs.append(trial)
        value = trial.min(axis=1)
        value[goal_id] = 0.0

    def act(position, symbol, remaining):
        place = _encode(lookup, symbol, position)
        return costs[remaining - 1][place]

    return act


def _tile_policy(observed, goal):
    """Shortest paths over the directed moves actually taken during exploration."""
    tiles = {tuple(t["position"]): t["symbol"] for t in observed["tiles"]}
    dest = _landmark(tiles, goal)
    neighbors = {}
    parents = defaultdict(set)
    for link in observed["links"]:
        src, dst = tuple(link["source"]), tuple(link["target"])
        neighbors[src, link["action"]] = dst
        parents[dst].add(src)
    distance = {dest: 0}
    queue = deque([dest])
    while queue:
        dst = queue.popleft()
        for src in parents[dst]:
            if src not in distance:
                distance[src] = distance[dst] + 1
                queue.append(src)

    def act(position, symbol, remaining):
        return [1 + distance.get(neighbors.get((position, a)), float("inf")) for a in ACTIONS]

    return act


def navigate(case, arm):
    """Evaluate one frozen planner. The tile simulator and the policy have separate inputs."""
    if arm not in ("appearance", "split", "tile"):
        raise ValueError(f"unknown navigation arm {arm!r}")
    settings = load_protocol()["navigation"]
    observed = case["observed_map"]
    tiles = {tuple(t["position"]): t["symbol"] for t in observed["tiles"]}
    budget = settings["budget"]
    routes = []
    for task in settings["tasks"]:
        start, goal = task["start"], task["goal"]
        policy = (_tile_policy(observed, goal) if arm == "tile"
                  else _place_policy(case["candidates"][arm], goal, budget))
        heading = _turn(task["heading"], case["rotation"] // 90)
        walk = Traversal(TileWorld(case["layout"]), task["id"], "navigation", start,
                         _landmark(tiles, start), heading)
        for remaining in range(budget, 0, -1):
            if walk.symbol == goal:
                break
            costs = policy(walk.position, walk.symbol, remaining)
            best = min(costs)
            order = [_turn(walk.heading, k) for k in range(4)]
            action = next(a for a in order if costs[ACTIONS.index(a)] <= best + 1e-12)
            walk.apply(action)
        success = walk.symbol == goal
        walk.tape.update(arm=arm, goal=goal, budget=budget, success=success,
                         reason="goal" if success else "budget")
        routes.append(walk.tape)
    return routes
