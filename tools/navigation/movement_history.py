"""Fit observation and achieved-motion histories; predict and route with frozen counts."""

from collections import Counter, defaultdict
import math

import numpy as np

from fixture import ACTIONS


def context(past):
    """Three observed symbols connected by two actual, possibly blocked, commands."""
    if len(past) < 2:
        return ()
    first, second = past[-2:]
    if first["next_symbol"] != second["symbol"]:
        raise ValueError("history crosses a discontinuity")
    return (first["symbol"], first["action"], *first["displacement"], first["next_symbol"],
            second["action"], *second["displacement"], second["next_symbol"])


def origins(tape):
    """Yield each pre-action observation and a bounded slice of its own episode."""
    rows = tape["steps"]
    for t, row in enumerate(rows):
        yield row["symbol"], context(rows[max(t - 2, 0):t]), tuple(row["position"])


def smoothed(frequencies, size):
    probabilities = np.zeros(size, dtype=np.float64)
    total = sum(frequencies.values())
    if not total:
        return np.full(size, 1.0 / size)
    for outcome, count in frequencies.items():
        probabilities[outcome] = .99 * count / total
    return probabilities + .01 / size


class Places:
    def __init__(self, record):
        self.record = record
        self.alphabet = tuple(record["alphabet"])
        self.symbols = {s: i for i, s in enumerate(self.alphabet)}
        self.keys = tuple(map(tuple, record["keys"]))
        self.ids = {key: i for i, key in enumerate(self.keys)}
        self.seen = set(map(tuple, record["histories_seen"]))
        self.aliases = set(record["aliases"])
        self.predictions = {
            (source, commands): dict(outcomes)
            for source, commands, outcomes in record["prediction_counts"]
        }
        self.locations = defaultdict(Counter)
        for state, r, c, count in record["locations"]:
            self.locations[state][(r, c)] = count

    def classify(self, symbol, history):
        return self.ids.get(history, self.symbols[symbol])

    def predict(self, symbol, history, commands):
        state = self.classify(symbol, history)
        source = state
        row = self.predictions.get((source, commands))
        if row is None:
            source = self.symbols[symbol]
            row = self.predictions.get((source, commands))
        probabilities = smoothed(row or {}, len(self.symbols))
        used = ("uniform" if row is None else "appearance" if source < len(self.symbols)
                else "history")
        return probabilities, used

    def place_entropy(self, state):
        counts = self.locations.get(state, {})
        total = sum(counts.values())
        return -math.fsum((c / total) * math.log(c / total) for c in counts.values())

    def route_policy(self, goal, budget):
        """Expected travel cost on fitted, sparse one-step kernel including stays."""
        n = len(self.keys)
        transitions = np.array(self.record["transitions"], dtype=np.int64)
        src, action, dst, counts = transitions.T
        rows = src * len(ACTIONS) + action
        totals = np.bincount(rows, weights=counts, minlength=n * len(ACTIONS))
        fallback = np.array([self.symbols[key[-1]] for key in self.keys])
        goal_states = np.array([key[-1] == goal for key in self.keys])
        value = np.where(goal_states, 0., budget + 1.)
        costs = []
        for _ in range(budget):
            summed = np.bincount(rows, weights=counts * value[dst], minlength=len(totals))
            expected = np.divide(summed, totals, where=totals > 0, out=np.zeros_like(summed))
            expected = expected.reshape(n, len(ACTIONS))
            known = totals.reshape(n, len(ACTIONS)) > 0
            backoff = np.where(known[fallback], expected[fallback], value[:, None])
            trial = 1. + np.where(known, expected, backoff)
            costs.append(trial)
            value = np.where(goal_states, 0., trial.min(axis=1))
        return lambda symbol, history, remaining: costs[remaining - 1][self.classify(symbol, history)]


def fit(tapes, alphabet, *, histories):
    """Only callers that hold exploration tapes can add rows or nominate a context."""
    if not tapes or any(tape["phase"] != "fit" for tape in tapes):
        raise ValueError("fit requires nonempty exploration tapes")
    locations = defaultdict(set)
    observed = Counter()
    for tape in tapes:
        for symbol, key, position in origins(tape):
            locations[symbol].add(position)
            if key:
                observed[key] += 1
        if tape["steps"]:
            last = tape["steps"][-1]
            locations[last["next_symbol"]].add(tuple(last["next_position"]))
    aliases = {symbol for symbol, positions in locations.items() if len(positions) > 1}
    retained = sorted(key for key, count in observed.items() if count >= 2 and key[-1] in aliases)
    keys = [(symbol,) for symbol in alphabet] + (retained if histories else [])
    ids = {key: i for i, key in enumerate(keys)}
    counts = defaultdict(Counter)
    transitions = Counter()
    positions = Counter()
    for tape in tapes:
        rows = tape["steps"]
        if not rows:
            continue
        data = list(origins(tape))
        last = rows[-1]
        data.append((last["next_symbol"], context(rows[-2:]), tuple(last["next_position"])))
        states = [ids.get(key, ids[(symbol,)]) for symbol, key, _ in data]
        for t, row in enumerate(rows):
            symbol, _, (r, c) = data[t]
            appearance = ids[(symbol,)]
            for source in {appearance, states[t]}:
                positions[source, r, c] += 1
                transitions[source, ACTIONS.index(row["action"]), states[t + 1]] += 1
                for lead in (1, 3):
                    if t + lead > len(rows):
                        continue
                    commands = "".join(step["action"] for step in rows[t:t + lead])
                    end_symbol = rows[t + lead - 1]["next_symbol"]
                    counts[source, commands][ids[(end_symbol,)]] += 1
    return Places({
        "arm": "history" if histories else "appearance", "alphabet": alphabet, "keys": keys,
        "aliases": sorted(aliases), "histories_seen": sorted(observed) if histories else [],
        "locations": [list(k) + [v] for k, v in sorted(positions.items())],
        "prediction_counts": [[s, actions, sorted(row.items())]
                              for (s, actions), row in sorted(counts.items())],
        "transitions": [list(k) + [v] for k, v in sorted(transitions.items())],
    })


class ObservedMap:
    def __init__(self, observed):
        self.tiles = {tuple(t["position"]): t["symbol"] for t in observed["tiles"]}
        self.links = {(tuple(edge["source"]), edge["action"]): tuple(edge["target"])
                      for edge in observed["links"]}

    def predict(self, position, commands, alphabet):
        for action in commands:
            position = self.links.get((position, action))
            if position is None:
                return smoothed({}, len(alphabet)), "uniform"
        return smoothed({alphabet.index(self.tiles[position]): 1}, len(alphabet)), "map"
