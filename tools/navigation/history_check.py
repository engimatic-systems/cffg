"""Independent hand-enumerated histories, forecasting, control and failure checks."""
from copy import deepcopy
import math

import jax
import numpy as np

from history_study import NativeScores
from movement_history import ObservedMap, context, fit


def tape(name, x, start, goal):
    labels = [start, "mid", "J", goal, "J", goal]
    rows = []
    position = (0, x)
    for t, (a, d) in enumerate(zip("EENSN", [(0,1), (0,1), (-1,0), (1,0), (-1,0)])):
        dest = tuple(np.add(position, d).tolist())
        rows.append(dict(symbol=labels[t], action=a, displacement=list(d),
                         next_symbol=labels[t+1], position=list(position), next_position=list(dest)))
        position = dest
    return {"id": name, "phase": "fit", "steps": rows}


def check():
    alphabet = ["J", "a", "b", "left", "mid", "right"]
    left, right = tape("left", 0, "a", "left"), tape("right", 10, "b", "right")
    tapes = [left, right, deepcopy(left), deepcopy(right)]
    history = fit(tapes, alphabet, histories=True)
    plain = fit(tapes, alphabet, histories=False)
    kr = context(right["steps"][:2])
    kl = context(left["steps"][:2])
    rare = fit([left, right], alphabet, histories=True)
    assert kl not in rare.ids and kr not in rare.ids
    assert kl != kr and history.classify("J", kl) != history.classify("J", kr)
    for commands in ("N", "NSN"):
        # Each of the two latent locations is visited equally often with these commands.
        for model, key, weight in ((plain, kl, .5), (history, kl, 1.), (history, kr, 0.)):
            p, used = model.predict("J", key, commands)
            assert math.isclose(p[alphabet.index("left")], .99*weight + .01/6)
            assert math.isclose(sum(p), 1.)
            assert used == ("appearance" if model is plain else "history")
    assert history.predict("J", (), "N")[1] == "appearance"
    p, used = history.predict("J", kl, "WWW")
    np.testing.assert_array_equal(p, np.full(6, 1/6))
    assert used == "uniform"
    action_costs = history.route_policy("left", 3)("J", kl, 3)
    assert np.argmin(action_costs) == 0
    failed = deepcopy(left["steps"][:2])
    failed[-1]["displacement"] = [0, 0]
    assert context(failed) != kl
    assert context(left["steps"][:1]) == ()

    tile = ObservedMap({"tiles":[{"position":[0,0],"symbol":"a"},{"position":[0,1],"symbol":"b"}],
                        "links":[{"source":[0,0],"action":"N","target":[0,0]},
                                 {"source":[0,0],"action":"E","target":[0,1]}]})
    p, used = tile.predict((0, 0), "NNE", alphabet)
    assert used == "map" and math.isclose(p[2], .99+.01/6)
    assert tile.predict((0, 1), "N", alphabet)[1] == "uniform"
    try:
        fit([dict(left, phase="evaluation")], alphabet, histories=True)
    except ValueError:
        pass
    else:
        raise AssertionError("fitter consumed evaluation data")

    with jax.enable_x64(True):
        tree = NativeScores(alphabet)
        assert abs(tree.score(p, 2) + math.log(.99+.01/6)) < 1e-13
        assert abs(tree.score(p, 1) + math.log(.01/6)) < 1e-12
    return {"status": "passed", "native_nll_error": tree.maximum_error}


if __name__ == "__main__":
    print(check())
