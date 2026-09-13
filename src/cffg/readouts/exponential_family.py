"""exponential family readouts over explicit structure and current numerical inputs."""

from __future__ import annotations

import numpy as np

from cffg import ef
from cffg.readouts.types import (
    EfMarginal,
)
from cffg.readouts.views import (
    EdgesView,
)
from cffg.runtime_types import State


def ef_marginals(view: EdgesView, state: State) -> dict[str, EfMarginal]:
    """Edge name -> EF marginal in standard parameters. Observation-family
    edges (poisson/bernoulli DATA ports) carry no messages and are
    skipped. An improper combined belief (e.g. before any sweep) reports
    proper=False with NaN moments — check it before trusting `mean`."""
    out: dict[str, EfMarginal] = {}
    eta = np.asarray(state.ef_eta)
    for e in view.edges.values():
        if not e.ef or e.ef not in ef.FAMILIES:
            continue
        fam = ef.FAMILIES[e.ef]
        eta_q = fam.product(eta[e.slot0], eta[e.slot1])
        proper = bool(fam.is_proper(eta_q))
        out[e.name] = EfMarginal(
            family=e.ef,
            params=tuple(float(p) for p in fam.from_natural(eta_q)),
            mean=float(fam.mean(eta_q)) if proper else float("nan"),
            proper=proper,
        )
    return out
