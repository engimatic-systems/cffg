"""The generic discrete POMDP rollout template (the papers' Fig. 26 pattern).

One slice per timestep: control prior → transition mixture → equality branch
→ observation. The first `past` slices are perception-style (control and
observation exposed as DATA ports `act{k}` / `obs{k}`); the rest are
planning-style (control prior + P-substituted goal-observation composite).
As the agent's clock advances, a slice's beads flip from goal to data with
identical wiring — the receding-horizon constraint flip.

Any (d, A, c, e, Bs) instantiates it: the T-maze and the gridworld are both
one call away. Blob names are fixed (d/A/Ac/c/e/Bs) so runtimes can swap
parameters per tick via the params dict without recompiling.
"""

from __future__ import annotations

import numpy as np

from cffg.core.ir import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)

# Which blobs each declarable learnable owns (cff-x0sv). A and B appear twice
# because the same parameter is read in two forms: probabilities by the plain
# message rules, raw Dirichlet counts by the GFE composites that differentiate
# them for parameter novelty.
LEARNABLE_BLOBS = {
    "d": ("d",),  # initial-state prior
    "A": ("A", "Ac"),  # observation model
    "B": ("Bs", "Bc"),  # transition model, per control
    "e": ("e",),  # habits: the control prior
}


def build_rollout(
    name: str,
    *,
    steps: int,
    past: int = 0,
    d: np.ndarray,
    A: np.ndarray,
    c: np.ndarray,
    e: np.ndarray,
    Bs: np.ndarray,
    gfe: bool = True,
    map_policy: bool = False,
    learned_a: bool = False,
    epistemic_b: bool = False,
    learn: tuple[str, ...] = (),
) -> tuple[Template, dict[str, np.ndarray]]:
    n_state, n_obs, n_ctrl = len(d), A.shape[0], len(e)
    assert Bs.shape == (n_ctrl, n_state, n_state), f"Bs shape {Bs.shape}"
    assert A.shape == (n_obs, n_state) and c.shape == (n_obs,)

    t = Template(name=name)
    t.blobs = {
        blob: BlobRef(blob, f"{blob}.npy")
        for blob in ("d", "A", "Ac", "c", "e", "Bs", "Bc")
    }
    t.variables["z0"] = VariableSpec("z0", n_state)
    t.factors["prior"] = FactorSpec(
        "prior", "categorical", edges={"out": "z0"}, param_blobs={"statistic": "d"}
    )
    prev = "z0"
    for k in range(1, steps + 1):
        u, zm, zo, zt, x = f"u{k}", f"zm{k}", f"zo{k}", f"zt{k}", f"x{k}"
        t.variables |= {
            u: VariableSpec(u, n_ctrl),
            zm: VariableSpec(zm, n_state),
            zo: VariableSpec(zo, n_state),
            zt: VariableSpec(zt, n_state),
            x: VariableSpec(x, n_obs),
        }
        t.factors |= {
            f"tm{k}": FactorSpec(
                f"tm{k}", "transition_mixture",
                edges={"out": zm, "in": prev, "selector": u},
                # Curiosity-enabled planning slices read Dirichlet counts
                # ("Bc", the dynamics posterior); perception slices and plain
                # models read probabilities ("Bs").
                param_blobs={"matrices": "Bc" if (epistemic_b and k > past) else "Bs"},
            ),
            f"eq{k}": FactorSpec(f"eq{k}", "equality", edges={"a": zm, "b": zo, "c": zt}),
            f"obs{k}": FactorSpec(
                f"obs{k}", "matrix_cat", edges={"out": x, "in": zo},
                # Learned-A planning slices read Dirichlet counts ("Ac") —
                # but only under GFE, which is what differentiates them for
                # parameter novelty. Perception slices, fixed-A models, and
                # models learning A without an epistemic channel read
                # probabilities ("A"), fed E[log A] by the learnables API.
                param_blobs={"matrix": "Ac" if (learned_a and gfe and k > past) else "A"},
            ),
        }
        if k <= past:  # perception slice: executed control + observed outcome
            t.ports[f"act{k}"] = PortSpec(f"act{k}", u, PortRole.DATA)
            t.ports[f"obs{k}"] = PortSpec(f"obs{k}", x, PortRole.DATA)
        else:  # planning slice: control prior + goal-constrained observation
            t.factors[f"e{k}"] = FactorSpec(
                f"e{k}", "categorical", edges={"out": u}, param_blobs={"statistic": "e"}
            )
            t.factors[f"goal{k}"] = FactorSpec(
                f"goal{k}", "categorical", edges={"out": x}, param_blobs={"statistic": "c"}
            )
            if gfe:
                detail = [("substituted_role", "out")]
                if learned_a:
                    detail.append(("parameter_uncertainty", "dirichlet"))
                t.constraints.append(
                    ConstraintSpec(ConstraintKind.P_SUBSTITUTION, f"obs{k}", tuple(sorted(detail)))
                )
            if epistemic_b:
                # Dynamics curiosity (cff-fk2u): P-substituted transition with
                # Dirichlet parameter uncertainty — the GFE-TM composite.
                t.constraints.append(
                    ConstraintSpec(
                        ConstraintKind.P_SUBSTITUTION,
                        f"tm{k}",
                        tuple(sorted([
                            ("substituted_role", "out"),
                            ("parameter_uncertainty", "dirichlet"),
                        ])),
                    )
                )
            if map_policy:
                t.constraints.append(ConstraintSpec(ConstraintKind.DELTA, u))
        prev = zt
    t.ports["horizon"] = PortSpec("horizon", prev, PortRole.GLUE)  # open half-edge

    # Declared learnables (cff-x0sv). Only what this window actually reads:
    # the perception window has no control priors, the planning window has no
    # perception slices, and a learnable with no reader is a compile error.
    read = {b for f in t.factors.values() for b in f.param_blobs.values()}
    for lname in learn:
        if lname not in LEARNABLE_BLOBS:
            raise ValueError(f"build_rollout: unknown learnable {lname!r}")
        blobs = tuple(b for b in LEARNABLE_BLOBS[lname] if b in read)
        if not blobs:
            continue
        # Habits accumulate the *executed* control, not the whole horizon's
        # policy: every planning slice reads e (the prior applies at every
        # step), one slice supplies the statistic.
        sources = ("e1",) if lname == "e" and "e1" in t.factors else ()
        t.learnables[lname] = LearnableSpec(lname, "dirichlet", blobs, sources)

    arrays = {
        "d": d,
        "A": A,
        "Ac": np.ones_like(A),  # placeholder counts; swapped at runtime
        "c": c,
        "e": e,
        "Bs": Bs,
        "Bc": np.ones_like(Bs),  # placeholder counts; swapped at runtime
    }
    return t, arrays
