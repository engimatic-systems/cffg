"""The classic T-maze (Friston et al. 2015), Part I §7 / Part II §6 layout.

Positions P = (O, L, R, C): start, left arm, right arm, cue. Hidden reward
context R = (RL, RR). State = position ⊗ context (position-major, 8). An
observation is position ⊗ outcome with outcomes (CL, CR, RW, NR) (16).
Controls select one of four transition matrices B_u = P_u ⊗ I₂; invalid moves
bounce to O, reward arms are followed by a mandatory move back.

`build_tmaze` produces the policy-inference model of Part I Fig. 26: state
prior d at the current time, then per future step a control prior, a
transition-mixture node, an equality branch, and a P-substituted
goal-observation composite. With gfe=False the P-beads are omitted — the
Part II §7.1 BFE-ablation agent that lacks the epistemic drive.
"""

from __future__ import annotations

import numpy as np

from cffg.core.ir import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)

O, L, R, C = 0, 1, 2, 3  # noqa: E741  (paper's position names)
N_POS, N_CTX, N_OUT = 4, 2, 4
N_STATE, N_OBS = N_POS * N_CTX, N_POS * N_OUT


def transition_stack() -> np.ndarray:
    """[4, 8, 8] stack of B_u = P_u ⊗ I₂ (Part II §6.2)."""
    p = np.zeros((4, N_POS, N_POS))
    p[O] = [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    p[L] = [[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    p[R] = [[0, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]]
    p[C] = [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
    return np.stack([np.kron(p[u], np.eye(N_CTX)) for u in range(4)])


def observation_matrix(alpha: float) -> np.ndarray:
    """[16, 8] block-diagonal likelihood (Part II Table 3), position-major."""
    blocks = {
        O: [[0.5, 0.5], [0.5, 0.5], [0, 0], [0, 0]],
        L: [[0, 0], [0, 0], [alpha, 1 - alpha], [1 - alpha, alpha]],
        R: [[0, 0], [0, 0], [1 - alpha, alpha], [alpha, 1 - alpha]],
        C: [[1, 0], [0, 1], [0, 0], [0, 0]],
    }
    a = np.zeros((N_OBS, N_STATE))
    for pos in range(N_POS):
        a[pos * N_OUT : (pos + 1) * N_OUT, pos * N_CTX : (pos + 1) * N_CTX] = blocks[pos]
    return a


def goal_statistic(c_util: float) -> np.ndarray:
    """softmax of per-outcome utilities (0, 0, +c, −c), tiled per position."""
    u = np.tile([0.0, 0.0, c_util, -c_util], N_POS)
    e = np.exp(u - u.max())
    return e / e.sum()


def initial_state() -> np.ndarray:
    d = np.zeros(N_STATE)
    d[O * N_CTX : O * N_CTX + N_CTX] = 0.5  # at O, context unknown
    return d


def build_tmaze(
    steps: int = 2,
    alpha: float = 0.9,
    c_util: float = 2.0,
    *,
    past: int = 0,
    gfe: bool = True,
    map_policy: bool = False,
    learned_a: bool = False,
) -> tuple[Template, dict[str, np.ndarray]]:
    """Policy-inference CFFG over `steps` timesteps, plus its arrays.

    The first `past` slices are perception-style: the control edge becomes a
    DATA port (`act{k}`, the executed move) and the observation branch is a
    plain likelihood with a DATA port (`obs{k}`) — no control prior, no goal,
    no P-substitution. Remaining slices are future/planning-style as before.
    This is the receding-horizon constraint flip of the agent runtime: as time
    advances, a slice's beads change from goal to data (same wiring)."""
    t = Template(name=f"tmaze[{past}|{steps}]")
    t.blobs = {
        "d": BlobRef("d", "d.npy"),
        "A": BlobRef("A", "A.npy"),
        "Ac": BlobRef("Ac", "Ac.npy"),
        "c": BlobRef("c", "c.npy"),
        "e": BlobRef("e", "e.npy"),
        "Bs": BlobRef("Bs", "Bs.npy"),
    }
    t.variables["z0"] = VariableSpec("z0", N_STATE)
    t.factors["prior"] = FactorSpec(
        "prior", "categorical", edges={"out": "z0"}, param_blobs={"statistic": "d"}
    )
    prev = "z0"
    for k in range(1, steps + 1):
        u, zm, zo, zt, x = f"u{k}", f"zm{k}", f"zo{k}", f"zt{k}", f"x{k}"
        t.variables |= {
            u: VariableSpec(u, N_POS),
            zm: VariableSpec(zm, N_STATE),
            zo: VariableSpec(zo, N_STATE),
            zt: VariableSpec(zt, N_STATE),
            x: VariableSpec(x, N_OBS),
        }
        t.factors |= {
            f"tm{k}": FactorSpec(
                f"tm{k}", "transition_mixture",
                edges={"out": zm, "in": prev, "selector": u},
                param_blobs={"matrices": "Bs"},
            ),
            f"eq{k}": FactorSpec(f"eq{k}", "equality", edges={"a": zm, "b": zo, "c": zt}),
            f"obs{k}": FactorSpec(
                f"obs{k}", "matrix_cat", edges={"out": x, "in": zo},
                # Learned-A planning slices read Dirichlet counts ("Ac");
                # perception slices and fixed-A models read probabilities ("A").
                param_blobs={"matrix": "Ac" if (learned_a and k > past) else "A"},
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
                    ConstraintSpec(
                        ConstraintKind.P_SUBSTITUTION, f"obs{k}", tuple(sorted(detail))
                    )
                )
            if map_policy:
                t.constraints.append(ConstraintSpec(ConstraintKind.DELTA, u))
        prev = zt
    t.ports["horizon"] = PortSpec("horizon", prev, PortRole.GLUE)  # open half-edge

    arrays = {
        "d": initial_state(),
        "A": observation_matrix(alpha),
        "Ac": np.ones((N_OBS, N_STATE)),  # placeholder counts; swapped per trial
        "c": goal_statistic(c_util),
        "e": np.full(N_POS, 0.25),
        "Bs": transition_stack(),
    }
    return t, arrays
