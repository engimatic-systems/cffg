"""Lower one square, discrete HMM scan stage to its one-register recurrence."""

import jax
import jax.numpy as jnp
import numpy as np

from cffg import efrules
from cffg.compiled import NEG
from cffg.execution.failures import FailureIndex
from cffg.failure_facts import FailureFact, FailureReason, nonfinite_rows
from cffg.kernels import BUILTIN_MATRIX_CAT
from cffg.rules import BP_UPDATES, bp_equality, bp_matrix_cat, matrix_to_in, matrix_to_out


def _column(plan, column):
    if plan[0] == "table":
        return None
    if plan[0] == "direct":
        _, element, role, delay = plan
        return element, role, column, delay
    item = plan[1][column]
    return item[1:] if item[0] == "r" else None


def make_hmm_scan(arena, stage, plans):
    """Recognize the actual data ports, phase order and every register dependency."""
    if (arena.semiring != "sum" or len(stage.elems) != 2
            or any(e.kind != "disc" for e in stage.elems)
            or BP_UPDATES["matrix_cat"] is not bp_matrix_cat
            or BP_UPDATES["equality"] is not bp_equality):
        return None
    groups = [arena.groups[e.group] for e in stage.elems]
    if sorted(g.type for g in groups) != ["equality", "matrix_cat"]:
        return None
    m = next(i for i, g in enumerate(groups) if g.type == "matrix_cat")
    q = 1 - m
    matrix, equality = groups[m], groups[q]
    mr, er = stage.elems[m].rows, stage.elems[q].rows
    k = matrix.cards[0]
    if (matrix.kernel is not BUILTIN_MATRIX_CAT or equality.kernel is not None
            or matrix.mode != "bp" or equality.mode != "bp"
            or matrix.roles != ("out", "in") or equality.roles != ("a", "b", "c")
            or matrix.cards != (k, k) or equality.cards != (k, k, k)
            or mr.shape != (stage.length, 2) or er.shape != (stage.length, 1)):
        return None
    data = set(arena.data_slots.values())
    observed = [c for c in range(2) if all(slot in data for slot in matrix.gather[mr[:, c], 0])]
    if len(observed) != 1:
        return None
    o = observed[0]
    t = 1 - o
    deps = [_column(plans[m][r], c) for c in (o, t) for r in range(2)]
    deps.extend(_column(p, 0) for p in plans[q])
    if m == 0:
        expected = [None, None, None, (q, 1, 0, 1),
                    (m, 0, t, 0), None, (m, 1, o, 0)]
    else:
        expected = [None, (q, 2, 0, 0), (q, 0, 0, 0), None,
                    None, (m, 1, t, 1), None]
    if deps != expected:
        return None
    obs, trans, eq = mr[:, o], mr[:, t], er[:, 0]
    # Static reads can also come from a different branch. Prove the reverse
    # links as well as the registers used by the current traversal.
    ordered = slice(None) if m == 0 else slice(None, None, -1)
    link_eq, link_trans = eq[ordered][:-1], trans[ordered][1:]
    for a, i, ar, b, j, br in (
        (matrix, obs, 1, equality, eq, 2),
        (matrix, trans, 0, equality, eq, 0),
        (equality, link_eq, 1, matrix, link_trans, 1),
    ):
        if not (np.array_equal(a.gather[i, ar], b.scatter[j, br])
                and np.array_equal(a.scatter[i, ar], b.gather[j, br])):
            return None
    return _hmm_stage(arena, matrix, equality, obs, trans, eq, m == 0)


def _normal(log_values):
    return jnp.maximum(efrules.CATEGORICAL.normalize(log_values), NEG)


def _forward(log_matrix, incoming):
    return _normal(matrix_to_out(log_matrix, incoming))


def _backward(log_matrix, incoming):
    return _normal(matrix_to_in(log_matrix, incoming))


def _hmm_stage(arena, matrix, equality, obs, trans, eq, inward):
    # Blocks are always in emission out/in, transition out/in, equality a/b/c order.
    def slots(attribute):
        mats = getattr(matrix, attribute)
        eqs = getattr(equality, attribute)
        return arena.message_indices(
            np.concatenate((mats[obs], mats[trans], eqs[eq]), axis=1), matrix.cards[0]
        )

    gather, scatter = slots("gather"), slots("scatter")
    index = FailureIndex(arena)
    ranks = np.stack((index.ranks(matrix.members)[obs], index.ranks(matrix.members)[trans],
                      index.ranks(equality.members)[eq]), axis=1)
    param = dict(matrix.parameter_paths)["logf"]

    def propagate(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
        a, b = params[param][obs], params[param][trans]

        def read(role):
            return messages[gather[:, role]]

        oi = _backward(a, read(0))
        if inward:
            oo = _forward(a, read(1))
            ti = _backward(b, read(2))
            side = read(5)
            ea = _normal(side + oi)

            def step(previous, row):
                transition, evidence = row
                to = _forward(transition, previous)
                eb = _normal(to + evidence)
                return eb, (to, eb)

            _, (to, eb) = jax.lax.scan(step, messages[gather[0, 3]], (b, oi))
            ec = _normal(to + side)
        else:
            left, evidence = read(4), read(6)
            eb = _normal(left + evidence)
            to = _forward(b, read(3))

            def step(previous, row):
                transition, local = row
                ea = _normal(previous + local)
                ti = _backward(transition, ea)
                return ti, (previous, ea, ti)

            _, (previous, ea, ti) = jax.lax.scan(step, messages[gather[0, 5]], (b, evidence))
            ec = _normal(left + previous)
            oo = _forward(a, ec)

        rows = jnp.stack((oo, oi, to, ti, ea, eb, ec), axis=1).astype(messages.dtype)
        # Failure ordering follows step, then element, then lowest simultaneous factor rank.
        invalid = [nonfinite_rows(block, log_support=True)
                   for block in (rows[:, :2], rows[:, 2:4], rows[:, 4:])]
        absent = jnp.iinfo(jnp.int32).max
        failed = jnp.where(jnp.stack(invalid, axis=1), ranks, absent)
        mat, eq = jnp.min(failed[:, :2], axis=1), failed[:, 2]
        first, second = (mat, eq) if inward else (eq, mat)
        per_step = jnp.where(first != absent, first, second)
        owner = per_step[jnp.argmax(per_step != absent)]
        failure = FailureFact(jnp.where(owner == absent, -1, owner),
                              jnp.where(owner == absent, 0, FailureReason.NONFINITE_OUTPUT).astype(jnp.int32),
                              jnp.int32(0))
        return messages.at[scatter].set(rows), g_eta, g_lam, ef_eta, failure

    return propagate
