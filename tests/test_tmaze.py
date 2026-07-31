"""The T-maze: the canonical epistemic benchmark (Part I §7).

The hallmark result: with everything symmetric between the arms, a GFE agent's
first move is to the cue — information has value — while the one-bead BFE
ablation goes straight for an arm. Model symmetry under context-swap ⊗
arm-swap makes exact L/R equality at the converged fixed point a hard
invariant, which is a stronger check than matching the paper's two-sweep
schedule numbers (golden-parity ticket: cff-fdcl).
"""

import numpy as np

from cffg.compiler import compile_template
from cffg.engine import Engine
from cffg.models.tmaze import C, L, O, R, build_tmaze

N_ITERS = 400


def solve(**kwargs):
    template, arrays = build_tmaze(**kwargs)
    engine = Engine(compile_template(template, arrays=arrays))
    state, deltas = engine.run(
        engine.init_state(), N_ITERS, tol=1e-13
    ).require_converged_with_trace()
    assert deltas[-1] < 1e-10, f"not converged: {deltas[-6:]}"
    return engine, state


def test_gfe_agent_seeks_the_cue_first():
    engine, state = solve()
    q = engine.marginals(state)
    u1, u2 = q["u1"], q["u2"]

    assert int(np.argmax(u1)) == C, f"step-1 policy not epistemic: {u1}"
    # Step 2: exploit an arm; L/R exactly symmetric by model symmetry.
    np.testing.assert_allclose(u2[L], u2[R], atol=1e-9)
    assert min(u2[L], u2[R]) > max(u2[O], u2[C]), f"step-2 not arm-seeking: {u2}"


def test_context_marginal_stays_symmetric():
    engine, state = solve()
    q = engine.marginals(state)
    for name in ("z0", "zm1", "zm2"):
        ctx = np.asarray(q[name]).reshape(4, 2).sum(axis=0)
        np.testing.assert_allclose(ctx, [0.5, 0.5], atol=1e-9)


def test_bfe_ablation_skips_the_cue():
    """Remove the P-beads (one structural bit) and the epistemic drive dies:
    the KL-control agent heads straight for a reward arm."""
    engine, state = solve(gfe=False)
    u1 = engine.marginals(state)["u1"]
    assert int(np.argmax(u1)) in (L, R), f"BFE agent should be greedy: {u1}"

    engine_g, state_g = solve()
    u1_g = engine_g.marginals(state_g)["u1"]
    assert u1_g[C] > u1[C], "GFE must value the cue strictly more than BFE"


def test_map_policy_variant():
    """Fig. 29/30: δ-beads on the controls make the readout a decision.
    Two-phase run: soft to coherence, then harden (a premature pointmass
    would lock in its own arbitrary argmax)."""
    template, arrays = build_tmaze(map_policy=True)
    engine = Engine(compile_template(template, arrays=arrays))
    state, deltas = engine.run(
        engine.init_state(), 900, tol=1e-13, harden_last=450
    ).require_converged_with_trace()
    assert deltas[-1] < 1e-10, f"not converged: {deltas[-4:]}"
    q = engine.marginals(state)
    np.testing.assert_allclose(sorted(q["u1"])[-1], 1.0, atol=1e-12)
    assert int(np.argmax(q["u1"])) == C
    np.testing.assert_allclose(sorted(q["u2"])[-1], 1.0, atol=1e-12)
    assert int(np.argmax(q["u2"])) in (L, R)


def test_gfe_is_finite_and_stable():
    engine, state = solve()
    g1 = engine.bfe(state)
    state2 = engine.run(state, 5).state
    g2 = engine.bfe(state2)
    assert np.isfinite(g1) and abs(g1 - g2) < 1e-9


def test_longer_horizon_compiles_and_stays_epistemic():
    engine, state = solve(steps=3)
    u1 = engine.marginals(state)["u1"]
    assert int(np.argmax(u1)) == C


# Converged posteriors from the authors' Julia reference (biaslab/LAIF @
# 9453782, RxInfer 3.6.0, Part1/Policy_Inference.jl run deterministically,
# 10 VMP iterations, 20 Newton steps; extraction: ticket cff-fdcl). These are
# the values Fig. 28 rounds to. Tolerance covers the reference's numeric
# quirks we deliberately do not copy (A .+= 1e-12, log(d .+ 1e-6),
# clamped softmax) — agreement is ~1e-4 without them.
LAIF_U1 = [0.2500982317513749, 0.19810472849235258, 0.19810472849235264, 0.3536923112639199]
LAIF_U2 = [0.13187528818577465, 0.30062770805584765, 0.3006277080558477, 0.26686929570253]


def test_golden_parity_with_laif_reference():
    engine, state = solve()
    q = engine.marginals(state)
    np.testing.assert_allclose(q["u1"], LAIF_U1, atol=5e-4)
    np.testing.assert_allclose(q["u2"], LAIF_U2, atol=5e-4)
