---
id: cff-i45j
status: in_progress
deps: []
links: [cff-cj5h]
created: 2026-07-08T22:00:50Z
type: bug
priority: 1
assignee: coopwilliams
---
# GFE planner diverges to NaN on large sparse-support models; run() reports the garbage as converged

Found 2026-07-08 stress-testing a receding-horizon GFE agent on a 576-state
POMDP (mostly-deterministic transitions, a distance-decaying noisy sensor,
horizon 4, damp 0.5). Symptom: the agent repeats one low-information sensing
action every tick, never acting pragmatically. The T-maze/gridworld tiers
never see this regime (small state spaces, a perfect cue).

DIAGNOSIS (top-level analysis of a sub-agent's instrumented runs):

1. ROOT CAUSE — GFE message overflow on sparse supports. The planner's
   messages saturate at the ±1e30 clamp and go NaN by sweep ~6. Mechanism:
   `gfe_update` computes mu2 = log(zbar) − log_d, and with deterministic
   transitions log_d contains exact NEG (−1e30) zeros, so mu2 hits +1e30 at
   impossible states; `_norm_pad`'s logsumexp is then dominated by the junk
   and destroys the whole message. The BFE ablation (gfe=False) converges
   cleanly in 55 sweeps — the composite is the sole source.
2. MASKING BUG — Engine.run's tol-break accepts non-finite deltas: the
   delta trace goes 5e29, 2.5e29, −inf and the −inf "passes" tol, returning
   a sweep-4 transient as if converged (220 and 1000 sweeps byte-identical).
   The read-off posterior is a near-flat transient (top 0.171 vs uniform
   0.111; two symmetric actions get different mass — a non-convergence
   fingerprint) and argmax picks noise.
3. TO RE-TEST AFTER THE FIX — epistemic-term audit: in the pre-NaN transient
   the composite pushed mass toward the LOWEST-MI sensing action (exact MI
   0.009 nats vs 0.069 for the better sensors — an inversion). If that
   ranking persists once messages are finite and converged, the P-sub term
   is mis-weighting ambiguity retention as value; that becomes its own
   derivation bug. Also secondary: with weak goal utilities the
   move-then-act plan cannot compete inside the horizon (utility-scale
   sweep flipped the argmax to movement) — a modeling-recipe note, plus the
   absorbing-mode/termination question for episodic tasks.

FIX PLAN: (a) engine.run: never treat a non-finite delta as convergence;
surface divergence loudly. (b) gfe_update: mask mu2 to NEG where log_d has
no support BEFORE normalization (value there is irrelevant downstream —
the receiving product re-zeroes it — but it must not poison logsumexp).
(c) minimal repro test: a GFE composite fed a log_d with exact zeros
currently NaNs; must converge finite after. (d) re-run the epistemic-term
audit on the fixed engine and update this ticket with the verdict.

DELEGATION: fixes (a)-(c) and the term audit are Fable-level (FE/message
semantics); parameter sweeps for the audit can go to a Sonnet sub-agent
once the engine is fixed.

## Acceptance Criteria

Non-finite deltas never satisfy tol (test); GFE messages stay finite on
sparse-support models (repro test converges); epistemic-term audit re-run
on the fixed engine with a written verdict in this ticket — either the MI
inversion disappears (close) or it becomes a derivation ticket (link it).
