---
id: cff-2gev
status: open
deps: [cff-hp3y, cff-4oq8, cff-5ipi]
links: []
created: 2026-07-10T03:34:54Z
type: feature
priority: 3
assignee: coopwilliams
parent: cff-mwjo
---
# GFE-TM novelty weight: anneal with accumulated count mass

## Plain English

This ticket gives transition curiosity a lifecycle. Unfamiliar transition
parameters receive a stronger novelty incentive, accumulated evidence turns
that incentive down, and a familiar model returns to the cheaper ordinary
planning path. The preregistered experiment measures both task performance
and whether the agent actually learns a better transition model.

Corrected post-cff-hp3y Hallway rerun (L=8, H=10, 120 sweeps, seed 0,
25 episodes): with near-uniform Dirichlet counts the dynamics-novelty channel
dominates the goal channel — task success is 0.16 versus 0.76 for the learn-only
ablation, and visitation-weighted transition KL is also worse (2.650 versus
2.534). The earlier pre-fix 0.40/0.80 success comparison and claimed marginal
KL benefit are invalid and must not be cited. That flat model also has no
parameter tying for directed exploration to exploit and its aliased sensor
smears learned statistics across states, so it is negative evidence about that
model, not a protocol for this ticket.

After cff-b47g supplies a factored two-chain rollout and cff-5ipi records a
replacement protocol before any run, add the protocol's declared-learnable
novelty weights and the option to drop to the plain composite when its
reachable/active parameter blocks saturate. The GFE-TM composite currently
costs about 4x a plain GFE act() whether novelty remains. The controlled arm
must recover goal performance without claiming useful exploration unless it
also improves the preregistered independent model-error metric. The weight is
a Q-side constraint knob; document its modified-GFE interpretation and keep
the optimistic-bound limitation in S1c.

## Acceptance Criteria

The implementation follows the live cff-5ipi protocol without changing its
weights, masks, arms, windows, thresholds, or outcome language after a run
starts. Lambda=1 reproduces the current learned GFE-TM rule and lambda=0
reproduces its fixed-parameter objective to tolerance. Weight/count routing is
by declared learnable and tied parameter block, never hard-coded node names.
Drop-back reaches plain-GFE cost when all protocol-declared active blocks
saturate; unreachable columns cannot prevent it. Exact-zero counts/support,
vacuous initialization, float32, uncomfortable parameter scale, finite FE and
truthful RunStatus are tested. Report the preregistered task, model-error and
cost outcomes without a useful-exploration claim unless its primary test
passes; update S1c either way.

## Notes

**2026-07-28T04:21:51Z**

2026-07-27 clean-slate resequencing: PR #35 was closed unmerged and no replacement run occurred. The flat Hallway numbers remain negative evidence only. Current path is cff-x0sv -> cff-nhxf -> cff-b47g -> cff-5ipi -> cff-2gev. cff-5ipi must install one live protocol after the builder exists and before any result is observed; do not recover the void PR #35 protocol from git history.
