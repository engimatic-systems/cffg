---
id: ctcp-6msb
status: closed
deps: []
links: []
created: 2026-11-28T04:50:36Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-2gev
---
# Derive and independently check the posterior for a changing transition column

A familiar transition can change after many observations. We need a posterior update that can reconsider the learned transition law when observations change. Flat discounting, inferring a change point and learning recurring latent regimes imply different behavior for an unvisited transition.

## Design

Derivations and independent oracles only. Read Part II and D1 for transition novelty. Compare the bounded moment projection to an exact growing mixture. Capture the semantically different fractional-count update and its limitations before implementing the runtime; do not attach timing or performance multipliers to free energy.

## Acceptance Criteria

Choose one generative mechanism and an explicit clock. Derive the update, the h=0 stationary limit, structural-zero and improper-input behavior, and what posterior moments the GFE-TM planner receives. Independently check small hard-count cases against complete reset-history enumeration. Quantify the approximation on a fixed prior-only example; make no claims on the comparative study.


## Notes

**2026-11-28T04:50:57Z**

Change paths for this read-only oracle: runtime numerical invariant and learnable/parameter semantics. Positive proper concentrations live on an explicitly fixed allowed face. Test h=0 and h=1, missing likelihood, one-category support, impossible outcomes, f32/f64, and larger cardinality. No JIT/executor or codec is added here. The runtime ticket separately owns both paired execution surfaces and legacy/current checkpoint admission.

**2026-11-28T04:53:53Z**

Clock choice before evaluation: one physically committed update of the named transition block. Each of its columns independently resets with probability h before generating this event. Even columns with empty likelihood undergo that declared predictive hazard. An unexecuted block does not advance, and planning is read-only. Known ties pool records from the same boundary once. This observable block clock uses no hidden source or change flag. Gauss independently checked the trace/covariance derivation, singleton face, exact h=0/1 limits and loss of expected categorical entropy under projection (D1 Propositions 2–3). Expose next-event predicted moments separately from the committed posterior.

**2026-11-28T05:49:23Z**

Independent final dashboard gate for 7886699256222391d11944a7a17b2867e271d01c: 2,843 passed, zero failed, four workers, 439.15 s pytest, 439.930 s recorded dashboard wall and 440.095 s command wall. The five focused derivation cases also passed. Exact receipt and command log: ~/agent-share/changing-transition-contract-m7Ti0v/{runtime.json,gate.log}. D1a establishes the fixed-sequence reference and bounded projection. The changing-column runtime (ctcp-w305) and preregistered study (ctcp-qodi) stay open under cff-2gev.
