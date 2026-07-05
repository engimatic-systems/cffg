---
id: cff-alvf
status: closed
deps: [cff-5wqw]
links: []
created: 2026-07-05T22:44:11Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-rlr2
---
# Exact BFE ledger + tree identity

Node beliefs per group, F = sum_a F[q_a] + sum_interior H[q_i]; where-guarded 0log0; on trees BFE == -log Z (forward likelihood). This is the 'true BFE' invariant.


## Notes

**2026-07-05T22:49:06Z**

BFE ledger: F = sum_a F[q_a] + sum_interior H[q_i], where-guarded 0log0, energies from rule registry. Verified: BFE == -log Z on tree CSP (1e-9), == -loglik on HMM chain, == 0 with no data (normalized model). NOTE: paper eq(8) OCR sign ambiguity resolved to standard Yedidia form (+ (d_i - 1) H); derivation checked by hand on prior+likelihood micro-model.
