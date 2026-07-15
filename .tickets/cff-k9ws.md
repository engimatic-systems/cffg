---
id: cff-k9ws
status: open
deps: []
links: [cff-o7hs, cff-pksr, cff-lquu]
created: 2026-07-15T01:45:43Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [hybrid, factor-family, rslds]
---
# Recurrence-link factor: stick-breaking softmax transition (continuous drives discrete)

The missing rSLDS ingredient: a factor p(z' | z, x) where a Gaussian edge modulates a discrete transition via a stick-breaking logistic (softmax) link — continuous state steering regime switches. family=hybrid, edges (out: discrete z', in: discrete z, drive: gaussian x), params (weights R [K-1, d], offsets r [K-1], per stick-breaking coordinate). Route: dedicated derivation with a provable variational bound — Pólya-Gamma augmentation (conditionally conjugate: messages to the drive edge are exactly Gaussian given E[omega], with closed-form tanh updates) or equivalently the Jaakkola-Jordan logistic bound. NOT via the generic nonlinear machinery (cff-o7hs) — see the scope note there and the cff-fxmj audit directive: declared subclasses with bounds, not universal approximate machinery. Precedent to copy: the mixture-collapse landing (cff-pksr) — exact-for-held-beliefs where possible, one clearly identified approximated term with a PINNED bound direction, tests against exact enumeration. Hostile-inputs checklist due at planning time per the numerical robustness contract: sparse/zero selector support, vacuous/improper drive messages, float32, 10-100x K and d. Implementation branch chains after cff-44we-shape-segmented-arenas (PR #27) until it merges; State arrays are 1-D — use Arena.*_indices.

## Acceptance Criteria

1. DERIVATIONS.md entry: stick-breaking link, PG/JJ bound, all three messages (to z', to z, to drive x), node energy; bound direction proved and stated. 2. Messages registered as a hybrid factor family; BFE ledger term labeled, bound direction pinned by a test (as test_mixg does for collapse). 3. Posterior parity vs dense-grid/MC ground truth on a small model within documented tolerance. 4. Hostile checklist answered before implementation and converted to tests: sparse support, vacuous messages, float32 sweep finite, 10-100x domain scale. 5. Coordinates with, does not wait for, cff-ri8m.

