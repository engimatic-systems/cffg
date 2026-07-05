---
id: cff-kv7p
status: closed
deps: [cff-5wqw]
links: []
created: 2026-07-05T22:44:11Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-rlr2
---
# VMP mean-field bead support (deferred tail of M2)

MEAN_FIELD bead switches factor to VMP geometric messages (incoming = edge marginal, not opposite message); node-belief formula changes to product-of-marginals; test: single-latent conjugate model where MF is exact. Feeds M3 (GFE nodes reduce to VMP for data-constrained observations, Part II Fig.3).


## Notes

**2026-07-05T23:07:38Z**

Landed: MEAN_FIELD bead -> vmp groups gathering both slots (edge marginals), geometric updates + energies for categorical/matrix_cat/TM; equality bead rejected at compile. Tests: conjugate exactness (marginals + BFE=-log evidence), TM clamped-parents BP parity, two-free-edges BP divergence (bead provably not a no-op). M2 epic can close.
