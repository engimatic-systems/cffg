---
id: cff-9n7o
status: closed
deps: []
links: []
created: 2026-07-05T22:12:45Z
type: task
priority: 2
assignee: coopwilliams
---
# Research: reusable FE functions in newest PyMDP (JAX)

Can we import free-energy / message primitives from pymdp 1.x (JAX rewrite by erstwhile VERSES team)? Findings land here as comment per sub-agent protocol.


## Notes

**2026-07-05T22:19:08Z**

SUB-AGENT FINDINGS (confidence high, sources: PyPI JSON API, repo cloned at v1.0.3, live install):
Q: Can we import FE functions from newest pymdp?
A: Partially - as vendored snippets, NOT as a dependency.
- pymdp 1.0.3 (2026-06-15) IS the JAX rewrite, on PyPI as inferactively-pymdp; MIT; active (4 releases in 4 months); deps incl equinox but ALSO matplotlib+seaborn+mctx+networkx as HARD deps.
- Pure functions importable without Agent: pymdp.maths.calc_vfe (has return_decomposition=True), compute_accuracy, dirichlet_kl_divergence, dirichlet_expected_value, spm_wnorm(exact_param_info_gain=True), log_stable/stable_xlogx/stable_entropy/stable_cross_entropy, factor_dot; pymdp.control.compute_info_gain / compute_expected_utility / compute_neg_efe_policy; pymdp.learning.update_*_dirichlet.
- CRITICAL: zero Bethe/Forney/GFE/CFFG content. Fixed-form POMDP (A/B/C/D) only; 'message passing' = chain-HMM VMP/MMP over time. No node-local BFE, no edge-entropy corrections, no P-substitution. Grep for bethe/forney/GFE/overcount: 0 hits. Our core machinery must be built independently (RxInfer.jl lineage is the prior art).
- No Dirichlet expected-log helper E[ln theta]=digamma(a)-digamma(a0) exists there; we write our own.
- Red flags: young post-rewrite API, announced sign-convention churn (spm_wnorm minus), eps-clipping numerics that can inflate info gain on zeroed Dirichlet entries.
DECISION: read-only reference + selective vendoring (<50 lines each, MIT, with attribution) into cffg/vendor/; calc_vfe decomposition + VMP/MMP sequence code = cross-validation ORACLES for our M2/M3 FE tests. Recorded in ARCHITECTURE.md dependency policy.
