---
id: cff-l6i5
status: open
deps: [cff-8s6a, cff-ri8m]
links: [cff-1o72, cff-fxmj, cff-o7hs, cff-ri8m, cff-3hgi, cff-6tek]
created: 2026-07-09T17:16:15Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Zero-regression refactor: Cat/Dir/Gaussian message families onto the EF natural-parameter abstraction

Re-express the discrete (categorical/Dirichlet log-space) and Gaussian (information-form) message families as instances of the EF contract (DERIVATIONS.md D2, cffg/ef.py + cffg/efrules.py): products are natural-parameter addition, moments are grad-A, energies/entropies are A-algebra. Pure refactor of internals — the arenas, rule signatures, and every existing test stay byte-identical in behavior.

## Acceptance Criteria

Full test suite passes unchanged (zero test-file edits except imports if a module moves); structure hashes unchanged; no new dependencies; BFE values bit-comparable on the HMM/Kalman/T-maze goldens.


## Notes

**2026-07-13T03:03:27Z**

Follow-up from cff-3hgi judging: conj.py _spd still uses an absolute symmetry tolerance (np.allclose atol=1e-10) in the learning-side SPD assert. It fails loudly (raises) rather than silently accepting, but it is the same absolute-tolerance family cff-3hgi removed from gauss.py/mixg.py and can spuriously reject large-unit accumulated statistics. When migrating the Gaussian family onto the EF natural-parameter abstraction, replace it with the matrix-scale/dtype-relative symmetry check (gauss._symmetry_ok pattern).
