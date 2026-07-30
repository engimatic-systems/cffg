---
id: cff-l6i5
status: in_progress
deps: [cff-8s6a, cff-ri8m]
links: [cff-1o72, cff-fxmj, cff-o7hs, cff-ri8m, cff-3hgi, cff-6tek, cff-209l]
created: 2026-07-09T17:16:15Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Zero-regression refactor: Cat/Dir/Gaussian message families onto the EF natural-parameter abstraction

## Plain English

Categorical, Dirichlet, and Gaussian messages will share the exponential-family natural-parameter algebra already used by newer families. Their dedicated arenas and public behavior stay unchanged, so this consolidates internals without changing models, hashes, or numerical results.

Re-express the discrete (categorical/Dirichlet log-space) and Gaussian (information-form) message families as instances of the EF contract (DERIVATIONS.md D2, cffg/ef.py + cffg/efrules.py): products are natural-parameter addition, moments are grad-A, energies/entropies are A-algebra. Pure refactor of internals — the arenas, rule signatures, and every existing test stay byte-identical in behavior.

## Change-path handoff

Applicable change-path index row: **New factor or rule family**, applied as a refactor of existing families.

- Authority: `src/cffg/ef.py` and `src/cffg/efrules.py` own reusable host/JAX natural-parameter algebra; `rules.py`, `gauss.py`, `learn.py`, and `conj.py` remain family adapters.
- Compiler grouping, slot addressing, parameter packing, and `Arena`: checked, behavior-preserving only; no schema or layout changes.
- Synchronous `_make_step` and scheduled `_batch_stage_fn` / `_scan_stage_fn`: checked pair; both must consume the same unchanged rule signatures.
- `_make_bfe`: affected through shared family energy/entropy algebra; HMM, Kalman, and T-maze values must remain bit-comparable.
- Schedule classification, readouts, serialization manifest, estimator, and public exports: checked N/A unless an internal import is needed; structure hashes must remain unchanged.
- Duplicated surfaces: host/JIT affected; flooding/scheduled and single-instance/`vmap` checked through their shared rule paths; current/legacy checkpoint decoding N/A.

## Hostile-input plan

- Sparse/zero support: preserve exact categorical log-zeros and the reserved support sentinel without flooring.
- Improper or vacuous incoming messages: preserve valid vacuous additive identities; improper Dirichlet/Gaussian ledger states remain loud and non-NaN.
- float32: preserve dtype-scaled guards and replace `conj.py` absolute symmetry tolerance with the existing matrix-scale/dtype-relative Gaussian check.
- Uncomfortable scale: algebra remains vectorized with no iteration budgets or size-dependent constants; existing large-domain hostile tests remain unchanged.

## Acceptance Criteria

Full test suite passes unchanged (zero test-file edits except imports if a module moves); structure hashes unchanged; no new dependencies; BFE values bit-comparable on the HMM/Kalman/T-maze goldens.


## Notes

**2026-07-13T03:03:27Z**

Follow-up from cff-3hgi judging: conj.py _spd still uses an absolute symmetry tolerance (np.allclose atol=1e-10) in the learning-side SPD assert. It fails loudly (raises) rather than silently accepting, but it is the same absolute-tolerance family cff-3hgi removed from gauss.py/mixg.py and can spuriously reject large-unit accumulated statistics. When migrating the Gaussian family onto the EF natural-parameter abstraction, replace it with the matrix-scale/dtype-relative symmetry check (gauss._symmetry_ok pattern).
