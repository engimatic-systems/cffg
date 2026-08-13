---
id: cff-4apw
status: closed
deps: [cff-k9ws, cff-lquu]
links: []
created: 2026-07-15T01:45:57Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [hybrid, learning, rslds]
---
# Recurrence-weight conjugate learning via Pólya-Gamma augmentation

## Plain English

Recurrence links will learn how continuous state changes regime-switching
probabilities without hard-assigning either regime. The engine will turn its
existing soft recurrence belief into Pólya–Gamma linear and quadratic
statistics, and a Gaussian posterior will accumulate them between trials.
Rolling the posterior mean forward supplies the next trial's stick weights and
offsets on the direct path to the rSLDS integration tier.

## Outcome and semantics

Add `Engine.recurrence_weight_stats(state)`, keyed by recurrence factor name.
It consumes the `cff-lquu` belief boundary and current factor parameters and
returns:

- `omega[K,K-1]`: the coordinate-optimal PG mean per at-risk trial;
- `exposure[K,K-1]`: soft source responsibility times destination at-risk
  mass;
- `linear[K,K-1,d+1]`: the Gaussian natural-vector increment for augmented
  feature `[x;1]`;
- `quadratic[K,K-1,d+1,d+1]`: the Gaussian precision increment; and
- `proper`: finite, normalized beliefs and valid drive moments produced a
  usable record.

The statistics use `q(in) q(out) q(x)`, never an argmax regime. `omega` is
evaluated at the point weights/offsets used by the converged inference state,
matching the repository's between-trial empirical-Bayes rollover convention.
The readout is derived and must not alter state, support, status, or FE.

Add `conj.PGWeights`, a product of Gaussian posteriors over each augmented
stick coefficient `[R_il; r_il]`. `update(linear, quadratic)` adds natural
statistics, `rollover()` returns point weights and offsets for the next trial,
and `vague_pg_weights(K,d)` constructs a proper weak prior. This is a direct
between-trial helper like NIW/MNIW, not registration in declared
`LearnableSpec`; generic non-discrete `member_of` remains cff-6tek.

## Change-path handoff

Applicable row: **Learnable or checkpoint feature**, restricted to the
standalone conjugate helper and recurrence readout.

- Authority: DERIVATIONS D8 plus the new PG posterior subsection;
  `src/cffg/recurrence.py` owns the vectorized statistic algebra and
  `src/cffg/conj.py` owns between-trial natural-parameter accumulation.
- Derived surfaces: host assembly in `Engine.recurrence_weight_stats`, README
  cadence example, ARCHITECTURE's recurrent learning boundary, and compact
  recurrence/conjugacy tests.
- N/A surfaces: IR/serialization/hash, validation, plates, compiler grouping
  and parameter packing, `Arena.member_of`, `Learnables`, agent cadence,
  checkpoint migration, bundles, synchronous/scheduled message updates,
  ledger, estimator, and visualization. Those are either unchanged derived
  reads or explicitly deferred to cff-6tek/cff-y4nx.
- Duplicated pairs: JAX grouped/`vmap` statistics and the host per-factor map
  must agree. Flood/mixed, host/JIT solver paths, colony batching, and
  current/legacy checkpoints are N/A because no inference or persisted state
  surface changes.
- Canonical tests: extend `tests/test_recurrence_link.py` and
  `tests/test_conj.py`; add no new test file and no long per-trial engine loop.

## Hostile inputs

- **Sparse/zero support:** an exact-zero source responsibility produces exact
  zero blocks for all of that source's sticks. An output zero contributes no
  success or at-risk mass of its own; a stick with zero total at-risk mass has
  exact zero exposure and blocks. Empty support remains `proper=False`; no
  hard-assignment fallback exists.
- **Improper/vacuous inputs:** non-finite or unnormalized probabilities,
  non-symmetric/indefinite drive covariance, non-finite PG statistics, or a
  non-PD posterior precision fail loudly. A zero-exposure observation is a
  valid no-op on a proper prior.
- **Float32:** the JAX statistic helper uses the existing dtype-scaled PG
  zero-limit and second-moment tolerance. It introduces no probability floor,
  clamp, or ridge; the NumPy posterior follows `cffg.conj` by accumulating in
  float64.
- **Uncomfortable scale:** exercise `K=32,d=16`. Storage is
  `O(K^2 d^2)` for the declared independent-stick Gaussian posteriors and no
  source/destination/stick cubic tensor is allocated.
- **Large logits:** the hyperbolic-tangent PG mean remains finite as `xi`
  grows, with its analytic dtype-scaled series at zero.

## Acceptance Criteria

1. DERIVATIONS documents the augmented-feature PG statistics, Gaussian
   posterior, point rollover, and empirical-Bayes boundary.
2. The engine readout matches a compact independent NumPy reconstruction and
   uses fractional source/destination responsibilities without argmax.
3. A deterministic vectorized synthetic switching fixture recovers known
   scalar recurrence weights/offsets with documented tolerance using a small
   number of batch coordinate iterations; no per-trial engine loop is added.
4. `PGWeights` validates shape/propriety, accumulates natural statistics,
   rolls over correctly, and rejects broken posterior inputs without ridging.
5. Exact-zero, empty-support, invalid-drive, float32, large-logit, and
   `K=32,d=16` outcomes match the hostile contract. Existing recurrence
   message/FE tests remain the authority for inference behavior.
6. README and ARCHITECTURE document the between-trial API; LIMITATIONS states
   the point-rollover approximation. The new coverage stays in the two
   existing focused test files.

## Dependency map

Legend: prerequisite --> dependent

    [x] cff-k9ws recurrence factor
              |
              +--> [x] cff-lquu hybrid belief readouts
                          |
                          +--> [ ] cff-4apw PG weight learning (ready)
                                      |
                                      +--> [ ] cff-y4nx rSLDS integration

## Notes

**2026-08-01T03:52:47Z**

Implemented on agent/cff-4apw-pg-learning in commit 28a7e8a. Engine.recurrence_weight_stats emits soft PG omega/exposure and additive Gaussian natural statistics; conj.PGWeights validates, accumulates, reports covariance, and rolls point weights/offsets forward. D8.4 plus README/ARCHITECTURE/LIMITATIONS document the empirical-Bayes boundary. Validation stayed compact: exactly three new tests across the existing conjugacy and recurrence files; focused 20 passed, and final default suite 1283 passed / 38 deselected in 359.21s with the dashboard refreshed. cff-y4nx is now unblocked; generic LearnableSpec registration remains deliberately with cff-6tek.
