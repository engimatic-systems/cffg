---
id: cff-lquu
status: in_progress
deps: [cff-3hgi]
links: [cff-o7hs, cff-pksr, cff-k9ws]
created: 2026-07-10T03:31:09Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Hybrid node readouts: soft sufficient statistics + posterior predictive

## Plain English

Hybrid factors will expose the soft regime responsibilities and continuous
moments that inference already computes instead of making learning code rebuild
them or discard uncertainty through hard assignments. Switching-model learners
will receive factor-local sufficient statistics, and consumers will receive a
one-step Gaussian-mixture predictive with an explicit moment-matched collapse.
The same readout conventions will give recurrence-weight learning a clean,
sound input boundary on the direct path to the rSLDS integration tier.

## Outcome and semantics

`gaussian_mixture` and `gaussian_mixture_linear` are currently skipped by
`Engine.gaussian_node_stats`, even though the D3 node-energy path already
constructs their per-regime joint Gaussian beliefs and responsibilities. That
omission forces switching-model learning to hard-assign regimes or duplicate
private arena algebra. The recurrence factor likewise has no factor-local
belief readout, so `cff-4apw` would otherwise stitch categorical and Gaussian
edge marginals together in consumer code.

Extend `Engine.gaussian_node_stats(state)` with records for all current hybrid
plans:

- `gaussian_mixture_linear`: `responsibility[K]` plus responsibility-weighted
  `Syy[K,dy,dy]`, `Syx[K,dy,dx+1]`, and
  `Sxx[K,dx+1,dx+1]`, computed from the actual D3 factor belief
  `q_s(y,x)` rather than a product of endpoint marginals.
- `gaussian_mixture`: `responsibility[K]` plus responsibility-weighted
  component first and second moments. Names must state that the component
  arrays are weighted; a consumer must not multiply by responsibility twice.
- `recurrence_link`: the coordinate belief boundary needed by `cff-4apw` —
  `out_probability[K]`, `in_probability[K]`, Gaussian drive mean and second
  moment, exact support, and `proper`. PG expectations and weight-learning
  sufficient-statistic algebra remain in `cff-4apw`, not here.

Every record retains the existing `proper` convention: a valid record contains
finite moments; an invalid factor belief contains shape-correct NaNs and
`proper=False`, never plausible zeros. The method is a readout of the converged
factor belief and must not modify messages, parameters, support, status, or FE.

Add `Engine.gaussian_mixture_predictives(state)` keyed by factor name. Each
record contains `weight[K]`, `component_mean[K,d]`,
`component_covariance[K,d,d]`, and collapsed `mean[d]`/`covariance[d,d]` plus
`proper`. It is the factor-to-`out` **cavity predictive**: it uses the incoming
selector and, for the linear factor, incoming Gaussian state belief, while
excluding the `out`-facing message so observed downstream evidence is not
counted twice. Its collapsed moments must equal the existing D3 forward
message before information-form conversion. This is distinct from a posterior
edge marginal and must be named and documented accordingly.

## Change-path handoff

Applicable row: **New factor or rule family**, restricted to the row's readout
surface because all factor types and inference rules already exist.

- Authority: `docs/DERIVATIONS.md` D3 and D8; `src/cffg/mixg.py` factor-belief
  and collapse algebra; `src/cffg/recurrence.py` coordinate beliefs; and the
  established `Engine.gaussian_node_stats` loudness/schema convention.
- Derived surfaces: factor-local moment helpers, host readout assembly in
  `src/cffg/engine.py`, public API documentation, learning consumers, and
  focused readout/oracle tests.
- N/A surfaces: IR and serialization, validation, compiler grouping and slot
  layouts, parameter packing, `member_of`, synchronous/scheduled message
  updates, `_make_bfe`, schedule selection, estimator, checkpoint migration,
  and visualization. This ticket reads existing arenas and changes none of
  them.
- Duplicated pairs: flood/mixed results must agree when given the same fixed
  point. Host/JIT is N/A for the public NumPy readout, though any JAX
  factor-moment helper must retain grouped/`vmap` parity. Colony state batching
  and current/legacy checkpoints are N/A because the API consumes one ordinary
  `State` and stores no derived readout.
- Canonical tests: extend `tests/test_conj.py` and the hybrid engine/switching
  tiers; add a dedicated test file only if the combined readout cases would
  obscure those existing contracts. Register any new file in the dashboard.

## Hostile inputs

- **Sparse/zero support:** an exactly impossible selector regime has exactly
  zero responsibility and exactly zero weighted sufficient statistics. It
  must not create `0 * NaN`; empty joint support is improper/infeasible, never
  renormalized onto a larger support. Recurrence categorical zeros remain
  exact in both probability readouts.
- **Improper/vacuous inputs:** non-PD component parameters, improper D3 joint
  beliefs, or an improper recurrence drive return `proper=False` with
  shape-correct NaNs. A `gaussian_mixture_linear` cavity predictive with a
  vacuous input Gaussian is not a normalized predictive and is loud; the
  prior-only `gaussian_mixture` remains proper when its component covariances
  are proper.
- **Float32:** use the existing dtype-scaled PD/PSD guards and stable
  normalization. Introduce no absolute ridge, probability floor, or new
  tolerance. Exact-zero responsibilities survive host conversion.
- **Uncomfortable scale:** exercise at least `K=32`, `d=16` with finite
  float32 readouts. Storage is the declared `O(K d²)` output, with no dense
  regime-pair or whole-graph covariance tensor.
- **Semantic leakage:** changing only the message facing the predictive
  factor's `out` role must not change the cavity predictive, while changing
  the input or selector cavity must. This negative test pins the distinction
  from the posterior edge marginal.

## Dependency map

Legend: prerequisite --> dependent

    [x] cff-3hgi Gaussian numerical foundation
              |
              +--> [ ] cff-lquu hybrid readouts (ready)
                          |
                          +--> [ ] cff-4apw PG recurrence learning
                          |           |
                          |           +--> [ ] cff-y4nx rSLDS integration
                          |
                          +--> [ ] cff-ahps factor-library epic
                                (also waits on parallel roadmap branches)

Parallel ready performance work (`cff-g7rc`, `cff-wbhu`) shares no dependency
with this path. `cff-209l` explicitly excludes generic hybrid registration and
is not a prerequisite.

## Acceptance Criteria

1. `gaussian_node_stats` returns the declared mixture and recurrence records;
   D3 responsibilities and weighted moments match an independent NumPy
   reconstruction of the factor belief, including augmented offset moments.
2. Summing per-regime weighted moments reproduces the factor-belief aggregate,
   and the `K=1` linear-mixture record matches the ordinary
   `gaussian_linear` statistics at existing numerical tolerance.
3. A fixed-seed low-separation switching fixture uses the soft statistics in
   the existing MNIW between-trial loop and beats a documented hard-assignment
   baseline on held-out parameter error with headroom. Mark it `slow` if its
   multi-trial runtime warrants it.
4. `gaussian_mixture_predictives` matches independent analytic component
   moments and dense scalar quadrature, and its collapsed moments match the
   existing forward message. The output-facing-message negative test proves
   cavity semantics.
5. Recurrence records expose only beliefs and moments; `cff-4apw` can compute
   its PG sufficient statistics without reading private arena plans or
   reconstructing edge products. No PG posterior/update implementation lands
   in this ticket.
6. Exact zeros, empty support, improper/vacuous beliefs, float32, and
   `K=32,d=16` satisfy the hostile-input outcomes above. No readout can return
   `proper=True` with a non-finite value.
7. Existing message, marginal, BFE, schedule, structure-hash, and serialization
   tests remain unchanged. README documents both public readouts and
   ARCHITECTURE records their factor-belief/cavity boundary; LIMITATIONS keeps
   the D3 assumed-density qualification accurate.

## Notes

**2026-08-01T02:39:09Z**

Planning handoff complete. Implementation should extend the existing factor-belief algebra rather than reconstructing hybrid statistics from endpoint marginals. The predictive readout is pinned as the factor-to-out cavity predictive and must exclude the out-facing message; recurrence scope is belief/moment exposure only, with PG expectations and posterior updates left to cff-4apw. The hostile contract covers exact support, vacuous/improper beliefs, float32, K=32/d=16, and evidence-leakage semantics. Next step is implementation and focused oracle tests. Ticket-only validation: tk dep tree --full cff-y4nx confirms the path cff-lquu -> cff-4apw -> cff-y4nx; tk ready/blocked agree; tk dep cycle and git diff --check pass. No short suite is required for this ticket-only PR.
