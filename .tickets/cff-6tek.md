---
id: cff-6tek
status: open
deps: [cff-x0sv]
links: [cff-l6i5]
created: 2026-07-27T20:52:13Z
type: feature
priority: 2
assignee: coopwilliams
---
# Register Gaussian learnables and named non-discrete parameter writes

## Background in Plain English

The declared-learning API can update categorical parameters, but it cannot yet
address or learn Gaussian parameters. A Gaussian-linear factor stores its
matrix, offset, and covariance in separate runtime arrays, while the
`cff-x0sv` `Arena.member_of` representation points to only one array row.
Consumers therefore
edit stacked arrays by position or maintain model-specific installers.

This ticket owns the multi-role runtime address, atomic parameter installation
by factor name, and the NIW and MNIW adapters needed for ordinary Gaussian and
Gaussian-linear factors. It also owns both learning cadences required by the
roadmap: adding evidence from distinct trials and replacing a one-sequence
coordinate from fixed priors. Ticket `cff-l7y4` owns when the rSLDS fitter uses
those operations; this ticket owns their generic behavior.

## Outcomes / Acceptance Criteria

1. The compiler replaces each `Arena.member_of` value with one frozen
   `MemberAddress` containing the factor type and a tuple of frozen
   `ParameterAddress` records. Each `ParameterAddress` contains `role`,
   `blob`, `key`, `index`, and `transform` fields.
   1.1. `role` is the serialized `FactorSpec.param_blobs` role, `blob` is the
   referenced blob name, `key` and `index` locate one runtime row, and
   `transform` is one of `identity`, `probability_to_log`, or
   `covariance_to_precision`.
   1.2. Every package consumer of the old `(key, index)` pair selects an
   explicit role from the new address.
2. Member addresses cover these factor types and parameter roles:
   2.1. `categorical(statistic)`, `matrix_cat(matrix)`, and
   `transition_mixture(matrices)` retain their pre-ticket discrete coverage.
   2.2. `gaussian(mean, cov)` and
   `gaussian_linear(matrix, offset, cov)` receive Gaussian coverage; a
   conditioned `gaussian_linear_obs` runtime group uses the serialized
   `gaussian_linear` roles.
   2.3. `gaussian_mixture(mean, cov)`,
   `gaussian_mixture_linear(matrix, offset, cov)`,
   `recurrence_link(weights, offsets)`, and `gaussian_prec(mean)` receive
   hybrid coverage for named writes.
   2.4. Plate expansion is not a separate parameter family. Every scoped
   plate copy of a covered factor receives its own address automatically.
   2.5. Parameterless equality factors are N/A. Constraint parameters are
   compile-time structure and require recompilation, so named runtime writes
   are N/A. EF kernel parameters and direct writes to synthetic GFE composite
   members are N/A for this ticket and remain with `cff-ahps`; the internal
   addresses used by declared learning retain their existing behavior.
3. The package exports
   `set_params(engine, params, updates) -> dict[str, Array]`, where `updates`
   maps factor names to maps from the exact roles in criterion 2 to numeric
   arrays.
   3.1. The returned mapping is new; neither the input mapping nor its arrays
   are changed.
   3.2. Probability roles validate their documented simplex axes and convert
   probabilities to runtime log parameters while preserving literal zero as
   the runtime negative-infinity sentinel.
   3.3. Each `cov` value must be finite, symmetric, and positive definite and
   is converted to the precision consumed by the Gaussian runtime.
   3.4. Identity roles are finite and retain their serialized value.
   3.5. Values must match the serialized blob shape. Accepted values are cast
   to each destination dtype only after validation.
   3.6. Updating one factor role updates every runtime row that reads the same
   blob in the same role. An update cannot silently break template-level
   parameter tying; callers that need separate values must edit the template
   blob references and recompile.
   3.7. Two requested updates that reach one tied blob with unequal values are
   rejected.
   3.8. Unknown factors, unsupported roles, non-numeric values, invalid
   shapes, non-finite values, invalid probabilities, and invalid covariances
   fail before any output array is changed.
   3.9. Runtime members not reached through an updated blob and role retain
   their original values and object-compatible shapes.
4. `Template.learnables` accepts exactly these new family and factor pairs:
   `niw` with `gaussian`, and `mniw` with `gaussian_linear`, including the
   conditioned runtime form of that factor. Other NIW or MNIW pairings fail at
   compile time with the declaration name and unsupported pair.
   4.1. Generic NIW or MNIW learning for `gaussian_mixture` and
   `gaussian_mixture_linear` is outside this ticket because it needs a
   per-regime posterior-axis contract. `cff-y4nx` retains its model-specific
   per-regime MNIW dynamics learner, and `cff-ahps` owns broader generic
   registration.
   4.2. Recurrence, precision, and EF factors receive no new learning-family
   adapter in this ticket.
5. The NIW and MNIW adapters read `Engine.gaussian_node_stats`, validate prior
   and statistic keys and shapes, expose posterior means for reporting, and
   route every rollover output through the role-bearing `LearnTarget` records.
6. Tied sources have one statistical meaning. Every eligible factor that
   reads a learnable's blobs contributes one factor-local statistic record,
   all such records are pooled into one posterior update, and one rollover is
   written to every tied target. A declaration with an explicit empty source
   set contributes no evidence but may still write its prior rollover.
7. `Learnables` retains an immutable validated copy of each initial prior and
   provides two atomic update operations:
   7.1. `accumulate` starts from the current posterior and adds one new set of
   source statistics for distinct-trial learning.
   7.2. `replace` starts from the immutable initial prior and recomputes the
   posterior from the current source statistics for one-sequence coordinate
   fitting.
   7.3. Both operations accept the named `only` restriction and
   require a converged source state.
   7.4. Both operations compute and validate every selected posterior in
   temporary storage. If one selected source or posterior is invalid, no
   selected posterior changes.
   7.5. Calling `replace` twice with unchanged statistics produces identical
   hyperparameters and does not count the sequence twice.
8. `Learnables.write` is atomic across all selected targets and returns a new
   parameter mapping. NIW and MNIW `cov` roles receive the inverse expected
   precision `psi / nu` defined by `cffg.conj`, rather than the posterior mean
   covariance.
9. Each supported family/factor pair has independent behavioral evidence:
   9.1. Pooled `niw`/`gaussian` statistics produce the same posterior
   hyperparameters and rollover as a direct `cffg.conj.NIW` update under the
   test dtype's tolerance.
   9.2. An end-to-end `mniw`/`gaussian_linear` trial loop uses
   `Learnables.accumulate` and `Learnables.write`, without manual statistic
   reads or row edits, and meets the seeded scalar LGSSM limits in
   `tests/test_conj.py::test_trial_loop_recovers_lgssm_parameters`:
   matrix error below `0.05`, offset magnitude below `0.05`, and covariance
   error below `0.03`.
   9.3. A tied observed-emission fixture pools soft statistics from every time
   step and one `replace` call matches a direct pooled MNIW update.
10. Hostile inputs have explicit outcomes for both supported pairs. An
    improper prior, non-finite statistic, `proper=False` record, incompatible
    statistic shape, or non-positive-definite posterior fails before posterior
    or runtime parameters change. A masked Gaussian-linear observation adds
    no evidence. Literal zero moment entries remain zero without a probability
    floor.
11. Float32 accumulation, replacement, and writes remain finite and preserve
    float32 destinations. Every new numerical guard or tolerance derives from
    `finfo` for the active dtype and states the invariant it protects.
12. A fixture with at least 16 continuous dimensions and 64 tied or grouped
    members constructs addresses, pools statistics, and installs named values
    without an array coupling unrelated group members or a Python loop in the
    numerical update over individual group rows.
13. The versioned learning checkpoint round-trips NIW and MNIW keyed
    hyperparameters. Template serialization and the structure hash include
    family declarations and blob roles but exclude derived runtime arrays and
    the transient member-address map.
14. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` document the
    address and named-write contracts, tied-source pooling, both learning
    cadences, inverse-expected-precision rollover, and the unsupported
    learning families left after this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another method
that satisfies the outcomes.

- Use the **Learnable or checkpoint feature** change-path row. Treat
  `src/cffg/compiler.py` as the address authority, `src/cffg/learn.py` as the
  cadence and adapter authority, and `src/cffg/conj.py` as the posterior-
  algebra authority.
- Derive `set_params` and `Learnables.write` from the same address records so
  direct writes and posterior rollovers cannot disagree about roles, row
  layout, covariance conversion, scoped names, or tied blobs.
- Preserve serialized blob references as the tying authority. A reverse
  blob-and-role index may make validation and fan-out explicit without
  embedding runtime arrays in the template.
- Check validation, grounding, compilation, `Arena.learn_plans`, readouts,
  checkpoint encoding, bundle behavior, package exports, and the three
  canonical documents. Message equations, free-energy evaluation, and
  schedule selection are N/A because updates occur between converged runs.
- Exercise unscoped and plate-scoped names, singleton and stacked groups,
  distinct-trial accumulation and one-sequence replacement, float32, masked
  observations, improper statistics, and tied emissions. Compare flooding and
  scheduled statistic readouts only after each run reports convergence.
- Extend `tests/test_learnables.py` for contracts and use the existing seeded
  recovery case in `tests/test_conj.py` as the MNIW behavioral target. Run
  focused tests during implementation and the short suite once after the
  final code change.

## Future work

`cff-l7y4` consumes generic MNIW emission learning and reset-from-prior
replacement while owning rSLDS outer-loop scheduling, initialization, status,
and scientific comparison. `cff-y4nx` retains its deliberate model-specific
per-regime dynamics and recurrence learner.

`cff-ahps` owns generic learnable adapters for mixture, recurrence, precision,
and EF families, plus named writes for EF kernel parameters and any synthetic
GFE destinations that need a public member identity.

## Definitions

- **Atomic update:** An operation that changes none of its selected outputs if
  validation of any selected input or result fails.
- **Distinct-trial accumulation:** Adding statistics from a new independent
  sequence to posterior evidence retained from earlier sequences.
- **EF (exponential family):** A distribution family represented through
  sufficient statistics and natural parameters.
- **GFE (generalised free energy):** The active-inference objective whose
  compiled composites may absorb parameter uncertainty.
- **MNIW (Matrix-Normal-Inverse-Wishart):** A conjugate posterior for a linear
  matrix, offset, and covariance.
- **NIW (Normal-Inverse-Wishart):** A conjugate posterior for a Gaussian mean
  and covariance.
- **One-sequence coordinate replacement:** Recomputing a posterior coordinate
  from fixed priors and the current statistics of one sequence instead of
  adding that sequence again.
- **Parameter role:** The serialized meaning of a parameter blob within one
  factor, such as `matrix`, `offset`, or `cov`.
- **Runtime row:** One member's slice of a stacked parameter array in a
  compiled arena.
- **Tied factors:** Factor instances whose parameter roles reference the same
  serialized blob and therefore denote one shared parameter.
- **VMP (variational message passing):** Message passing that consumes
  expectations under variational posterior beliefs.

## Notes

**2026-07-27T21:57:56Z**

Scope narrowed by review of cff-x0sv (2026-07-27).

The structural half of this ticket landed in cff-x0sv instead, because a
reviewer correctly pointed out that deferring it meant the contract did not
actually admit NIW/MNIW, whatever the docstring claimed. Now done:

- hyperparameters, statistics and rollover outputs are keyed dicts, not bare
  arrays, so a posterior with four hyperparameters and three statistics fits
- a Family names which engine readout supplies its statistics
  (cffg.learn.READOUTS), so a family reading gaussian_node_stats rather than
  node_beliefs needs no change to the runtime
- LearnTarget carries the blob it writes and the parameter role that blob
  fills, so one MNIW rollover routes into a gaussian_linear's separate
  matrix, offset and cov blobs
- LearnPlan carries per-blob shapes, which differ across a joint family

What remains here is genuinely registration plus the map it needs:

1. extend Arena.member_of past the discrete families, so every parameterized
   factor has an addressable per-member slot (Gaussian, EF, hybrid and
   plate-expanded group builders all skip it today). A Gaussian learnable
   currently raises a compile error naming exactly this.
2. register NIW and MNIW against the Family protocol, wrapping the updates
   already in cffg.conj.
3. the bulk set_params(factor_name -> arrays) helper from the 2026-07-10
   evaluation note, which falls out of the same map.

**2026-08-01T05:06:38Z**

NASCAR consumer finding, 2026-07-31: the independent cff-y4nx harness at /Users/ravd/projects/experiments/cffg-nascar-validation commit 0543683 could run fixed-emission oracle inference but could not express the canonical observed-only fit through packaged learning. Shared gaussian_linear emission MNIW registration is now a concrete downstream requirement of cff-l7y4, not only an API-completeness argument. Source: results.json canonical_fit_readiness and primary-agent code audit. Confidence: high.

**2026-08-08T17:16:27Z**

Readiness audit, 2026-08-08: the rSLDS milestone is not implementation-ready. The remaining design and sequencing questions are recorded in docs/todo-notes/cff-6tek-rslds-milestone-questions.md for the refiner. cff-y4nx is substantively clear; cff-6tek still needs the structured rewrite, and cff-l7y4/cff-r6bl retain the cross-ticket questions named there.

**2026-08-08T18:10:00Z**

Readiness resolution, 2026-08-08. This rewrite supersedes the preceding readiness verdict: the ticket now fixes the multi-role address shape, covered and excluded factor families, atomic and tie-preserving `set_params` behavior, NIW/MNIW pairings, pooled tied statistics, and both accumulation and replacement cadences. `cff-6tek` depends only on closed `cff-x0sv`, so implementation may start without `cff-y4nx` closing.

Legend: prerequisite --> dependent

    [x] cff-x0sv --> [ ] cff-6tek (ready) --+
                                                |
    [x] cff-k9ws --> [x] cff-4apw --> [~] cff-y4nx --+--> [ ] cff-l7y4
                                                     |
    [ ] cff-r6bl (ready, parallel) ------------------+--> [ ] cff-ahps

The `tk` records remain authoritative. The bounded current scan retained intentional terms with these meanings: canonical=project documents or literature model; end_to_end=component-chain integration test; fixed=fixed prior or constant parameter; generic=model-general protocol or registered family coverage; n_a=structurally irrelevant or delegated surface; preserve=literal-value, dtype, tying, or compatibility invariant; public=documented API; register=runtime family coverage; same=blob identity, value equality, or shared address authority; expose=reporting access; implement=planned component integration or preserved historical statement; provide=ticket deliverable; support=registered pair or probability support; done and land=preserved historical completion; ready=dependency or design readiness; remaining=preserved state or named delegated work; sufficient=mathematical statistics; check=manual inspection or automated check; converge=runtime status; correct=preserved review finding; evidence=test or provenance artifact; exact=literal role, zero, identifier, or threshold; match=tolerance comparison; oracle=generating-parameter workflow; pass=message passing or test success; test=test artifact or execution; validate=input or posterior guard; accumulate=distinct-trial evidence addition; fit=parameter estimation; infer=posterior inference; learn and learnable=declared-learning workflow or registered family; recover=parameter recovery; rollover=posterior-to-runtime conversion; trial=independent dataset; can and could=present or historical capability selected by the sentence; expected=mathematical expectation; may=permission or optional method; explicit_owner and named_ticket_handoff=the ticket, component, or named roadmap ticket in the sentence; all, any, every, and only=the locally enumerated set or explicit exclusion; at_least=inclusive lower bound; current=runtime-active posterior or statistics; deferred=named historical delegation; existing=pre-ticket compatibility artifact; read=machine statistic access. Capability ownership was checked separately. Intelligible-ticket review: complete.

Post-note scan addition: integrate=the named component-chain integration test; complete=the intelligible-ticket review marker; close=ticket closure in the sequencing statement; estimate=parameter estimation. Historical slice: 29 candidates, 61 occurrences, zero classified problems.
