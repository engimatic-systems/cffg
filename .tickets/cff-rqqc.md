---
id: cff-rqqc
status: open
deps: [cff-iwgl]
links: [cff-wbhu, cff-iwgl, cff-sj1w]
created: 2026-07-28T18:06:35Z
type: feature
priority: 2
assignee: coopwilliams
---
# Colony: per-instance PRNG streams and mu3 stochastic status batching

## Background in Plain English

`Engine` already owns the `mu3` evaluator's random sampling contract. It draws
one particle set from an explicit caller key, keeps those particles unchanged
during an inference attempt, returns the next key, and reports effective sample
size and stochastic outcome for each evaluator group. `Colony` owns graph
instance creation, removal, state, structure grouping, and result routing. At
the repository state inspected on 2026-08-07, `Colony.spawn` rejects every graph
that contains a `mu3` evaluator.

This ticket makes `Colony` responsible for carrying each graph instance's
caller-supplied key cursor, prepared particle state, and stochastic result. The
`Engine` remains responsible for sampling and `mu3` mathematics. An instance
that exhausts its sampling budget or diverges must stop without changing the
state or outcome of another instance in its structure group.

Random state is graph-instance runtime data. It is not template structure, it
does not enter a structure hash, and it does not acquire temporal meaning from
`PortRole.GLUE`. The existing deterministic `Colony` interface remains usable
without a random key. This ticket establishes correctness and isolation for
stochastic Colony execution; cff-wbhu owns the later compiled-loop throughput
work.

## Outcomes / Acceptance Criteria

### 1. Colony has an explicit per-instance random-state contract

- [ ] 1.1 `Colony` accepts a caller-supplied JAX PRNG key when it creates or
  prepares a graph instance whose compiled arena contains a `mu3` evaluator.
- [ ] 1.2 `Colony` rejects a `mu3` instance that has no caller-supplied key
  before inference starts, and the error identifies the missing per-instance
  key.
- [ ] 1.3 A graph instance without a stochastic evaluator requires no PRNG key.
- [ ] 1.4 Equal key values on two instances are retained as caller intent and
  are not replaced with distinct values by `Colony`.
- [ ] 1.5 The random-state API does not inspect `PortRole.GLUE` or assign time,
  hierarchy, space, or another domain meaning to that port role.

### 2. Each instance owns its prepared evaluator state

- [ ] 2.1 Each `mu3` instance retains the particle array produced from its
  supplied key until the caller requests a refresh.
- [ ] 2.2 Each `mu3` instance retains its importance weights without sharing
  them with another instance.
- [ ] 2.3 Each `mu3` instance retains its log normalizer without sharing it with
  another instance.
- [ ] 2.4 Each `mu3` instance retains its effective sample size without sharing
  it with another instance.
- [ ] 2.5 Each `mu3` instance retains its stochastic outcome without sharing it
  with another instance.
- [ ] 2.6 Each `mu3` instance retains the next key returned by
  `Engine.prepare_mu3` separately from the string used to index the structure
  cache.
- [ ] 2.7 A call to `Colony.run` does not draw a new particle set.
- [ ] 2.8 A second `Colony.run` on the prior result does not draw a new particle
  set.
- [ ] 2.9 An explicit refresh consumes the stored key cursor once for each
  `mu3` evaluator group and stores the returned cursor for the next refresh.
- [ ] 2.10 Replaying an instance from its original graph inputs and original key
  produces a bitwise-identical particle array after the declared inference
  budget.
- [ ] 2.11 That replay produces bitwise-identical importance weights after the
  declared inference budget.
- [ ] 2.12 That replay produces bitwise-identical state leaves after the
  declared inference budget.
- [ ] 2.13 That replay returns a bitwise-identical next key after the declared
  inference budget.

### 3. Colony results agree with independent Engine results

- [ ] 3.1 For a homogeneous structure group with distinct keys, each Colony
  instance's particle array is bitwise identical to the array from an
  independent `Engine.prepare_mu3` call with that instance's key and inputs.
- [ ] 3.2 For that comparison, each Colony instance's returned key is bitwise
  identical to the key returned by the independent preparation call.
- [ ] 3.3 For that comparison, each Colony instance's final state leaves agree
  with the independent `Engine.run` leaves under a tolerance declared from the
  active dtype.
- [ ] 3.4 For that comparison, each Colony instance's free-energy trace agrees
  with the independent `Engine.run` trace under the declared dtype-aware
  tolerance.
- [ ] 3.5 For that comparison, each Colony instance's weighted moments agree
  with the independent engine diagnostics under the declared dtype-aware
  tolerance.
- [ ] 3.6 For that comparison, each Colony instance's log normalizers agree
  with the independent engine diagnostics under the declared dtype-aware
  tolerance.
- [ ] 3.7 For that comparison, each Colony instance's effective sample sizes
  agree with the independent engine diagnostics under the declared dtype-aware
  tolerance.
- [ ] 3.8 For that comparison, each Colony instance reports the `RunStatus`
  reported by the independent engine.
- [ ] 3.9 For that comparison, each Colony instance reports each per-evaluator
  stochastic outcome reported by the independent engine.

### 4. Instance lifecycle and stopping outcomes remain isolated

- [ ] 4.1 One structure group can contain instances whose final outcomes are
  `CONVERGED`, `BUDGET_EXHAUSTED`, and `DIVERGED`.
- [ ] 4.2 After one instance stops, later iteration work for an active peer does
  not change the stopped instance's state or diagnostic values.
- [ ] 4.3 A low-effective-sample-size outcome stops only the affected instance.
- [ ] 4.4 An all-nonfinite-weight outcome stops only the affected instance.
- [ ] 4.5 Removing one instance does not change another instance's stored key
  cursor or prepared evaluator state.
- [ ] 4.6 Creating an instance after a removal does not reuse the removed
  instance's key cursor or prepared evaluator state.
- [ ] 4.7 Rebuilding structure groups after creation or removal retains the
  association between each instance identifier and its key cursor.
- [ ] 4.8 Rebuilding structure groups after creation or removal retains the
  association between each instance identifier and its prepared state.
- [ ] 4.9 Rebuilding structure groups after creation or removal retains the
  association between each instance identifier and its latest result.

### 5. Deterministic and mixed-structure Colony behavior remains intact

- [ ] 5.1 A Colony containing both `mu3` and deterministic structure groups
  runs each group and returns one result for each live instance.
- [ ] 5.2 Deterministic instances produce the pre-change state on the frozen
  Colony fixtures.
- [ ] 5.3 Deterministic instances produce the pre-change free-energy trace on
  the frozen Colony fixtures.
- [ ] 5.4 Deterministic instances produce the pre-change `RunStatus` on the
  frozen Colony fixtures.
- [ ] 5.5 `Colony.spawn` accepts its pre-change deterministic call forms and
  retains their return semantics.
- [ ] 5.6 `Colony.run` accepts its pre-change deterministic call forms and
  retains their return semantics.
- [ ] 5.7 `Colony.bind` accepts its pre-change call forms and retains their
  state-update semantics.
- [ ] 5.8 `Colony.marginals` accepts its pre-change call forms and retains its
  marginal-read semantics.
- [ ] 5.9 `Colony.kill` accepts its pre-change call forms and retains its
  removal semantics.
- [ ] 5.10 This ticket makes no throughput claim for stochastic batches and does
  not require the compiled sweep loop delegated to cff-wbhu.

### 6. Hostile stochastic inputs have instance-local, dtype-aware behavior

- [ ] 6.1 A `mu3` proposal with a non-finite concentration fails before
  sampling and names the affected evaluator group.
- [ ] 6.2 A `mu3` proposal with a concentration that is not strictly positive
  fails before sampling and names the affected evaluator group.
- [ ] 6.3 A base posterior changed after preparation fails before further
  inference and names the affected evaluator group.
- [ ] 6.4 A valid sparse Dirichlet proposal follows the independent Engine
  result without adding an absolute probability floor.
- [ ] 6.5 Literal zero likelihood or goal support follows the independent
  Engine result without replacing the zero with a positive value.
- [ ] 6.6 All-nonfinite importance weights produce `DIVERGED` for the affected
  instance.
- [ ] 6.7 Effective sample size below the configured threshold produces
  `BUDGET_EXHAUSTED` for the affected instance.
- [ ] 6.8 A float32 Colony fixture with 20 observation outcomes, 40 hidden
  states, and 256 particles returns finite values wherever the independent
  Engine contract requires finite values.
- [ ] 6.9 Structure groups containing 32 and 128 live instances retain each
  per-instance key association.
- [ ] 6.10 Those structure groups retain each per-instance state association.
- [ ] 6.11 Those structure groups retain each per-instance diagnostic
  association.
- [ ] 6.12 Those structure groups retain each per-instance result association.
- [ ] 6.13 Every new numeric guard derives from `jnp.finfo` for the active dtype
  and has a short source comment naming the invariant it protects.

### 7. Serialization and documentation state the ownership boundary

- [ ] 7.1 Template JSON contains no PRNG key, particle array, importance
  weight, effective sample size, stochastic outcome, or next-key cursor.
- [ ] 7.2 Changing only an instance's PRNG key or prepared evaluator state does
  not change the template structure hash.
- [ ] 7.3 `README.md` documents the public per-instance key, run, diagnostic,
  and refresh workflow for a `mu3` Colony.
- [ ] 7.4 `docs/ARCHITECTURE.md` describes PRNG cursors, particles, weights, and
  stochastic outcomes as graph-instance runtime state owned by `Colony`.
- [ ] 7.5 `docs/LIMITATIONS.md` removes the statement that Colony rejects
  `mu3`.
- [ ] 7.6 `docs/LIMITATIONS.md` retains the documented importance-sampling
  limits that this ticket does not change.
- [ ] 7.7 The documentation states that `PortRole.GLUE` remains a
  domain-neutral composition boundary.

### 8. Every affected execution surface is reviewed and verified

- [ ] 8.1 The implementation handoff walks the `Runtime numerical or status
  invariant` row in `docs/ARCHITECTURE.md` and records each named surface as
  changed or not applicable, with a reason.
- [ ] 8.2 The independent `Engine.prepare_mu3`/`Engine.run` path and the Colony
  path are exercised on the same per-instance inputs.
- [ ] 8.3 The one-instance Colony path and the many-instance structure-group
  path are exercised on the same model contract.
- [ ] 8.4 Host-side active-instance and result routing is exercised together
  with the vmapped evaluator work used by the selected implementation.
- [ ] 8.5 Flooding and scheduled message paths are marked not applicable unless
  the implementation changes message-rule or free-energy mathematics.
- [ ] 8.6 If message-rule mathematics changes, flooding and scheduled paths
  produce equal messages under the declared dtype-aware tolerance.
- [ ] 8.7 If free-energy mathematics changes, flooding and scheduled paths
  produce equal free energy under the declared dtype-aware tolerance.
- [ ] 8.8 If message-rule or free-energy mathematics changes, flooding and
  scheduled paths report the identical runtime outcome.
- [ ] 8.9 Current and legacy checkpoint decoding are marked not applicable when
  the implementation adds no Colony checkpoint format.
- [ ] 8.10 If the implementation adds a Colony checkpoint format, current and
  legacy decoding paths receive explicit acceptance coverage before closure.
- [ ] 8.11 Focused coverage includes `tests/test_colony.py`, `tests/test_mu3.py`,
  `tests/test_runtime_status.py`, and any changed schedule test.
- [ ] 8.12 The final code change is followed by one passing default short-suite
  run through the repository dashboard workflow.

## Methodological Suggestions

The implementer may choose another sound method that satisfies every outcome.
One direct design is to extend `_Instance` with a clearly named structure-cache
identifier and a separate optional PRNG cursor, then let `Colony.spawn` prepare
`mu3` state through the cached `Engine`. A separate `refresh` operation can
call `Engine.prepare_mu3` with the stored cursor and replace both the instance
state and cursor together. Keep refresh atomic so a failed proposal does not
partly replace prepared state.

Reuse `Engine.prepare_mu3`, `Engine.run`, and `Engine.mu3_diagnostics` as the
single-instance references. Do not duplicate the importance-sampling formulas
in `colony.py`. The current mixed-GFE Colony path already retains one host
controller per instance. It is a valid correctness-first route for stochastic
instances; vectorize pure evaluator work only where stopped-state freezing and
per-instance diagnostics remain explicit. Leave the compiled sweep/status
carry and throughput target to cff-wbhu.

Use exact array equality for keys and replayed random artifacts. Before writing
numeric comparison tests, declare tolerances as multiples of the compared
dtype's machine epsilon and account for any reduction-order difference caused
by `vmap`. Use independent Engine runs as the behavioral reference, and build
one structure group whose per-instance inputs deliberately produce each
stochastic outcome.

The applicable change-path row is `Runtime numerical or status invariant` in
`docs/ARCHITECTURE.md`. Walk `RunStatus`, `RunResult`, `Engine.run`, the `mu3`
status fields, Colony active-instance bookkeeping, result routing, the resource
estimator, and the dashboard consumer. `New IR field` is not applicable because
the template schema and structure hash must not contain runtime random state.
The duplicated execution surfaces are independent Engine versus Colony,
one-instance versus grouped Colony, and host active-set routing versus any
vmapped evaluator work. Flooding versus scheduled rules is not applicable if
the message and free-energy formulas do not change. Current versus legacy
checkpoint decoding is not applicable because Colony has no checkpoint
format.

Evidence anchors:

- `src/cffg/engine.py` defines `prepare_mu3`, fixed-particle checks,
  base-posterior validation, diagnostics, and stochastic outcomes.
- `tests/test_mu3.py` anchors explicit-key replay, preparation refusal,
  refreshed-particle scoring, low effective sample size, collapsed weights,
  exact-zero support, float32, and the 20-by-40 scale case.
- `tests/test_mu3_derivation.py` independently anchors the importance-sampling
  formulas and hostile inputs.
- `src/cffg/colony.py` owns instance lifecycle and structure grouping and
  contains the current `mu3` rejection.
- `tests/test_colony.py` anchors structure-cache grouping, result equivalence,
  mixed structures, removal, and host data routing for deterministic models.
- `docs/ARCHITECTURE.md` section 5b records the current Colony boundary and its
  target direction.
- `README.md` documents the single-Engine `mu3` workflow and deterministic
  Colony workflow.
- `docs/LIMITATIONS.md` S2 records the current Colony rejection and the
  importance-sampling limits that remain after this ticket.

Legend: `-->` means prerequisite to dependent; `~~~` means related work without
a dependency edge.

```text
[x] cff-sj1w Colony instance and structure-group foundation ~~~+
                                                            |
[x] cff-iwgl Engine mu3 evaluator and explicit-key contract --> [ ] cff-rqqc (ready)
                                                            |
[ ] cff-wbhu compiled-loop throughput ~~~~~~~~~~~~~~~~~~~~~~~+

[ ] cff-rqqc --> no declared downstream feature or epic
```

## Future work

cff-wbhu owns moving the general sweep loop and per-instance stopping carry
into a compiled JAX loop and owns its throughput threshold. It is linked work,
not a prerequisite for cff-rqqc. Better importance-sampling estimators,
including Laplace methods, quadrature, and control variates, remain research
outside this ticket; `docs/LIMITATIONS.md` S2 records their possible benefit.
No downstream feature or epic currently depends on cff-rqqc.

## Definitions

- **Colony**: the runtime that owns many graph instances, groups them by
  structure, and routes each instance's state and result.
- **Graph instance**: one runtime state and input set associated with one
  compiled graph structure.
- **JAX PRNG key**: explicit state consumed by JAX random operations so callers
  can reproduce and advance sampling.
- **Key cursor**: the next PRNG key stored for a later explicit refresh.
- **`mu3`**: the nonconjugate Part II functional message from future planning
  slices to the learned observation-model parameter `A`.
- **Stochastic evaluator group**: the future slices that share one learned
  parameter blob and therefore one `mu3` particle set and diagnostic record.
- **Particle**: one sampled observation-model matrix used by the `mu3`
  importance-sampling estimate.
- **Importance weight**: the normalized contribution of one particle to the
  estimated moments.
- **Effective sample size (ESS)**: a diagnostic derived from importance weights
  that reports how many equally weighted particles would carry comparable
  information.
- **Free energy (FE)**: the objective value used by the inference runtime as a
  convergence signal and policy-scoring quantity.
- **`RunStatus`**: the runtime classification for convergence, budget
  exhaustion, infeasible support, or divergence.
- **Base posterior**: the proper Dirichlet distribution from which `mu3`
  particles are drawn before the functional message reweights them.
- **Structure hash**: the content identifier used to cache an engine for graph
  structure; runtime keys and particles do not contribute to it.
- **`PortRole.GLUE`**: a domain-neutral open edge used to compose graph
  fragments.
- **Common random numbers**: deliberate reuse of equal keys so two instances
  receive equal random draws for a controlled comparison.
- **float32**: JAX's 32-bit floating-point dtype, whose machine precision sets
  the numeric guards and test tolerances for that execution.

## Notes

### 2026-07-28T18:06:47Z — Dependency evidence preserved from the original ticket

Dependency companion (Legend: prerequisite --> dependent):

```text
[x] cff-sj1w Colony batching foundation
[x] cff-iwgl mu3 runtime + explicit-key evaluator
        |
        +--> [ ] cff-rqqc per-instance stochastic Colony runtime (ready)
```

cff-wbhu is linked parallel performance work, not a prerequisite: key/status
correctness should not be coupled to the later `lax.scan` throughput refactor.
This ticket is not on the cff-mwjo completion path; it removes the documented
Colony integration limitation after the core M3 evaluator exists.

### 2026-08-07 — Rewrite audit

The bounded historical ambiguity slice returned 29 candidates across 50
occurrences and classified zero occurrences as problems. The final
post-rewrite scan returned 42 candidates across 186 occurrences. Manual review
found no unclear claim. The complete retained candidate-to-meaning map follows:

- `fixed`: a particle set is pinned for one inference attempt.
- `integrate`: the preserved note names internal Colony-to-Engine wiring.
- `preserve`: the original dependency note and its evidence are retained.
- `public`: the documented consumer call contract is meant.
- `reusable`: removed instance state is forbidden from artifact reuse, the
  named Engine routines are shared code, and equal-key experiments deliberately
  reuse random draws.
- `same`: the comparison uses equal per-instance inputs or one declared model
  contract.
- `implement`: planned code choices or the required implementation handoff are
  meant according to the sentence.
- `support`: the feasible probability set, including literal zero membership,
  is meant.
- `complete`: cff-mwjo work-item completion or this ticket's review completion
  marker is meant.
- `ready`: cff-rqqc has no open declared prerequisite.
- `remaining`: a named responsibility or semantic property continues, or named
  research stays outside this ticket.
- `close`: acceptance coverage is required before work-item closure.
- `check`: an Engine runtime input guard is meant.
- `converge`: the public runtime stopping classification or its FE signal is
  meant.
- `correct`: conformance to the explicit key, state-isolation, and status
  contract is meant.
- `equivalent`: deterministic Colony and independent Engine behavior agree on
  the named fixture.
- `estimate`: the `mu3` importance-sampling computation, the resource estimator, or a
  proposed alternative estimator is meant according to the sentence.
- `evidence`: named repository sources establish the ticket's provenance.
- `exact`: key and random-artifact identity is bitwise, while zero support is
  literal.
- `pass`: the default short-suite command returns test success.
- `reproduce`: replaying the explicit PRNG state recreates its random draws.
- `sound`: an optional method satisfies this ticket's acceptance contract.
- `test`: a named executable check or numeric comparison check is meant.
- `validate`: `Engine` checks the base posterior before sampling or reuse.
- `verify`: a future acceptance check is required.
- `infer`: state or policy inference at installed parameter values is meant.
- `learn`: `A` is the observation-model parameter updated through the declared
  learning path.
- `performance`: cff-wbhu's warmed repeated-execution throughput work is meant.
- `scale`: the 20-observation by 40-state domain size is meant.
- `can`: criterion 4.1 requires a demonstrated batch capability; suggestion and
  definition uses describe a permitted design capability.
- `may`: the implementer has permission to select another method that satisfies
  the outcomes.
- `should`: the preserved historical note states a design intent.
- `explicit_owner`: `Engine`, `Colony`, this ticket, and cff-wbhu are the named
  capability owners in their surrounding clauses.
- `named_ticket_handoff`: cff-wbhu is the concrete future owner of compiled-loop
  throughput.
- `all`: every importance weight in the affected instance is non-finite.
- `any`: the implementer freely selects the applicable changed schedule check,
  reduction-order difference, or vmapped evaluator work.
- `every`: the requirement applies independently to each graph, numeric guard,
  execution surface, or acceptance outcome in the named class.
- `only`: instance-local stopping excludes peers, a runtime-state-only change
  excludes structure, and vectorization is limited by the stated isolation
  condition.
- `current`: repository claims refer to the 2026-08-07 inspected tree and
  current-versus-legacy names select the active checkpoint variant.
- `existing`: deterministic Colony calls predate this ticket and form the
  compatibility baseline; the preserved note describes a limitation present
  when it was written.
- `later`: the action follows the named earlier step or is delegated to future
  work.
- `read`: `Colony.marginals` retrieves marginal data from runtime state.

Intelligible-ticket review: complete.
