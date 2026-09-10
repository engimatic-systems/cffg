---
id: cff-cfqg
status: closed
deps: [cff-ofv1]
links: [cff-ahps, cff-209l, cff-edoz, cff-ws3c, cff-ty0v]
created: 2026-08-13T03:59:30Z
type: epic
priority: 1
assignee: coopwilliams
external-ref: gh-108
tags: [architecture, trustworthiness, extensibility]
---
# Epic: Reduce the central extension surface

## Background in Plain English — WHY

After the rSLDS clean-port stack, CFFG can run a broad mix of discrete,
Gaussian, exponential-family, hybrid, recurrence, constraint, and GFE paths.
Adding or changing one family still requires careful edits across large central
modules. Commit e488df3 is this program's authoring snapshot, source-region
map, and external-parity reference: `src/cffg/compiler.py` has 2,368 lines and
`src/cffg/engine.py` has 3,973 lines there. Both interpret factor types in
several places. That makes a local feature look like an engine-wide change and
makes it easy for synchronous execution, schedules, objective terms, readouts,
and serialization-derived plans to drift apart.

This epic makes the existing library easier to trust and extend. It first
finishes the registered same-arena factor path. It then gives compiler stages,
runtime execution, schedules, objective accounting, and readouts focused
owners behind typed internal contracts. The final gate proves the resulting
extension paths with executable evidence.

This is an architecture program with execution-structure preservation gates.
It protects the existing flat storage, chain scans, traced-executable reuse,
and actual Colony batches while improving extension ownership. It does not
repeat rSLDS scientific validation, add a factor family, promise a speedup,
define a public release, or change an existing model's mathematics.

## Outcomes / Acceptance Criteria — WHAT

1. The direct dependency `cff-ofv1` is closed because its acceptance criteria
   are satisfied. Its dependency chain closes `cff-209l`, `cff-edoz`, and
   `cff-ws3c` first.
2. Registered factors that use one discrete, Gaussian, or exponential-family
   arena have one documented extension path for structure, messages, energy,
   parameters, observations, readouts, and serialized implementation identity.
   Mode-specific input meaning is explicit: arena storage does not decide
   whether a rule consumes cavities, factorized beliefs, or a specialized
   variational construction. Max-product assignment scoring has its own
   declared capability, distinct from BFE/GFE.
3. A representative same-arena factor can be added and exercised without
   adding its type name to central compiler, engine, validation, schedule, or
   objective dispatch.
4. Factors that need more than one arena, a GFE composite, or a structural
   constraint keep explicit specialized plans.
   4.1. Each specialized plan has one compiler-lowering owner.
   4.2. Each specialized plan has one runtime-execution owner.
   4.3. Each specialized plan has one objective-accounting owner.
   4.4. No universal callback is introduced by passing irrelevant state for
        unrelated factor families.
5. Compiler planning uses context-sized typed stages for topology, registered
   lowering, specialized lowering, data/learning resolution, and assembly.
   `cff-ecw0` characterizes the exact commit that closes `cff-209l`, not the
   older authoring snapshot.
   Specialized discovery precedes ordinary grouping; the same owner then
   completes specialized drafts using registered membership. Raw topology,
   semantic edge tags, and blob-derived payloads have distinct owners.
   Assembly verifies factor/composite and objective ownership exactly once.
6. Engine execution, schedules, state/data/mu3 lifecycle, ordinary iteration,
   explicit outer/exact strategies, objective accounting, and readouts use
   focused owners after `cff-sf29` characterizes the exact commit that closes
   compiler track `cff-edoz`.
   Execution accepts local blocks from flat arrays or scan registers and
   returns local solver facts alongside writes. Engine and Colony share
   termination interpretation and result construction below the facade.
7. The refactor preserves the exact compiler and engine characterization
   baselines. For public, serialized, and numerical surfaces where `cff-209l`
   promises parity, this also preserves e488df3 imports and call forms,
   template JSON, structure hashes, CFFGZ behavior, addresses, checkpoints,
   migrations, messages, objective terms, schedules, and statuses.
8. The new same-arena registration behavior delivered by `cff-209l` is part of
   the compiler baseline and remains present through the final proof; it is not
   incorrectly attributed to e488df3.
9. Literal zero support, improper or vacuous inputs, float32, and
   uncomfortable-scale cases retain their documented values or loud status.
   No child ticket weakens the numerical robustness contract to make a
   structural extraction pass.
10. `docs/ARCHITECTURE.md` names the final owners and gives contributors a
    bounded change path for same-arena factors and each specialized plan class.
    `README.md` and `docs/LIMITATIONS.md` change only when a documented
    import or extension boundary actually changes.
11. Completion is described as a trustworthy architecture and bounded
    extension baseline. It is not described as proof of paper faithfulness,
    rSLDS recovery, runtime speed, release readiness, or universal factor
    coverage.
12. Existing discrete and Gaussian chains retain flat persistent bytes,
    registerized scan carry/scatter behavior, and traced-executable reuse when
    runtime values change at unchanged shapes/dtypes/static controls.
13. A Colony fixture with at least two distinct-input instances sharing a
    structure plus a different-structure instance retains the recorded Engine
    comparison, grouping, and isolation contract. An invalid/stopped neighbor
    does not change a healthy instance's state or termination facts.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Land the cff-wera discrete migration, then cff-zq25 Gaussian migration,
  closing the cff-209l foundation before compiler extraction. This ordering
  coordinates overlapping edits; it is not a mathematical arena dependency.
- Reuse existing mathematical authorities and add old/new observable
  characterization only for an uncovered refactor seam. A newly discovered
  scientific gap gets a separate ticket unless it proves the move unsafe.
- Move one owner at a time and keep compatibility imports until consumers
  migrate.
- Prefer typed values named for their responsibility over broad utility
  modules or partially initialized arena objects.
- Keep same-arena registration and specialized-plan lowering as separate,
  explicit extension classes.
- Use the **New factor or rule family**, **New IR field**, **Runtime numerical
  or status invariant**, and **Learnable or checkpoint feature** rows in the
  architecture change-path index as preservation checklists. Each child ticket
  states which duplicated execution surfaces apply.

## Future work

The following tickets remain outside this epic:

- `cff-xabn -> cff-enhs -> cff-jilz -> cff-cz6v` is the parallel persistent-
  learning/checkpoint chain.
- `cff-j7mm` owns resource-estimator correctness.
- `cff-wbhu` owns compiled-loop and Colony throughput.
- `cff-rqqc` owns per-instance stochastic keys/state/statuses; this epic retains
  Colony's explicit rejection of unsupported stochastic evaluators.
- `cff-5edo` owns long-rollout convergence and warm starts.
- `cff-3kdk` owns readout vectorization.
- `cff-cj5h` owns epistemic diagnostics.
- `cff-o7hs` is the immediate nonlinear-family implementation consumer and
  waits for this epic. Research and derivation that do not implement a factor
  may continue independently.
- `cff-ahps` remains the downstream factor-breadth roadmap and waits for this
  architecture gate.

## Definitions

- **Central dispatch:** Code in a shared compiler, engine, schedule, validation,
  or objective module that selects behavior by a factor type name.
- **Extension surface:** The production modules and contracts that must change
  to add one factor or feature.
- **Focused owner:** One module responsible for one named interpretation or
  operation.
- **Same-arena factor:** A factor whose variable messages all use one discrete,
  Gaussian, or exponential-family representation.
- **Specialized plan:** An explicit compiled plan for behavior that cannot use a
  same-arena kernel contract, such as a hybrid factor, GFE composite, or
  structural constraint.
- **Typed stage:** A compiler or runtime step with a named input and output
  contract instead of hidden shared state.

## Notes

**2026-09-08 PR #108 review handoff (supersedes the September 2 plan)**

Source of truth: `tk dep tree --full`, `tk ready`, and `tk blocked` at this
revision.

Baseline semantics: e488df3 is the authoring snapshot, source-region map, and
external-parity reference. `cff-ecw0` characterizes the exact commit that
closes `cff-209l`; `cff-sf29` characterizes the exact commit that closes
compiler track `cff-edoz`. Later children preserve those prerequisite-aware
baselines rather than attempting to characterize the older snapshot directly.

Legend: prerequisite --> dependent

```text
[ ] cff-wera (ready)
  --> [ ] cff-zq25
  --> [ ] cff-209l
  --> [ ] cff-ecw0
  --> [ ] cff-544x
  --> [ ] cff-8t4d
  --> [ ] cff-kfo7
  --> [ ] cff-ejib
  --> [ ] cff-bfr9
  --> [ ] cff-ddsq
  --> [ ] cff-edoz
  --> [ ] cff-sf29
  --> [ ] cff-ny7f
  --> [ ] cff-nyad
  --> [ ] cff-5xdy
  --> [ ] cff-ii9t
  --> [ ] cff-jrbw
  --> [ ] cff-gl89
  --> [ ] cff-3ola
  --> [ ] cff-ws3c
  --> [ ] cff-ofv1
  --> [ ] cff-cfqg
        |
        +--> [ ] cff-o7hs --> [ ] cff-1o72 --> [ ] cff-ahps
        |
        +-------------------------------------> [ ] cff-ahps

Parallel or downstream consumers outside this epic:
[ ] cff-xabn --> [ ] cff-enhs --> [ ] cff-jilz --> [ ] cff-cz6v
[ ] cff-j7mm
[ ] cff-wbhu --> [ ] cff-5edo
[ ] cff-rqqc
[ ] cff-3kdk
[ ] cff-cj5h

Parallel source-recovery research/specification:
[ ] cff-iur0 --> [ ] cff-ocxl --> [ ] cff-x0if --> [ ] cff-zov5
```

Capability ownership:

- `cff-209l` is the registration completion gate. cff-wera delivers the
  discrete binding; cff-zq25 delivers the Gaussian binding and combined proof.
- `cff-ecw0` and `cff-544x` own compiler characterization and compiled values;
  `cff-8t4d`, `cff-ejib`, `cff-kfo7`, `cff-bfr9`, and `cff-ddsq` own topology,
  registered lowering, specialized lowering, data/learning resolution, and
  final assembly; `cff-edoz` closes that track.
- `cff-sf29` owns bounded engine characterization. `cff-ny7f`, `cff-nyad`,
  `cff-5xdy`, `cff-ii9t`, `cff-jrbw`, `cff-gl89`, and `cff-3ola` own execution,
  schedules, lifecycle, ordinary control, explicit outer/exact control, the
  objective ledger, and readouts; `cff-ws3c` closes that track.
- `cff-ofv1` owns the measured extension-surface proof. It uses the
  `cff-209l` test-only factors to prove same-arena registration without
  central type-name edits and an explicit matrix for multi-arena recurrence,
  GFE/P-substitution, and constraints.
- `cff-o7hs` is the first new-family implementation consumer. The learning,
  resource, throughput, rollout, vectorization, diagnostics, and
  source-recovery tickets shown outside the epic retain their own scopes.

Architecture change-path audit:

- Compiler characterization and extraction use **New factor or rule family**,
  **New IR field**, and **Learnable or checkpoint feature**. Moving compiled
  values also checks **Runtime numerical or status invariant** because runtime
  consumers move imports.
- Engine characterization and extraction use **Runtime numerical or status
  invariant** and **New factor or rule family**. Restored-agent and
  learning-statistic consumers also use **Learnable or checkpoint feature**.
- The final proof records all four rows in its extension matrix.

Duplicated-surface and robustness audit:

- Compiler tickets keep flooding/scheduled, direct/`vmap`, and
  latest-schema/supported-legacy consumers on one compiled contract. Host/JIT
  compilation is N/A because compilation is host-side.
- Engine tickets pin flooding/scheduled, direct/multi-instance Colony, host/JIT
  status, and latest-schema/supported-legacy checkpoint pairs.
- Compiler hostile values are runtime N/A; dtype-stable structure and a
  32-group fixture are required. Engine evidence covers literal zero support,
  vacuous and improper inputs, non-finite updates, float32, and a domain or
  horizon at least ten times its smallest fixture.
- Specialized multi-arena, GFE, evaluator, and constraint plans remain
  explicit. This program does not force a universal callback.

Scope boundary: no child owns scientific recovery claims, a speedup target,
release readiness, or new factor-family behavior. Execution structure and
batch isolation are architecture acceptance criteria. The serial ticket chain
is a safe landing order for overlapping files; it is not the runtime call graph.

**2026-09-10T09:52:50Z**

Architecture epic complete on the review stack, not merged to main. All four gate children are closed: registration cff-209l, compiler cff-edoz, runtime cff-ws3c and extension proof cff-ofv1. The final source bc35dc6 passes 2,151 default-suite tests, including the 30 final ownership checks; 41 slow tests remain opt-in and four existing Gym warnings remain. Compiler and Engine facades are 138 and 659 physical lines; the largest new production owner is 808 lines. Same-arena external discrete/Gaussian/EF factors reuse declared registration for compilation, execution, energy, readouts and serialized implementation identity. Explicit recurrence/GFE/constraint owners retain specialized meanings and once-only composite/shared-parameter accounting. Existing oracles, frozen compiled plans, scan carry/scatter and trace-reuse tests, distinct-input Colony isolation, shared result finalization and supported reconstruction contracts remain green. The final contributor matrix covers inference capabilities, initialization, writes, learning, batching, bytes, manifests and checkpoints. The separate correctness repairs are documented in their own tickets/PRs; preservation does not mean restoring known pre-repair defects. This completion is a bounded architecture/extension baseline, not a new scientific validation, factor family, speedup, public API stabilization or production-release claim. Remaining parallel work includes cff-xabn learning persistence, cff-wbhu throughput, cff-rqqc stochastic Colony and cff-3kdk readout vectorization. cff-o7hs is the next new-family consumer; it is not started as part of this epic.
