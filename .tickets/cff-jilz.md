---
id: cff-jilz
status: open
deps: [cff-enhs]
links: []
created: 2026-08-31T03:16:28Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-cz6v
tags: [serialization, learning, validation]
---
# Prove CFFGZ learning resume across shipped families

## Background in Plain English — WHY

A family-independent archive layout is not sufficient evidence that learning resumes
correctly. The repository must show that loading a bundle preserves the next
posterior update and installed point parameters for the five family entries in
`cffg.learn.FAMILIES` on base commit `ca87069`. It must also show users a
representative archive that contains the
new learning state.

Ticket `cff-enhs` provides the format and API. This ticket supplies integration
evidence, builder/export handoff, and canonical documentation at the complete
workflow boundary.

## Outcomes / Acceptance Criteria — WHAT

1. A newly initialized learnable model can be saved, loaded without external
   priors, updated from a later converged trial, and saved again.
2. For Dirichlet, NIW, MNIW, component MNIW, and Pólya–Gamma weights, an
   uninterrupted learner and a save/load-resumed learner produce equal next
   posterior hyperparameters, reporting means, rollover values, and installed
   point parameters with equal dtypes, shapes, and values under
   `numpy.testing.assert_array_equal`.
3. The comparison covers both distinct-trial accumulation and one-sequence
   replacement from immutable priors.
4. A representative rSLDS bundle covers initial-regime Dirichlet, component-
   MNIW dynamics, tied MNIW emissions, and D9 Pólya–Gamma recurrence. A separate
   Gaussian-prior fixture covers NIW.
5. D9 recurrence restoration uses the family-independent semantic descriptor and retains
   the same rejection boundary for D8, unidentified, incomplete, tampered,
   wrong-shape, and wrong-dtype state.
6. Scoped learnable names, tied sources, multi-blob targets, and plate-expanded
   names round-trip without changing ownership or pooling behavior.
7. Exact-zero statistic contributions remain exact after resume. Improper or
   non-finite prior/posterior arrays fail before mutation. Float32 state and
   point arrays retain dtype and finite next-update behavior.
8. A fixture with at least 64 declared learnables demonstrates that resume does
   not depend on family-specific archive code or unscoped names.
9. The test introduced by `cff-yoxz` regenerates
   `tests/.artifacts/cffgz/representative-rslds.cffgz` as a schema-current,
   learning-resumable specimen and validates its complete inventory.
10. Original model-only specimens and hand-built incomplete schema-current
    archives exercise the documented unsupported-version and incomplete-state
    refusal rules. Cooper waived legacy bundle preservation on 2026-09-13.
11. `README.md` gives a complete save, load, inference, update, and resave
    example. `docs/ARCHITECTURE.md` identifies each state owner.
    `docs/LIMITATIONS.md` states that `.cffgz` resumes at a committed learning
    boundary, while exact mid-run and agent replay still require checkpoints.
12. Focused bundle, learnable, recurrence-checkpoint, and agent-boundary tests
    pass. One final short-suite run passes after the last code change.

## Methodological Suggestions — HOW

The implementer may choose another sound proof structure. Compare two learners
from one common pre-save state: let one update directly and let the other save,
load, then consume the identical converged result. Compare named state rather
than ZIP bytes because creation timestamps may differ. Use the existing rSLDS
specimen for four families and the existing ordinary Gaussian declared-learning
fixtures for NIW.

Use the **Learnable or checkpoint feature** change-path row. Review template
declarations, compiler learn plans, family validators, bundle and checkpoint
encode/decode, legacy migration, point-parameter writes, and public exports.
Check schema-current/legacy persistence directly. Learning updates remain host-side;
host/JIT is relevant to the inference result feeding the update,
flooding/scheduled statistics must agree where both execute, and the 64-member
fixture covers grouped or scoped state without introducing a new `vmap`
mutation path.

The numerical robustness contract is exercised through criterion 7 and the
existing family statistic tests. Do not create new probability floors, ridges,
or clamps in serialization code.

## Future work

Closing this ticket unblocks epic `cff-cz6v`. Open ticket `cff-9vab` depends on
that epic so balanced-tree rSLDS integration uses the declared-learning
resumable bundle instead of a topology-specific learning archive. Neural optimizer state,
executing hierarchy state, and per-patient exact replay remain owned by linked
tickets `cff-1o72`, `cff-pyy9`, and `cff-t2g3`.

## Definitions

- **Next-update equivalence:** Posterior and installed parameter arrays with
  equal dtypes, shapes, and values under `numpy.testing.assert_array_equal`
  when uninterrupted and restored learners consume the same later evidence.
- **One-sequence replacement:** Recomputing a posterior from the immutable
  prior and one sequence's current sufficient statistics.
- **Registered family:** One entry in `cffg.learn.FAMILIES` accepted by the
  declared-learning compiler.
- **Tied source:** One of several factor instances that contributes statistics
  to a posterior shared through common parameter blobs.

## Notes

**2026-08-31T03:25:27Z**

Retained bounded-scan candidates and intended meanings: canonical=the six project documents as authorities; integrate=builder-to-save/load-to-update public workflow or balanced-tree downstream integration; preserve=value identity for the next update; public=exported/documented API; register=the five runtime family entries named at ca87069; same=unchanged rejection boundary or identical later evidence; provide=cff-enhs ticket promise; complete=exhaustive specimen inventory or the listed consumer workflow; remaining=preserved literal zeros, host-side execution, or deliberately excluded runtime state; close=acceptance-based ticket and epic completion; sufficient=mathematical sufficient statistics; check=automated schema-current/legacy persistence comparison; converge=RunResult converged status; correct=contract conformance established by named comparisons; equivalent=bitwise next-update equality; evidence=automated workflow result or observations added to a posterior; exact=literal zero, bitwise array identity, or exact runtime replay boundary; pass=tests-succeed and short-suite success; robust=the named hostile-input contract; sound=reasonable proof structure; test=automated artifact, fixture, or execution; validate=artifact inventory validation; accumulate=statistics from a distinct later trial; infer=fixed-parameter state inference; learn=new-evidence accumulation or integrated workflow; learnable=serialized declaration or registered fitting family; rollover=point-parameter installation; trial=independent dataset; can=required demonstrated workflow; may=possible timestamp difference; explicit_owner=the linked tickets named in Future work; at_least=inclusive 64-learnable lower bound; only=legacy model-only scope; current=runtime-active one-sequence statistics or schema-current refusal; existing=the named rSLDS, Gaussian, and family-statistic fixtures on ca87069; later=ordered independent evidence. Capability ownership was checked separately: this ticket owns integration evidence and user workflow, not schema or ZIP mechanics.

**2026-08-31T03:25:56Z**

Post-handoff scan additions: fixed=the fixed-parameter inference input or legacy artifact; appropriate=the named integration-proof boundary; fit=one-sequence or registered-family parameter estimation; intended=the locally stated classification meaning.

**2026-08-31T03:26:14Z**

Final classification addition: estimate=registered-family parameter estimation.

**2026-09-13T15:23:37Z**

Alpha policy clarification from Cooper: legacy bundle preservation is not required at this version. Criterion 10 now verifies unsupported-schema refusal. cff-1xig tracks the compatibility and migration contract required before stabilizing the bundle format.
