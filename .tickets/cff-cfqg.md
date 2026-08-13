---
id: cff-cfqg
status: open
deps: [cff-3kdk, cff-cj5h, cff-j7mm, cff-209l, cff-5edo, cff-11jj, cff-ws3c]
links: [cff-bpgm, cff-uy48]
created: 2026-08-13T03:59:30Z
type: epic
priority: 1
assignee: coopwilliams
tags: [trustworthiness, engine]
---
# Epic: Make the existing engine trustworthy to operate and extend

## Background in Plain English

CFFG has strong mathematical and numerical tests, but several public workflows
still expose avoidable trust gaps. Some readouts and planning loops do not yet
meet their measured runtime targets. The resource estimator does not describe
the compiled non-discrete arenas accurately. Ordinary discrete and Gaussian
factor execution still crosses central type-name branches. The rSLDS fitting
workflow also bypasses part of the declared-learning API and keeps generally
useful solver coordination inside a model-specific module.

This epic is the release gate for those existing-engine problems. It connects
the six operational tickets already selected by the 2026-08-09 trajectory
survey with two narrow learning and fitting tickets and a bounded
compiler/engine modularization sequence. It does not add a new factor family
or claim that CFFG is a universal solver. It also does not replace the separate
paper-faithfulness and rSLDS scientific-evidence work.

## Outcomes / Acceptance Criteria

1. Every ticket in this epic's dependency metadata is closed because its own
   acceptance criteria are satisfied.
2. Public posterior readouts perform their numerical work in shape batches and
   preserve their documented support, dtype, and improper-belief behavior.
3. The public resource estimator reports the supported discrete, Gaussian,
   exponential-family, hybrid, and compiled-executable costs within its named
   evidence bounds, and reports unsupported estimates as unavailable rather
   than as confident numbers.
4. Ordinary discrete and Gaussian factor execution uses the registered kernel
   contract across compilation, messages, objective terms, observations,
   readouts, serialization, and resource estimates.
5. Every parameter family fitted by the current rSLDS workflow uses the
   declared-learning contract. Model code does not discover statistical
   meaning from factor-name prefixes.
6. Repeated inference, retry, replacement, and terminal-certification code is
   moved out of an rSLDS module only when a second model fixture proves the
   same contract. Model-specific initialization and diagnostics remain
   model-specific.
6.1. Compiler planning and lowering have typed one-way owners after
     characterization tests pin the existing arena and address contracts.
6.2. Engine execution, schedule control, objective assembly, and readouts have
     focused owners after characterization tests pin the existing numerical
     and status contracts.
7. Single-engine and colony loops preserve per-instance termination status and
   meet the throughput evidence required by `cff-wbhu` before long-rollout
   planning relies on them.
8. Long-rollout planning derives its budget from compiled depth, gates actions
   on a successful status, and records the measured warm- and cold-start
   behavior required by `cff-5edo`.
9. Public epistemic diagnostics make absent, inactive, and numerically invalid
   information-gain channels distinguishable.
10. The completed work preserves literal zero support, dtype-scaled guards,
    loud non-finite failure, true node-and-edge objective accounting, and the
    source-of-truth role of serialized templates.
11. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` agree on the
    delivered runtime envelope, unsupported combinations, and stable public
    entry points.
12. Completion of this epic is described as a trustworthy operating and
    extension baseline. It is not described as proof of every construction in
    the CFFG papers, rSLDS superiority, or universal factor coverage.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Use each dependency ticket's named change-path row and duplicated-surface
  checklist. This epic should integrate their evidence rather than duplicate
  their implementations.
- Preserve pre-change numerical and serialization artifacts before moving a
  dispatch, learning, or fitting boundary. Compare the proposed head with those
  artifacts after the move.
- Keep model recipes in `cffg.models`, factor semantics in registered rule or
  learning families, and generally reusable runtime policy outside model
  modules only after two consumers establish the common contract.
- Treat `cff-ahps` as the downstream finite integration baseline. It must not
  close by counting a model-specific rSLDS learner as generic learning support
  or by adding another family instead of satisfying a missing surface.
- Use this compact dependency view when the roadmap changes:

  ```text
  Legend: prerequisite --> dependent

  [ ] cff-3kdk (ready) ------------------------------------+
  [ ] cff-cj5h (ready) ------------------------------------+
  [ ] cff-j7mm (ready) ------------------------------------+
  [ ] cff-209l (ready) ------------------------------------+
  [ ] cff-wjqg (ready) --> [ ] cff-xdip --> [ ] cff-8whp   |
                                              --> [ ] cff-5edo
  [ ] cff-wjqg --> [ ] cff-0fw2 --> [ ] cff-wbhu ----------+
  [ ] cff-209l --> [ ] cff-edoz ----------------------------+
  [ ] cff-wbhu + cff-cj5h + cff-3kdk + cff-edoz            |
                                --> [ ] cff-ws3c ------------+
  [x] 6tek + 4apw + lquu --> [ ] cff-dqk2 (ready)          |
                                --> [ ] cff-11jj -----------+
                                                               |
                                                               +--> [ ] cff-cfqg
                                                                        |
                                                                        +--> [ ] cff-ahps

  Parallel scientific path:
  [ ] cff-akmp (ready) --> [ ] cff-64ri --+
                                              +--> [ ] cff-0l31 --> [ ] cff-oyie --> [~] cff-nkbj
  [ ] cff-dqk2 (ready) --> [ ] cff-11jj -----+
  ```

## Future work

- `cff-dqk2` owns model-independent declared-learning adapters for the existing
  `gaussian_mixture_linear` and `recurrence_link` factors.
- `cff-11jj` consumes those adapters in both rSLDS learning cadences and
  separates only proven reusable fitting coordination from model policy.
- `cff-3kdk`, `cff-cj5h`, `cff-j7mm`, and `cff-209l` retain their existing
  acceptance criteria. The `cff-wbhu` and `cff-5edo` milestones delegate their
  former bundled criteria to dependency-ordered child tickets. This epic
  supplies their common trustworthiness gate; it does not absorb their
  implementation scope.
- `cff-edoz` owns typed compiler planning and lowering boundaries after
  `cff-209l`. `cff-ws3c` owns the engine execution, objective, and readout
  extraction after the affected public contracts are characterized.
- `cff-uy48` is the separate paper-faithfulness and minimum-usable-release gate
  recommended by `cff-bpgm`; it remains parallel to this epic.
- Nonlinear, neural, balanced-tree recurrence, structure-learning, and new
  model-family work remain downstream or parked under their existing tickets.

## Definitions

- **Arena:** Shape-grouped runtime arrays and compiled index plans derived from
  a serialized graph template.
- **Declared learning:** A template names each learnable parameter group and
  conjugate family so the compiler and `Learnables` runtime can find its
  statistics, posterior, checkpoint values, and parameter targets.
- **Engine trustworthiness:** The bounded claim that documented engine
  workflows report truthful values and statuses, preserve their mathematical
  contracts, and can be extended through the stated interfaces.
- **Factor family:** One mathematical factor type together with its messages,
  objective term, parameters, and supported observations or readouts.
- **rSLDS:** Recurrent switching linear dynamical system.
- **Runtime status:** The explicit converged, budget-exhausted, diverged, or
  infeasible result returned by an inference run.
- **Shape batch:** A group of runtime items with compatible array shapes that
  can be evaluated together instead of one item at a time in Python.
- **Universal solver:** A system claiming executable support for arbitrary
  factor, constraint, and approximation combinations. This epic makes no such
  claim.

## Notes

**2026-08-13T04:37:28Z**

Roadmap refinement, 2026-08-12. Legend: prerequisite --> dependent.

[ ] 209l (ready) --> [ ] edoz --+
[ ] wjqg (ready) --> [ ] 0fw2 --> [ ] wbhu --+--> [ ] ws3c --> [ ] cfqg
[ ] 3kdk (ready) ----------------------------+
[ ] cj5h (ready) ----------------------------+
[ ] wjqg --> [ ] xdip --> [ ] 8whp --> [ ] 5edo --> [ ] cfqg
[ ] dqk2 (ready) --> [ ] 11jj ---------------------------> [ ] cfqg
[ ] j7mm (ready) ----------------------------------------> [ ] cfqg

Immediate blockers are shown by the first open prerequisite on each path. New family implementation waits downstream of cfqg.

**2026-08-13T04:40:34Z**

Authoring handoff, 2026-08-12: cff-edoz and cff-ws3c add the characterization-first compiler/engine modularization sequence. cff-wbhu and cff-5edo now delegate their bundled ordinary/mixed-controller and status/warm-start work to ordered children. Ticket scans were reviewed; dependency-cycle and diff-hygiene checks passed. No pytest run was required for ticket/document-only changes.
