---
id: cff-yoxz
status: closed
deps: []
links: []
created: 2026-08-31T02:44:46Z
type: task
priority: 1
assignee: coopwilliams
---
# Generate an inspectable representative CFFGZ specimen

## Background in Plain English — WHY

At ticket creation on `agent/cff-4jk1-outer-convergence-port`, the bundle tests
write temporary `.cffgz` files that disappear after the pytest process exits.
Cooper needs a fixed-path specimen to inspect while evaluating a proposal to make
each bundle saved by non-legacy code capable of resuming between-trial posterior
updates. The specimen must reveal the format boundary on that branch: it contains
a rich model topology, declared `LearnableSpec` families, and point-parameter
arrays, while using only the format emitted by `cffg.bundle.save_bundle`. The
generated binary must stay outside version control.

## Outcomes / Acceptance Criteria — WHAT

1. One named pytest test writes
   `tests/.artifacts/cffgz/representative-rslds.cffgz` and checks archive integrity
   by loading it with `cffg.bundle.load_bundle`.
2. The specimen is a recurrent switching linear dynamical system with exactly
   three regimes, two latent dimensions, three observation dimensions, five time
   slices, and the four declarations named `initial_regime`, `dynamics`,
   `recurrence`, and `emission`. Its graph also contains tied parameter blobs,
   five data ports, recurrence factors, switching dynamics, and Gaussian
   emissions.
3. The pytest assertions cover the archive member names and the dimensions,
   counts, family names, factor kinds, and blob ties listed in outcome 2, so
   pytest success rejects a trivial replacement graph.
4. `git check-ignore` confirms the specimen path is ignored, and generating it
   leaves the binary absent from `git status --short`.
5. `tests/test_bundle.py` passes after the change. The final proposed stack head
   passes the repository's required short suite once after the last code change.

## Methodological Suggestions — HOW

Use explicit NumPy arrays so the model is easy to reason about and repeatable.
Create parent directories in the pytest test, save to a temporary sibling, check
that file with `load_bundle`, then replace the fixed-path specimen atomically.
Keep the output under `tests/.artifacts/`, which `.gitignore` already excludes on
the ticket base, and do not add the binary to Git.

The applicable `docs/ARCHITECTURE.md` change-path row is **Learnable or
checkpoint feature**. This task exercises bundle JSON/hash serialization,
`LearnableSpec` declarations, and representative family topology. It does not
change compiler resolution, arena layout, engine parameter writes, agent cadence,
checkpoint encoding or migration, host/JIT execution, flooding/scheduled
execution, single-instance/`vmap` behavior, or numerical rules. Those execution
pairs are structurally untouched, and the hostile-input checklist is not
applicable because the task adds a serializer fixture without changing numeric
runtime behavior.

## Future work

Do not add posterior hyperparameters or resumable-learning metadata in this
ticket. The specimen exists to make that separate bundle-format design concrete
and inspectable.

## Definitions

- **CFFGZ bundle**: the ZIP-based model artifact containing `model.json`,
  `manifest.json`, and hashed NumPy parameter blobs.
- **Specimen**: a fixed-path example produced by pytest for human inspection but
  ignored by version control.
- **Learning family**: a serialized `LearnableSpec` that identifies a conjugate
  parameter family, its target blobs, and its contributing factors.
- **Resuming between-trial posterior updates**: restoring the accumulated
  posterior hyperparameters and sufficient-statistic schema, then incorporating
  evidence from later trials without restarting from immutable priors.

## Notes

**2026-08-31T02:58:32Z**

Implemented a fixed-path representative rSLDS bundle generator in tests/test_bundle.py. Focused validation: 17 passed, 1 deselected. The archive contains model.json, manifest.json, and 11 parameter blobs; load_bundle integrity checks and compilation pass. git check-ignore confirms tests/.artifacts/cffgz/representative-rslds.cffgz is ignored. Final short suite at 7adca36: 1464 passed, 39 deselected, 0 failed in 574.742 wall seconds. No remaining work in this ticket; resumable-learning bundle state remains deliberately deferred.
