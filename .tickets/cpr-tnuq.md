---
id: cpr-tnuq
status: closed
deps: []
links: []
created: 2026-11-22T02:01:05Z
type: bug
priority: 2
assignee: coopwilliams
tags: [premerge, training, serialization]
---
# Export every resolved blob when saving a neural trainer bundle

## Background in Plain English — WHY

A public NeuralTrainer can use a declared initializer to supply a missing prior mean. On PR 186 the fit accepts this model, then saving even the initial trainer state raises BundleError because the bundle is missing that same mean. PR 157 introduced the training/export path.

## Outcomes / Acceptance Criteria — WHAT

1. A valid trainer created from a template with an initializer-only prior mean saves and reloads a self-contained bundle.
2. Restoring the bundle reproduces the resolved arrays and trained weights, including an initializer that depends on a random key where supported.
3. Explicit caller overrides and previously supported bundles retain their behavior.

## Methodological Suggestions — HOW

Use the Learnable or checkpoint feature row of docs/ARCHITECTURE.md. Compare current and previously supported bundle decoding. Host/JIT, flooding/scheduled and single/vmap execution are N/A for the export inventory. Inspect training.py save_bundle, resolution/sources.py and bundle.py. Compare initializer-resolved, caller-supplied and trained arrays at save/load. Verify a fresh consumer can rebuild the model.

## Future work

Independent of the fixed-model reactor pilot. No new trainer or bundle format is requested.

## Definitions

A blob is a named parameter array in the template. An initializer supplies its value when the caller does not. A bundle packages a model with its resolved data.


## Notes

**2026-11-22T02:02:35Z**

Initializer-only m0 reproduced at PR 157 head 02344ba and at 80ae1f2 with the public trainer. Export path begins in 82765a7. See ~/agent-share/cffg-premerge-review/early/initialized_trainer_bundle.py and observed.txt.

**2026-11-23T03:13:45Z**

The original one-observation reproducer converged and failed export for missing m0 before the fix. NeuralTrainer now resolves missing arrays on the expanded template at construction, keeps explicit override dtypes, and passes the same values to compilation and bundle export. Export overlays the weights from the requested optimizer step. The version-1 checkpoint identity continues to hash the original supplied arrays. The model bundle format is unchanged.

Focused validation on review/premerge-195: the original reproducer creates a bundle. The new regression saved and loaded initial and Adam-updated weights with an initializer-only mean, a seeded random_spd prior covariance, and an explicit float32 override while initializer evaluation was disabled after construction. It passed. Existing float32 and float64 strict checkpoint/bundle continuation and exhausted-adjoint tests passed (3); existing self-contained, missing-array, and extracted-plate bundle tests passed (3). A compact two-copy neural plate preserved a private per-copy initializer and float32 override through export/reload. A version-1 checkpoint created with pre-fix training.py on the partial-input fixture loaded with the same identity. No outstanding ticket work. Parent owns integrated suite and merge.
