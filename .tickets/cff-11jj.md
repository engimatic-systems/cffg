---
id: cff-11jj
status: in_progress
deps: [cff-dqk2, cff-garu]
links: [cff-nkbj]
created: 2026-08-13T03:59:30Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-cfqg
tags: [trustworthiness, learning, rslds]
---
# Rebase rSLDS learning and fitting on declared engine contracts

## Background in Plain English

The rSLDS graph is already an ordinary composition of registered CFFG factors.
Its learning workflow is not yet equally compositional. `RSLDSLearning` finds
dynamics and recurrence records through `dynamics*` and `switch*` name
patterns, manually owns three posterior families, and installs them separately
from the generic emission learner. `fit_rslds` then coordinates this manual
learner with `Learnables`, owns retry and terminal-run policy, and reads the
global learning-family registry to report emission parameters.

Ticket `cff-dqk2` supplies the two missing generic family adapters. This ticket
uses them to express every parameter family fitted by the current rSLDS
workflow through `LearnableSpec` and `Learnables`. It also decides which parts
of the repeated fitting loop are truly model-independent. Code moves out of
the rSLDS module only when both an rSLDS and a non-switching Gaussian model
exercise the same contract.

The fitter delivered by `cff-l7y4` and stabilized by `cff-garu` updates the
initial **regime** distribution. It does not learn the continuous initial-state
mean and covariance. This ticket preserves that boundary; the PCA/ARHMM
initializer continues to supply the continuous initial-state prior.

## Outcomes / Acceptance Criteria

1. `cff-dqk2` is closed with generic declared-learning support for
   `gaussian_mixture_linear` dynamics and `recurrence_link` weights.
2. `cff-garu` is closed with the current deterministic retry, phase-evidence,
   terminal-solve, and truthful-status behavior available as the parity base.
3. The serialized rSLDS template declares each currently fitted posterior.
   3.1. A Dirichlet learnable owns the categorical initial-regime probability.
   3.2. A component-MNIW learnable owns the switching dynamics matrix, offset,
   and covariance.
   3.3. A Pólya–Gamma learnable owns the recurrence weights and offsets.
   3.4. The existing MNIW learnable owns the tied emission matrix, offset, and
   covariance.
   3.5. The continuous initial-state mean and covariance remain fixed
   initializer outputs and are not described as learned parameters.
4. Parameter tying and statistic sources come from serialized blob references
   and `LearnableSpec.sources`. No rSLDS learning or fitting path selects
   factors through prefixes, suffixes, numeric names, or sorted name order.
   A compact test renames every factor without changing fitted values.
5. Both rSLDS learning cadences use the public `Learnables` API.
   5.1. Distinct-trial learning uses `accumulate` from the current posterior.
   5.2. One-sequence coordinate fitting uses `replace` from immutable priors.
   5.3. All four posterior families update and write atomically for one
   coordinate.
   5.4. `RSLDSLearning` is removed or retained only as a compatibility facade
   that delegates statistics, posterior state, checkpointing, and writes to
   `Learnables`.
   5.5. Model modules do not import or index the global learning-family
   registry to compute reports or rollover parameters.
6. The implementation records a responsibility table for the existing fitting
   loop in `docs/ARCHITECTURE.md`.
   6.1. rSLDS-specific responsibilities include template construction,
   PCA/ARHMM initialization, observation-port binding, and any diagnostic that
   refers to recurrence or Gaussian-mixture beliefs.
   6.2. Candidate reusable responsibilities include status-gated inner runs,
   deterministic retry from an identical state, common attempt evidence,
   fixed-point comparison over declared posterior and point values, and a
   terminal solve after the last write.
   6.3. Each responsibility is labeled model-specific or reusable with the
   concrete evidence for that decision.
7. Code is moved to a shared fitting module only when one rSLDS fixture and one
   ordinary non-switching Gaussian declared-learning fixture use the same
   interface and match their direct pre-extraction loops on the values and
   tolerances listed in criterion 9.
   7.1. The shared interface accepts an `Engine`, status-bearing results,
   declared learnables, immutable priors, a parameter-bound starting state,
   and explicit iteration, tolerance, damping, and retry limits.
   7.2. The shared interface does not inspect factor names, factor types,
   rSLDS result classes, recurrence status, or Gaussian-specific readouts.
   7.3. Family- or model-specific evidence is attached through an explicit
   diagnostic callback or remains in the model wrapper.
   7.4. A block used only by rSLDS remains in the model layer and is not labeled
   generic merely to shorten the file.
8. Retry and terminal-certification behavior remains falsifiable.
   8.1. Only a finite `budget_exhausted` deterministic attempt may use the
   bounded damping-retry policy.
   8.2. Every retry starts from the identical parameter-bound state and
   preserves the fixed point, support, schedule, tolerance, and convergence
   certificate.
   8.3. Diverged, infeasible, or non-finite attempts do not retry.
   8.4. A stochastic engine state, including a functional `mu3` update, is
   rejected by this retry policy unless an explicit key and sample-reuse
   contract proves that retries compare the same stochastic realization.
   8.5. The returned state is inferred after the final parameter write and
   agrees with the returned point parameters.
9. Compact parity evidence covers the refactor.
   9.1. Existing rSLDS initialization, one-coordinate replacement, repeated
   replacement, terminal solve, phase evidence, parameter shapes, posterior
   values, runtime statuses, and free-energy traces retain their pre-refactor
   values within dtype-scaled tolerances.
   9.2. Distinct-trial accumulation retains the existing direct conjugate
   oracle values.
   9.3. The non-switching Gaussian fixture matches a direct loop that calls
   `Engine.run`, `Learnables.replace`, and `Learnables.write` without the
   shared coordinator.
   9.4. Flooding and scheduled inference supply equivalent statistics before
   host-side learning under their existing tolerance.
10. Hostile-input evidence covers literal zero selector support, an unused
    regime, vacuous and improper beliefs, non-finite free energy or posterior
    values, float32, and a horizon at least ten times the smallest fitting
    fixture. Every case either completes with its documented values or returns
    a truthful non-converged status without partial parameter mutation.
11. Public compatibility is explicit.
    11.1. Existing documented rSLDS construction and fitting imports either
    continue to work or receive a documented deprecation and migration path.
    11.2. Checkpoints round-trip the declared posterior state without a second
    model-specific checkpoint schema.
    11.3. `RSLDSFitResult` continues to separate inner Bethe free energy from
    the outer fixed-point residual and records every attempted run.
12. The refactor leaves each source file within the project line-count rule and
    gives initialization, model assembly, shared fit coordination, and
    rSLDS-specific result policy separate owners. It does not move code solely
    to evade the limit.
13. Inference messages, factor energies, objective-ledger meaning, recurrence
    topology, serialized model meaning, and the source-specific rSLDS
    scientific claims remain unchanged.
14. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` describe the
    declared rSLDS learning workflow, the shared-versus-model-specific fitting
    boundary, deterministic retry limit, fixed continuous initial-state prior,
    and remaining unsupported stochastic fitting behavior.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Use the **Learnable or checkpoint feature** and **Runtime numerical or status
  invariant** rows in the architecture change-path index.
- First replace manual rSLDS posterior and write code with `Learnables` while
  keeping the existing direct loop. Capture parity before extracting any fit
  coordination.
- Use a small learned linear Gaussian state-space model as the second client.
  Its analytic conjugate updates make a clearer reuse oracle than another
  switching-model variant.
- Split the PCA/ARHMM initializer from fit coordination along its existing
  protocol boundary. Keep phase diagnostics close to the result type or behind
  a callback rather than teaching a generic controller about recurrence.
- Prefer a small internal coordinator over a broad public framework. Promote it
  only after another real repository consumer needs the same stable API.

## Future work

`cff-0l31` freezes the confirmatory rSLDS protocol only after this ticket, so
the final scientific run evaluates the declared-learning production path.
Tickets `cff-oyie` and `cff-nkbj` retain the confirmatory run and conclusion.

This ticket does not learn the continuous initial-state Gaussian, add balanced
recurrence trees, change the rSLDS approximation, support stochastic `mu3`
fitting, or create a general non-conjugate optimizer. Each change would require
a separate mathematical and runtime contract.

## Definitions

- **Compatibility facade:** A small wrapper that preserves an older public API
  while delegating its behavior to the new implementation.
- **Declared learning:** A template-defined parameter-learning group executed
  through `Learnables`, including statistic sources, posterior state, writes,
  and checkpoints.
- **Deterministic retry:** Running inference again from the same state and
  parameters with a changed damping value but no changed random sample.
- **Fixed-point residual:** A scale-aware measure of change between consecutive
  posterior and point-parameter coordinates.
- **MNIW (Matrix-Normal-Inverse-Wishart):** A conjugate posterior for a linear
  matrix, offset, and covariance.
- **`mu3`:** The paper's functional message to an uncertain observation model,
  approximated in this repository with explicit random sampling.
- **PCA/ARHMM initializer:** An observation-only initializer that uses principal
  components, transition-pair clustering, and local autoregressive fits.
- **Pólya–Gamma learnable:** The conjugate Gaussian natural-parameter posterior
  over recurrent stick-breaking weights and offsets.
- **rSLDS:** Recurrent switching linear dynamical system.
- **Terminal solve:** Inference run after the last parameter write so returned
  beliefs and returned parameters describe the same coordinate.

## Notes

**2026-08-13T04:07:32Z**

Planning and ambiguity handoff, 2026-08-12. Applicable change paths: Learnable or checkpoint feature and Runtime numerical or status invariant. The duplicated surfaces are explicit in the outcomes: flooding/scheduled statistics, direct/shared host coordination, current/legacy checkpoints, deterministic/stochastic retry, compact/10x-horizon behavior, and rSLDS/non-switching-Gaussian consumers. Single-instance/vmap fitting remains N/A unless this ticket adds a batched contract.
