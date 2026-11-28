---
id: cff-3wsw
status: in_progress
deps: []
links: []
created: 2026-09-25T03:46:25Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Expose physical-function point training without neural-specific registration folklore

## Background in Plain English

NeuralTrainer can optimize physical constants when a registered mean function advertises network_parameters, but that undocumented neural-named metadata was discovered by reading code. Model authors need an honest public path for non-neural point training, including its lack of posterior uncertainty.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. A documented public registration contract admits a non-neural physical mean without relying on an unexplained magic metadata key.

2. A standalone physical-mean example demonstrates tied parameter updates, objective evaluation and status handling.

3. Existing neural registrations remain compatible or have an explicit migration path.

4. Checkpoint/resume preserves point parameters and optimizer continuation and never presents them as Bayesian uncertainty.

5. Validation covers malformed parameter shapes, non-finite constants, float32, identifiable/unidentifiable examples and a larger tied plate; invalid inputs fail explicitly.

6. README and LIMITATIONS distinguish expressive nonlinear dynamics, point learning and the separate finite-bank posterior capability.

## Methodological Suggestions

Inspect trainable.py:NeuralParameters.from_engine and existing MeanImplementation metadata. Architecture: Learnable or checkpoint feature; New IR field only if the registration identity changes. Pairs host/JIT parameter/gradient parity; flooding/scheduled inference reference; single/vmap tied member semantics; current/legacy checkpoint required.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

No additional future work is identified within this ticket; the completed outcome contributes to epic cff-xa6z.

## Definitions

Point estimate: one fitted parameter value rather than a distribution. Tied parameters: several factors sharing one value. Identifiability: whether the observations distinguish parameter values. Optimizer: the procedure that changes parameters to reduce an objective.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-28T04:20:19Z**

Learnable or checkpoint feature in docs/ARCHITECTURE.md. Thread an explicit opt-in from the host mean registration to its configured Gaussian binding, then reuse the existing tied-blob parameter layout, implicit certified derivative and Adam checkpoint. The function declaration and serialized template identity stay the same. Check the host objective and compiled derivative against a closed-form positive-noise cooling experiment, single versus tied plate rows, old neural capability and current checkpoint continuation. Flooding/scheduled message dispatch and conjugate learnable tap/bundle schemas have no changed numerical rules. A zero elapsed-time experiment identifies no rate. Cover it, wrong shapes, nonfinite constants, both dtypes, and a tenfold plate. Sparse categorical support and improper internal priors do not occur in this positive-variance Gaussian fixture. Keep parameter posterior probabilities on the separate finite-bank API.

**2026-11-28T04:26:31Z**

Ready for the combined review gate. Model authors now register MeanImplementation(..., point_trainable=True) and use MeanTrainer on the same certified mean, gradient, and Adam code as NeuralTrainer. An explicit false flag excludes a mean from selection even with the old network_parameters count. Omitted flags preserve legacy registrations. Checkpoints bind point values and optimizer continuation; the separate finite candidate bank supplies posterior probabilities.

Validation used the shared uv environment with PYTHONPATH=src and uv run --frozen --no-sync. Six physical tests in tests/test_physical_mean_training.py passed. A cooling experiment with two float64 or twenty float32 tied factors matched the closed-form marginal likelihood and gradient. Zero elapsed time gave the same objective and zero derivative at two log rates. tests/test_readme.py, tests/test_neural_training.py and tests/test_fixed_point_gradients.py passed all ten default checks; the five catalogue checks passed again after final documentation edits. In a separate process, a version 1 neural checkpoint created from git archive 2bfe84b resumed through MeanTrainer with exactly the original next Adam state and objective (-3.121703170960131, step 2). Running examples/physical_mean_training.py in default float32 gave a single fitted cooling rate of 0.40989 per second after five steps from 0.25, with objective 16.43666 to 2.43441. Dependency cycles and whitespace checks are clean.

The full gate is queued on the shared physical-training and recipes review head. Keep the ticket in progress until that gate passes and the review PR is published.

**2026-11-28T05:14:11Z**

The first combined short suite exposed one omitted public-catalogue expectation in tests/test_public_api.py (2862 passing, one failing). Added the canonical MeanTrainer owner to the sorted facade assertion and ran that test plus all five README checks. All six passed. The next gate is on the corrected head.
