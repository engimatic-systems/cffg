---
id: cff-iplr
status: closed
deps: [cff-wi2o]
links: []
created: 2026-09-25T03:46:26Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Implement the validated nonlinear parameter-particle learning contract

## Background in Plain English

After the particle design establishes its statistical and persistence contract, users need a reusable implementation for nonlinear model uncertainty. This work extends the successful finite-bank route without substituting point training or undocumented heuristic jitter.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The public API implements exactly the approved v1 target, resampling and rejuvenation contract and rejects unsupported targets explicitly.

2. Explicit random keys reproduce draws and continuation; no hidden global random generator is used.

3. Sequential observations are counted once and posterior/evidence comparisons pass the independent design oracles.

4. Degenerate weights, exact-zero support, non-finite likelihoods, float32 and larger particle sets have explicit tested outcomes.

5. A resumed serialized model/data/checkpoint produces the same continuation as an uninterrupted run.

6. An example explains approximation, uncertainty calibration, memory costs and non-identifiability without promising exact continuous inference.

## Methodological Suggestions

Follow the preceding derivation, using existing registered function and graph likelihood authorities where appropriate. Architecture: Learnable or checkpoint feature plus New IR field as dictated by the design. Pairs host/JIT, flooding/scheduled likelihoods, single/vmap and current/legacy checkpoint required.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Target: the declared posterior distribution. Degenerate weights: probability mass concentrated on too few or no valid particles. Continuation: subsequent learning after restoring saved state.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-28T05:01:55Z**

Bounded v1 from D6d: independent positive beta shapes on finite box intervals, identifiable versioned mean implementation, known SPD Gaussian observation covariance, explicit applied inputs, missing event is a no-op. Carry normalized host log weights; calculate source-dtype native normalized likelihood, deterministic-policy ESS stratified sampling and a fixed transformed-space MH kernel on BOTH paths. Retain all observed events and the complete key/counter; any unsupported target fails before consuming the key. Freeze continuation bit parity under same backend. Gaussian 64-node reference absolute budget 64 dtype eps; two-mode 128/1280 particle checks on 48 declared seeds use <=6 empirical SE and abs bounds [.05 times Z,.03,.03,.05] for Z, mean, second moment and side mass. 128 newly simulated data sets check 4-SE calibration of median and mean posterior rank. Include zero finite population vs provable impossibility, cross-method support, corrupted covariance/model/version/keys and a restart across a resampling event. Exact posterior or evidence is not claimed for a single continuous run.

**2026-11-28T05:59:47Z**

Implementation handoff. Use Architecture Learnable or checkpoint feature and Runtime numerical or status invariant. Public standalone posterior retains registered mean identity/configuration, bounded beta prior, known Gaussian observation covariance, explicit transformed particles/key and applied histories. Host owns atomic commits, normalized log weights and evidence. The compiled single/vectorized exact Gaussian conditional is shared with the observed Gaussian node; verify against a graph point likelihood and closed-form/independent quadrature. Engine flooding/scheduled state transitions, IR/bundle migration and Colony execution are N/A for a standalone conditional target. Validate versioned checkpoint continuation, reject older/unknown formats, numerical range and nonfinite scores before any state change. With finite x/y/means and proper SPD Gaussian noise, the true conditional has no exact zero; an overflow is an evaluation failure, not evidence of impossibility. Retain log-space support handling in the weight fold.

**2026-11-28T06:10:02Z**

Implementation qualification. The standalone conditional evaluates the registered mean directly at each physical parameter and uses native normalized Gaussian point density. It applies the same transformed-space MH policy on both ESS branches, conditions each reading once, and retains all applied history. It blocks unrepresentable arithmetic without changing state or key. A beta prior concentrated at a bound can produce physical values that round outside the representable open interval and are refused. The public Gaussian conditional has positive true support; exhausted population and invalid score are distinguished in the weight fold.

The initial direct Gaussian ensemble on seeds 0–47, ESS .8, one scale-1 move missed the f32/128 mean absolute limit (.045747 > .03) while within 6 SE. Other absolute metrics passed and f64/1280 passed. Qualification repeated with a separately declared policy: both modes are roughly 4 logit units apart, so use four scale-2.5 moves on both ESS branches and fresh seeds 20261128–20261175. Exact Gaussian quadrature Z=7.78985483; absolute f32/128 errors in Z, mean, second moment and positive-side mass: .228080, .003882, .000203, .002787. f64/1280: .009322, .003509, .000069, .002319. Both passed the unchanged limits and 6-SE criteria. The 128 prior-predictive data sets gave posterior rank mean .512822 and median .503215. Receipt /Users/ravd/agent-share/cffg-iplr/qualification.log. Public defaults remain one scale-1 move and ESS .5. Qualification is for the named seeded configurations.

Focused and opt-in checks: 39 passed across public particles, independent D6d references and finite hypotheses, using uv run --frozen --no-sync pytest -q -s -m 'slow or not slow' tests/test_static_parameter_particles.py tests/test_static_particle_derivation.py tests/test_finite_hypotheses.py. An added stratified ancestor oracle then passed both f32/f64 with exact resumed continuation across history growth. The example worked with JAX_ENABLE_X64=0. The combined suite is pending at this commit.

**2026-11-28T14:07:44Z**

The first exact-head full suite at 7ff254a had 2,880 passes and one failing README example-catalogue expectation. The failed log and runtime-failed.json remain in /Users/ravd/agent-share/cffg-iplr/gate-7ff254a-UYuiqq. Commit 900768e registered the example in the catalogue; all five README checks and the x64-off example passed. Final dashboard short gate at 900768e passed 2,881 tests, zero failures, in 435.245 seconds. /Users/ravd/agent-share/cffg-iplr/gate-900768e-mGUBAJ/runtime.json is byte-identical to the tracked runtime history. Scoped diff and dependency-cycle checks passed.
