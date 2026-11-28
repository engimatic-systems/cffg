---
id: cff-5k0n
status: closed
deps: []
links: []
created: 2026-09-25T03:46:25Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Measure physical-mean training setup and derivative costs on a retained fixture

## Background in Plain English

The agent reports 175 seconds constructing a trainer, 105 seconds for the first step and 5-10 seconds per warm step on 60 RK4 transitions. These are shared-machine observations, not an identified bottleneck. Optimization needs a reproducible breakdown and an independently checked gradient.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. A deterministic 60-transition physical-mean fixture is retained without PCGym or saved pickle dependencies.

2. Measurements separately report constructor/reference inference, derivative compilation, first step, warm step, forward inference and adjoint solve.

3. The report records hardware, versions, dtype, shapes, synchronization, repetitions and convergence statuses; concurrent-load timings are not used as fixed acceptance thresholds.

4. A small identifiable fixture checks directional derivatives independently; reduced objective alone is not claimed to recover non-identifiable constants.

5. The optimization ticket receives measured dominant costs and preregistered target workloads without changing objective or stopping semantics.

6. Results and reproduction command live in this ticket; no new research markdown file is created.

## Methodological Suggestions

Inspect training.py's certified reference solve and implicit.py's fixed-point derivative path. Architecture: Learnable or checkpoint feature; no semantic change intended. Pairs host/JIT derivative check; flooding/scheduled reference identified; single/vmap N/A unless used by fixture; checkpoint continuation control. Hostile zero/vacuous observations, float32 and larger transition count are measured or explicitly rejected.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-csmz — Remove measured redundant work from physical-mean training.

## Definitions

Adjoint solve: the linear solve used to propagate derivatives through a fixed point. Directional derivative: change along one parameter direction. Warm step: a training step after compilation and one-time setup.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-28T04:24:31Z**

Pre-implementation handoff for a774a54. Change path is Learnable or checkpoint feature. The benchmark runs the shipped trainer and inference unchanged. It measures the constructor reference solve, public first/warm Adam step, compiled certificate and fused derivative. A separate diagnostic splits the same matrix-free adjoint from the surrounding gradient operations and checks its result against the public derivative. The scalar oracle compares the scheduled/JIT path with a host Gaussian marginal calculation and checks current checkpoint continuation. Single/vmap comparisons are N/A for a fixed tied batch. No IR, bundle codec, ledger, schedule, stopping, or learning API changes.

**2026-11-28T04:29:44Z**

Frozen before the first 60-transition measurement. The retained graph has 60 independent four-coordinate Gaussian input priors and nonlinear transitions, one shared four-parameter blob, and two observed coordinates per transition. Use the CSTR RK4 mean from tools/cstr_runtime_fixture.py, four substeps over 26/60 minutes, with deterministic NumPy-generated concentration, temperature and coolant conditions. Original training observations, exact invocation, and working configuration were not retained. The new fixture uses linearization, float64, a budget of 50 inference iterations at tolerance 1e-8, 20 adjoint restarts with Krylov dimension 32, and step size 0.002. Time the public constructor and step with the actual forward/reference, certificate and fused derivative phases. A separate synchronized replay times the same GMRES system with split executables, checks its gradient against the public fused derivative, and stays outside public step totals. The scalar oracle uses five identifiable cooling transitions at float32/float64. Standalone tooling changes no package or test behavior and require the focused procedure only.

**2026-11-28T05:04:04Z**

Measured frozen source c1f9a60 on Apple M2 Max Mac14,6, 64 GiB, macOS 15.7.5, Python 3.12.13, JAX/jaxlib 0.10.2, NumPy 2.5.1, CPU float64 and disabled persistent cache. Held the registered CPU lock. The 60 by 4 input and 60 by 2 observation arrays produce a mixed schedule with 1680 root coordinates and one trained four-value blob read by all 60 transition factors. Arena.n_slots in the JSON counts discrete messages. The timings synchronize JAX results. The first step is followed by one warmup with changed parameters and three distinct warm steps. Receipt /Users/ravd/agent-share/cffg-physical-mean/retained-c1f9a60.json, SHA-256 f2daa6e4548f199b3cecf0c387a884f5484000d290c3471a90a0fb2d7c143c58. Reproduction from this worktree: python3 ~/agent-share/cffg-backlog-gates/with_cpu.py uv run --frozen --no-sync python tools/benchmark_physical_mean_training.py --case retained --dtype float64 --warm-repetitions 3 --output /path/to/fresh.json. The lock wrapper is for shared CPU scheduling and the Python benchmark can run directly.

Constructor 12.854 s inclusive, reference inference 11.491 s inclusive (8.855 s logged XLA compilation). First step 18.870 s inclusive, forward 0.072 s, derivative certificate 2.673 s (1.948 s XLA) and fused derivative 15.984 s (14.004 s XLA). Warm median 76.250 ms, forward 56.995 ms, derivative certificate 4.356 ms and fused derivative 11.832 ms. The separate synchronized GMRES replay compiled in 5.270 s, ran in 6.685 ms warm and recombined to exactly the production gradient. Its first wall was 5.545 s. Split executables have different fusion, so the replay is not part of the measured public step. All five updates applied, reference and each step converged, max derivative root residual 1.22e-14, max adjoint residual 2.96e-15. Use the current locked workload as a baseline. The historical reported times used a different revision, original observations and shared-machine conditions.

Focused validation: uv run --frozen --no-sync python tools/benchmark_physical_mean_training.py --case oracle --dtype float32 --output /path/to/fresh-f32.json and the same with float64. The independent Gaussian marginal yielded rank-two mean sensitivities and max directional errors 1.65e-6 and 7.11e-15 respectively. Public finite difference, bitwise exact checkpoint continuation at step two, and a zero-observation gradient within dtype tolerance passed. File names oracle-f32-dev.json and oracle-f64-dev.json under the same artifact root. The oracle scripts ran during development for correctness, outside the locked performance measurement. All run inputs are generated and include no held-out data. Scoped syntax and diff checks and tk dep cycle passed. Standalone benchmark only with ticket updates; no runtime, dependency, suite discovery or report aggregation changes, so the isolated-tooling validation applies.
