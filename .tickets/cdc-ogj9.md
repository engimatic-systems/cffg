---
id: cdc-ogj9
status: in_progress
deps: [cff-y15d]
links: []
created: 2026-09-14T20:25:51Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, performance, numerical]
---
# Measure early D12 compilation scaling and optional output costs

## Background in Plain English — WHY

Cooper asked for early evidence of how D12 compilation scales and whether users can avoid its cost when source recovery is not worthwhile. The tiny checkpoint passed, but its first synchronized calls took about five to six minutes and process samples reached 28.65 GiB. Those calls combine setup, compilation and execution. They cannot establish horizon scaling or the cost specifically attributable to learning. A small prospective compilation experiment should precede the larger qualification harness.

## Outcomes / Acceptance Criteria — WHAT

1. Commit a small machine-readable measurement specification before execution. Compare complete fixed-parameter solves at T=32 and T=100, K=4,d=2, using cff-a3wz's exact fixture formulas, identical qualified solver policy/budgets, dynamic numerical arguments and native float32. Treat this as two measured sizes, not a fitted asymptotic law. Do not use tiny float64 calls as a controlled scaling baseline.
2. Separate tracing/lowering and executable acquisition with retained Lowered/Compiled objects; disclose cache policy and observed reuse. Record source/spec/version/backend identifiers, actual float configuration and leaf dtypes, XLA argument/output/temporary/alias estimates and independently measured process high-water memory. Record compilation-only runs as execution not run. No optimizer warm-up, extra recovery fit, or numerical qualification claim is authorized by this ticket.
3. Prospectively bound an output-selection comparison at one of those sizes: the complete returned state versus a wrapper retaining inference coordinates, energies, certificates, statuses and counters. Preserve the same solver and status dependencies. Use the comparison to test compiler elimination; do not call it an implemented no-learning mode or alter the equations to manufacture a speedup. Preserve failures and resource stops before diagnosis, with no silent retries/tuning.
4. Explain where cost is paid: current unbound D12 versus existing public D9 compilation, fixed-parameter inference versus repeated parameter fitting, per-shape/static-layout compilation versus possible executable reuse. Distinguish measured facts, static call-graph facts, and hypotheses. Identify what optional reporting/export work can be omitted and which proposed savings would change inference or failure semantics.
5. Deliver a compact evidence-based recommendation for the next step and the later cost/benefit comparison with the unchanged registered recovery gate. Explicitly state that recovery quality is not yet established and compilation amortization needs actual moderate execution timing. Existing users must not silently incur D12 compilation; any future public activation remains explicit. A cheaper mode inside D12 is only proposed if its actual trade-off is understood and reviewed.
6. Keep the change to developer measurement/report artifacts and ticket/canonical documentation. Preserve production source, numerical thresholds, test selection/execution and prior evidence. Use a fresh bounded worker; commit the prospective setup and pause for review before the expensive measurements. Reproducible commands and changed-artifact checks must pass. If reusable instrumentation needs production or shared test changes, split that implementation prospectively rather than drifting into cdc-ztuf.

## Methodological Suggestions — HOW

Reuse the instrumentation research in cdc-ztuf and the existing hybrid_joint_storage complete-solve interface. A standalone developer measurement tool may be appropriate; isolate it from public execution, dependencies and the default suite. Assess that isolation explicitly. Prefer retained executables for later authorized qualification starts when possible, without overstating cross-process reuse. Declare practical per-compilation time/process-memory diagnostic limits before running. Retain partial evidence on a limit or compiler failure.

## Future work

cdc-ztuf still prepares genuine raw-zero/feasible/perturbed starts and the twelve-case harness. cff-a3wz owns their execution, float32 convergence, initialization sensitivity and actual iteration-storage qualification. Compiler integration and the exactly three registered recovery fits retain their authoritative dependencies and thresholds. A numerical contract, supported-scope or approximation change requires Cooper's decision.

## Definitions

- Executable acquisition: compile() including any persistent-cache lookup; it is not proven cold compilation without cache evidence.
- Output-selection comparison: changing retained outputs of the same computation, not changing its inference equations.
- Amortization: paying a setup cost once and reusing the executable across subsequent compatible calls.

## Change paths and hostile inputs

Developer reporting only; public compiler/engine, schedules, persistence, serialization, learning updates, numerical host/JIT and single/vmap behavior are unchanged. Measured complete solves retain the existing contract. Native float32 and T32/T100 are applicable measurement dimensions; sparse/empty support, improper inputs and large K20/d10 numerical survival remain in their existing gates and are not new test claims here. No full-suite claim may be inferred from compilation-only or artifact checks.



## Notes

**2026-09-14T20:27:54Z**

Read-only research at 96276c8, requested by Cooper's compilation-scaling/opt-out question; no files changed and no numerical work or suite polling occurred.

Question: which costs belong to recovery rather than inference? Answer: solve -> initialize/evaluate -> while_loop(iteration) -> line_search/evaluate; the conditional fallback linearizes the residual and runs bounded CGLS; certify evaluates again. joint.evaluate invokes separator evaluation and auxiliary_statistics. The latter fuses xi target/residual with centered learning linear/precision arrays; their finiteness contributes to status, which masks the xi target/residual. Thus their primal validity work is an inference dependency, although learning-array tangents are not themselves residual components. Physical-coordinate learning conversions are post-solve reporting. Compiler elimination/merging of duplicated calls or unused arrays is not established by the source graph. Sources: hybrid_joint_solver.py:142,235,255; hybrid_joint.py:210; hybrid_separator.py:226; tests/hybrid_joint_candidate.py:61. Confidence high for dependencies, unknown for achieved compiler elimination.

Question: do current users incur D12 compilation, and can they disable learning to avoid it? Answer: public compiler/Engine paths do not call hybrid_joint, and ARCHITECTURE says production remains D9. K6G5/7SW5 intend explicit declaration-derived activation and forbid silent approximation fallback. The unbound solver already holds model parameters fixed and performs no recovery fit or parameter-update loop; avoiding those activities saves repeated inference calls, not this existing compilation. There is no implemented public D12 no-recovery option. Potential separable work includes physical-coordinate conversion, export/payloads and a reviewed internal separation of learning-only statistics. The existing status dependencies prevent claiming that an output trim eliminates all learning computation. Sources: compiler.py:103; ARCHITECTURE.md:855; K6G5/7SW5 acceptance criteria; hybrid_joint.py:186; separator.py:226. Confidence high for current behavior; savings unmeasured.

Question: what would change the contract? Answer: freezing xi, collapsing source-conditioned moments, replacing D12 with D9, or removing FE/certification requirements changes approximation or inference semantics. Removing learning-finiteness checks changes existing failure behavior and needs explicit validation/review. A3WZ still requires learning statistics. None is authorized as a silent performance shortcut.

Question: smallest useful early measurement? Answer: compile the complete solve at fixed T32/T100,K4,d2 native-float32 configurations, separating lower() and compile(), reporting execution not run. A bounded full-output versus inference-output wrapper at one size can measure output-driven elimination without altering solve equations. Retain executables where possible for later authorized starts; no optimizer warm-up or extra fit. Use cdc-ztuf's precision/cache/timing/macOS RSS distinctions; a complete solve is required for outer-cap claims. Source: hybrid_joint_storage.py:57 and committed cdc-ztuf instrumentation note. Confidence high for isolation, no scaling result yet.

Dependency companion from tk dep tree --full cff-a3wz, tk ready and tk blocked. Legend: prerequisite --> dependent; TK records remain authoritative. Ready independent work is shown separately; implementation remains sequential.

[x] cdc-bwta --> [x] cff-y15d --> [ ] cdc-ogj9 (ready; early scaling, next)
       |                               |
       +--> [ ] cdc-c8kc (ready; root)  |
                    |                  |
                    +------------------+--> [ ] cdc-ztuf (blocked by both)
                                                   |
                                                   +--> [ ] cff-a3wz
                                                              |
[ ] cff-k6g5 (ready; independent declarations) ----------------+--> [ ] cff-7sw5

cff-7sw5/A3WZ feed the execution/state/learning/composition chain and then cff-x0if's unchanged three-fit recovery gate and cff-zov5. First complete cdc-ogj9's small evidence request, then root's suite-margin work, then the existing moderate harness/qualification chain. No scope, approximation, numerical threshold, recovery fit count or truth-sealing rule changed.

**2026-09-14T20:40:00Z**

Prospective setup completed; expensive measurements have NOT started. The four new developer artifacts are tools/d12_compile_spec.json, tools/d12_compile_fixture.py, tools/d12_compile_scaling.py, and tools/d12_compile_checks.py. The JSON plan fixes complete-solve compilation in this order: T32/K4/d2 full State, T32/K4/d2 inference fields, T100/K4/d2 full State. All use scalar initial_xi=0, the existing unperturbed feasible initializer, and unchanged outer/trial/linear budgets 200/24/128, backtrack=0.5, Armijo=0.0001. The exact frozen A3WZ formulas are evaluated on the host and rounded once into float32 argument arrays; Q is covariance and unary J is precision. Shape descriptors keep every numerical argument dynamic. Fixed support/origin/index maps are reported separately.

Before-measurement limits: each fresh CPU child has lowering 180 s, executable acquisition 900 s, and total lifetime 1200 s; the RSS diagnostic limit is 40 GiB (42949672960 bytes), sampled once per second. Termination grace is 5 s. Three sequential children give at most 3600 s of planned worker budgets, plus diagnostic/termination overhead. These are sampled soft stops, not guaranteed allocation caps. The first failure or resource stop ends the sequence, preserving the current worker checkpoint, source/spec identities, compiler log, samples, stop record, exit code, and independent parent wait4 high-water. No retry, tuning, continuation, complete-solve execution, optimizer start, or recovery fit is part of this plan. The full-output baseline means the native solver State; physical learning conversion and export are excluded in BOTH programs.

Native-float32 and cache evidence: the fresh child disables x64 before workload imports and rejects explicit x64 requests; record actual JAX/config/backend/version/environment identities, floating argument/metadata/output leaf manifests, and all StableHLO nested value types. No tiny runner is imported. Disable persistent compilation caching and auxiliary XLA caches; retain compiler logs and their cache-event counts. Retain Lowered and Compiled objects together through reporting, separately time lower() and compile(), never call Compiled, and disclose that the objects are released on child exit. No cross-process reuse or moderate execution amortization is claimed. XLA argument/output/temporary/alias estimates remain separate from logical leaves and parent os.wait4 process high-water; Darwin ru_maxrss is bytes, Linux is KiB. Sampled ps current RSS and process-lifetime high-water are not independent phase peaks.

Architecture/isolation review: the change-path index has no developer-reporting row. Derived component coordinate and Runtime numerical or status invariant are checked as consumers only: the existing unbound JIT complete-solve function is invoked solely by the opt-in tool, with no change to equations, defaults, failure checks, source files or public activation. Host/JIT semantic changes N/A; single/vmap N/A; flooding/scheduled N/A; current/legacy checkpoint decoding N/A; production compiler/engine, schedules, serialization, learning readouts and lifecycle N/A. The wrapper selects returned fields after the unchanged solver finishes and retains all status dependencies, including learning-array finiteness. Sparse/empty support, improper inputs and larger-domain numerical survival remain existing qualification work; this tool does not establish them. No shared dependency, test discovery/selection/execution, pass/fail interpretation, dashboard/runtime history or previous report/hash is edited. Checks live under tools/ and run directly with unittest, outside pytest discovery. This is isolated developer instrumentation and qualifies for focused artifact validation instead of the full short suite.

Focused checks: uv run python tools/d12_compile_checks.py passes 7 checks (0.236 s test time), covering all frozen fixture formulas/shapes, current solver budget parity, native precision and fixed layout without evaluation, identity-preserving inference-output selection, units/limit decisions, an actual terminated dummy subprocess with preserved telemetry, and scalar-addition-only lowering/compilation plus malformed-report rejection. The scalar executable is never called; no D12 program is lowered, compiled or solved. The first check run exposed only an incorrect expected selected-field count in the check itself (18 versus 17); it was fixed before the passing run. uv run python tools/d12_compile_scaling.py plan > /tmp/cdc-ogj9-plan.json and uv run python -m py_compile tools/d12_compile_scaling.py tools/d12_compile_fixture.py tools/d12_compile_checks.py pass. tk dep cycle reports no cycles; git diff --check passes. No full-suite result is claimed.

Resume point: commit this prospective setup and PAUSE for root's independent review. After review and any fixes, the frozen opt-in measurement command is uv run python tools/d12_compile_scaling.py run --output /tmp/cdc-ogj9-compile-001 (new external directory required; clean committed checkout required). Source/spec/version identities are saved at execution. Actual compile timing/memory comparison and the evidence-based recommendation are still pending; cdc-ogj9 remains in_progress. No concrete dependency forced numerical or shared-test changes. The unchanged registered recovery gate and public D12 activation remain downstream; recovery quality is not established.
