---
id: cff-a3wz
status: closed
deps: [cff-ffsg, cff-ocxl, cff-b0s3, cff-ej67, cff-y15d, cdc-ztuf, cdh-urn0]
links: []
created: 2026-09-14T02:43:33Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid, numerical]
---
# Qualify the complete hybrid message and auxiliary solver before integration

## Background in Plain English — WHY

D12 gives the equations for hybrid messages and the logistic-bound auxiliary values, but the existing NumPy experiment holds those auxiliary values fixed. Before compiler work makes this solver part of the engine, Cooper asks for an early numerical check that solves both sets of equations using the local JAX kernels from cff-ffsg.

This ticket extracts numerical feasibility from cff-f3jz. It uses small independent references and fixed synthetic chains. It does not fit the registered source-recovery datasets or change the mathematical approximation, supported model scope, or recovery thresholds.

## Outcomes / Acceptance Criteria — WHAT

1. An unbound solver uses the cff-ffsg kernels to solve the complete D12.7 message equations and D12.8 auxiliary equations at one returned coordinate. The coordinate removes the common scalar message gauge, respects symmetric precision, and fixes inactive auxiliary values to zero.
2. A committed pre-run specification fixes fixtures, initialization choices, tolerances, iteration and line-search budgets, linear-solve restart and storage limits, and reporting fields before the checkpoint solves begin. Numerical acceptance uses tau(dtype) = 64 * eps(dtype)**0.75, inherited from the fixed-auxiliary experiment, with eps the dtype machine epsilon. Budgets may be chosen from a stated implementation inventory before execution; they cannot be increased silently after a missed gate.
3. A converged result has each scaled natural-message residual, scaled conditional-mean and conditional-covariance residual, and auxiliary residual abs(xi-target)/(1+abs(xi)) at most tau. Tiny active label mass cannot hide conditional-moment error. It also has proper active aggregate and tilted beliefs, finite free energy, relative free-energy change at most tau, and primal/dual discrepancy divided by 1+abs(free energy) at most tau. All certificates and statistics are recomputed at the returned coordinate.
4. Feasible initialization and bounded globalization handle the fixed valid cases. Converged, budget-exhausted, and diverged remain distinct. Invalid active objectives/residuals cannot report convergence; improper trial steps may be rejected. A rejected or exhausted initialization is reported with its inputs and failure reason.
5. An independent tiny float64 reference solves the complete self-consistent auxiliary and message problem. It imports neither the candidate nor production mathematics. T=2 exact reductions may use independent path integration plus auxiliary closure; a coupled T=3 case uses an independent EC closure because exact path enumeration is a different temporal family. Both D11 source-mean twins, both variance assignments, D9 fixed-drive/drive-independent reductions, boundary messages, conditional cross-moments, full free energy, and source-conditioned learning increments are checked. The existing D12 temporal falsifier and fixed-auxiliary approximation ceilings remain unchanged and are reported separately.
6. Reference comparisons separate arithmetic error from stopping error and approximation error. At the same held coordinate, local arrays use atol=rtol=512*4*eps; total free-energy arithmetic uses absolute tolerance 512*max(4,5*T)*eps. Complete tiny reference solves satisfy tau(float64); solved-coordinate comparisons state and freeze any propagated stopping allowance before execution. Host/JIT and single/vmap checks cover the numerical solver and every returned certificate.
7. Fixed moderate cases use T=32 and T=100, K=4, d=2, in float32, with source-dependent nonzero slopes, destination-selected dynamics, and proper informative unary sites. The specification freezes the formula below. Each case exercises xi=0 and xi=1 starts with zero messages and the declared feasible initializer, plus a fixed non-oracle message perturbation. Report convergence, initialization sensitivity of messages/beliefs/free energy/learning increments, iteration counts, rejected trials, time, and memory. Independent initializations need not prove universal uniqueness, but materially different stationary outputs must be disclosed and diagnosed before claiming a positive checkpoint.
8. Actual iterative storage is measured, including state, residuals, line-search copies, differentiation buffers, and restarted linear-solver vectors where present. Compare at least two horizons and separately inventory K=20 and d=10 local/iteration shapes. Report array element counts and bytes plus compiled temporary memory where available; unavailable measurements are named with a substitute. No dense global Jacobian, dense trajectory covariance, retained unbounded optimization tape, or exponential path inventory is used outside the tiny reference. If a joint Krylov vector includes xi, its storage bound includes O(r*T*(K*d*d+K*K)); the message-only bound cannot be reported as its total.
9. Before implementation, the handoff states behavior for exact log-zero support, empty support, support changes, inactive xi, tiny positive underflow, signed/vacuous cavities, malformed/non-finite input, float32, and larger domains. Guards follow dtype and scale; signed messages are never clamped. Focused tests exercise these mechanisms and opt-in slow tests exercise the moderate complete solves.
10. The Derived component coordinate and Runtime numerical or status invariant rows cover unbound host/JIT and single/vmap evaluation/iteration/certification. Production compiler, scheduling, lifecycle, model selection, persistence, public learning readouts, and recovery fits are N/A. Numerical docstrings and D12 anchors describe the measured solver/storage boundary; dashboard test-file registration, focused tests, and one final short-suite gate pass after the final relevant edit.
11. A positive result records reproducible commands, fixed specifications, actual results, limits, and the reusable solver handoff, then closes this prerequisite. If the stated requirements still fail after a bounded diagnosis and ordinary implementation fixes, record the evidence and recommended next step and return the decision to Cooper. Do not revise D12 semantics, narrow supported scope, relax numerical or recovery gates, or proceed into compiler integration to avoid a failure.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

- Use local or matrix-free derivatives and explicit bounded damping/backtracking. Local auxiliary elimination is allowed only if its closure is certified and included in the complete residual.
- Use tests/hybrid_separator_reference.py for independent local integration and extend a separate tiny reference for joint EC stationarity. The existing src/cffg/experiments/hybrid_separator.py solver is a fixed-auxiliary comparison, not evidence that this gate passes.
- Freeze this deterministic moderate fixture with indices i,j in 0..3 and l in 0..2: W_il=[0.12*(i-1.5)+0.04*(l-1), -0.08*(i-1.5)+0.03*(l-1)]; offset_il=0.10*(i-1.5)-0.15*(l-1). A_j=[[0.65+0.04*j,0.03],[-0.02,0.80-0.03*j]]; dynamics bias=[0.10-0.05*j,-0.15+0.04*j]; Q_j=diag(0.8+0.1*j,1.1+0.05*j). Unary a_ti=-0.2-0.1*i; eta_ti=[0.1*(i-1.5)+0.05*(-1)**t,-0.08*(i-1.5)+0.03*((t mod 3)-1)]; J_ti=diag(0.9+0.1*i,1.1+0.05*i). Criterion 7 adopts these formulas; this paragraph gives their construction compactly.
- Keep experiment specifications/results in machine-readable artifacts and concise ticket notes; update canonical documentation in place. Do not add a research-report markdown file.

## Future work

cff-k6g5 declares serialized hybrid semantics. cff-7sw5 compiles component plans only after this numerical gate and those declarations. cff-f3jz integrates the qualified numerical solver, full objective ledger, and terminal status into the engine. cff-czdq, cff-ulbz, and cff-kul2 then supply learning identity, exterior mixtures, and public composition. cff-x0if alone owns the registered three-fit source-recovery gate and its one authorized diagnostic pass; its 0.150 thresholds and frozen datasets remain unchanged.

## Definitions

- D12: the hybrid-regime-ec-v1 mathematical contract in docs/DERIVATIONS.md.
- EC: expectation consistency, matching stated moments of neighboring regional beliefs.
- Auxiliary xi: the squared logistic-bound coordinate c*c, tied per source and stick.
- Stick: one binary decision in the ordered categorical link.
- Gauge: a redundant coordinate that does not change a normalized belief.
- Natural coordinates: a log potential written as a constant, linear term, and quadratic precision term.
- Proper belief: a normalizable probability distribution with valid covariance.
- Cavity: the product of messages entering a region from its other neighbors.
- Globalization: damping or line search that rejects invalid or unproductive proposed steps.
- Primal/dual: two expressions for the same stationary D12 objective.
- JAX/JIT/vmap: the array library, its compiled execution, and batched execution.
- Krylov restart: a bound on the number of directional vectors retained by an iterative linear solve.
- Unbound solver: numerical code that does not select or execute a compiled production model.
- Source-conditioned learning increments: sufficient statistics retaining the joint source-label/state moments.



## Notes

**2026-09-14T02:44:32Z**

Research question → what must the early solver gate exercise, and what references and storage bounds apply? Answer → D12.7 and D12.8 must both be certified at the same returned coordinate, including conditional moments independent of label mass, finite/stable D12.4 FE and stationary D12.9 agreement, plus all D11.1 learning increments. Existing NumPy solve is fixed-JJ only. Existing path enumeration is a valid local/T=2 reference; coupled temporal EC needs an independent EC closure. Proposed moderate fixtures and dtype tolerances are now frozen in this ticket before checkpoint execution. Actual monolithic Krylov vectors include xi: N=2*(T-1)*(K*(1+d+d*(d+1)/2)-1)+(T-1)*K*(K-1), giving N=58*(T-1) at K=4,d=2; storage is O(r*T*(K*d*d+K*K)), not the message-only expression. Source → independent solver_checkpoint_research report, docs/DERIVATIONS.md D12.3–7, src/cffg/experiments/hybrid_separator.py:207/222/297/367, tests/hybrid_separator_reference.py:14/92, tests/test_hybrid_separator_derivation.py:18/23/127/259/394. Confidence → high for equations/reference limitations/storage accounting; medium for proposed prospective numerical gates, which are not measured results. No unresolved EC equation was found. Inventory correction preserves the equations and approximation semantics.

**2026-09-14T02:44:32Z**

Supervision boundary: ordinary implementation and numerical-strategy fixes remain authorized after diagnosis, with every changed bounded budget/strategy and rerun recorded before execution. Do not silently replace a failed run or describe a tuned run as the original. A mere first-attempt exhaustion is not by itself a decision for Cooper. Escalate only if requirements remain unmet after bounded diagnosis and ordinary fixes, or evidence requires changed mathematics, supported scope, approximation semantics, or qualification thresholds. Check storage at more than one outer iteration cap to expose retained optimization history. No registered recovery fit is authorized by this checkpoint.

**2026-09-14T02:45:09Z**

Roadmap refinement built from tk dep tree --full cff-x0if, tk ready, and tk blocked. Legend: prerequisite --> dependent. tk records are authoritative; shared downstream arrows require all listed prerequisites.

[x] cff-ocxl (D12 accepted)
  +--> [~] cff-ffsg (ready; implementation running)
  |       --> [ ] cff-a3wz (joint numerical checkpoint; next)
  |                   |
  +--> [ ] cff-k6g5 (ready; independent declarations)
                      |
       a3wz + k6g5 --> [ ] cff-7sw5 (compiler plans)
                              --> [ ] cff-f3jz (engine integration)
                                    +--> [ ] cff-czdq (learning identity)
                                    +--> [ ] cff-ulbz (exterior mixture)
                           czdq + ulbz --> [ ] cff-kul2 (composition)
                                             --> [ ] cff-x0if (recovery)
                                                   --> [ ] cff-zov5 (epic)

Declaration work is structurally independent of the checkpoint; Cooper’s implementation sequence runs FFSG, then a3wz, then the remaining ready chain sequentially. Research may run alongside an implementer; independent review starts only after that implementer finishes. Documentation/ticket-only refinement: no numerical code or full-suite run; tk dep cycle reports no cycles and git diff --check passes.

**2026-09-14T03:25:56Z**

Numerical representation handoff after cff-b0s3 diagnosis: use its retained Gaussian constructors and fixed separator-label anchors throughout the solver. Anchors are initialized/frozen metadata, not additional unconstrained solve coordinates; preserve internal mean displacements in residuals/projections. The gauge-reduced message unknown count is unchanged, but actual buffer inventory must include O(T*K*d) anchor arrays and any retained physical parameters. State the chosen fixed basis in the pre-run specification and recompute all certificates/objectives/learning from the returned anchored coordinate. A centered FE helper alone is insufficient if tilts/messages still lose label information. b0s3 must pass before this checkpoint starts.

**2026-09-14T03:54:34Z**

Concrete local API handoff from b0s3 implementation a5a4747 (independent review pending): hybrid_natural.gaussian_site retains log mass/mean/covariance and optional fixed center; Natural.arrays() returns only a/h/J. Belief stores displacement+center and exposes world mean/second as readouts. hybrid_separator.evaluate takes stacked TransitionParameters, sites and two message banks. Auxiliary.linear/precision are now explicitly statistics for [x-center_i;1]; physical_linear/physical_precision transform them with B=[[I,center],[0,1]]. Use the physical views for original-coordinate independent references and also inspect retained centered statistics so a raw large second moment cannot hide small covariance. Freeze origins before packing solver unknowns and preserve their metadata on reconstruction; never flatten the entire NamedTuple as if status/origin fields were optimization variables.

**2026-09-14T04:41:18Z**

Bounded implementation split before starting: cff-j191 owns independent tiny EC+xi reference mathematics and the executable tiny specification; cff-y15d owns candidate packing, differentiation, initialization/globalization/status, complete certificates and tiny parity; this ticket retains all final qualification criteria, moderate runs, initialization sensitivity, actual buffers/compiled memory, larger K/d inventory and the positive/negative decision. No criterion or threshold is removed. Each stage commits its execution budgets and stopping accounting before its first applicable checkpoint solve; later phases inherit the fixed fixtures/tolerances and record their own prospective execution budgets before running. Earlier failed attempts remain evidence. Closing either prerequisite is not a claim that the numerical checkpoint passed.

Research question -> can reference, solver and moderate evidence fit one worker context? Answer -> they are three substantial correctness boundaries; use the sequential split above with a fresh worker per bounded ticket. Source -> independent solver_checkpoint_research audit of current local APIs and D12 checkpoint requirements; confidence high. No additional mathematics/scope/approximation decision is required by the split.

Dependency tree built from tk dep tree --full cff-a3wz, tk ready, and tk blocked. Legend: prerequisite --> dependent; tk records are authoritative.

[x] cff-ocxl
  +--> [ ] cff-ej67 (ready; next local numerical repair)
  |       --> [~] cff-b0s3 --> [~] cff-ffsg
  |            --> [ ] cff-j191 (independent tiny reference)
  |                 --> [ ] cff-y15d (candidate solver)
  |                      --> [ ] cff-a3wz (moderate qualification)
  +--> [ ] cff-k6g5 (parallel ready declarations, implementation deferred)
       a3wz + k6g5 --> [ ] cff-7sw5 --> [ ] cff-idhk --> [ ] cff-brcf
           --> [ ] cff-f3jz --> learning/composition --> [ ] cff-x0if
                --> [ ] cff-zov5

All implementation remains sequential, and compiler work still waits for complete local message+auxiliary qualification. Ticket-only artifact checks are tk dep cycle and git diff --check.

**2026-09-14T06:36:21Z**

Committed local API handoff at ff41a02 (independent review and full short-suite still pending): Belief.covariance is the high word; optional covariance_correction is the retained low word, with None exactly zero. correction exposes a zero-filled array when needed; full_covariance is a once-rounded physical readout. Cancellation-sensitive residuals and contractions must preserve both words instead of reconstructing from full_covariance. These words are derived from the message coordinate, not additional optimization variables. Natural.arrays() still returns only constant/linear/precision, and fixed origins remain metadata. Normalize reconstructed status arrays to int32: a weak int64 Python-default status combined with a batched int32 flag reproduced a JAX backend segfault, now guarded in local paths.

hybrid_covariance.solve uses custom_linear_solve with compensated residual refinement and a componentwise backward-error certificate, without a unit-scale floor; unsupported unresolved solve arithmetic returns NONFINITE. The logdet uses a whitened log1p identity and the exact trace differential. Candidate complete-residual JVP/transpose checks must include this retained covariance path, xi=0 and fixed nonzero origins. Local tests passed201/201 in1828.42s; this is not complete-solver qualification.

Reproducible local compiled inventory is uv run python tests/test_hybrid_projection.py, which lowers dynamic ShapeDtypeStruct arguments and emits four memory analyses. At T100/K20/d10, f32, combined projections and quotient messages have argument72,497,200/output5,490,608/temporary258,775,144 bytes, zero aliases; retained projection corrections total1,600,000 bytes. Whole pytest-process RSS observed12.4GiB during the long focused command is a separate process-level observation, not a per-iteration allocation. The qualification must still measure the complete iteration, differentiation/linear vectors, compilation cost and available process memory at its fixed T32/T100 cases and multiple iteration caps; do not extrapolate or sum unrelated local temporary peaks as if they were that measurement. No moderate checkpoint or recovery solve has run.

**2026-09-14T08:48:12Z**

Qualified local API update: FFSG, B0S3 and EJ67 are now closed in PR #143 (https://github.com/engimatic-systems/cffg/pull/143). Final numerical source is 07b18fd; independent review found no remaining issues, and the combined short suite passed 2489 tests, 41 deselected, in 3182.41s through tools/dashboard.py --run-suite. Earlier notes marked review/full-suite pending and the 201-test focused result are historical, not the current gate.

Natural.arrays still exposes only message coefficients; fixed origins remain metadata and retained covariance low words remain derived. Cancellation-sensitive covariance/quotient arithmetic still requires both representable words. Ordinary physical second moments use a distinct rounded product so an unused subnormal remainder does not reject an otherwise finite belief; both constructors and readouts share overflow behavior. This preserves the mathematical contract and all thresholds.

The reproducible local command uv run python tests/test_hybrid_projection.py now lowers six dynamic-argument programs. At T=100, K=20, d=10, float32 data with x64 enabled, combined projections/quotients retain argument/output/temporary bytes 72,497,200 / 5,490,608 / 258,775,144; grouped normalization is 70,560,400 / 70,720,856 / 131,200,984; grouped physical second moments are 70,560,000 / 64,000,000 / 0, with zero aliases. Zero temporary allocation reflects fusion into outputs, not zero work. The ordinary rounded product has one floating word and integer significand limbs, not the strict product's floating remainder/matrix accumulator. Full six-program details and final diagnostics live in EJ67 and D12.6. These remain local program measurements, not complete iterative buffers or process memory. Complete-residual differentiation, candidate convergence and actual iteration storage still require this sequential checkpoint chain. No complete reference/candidate/moderate solve or registered recovery fit has run at this handoff.

**2026-09-14T08:51:59Z**

Independent storage research — Question: what must complete matrix-free iteration inventory beyond packed unknowns and directional vectors? Answer: distinguish N coordinate-space vectors from M signed-residual vectors. With L=T-1 and s=d*(d+1)/2, full-support N=2*L*(K*(1+d+s)-1)+L*K*(K-1). Retaining all label masses and independent covariance entries gives M=N+2*L*K*(1+d+s), hence N=58*L and M=106*L at K4,d2. Dropping one redundant label-mass equation per incidence subtracts2*L; freeze the chosen layout explicitly. The existing outer.py fixed-point candidate treats its residual as a coordinate step and assumes M=N; do not copy it into a rectangular residual solver.

Inventory fixed context/parameter arrays, active maps/support/origins and materialized versus broadcast representations; packed and unpacked accepted/trial coordinates; M residuals and frozen scales plus N coordinate scales; grouped beliefs/factors/cavities O(T*K*K*d*d); separators/projections/covariance correction words O(T*K*d*d); auxiliary targets/exposures O(T*K*K); centered learning arrays O(T*K*K*(d+1)^2) and any materialized physical views. Include retained derivative closures (local factors/solutions/probabilities/corrections/masks/scans), N-sized gradient/step/linear residual/direction, M-sized Jp, normal-product and boundary/Cauchy temporaries, trial/candidate banks, integer significand limbs, status and bounded diagnostics. Steihaug's existing loop carries three N-vectors plus counters/flags; ordinary Steihaug has no stored restart basis, so count any added basis or preconditioner separately. Custom JVPs mean forward integer temporaries are not automatically all retained in differentiation.

Installed JAX 0.10.2 source shows separate linearize and vjp calls build separate linearization/residual collections. This is NOT evidence of twice the physical memory: inputs/compiler work may be shared. A linear_transpose of the already-linearized JVP can reuse its nonlinear forward data for the same frozen-scale residual, preserving J-transpose-J, subject to complete JVP/transpose identity checks. Compare actual enclosing iterations, not isolated closures alone.

Reproducible future measurement: dynamic ShapeDtypeStruct coordinate/context/scale/vector arguments; logical leaves/dtypes/bytes for accepted evaluation, derivative closures, normal product, complete linear step, accepted/trial iteration and terminal certification/statistics. Lower/compile the actual enclosing iteration and record argument/output/temporary/alias bytes; do not sum unrelated temporary peaks. Compare required horizons and outer/linear caps; report process/compiler RSS separately. Returning closure leaves lengthens lifetimes and is a logical inventory aid, not a peak-iteration measurement. Source -> independent ffsg research report; hybrid_natural.py:72, hybrid_separator.py:221/377, hybrid_covariance.py:26, outer.py:465/542, installed jax/_src/api.py linearize/vjp/linear_transpose, tests/test_hybrid_projection.py dynamic inventory. Confidence high for shapes/API semantics and procedure; physical sharing and full iteration memory remain unmeasured. No solves, compilation, edits to the reference, budget selection or threshold changes occurred in this research.

**2026-09-14T11:02:30Z**

Pre-run handoff clarification after cff-8ois diagnosis: native float32 value accuracy and complete convergence remain separate from promoted-input derivative/reference evidence. Report the actual JAX x64 configuration and dtypes of parameters, coordinates, internal floating leaves and results; the fixed moderate float32 gate must exercise native consumer float32 behavior, not rely on higher-precision intermediates hidden by the pytest-wide x64 setting. Preserve identical stored inputs when using promotion as an independent derivative diagnostic. Any derived precision/covariance words added by 8OIS belong in actual fixed-context/evaluation/iteration/derivative storage and compiled memory inventories, without becoming extra unknowns. Original fixtures, thresholds, initialization requirements and recovery separation are unchanged; no moderate solve has run.

**2026-09-14T17:35:38Z**

Pre-implementation research and prospective initialization clarification. Question -> which moderate starts are required, and how do the frozen arrays map onto existing interfaces? Answer -> criterion 7 and the candidate specification's moderate_qualification_unchanged section retain zero messages, feasible initialization and fixed perturbation. No earlier A3WZ note explicitly resolves whether raw zero is a distinct solve start. The candidate specification's top-level two-mode list belongs to Y15D's tiny protocol and cannot narrow this later gate. Root therefore requires an explicit prospective inventory of 12 moderate cases: T=32/100 x xi=0/1 x {raw-zero messages, one-sweep feasible messages, feasible messages plus the fixed non-oracle perturbation}. This covers every named mode; formulas, numerical thresholds, approximation semantics and registered recovery are unchanged. No moderate solve has run. Commit the exact construction and budgets before executing these cases.

Field mapping -> L=T-1 gives 31 or 99 transition regions. TransitionParameters.weights has shape [L,4,3,2], offsets and xi [L,4,3], matrix/noise [L,4,2,2], and bias [L,4,2]. i is the source regime, j selects destination dynamics, and the stick index is 0..2. Q is covariance; unary J is precision. Natural constant/linear/precision have shapes [T,4], [T,4,2], [T,4,2,2]; preserve the time-dependent unary linear terms and natural constants. Origins are fixed zeros [T,4,2]. Scalar tiny adapters append singleton axes and are not a moderate factory; the independent tiny model validator only accepts T=2/3. Sources -> hybrid_separator.py parameter and chain contracts, hybrid_joint_reference.py tiny limits, candidate moderate specification, and A3WZ criterion 7. Confidence -> high; read-only research by precision_8ois_implementer, independently checked and clarified by root.

Question -> how will the native float32 claim be proved? Answer -> use a fresh subprocess, disable jax_enable_x64 before numerical imports, assert the actual configuration is false, and construct explicit float32 inputs. Pytest and the tiny candidate runner enable x64 and cannot establish this consumer claim. Record source/spec hashes, Git revision, versions and leaf path/shape/dtype/element/byte manifests for parameters, fixed metadata, coordinates, evaluation, derivative captures, iteration state and results, including retained correction words. Integer statuses, masks and significand limbs stay integer/bool. The existing hybrid_joint_storage.arrays helper supplies the manifest. Sources -> tests/test_hybrid_precision.py native subprocess mechanism, hybrid_joint_storage.py, and the prior A3WZ dtype handoff. Confidence -> high; no code, compilation or solve was run in this research.

Implementation handoff -> the current Y15D solve wrapper exposes initial_xi and perturb and always calls its feasible initializer. A distinct raw-zero start must therefore be represented explicitly and exercised, not relabeled as an already-initialized start. Prefer a shared solver entry for an explicit accepted initial coordinate/state over copying the iteration or certification equations into the qualification harness. Any necessary ordinary API change is prospective implementation work; preserve prior immutable tiny reports and refresh affected qualification evidence when source hashes change. Finish the bounded harness/API implementation and focused checks, commit, and pause for root review before expensive moderate runs. If API work, harness construction and evidence cannot fit one worker context, split a prerequisite ticket before implementation. Root still owns the short-suite gate; compiler integration and registered recovery remain blocked on a positive A3WZ decision.

**2026-09-14T18:09:57Z**

Setup is split before implementation into cdc-ztuf: a fresh worker owns the shared explicit-coordinate entry, twelve-case native-float32 fixture/specification, and numerical/storage harness. cff-a3wz retains all actual moderate convergence, initialization sensitivity and iteration-memory execution/interpretation, plus the integration decision. Y15D tiny evidence and final short-suite publication finish first. No fixture formula, approximation, supported scope, tolerance, storage requirement or recovery gate changes.

Legend: prerequisite --> dependent. TK records remain authoritative.

[~] cdc-bwta (ready; root final suite gate)
  --> [~] cff-y15d (author fixes committed; re-review/evidence/gate pending)
        --> [ ] cdc-ztuf (setup; immediately blocked by Y15D)
              --> [ ] cff-a3wz (twelve moderate starts and actual storage)
                    --> [ ] cff-7sw5 (compiler plans)
                              ^
[ ] cff-k6g5 (ready; independent declarations) --+

cff-7sw5 and cff-a3wz feed the remaining integration/state/ledger chain before cff-f3jz, learning/composition, cff-x0if's three frozen recovery fits and cff-zov5. This companion comes from tk dep tree --full cff-a3wz, tk ready and tk blocked; no registered recovery fit has run.

**2026-09-14T23:05:04Z**

Independent cdc-ztuf report repairs exposed two bounded native covariance-projection discrepancies outside frozen arithmetic allowances. Added cdh-urn0 as an explicit prerequisite. The repaired validator rejects them; two strict expected-failure probes preserve them. No moderate solve/storage compilation or registered fit has run. See cdc-ztuf re-review handoff and cdh-urn0 for evidence, scope and the prerequisite tree.

**2026-09-15T03:12:53Z**

All setup prerequisites now pass and are closed. Final numerical source includes the reviewed compensated-centering repair and independent model-point reference. Immutable tiny attempt004 runs at aecd819, reportSHA b127d35227a32851ccfcd7721cc54fdab09c8311c63b47f9f63547520dd9a900; all40converged outputs,6expected divergences and1exhaustion pass independent checks. The shared final short suite at144fb21466894bfd91d2040ff2d5e1d003cc82e9 passes2832 tests in1126.534 seconds under the1200-second cap. The final ordinary setup repair cdh-wi2b isolates disabled-cache children from two suite cache settings; its strict descriptor/runtime checks pass review. Current UNEXECUTED moderate specification SHA3a6bab37fcbab2b3e3ffdac4bc6892f17e2be244c4f2e65754b4b3c3ee660c3b, descriptorSHA1b98bcfa48a3aeb8d6de3a6ca14c2c963f5e900faf019faa3f831935244672f9. All1026 leaves/runtime per shape match the prior descriptor; only its producer source binding changed. Twelve cases, budgets, limits, numerical tolerances, approximation and recovery gate are unchanged. Root will supply the published stack base and a clean fresh worktree to a fresh bounded author. This ticket still owns every actual moderate optimizer/storage run and their interpretation. No moderate qualification run or registered recovery fit has occurred. cdh-kvy5 remains nonblocking.

**2026-09-15T03:20:17Z**

Prospective attempt001 execution freeze, before any moderate complete-program lowering, compilation or solve. Fresh author works in /Users/ravd/projects/cffg-d12-moderate on cff-a3wz-moderate-qualification, starting clean HEAD58fd5363846cec397d29af44cdc8e6a86920784a from reviewed PR149. This note and tools/measurements/d12-moderate-001-prospective.json are committed before launch; the runner records their resulting clean Git revision. The32 source hashes are preserved in that JSON; canonical sorted compact source-map SHA256 is595ca4a3e3dba8644800ce1187183dde3320b1f00770dad079b4794a760a694c. Frozen spec SHA2563a6bab37fcbab2b3e3ffdac4bc6892f17e2be244c4f2e65754b4b3c3ee660c3b; descriptor SHA2561b98bcfa48a3aeb8d6de3a6ca14c2c963f5e900faf019faa3f831935244672f9. No producer, specification, fixture, threshold, approximation, model scope or numerical code changed.

Preflight: uv run python tools/d12_moderate_runner.py plan > /tmp/cff-a3wz-plan-001.json passed. uv installed the normal worktree environment from the unchanged lockfile: Python3.12.13, JAX/jaxlib0.10.2, NumPy2.5.1, macOS15.7.5 arm64. No inherited JAX_/XLA_ environment settings were present. A fresh child using the reviewed child_environment/native_jax/live_binding functions matched the descriptor runtime/source binding exactly, with zero complete-program lower/compile/execute calls. All12 immutable tiny004 source hashes remain current and its report SHA256b127d35227a32851ccfcd7721cc54fdab09c8311c63b47f9f63547520dd9a900 matches. Accepted focused/tiny/full-suite evidence is reused; no repeat tests or numerical warmups occurred.

Exact phase commands, run sequentially from this worktree:
uv run python tools/d12_moderate_runner.py run --phase solves --output /tmp/cff-a3wz-solves-001 > /tmp/cff-a3wz-solves-001.stdout.log 2> /tmp/cff-a3wz-solves-001.stderr.log
Only after successful solves and saved-report validation:
uv run python tools/d12_moderate_runner.py run --phase storage --output /tmp/cff-a3wz-storage-001 > /tmp/cff-a3wz-storage-001.stdout.log 2> /tmp/cff-a3wz-storage-001.stderr.log
Both output directories and all four parent log paths were verified absent. Preserve every runner JSON/NPZ/compiler/stdout/RSS artifact and parent log, including partial evidence. Independent final revalidation uses the reviewed validate command on the saved directories and makes no complete-program calls.

Inventory: exactly T32/T100 x xi0/1 x raw-zero/feasible/perturbed, twelve complete solves. Raw-zero supplies zero message coefficients without a feasible sweep; feasible uses its one simultaneous quotient sweep; perturbed adds0.01*sin(arange(message_size)+1) to feasible messages only. K4,d2, full support, zero fixed origins and all frozen formula inputs remain unchanged. Each horizon owns one Lowered+Compiled initializer pair and one Lowered+Compiled complete-solve pair, each lowered/compiled once and retained for six distinct calls. Planned total initializer calls12 and solve calls12; no warmups, retries, repeated starts, extra benchmark calls or cross-child reuse. First unsuccessful initializer prevents that optimizer call; first nonconverged terminal/resource/runtime/validation gate preserves actual counters and stops subsequent authorized work. Actual partial counts will supersede the planned totals if stopped.

After passing solves: one separate abstract inventory child for T32/K4/d2,T100/K4/d2,T100/K20/d10, zero complete-program lower/compile/execute calls. Then exactly ten fresh compilation workers: each T32/T100 compiles complete solve outer200/linear128, complete solve400/128, complete solve200/256, enclosing iteration200/128 and enclosing iteration200/256. Every storage executable has zero calls. No wide numerical solve is authorized. Inventory includes all1026 leaves per shape, context/metadata, high+low covariance and integer captures, rectangular residuals/scales, derivative closures, linear vectors, accepted/trial/outer state and terminal learning; N=58*(T-1), M=106*(T-1) at K4,d2. Report actual XLA argument/output/temp/alias bytes separately from logical leaf bytes and wait4 child lifetime high-water. Missing statistics remain unavailable. Compile RSS is not iteration allocation; logical alias repetitions are not physical storage. Complete outer-cap comparisons use complete enclosing solves.

Frozen numerical budgets: outer200,trials24,linear128,restarts0,backtrack0.5,Armijo0.0001. Resource caps: lower180s,compile900s,solve worker3600s,storage worker1200s,RSS42949672960 bytes sampled every1s,termination grace5s. Children explicitly use CPU/native float32,x64=false,explicit-x64 ERROR,disabled persistent/XLA auxiliary caches and no inherited compilation-cache directory/max-size/min-time controls. Record actual inputs/outputs/StableHLO floating types, runtime settings and compiler logs. Cache disabling and separate children do not establish absence of every possible OS/compiler reuse.

Acceptance stays fixed: every initial point valid; all seven certificates <=64*eps(float32)**0.75, proper active aggregate/tilted beliefs, finite absolute energies, stable FE change and primal-dual agreement, truthful work/rejection/status. The independent NumPy adapter reconstructs actual model/coordinate beliefs, targets, normalizers, source-conditioned learning and both absolute energies at every valid initial/terminal point. Local arithmetic atol=rtol=512*4*eps(float32); total-energy absolute allowance512*max(4,5*T)*eps(float32), no relative enlargement. Preserve all19 numerical groups and report all30 within-horizon initialization pairs. Material stationary differences require diagnosis, not an assumed uniqueness claim. Arithmetic, stopping and temporal approximation remain separate.

Architecture change paths: Derived component coordinate and Runtime numerical or status invariant. Affected evidence surfaces: unbound initialization/complete iteration/certification, saved host independent arithmetic, retained JIT programs, complete-enclosing/logical storage and native float32 status/readouts. Host/JIT and single/vmap tiny pairs reuse accepted current-source004 evidence; new moderate single/JIT cases follow the frozen plan, with no extra batched calls. Checked production pairs are N/A: synchronous/scheduled execution, current/legacy serialization, IR/JSON/hash/scoping/edits/composition, topology/specialized/registered/compiler/resolution/assembly/Arena, lifecycle, model selection, production ledger/readouts/learning replacement, persistence/checkpoint/CFFGZ, estimator and public exports. No production integration or registered recovery dataset/fit/warmup/restart is included.

Failure handoff: preserve the first failure, perform bounded read-only diagnosis, and create a separate ordinary repair prerequisite if needed; root assigns a fresh sequential author. No specification/budget/threshold retuning or silent numerical rerun. Escalation is reserved for a needed D12 mathematics/scope/approximation/threshold decision or unresolved failure after ordinary repairs. After evidence commit, pause for independent root review; root owns final gate, closure and publication. This prospective change is ticket/evidence-only; artifact checks are tk dep cycle and git diff --check.

**2026-09-15T04:10:03Z**

Attempt 001 author evidence is complete; pause for root's independent artifact review before closure or publication. Both frozen commands exited 0 at clean executed revision 298790bf857753d4a9393015b299e9e90ca1ac84. All 32 source hashes remain those prospectively committed; specification SHA256 is 3a6bab37fcbab2b3e3ffdac4bc6892f17e2be244c4f2e65754b4b3c3ee660c3b and descriptor SHA256 is 1b98bcfa48a3aeb8d6de3a6ca14c2c963f5e900faf019faa3f831935244672f9. No numerical, harness, fixture, tolerance, budget or approximation change occurred, and no failure or retry occurred. Original directories /tmp/cff-a3wz-solves-001 and /tmp/cff-a3wz-storage-001 and their four parent stdout/stderr logs remain intact.

Reviewable artifacts: tools/measurements/d12-moderate-001.json contains per-case certificates, counters, timing and independent absolute-energy comparisons at every initial and terminal coordinate; all 30 initialization comparisons; mappings for all 19 output groups; exact compilation/call counts; runtime/source identity; all logical storage group totals and cap comparisons. Its indexed raw archive, tools/measurements/d12-moderate-001.tar.gz, retains every raw report, NPZ snapshot, child compiler/stdout/RSS log and parent stdout/stderr log: 101 files, 9,792,622 raw bytes, 4,249,018 compressed bytes. Archive SHA256 is aa3d006f53f560ae30a6679bf595ccb4fbfb5c32f234716f383ec5b7b4bb53b7. Every archived member was read back and matched to the original byte length and SHA256. Each of the 24 initial/terminal snapshots has 77 leaves and all floating leaves are float32.

Numerical outcome: all twelve declared initializers were valid and all twelve complete solves converged. Raw-zero starts used six accepted updates each; feasible and perturbed starts used five. There were zero rejected trials, linear iterations or Gauss-Newton steps. Maximum terminal certificates were natural 1.662671566e-4, label mass 1.341104507e-5, conditional mean 1.123696566e-4, conditional covariance 2.665689863e-6, auxiliary 2.591043813e-5, primal-dual 1.045616955e-5 and relative energy change 1.129269904e-7, all below tau=4.105939588e-4. Complete independent saved-output validation passed, including proper active beliefs, reconstructed certificates/scales, native learning readouts and model-point normalizers/targets/absolute energies. Maximum absolute energy arithmetic error across initial and terminal points was 2.605349689e-5 at T32 and 7.003207170e-5 at T100, within the unchanged absolute allowances 0.009765625 and 0.030517578125. Every one of the 30 within-horizon initialization pairs was below the material-difference threshold; maxima by group were messages 2.925650550e-5, xi 3.349028558e-6, energy 8.884319463e-7, regions 1.539643330e-5, separators 2.514424902e-5 and learning 4.067969573e-6. No material stationary discrepancy requires diagnosis. Small differences remain consistent with the accepted finite stopping tolerance; this does not establish universal uniqueness or temporal exactness.

Storage outcome: the separate abstract child matched all 1,026 declared leaves at each of T32/K4/d2, T100/K4/d2 and T100/K20/d10. N/M are 1,798/3,286; 5,742/10,494; and 298,782/560,142. Logical derivative-capture bytes are 1,946,638; 6,202,758; and 2,205,902,774 respectively. The wide case is abstract-only. Actual enclosing-iteration XLA argument/output/temp/alias bytes were 171,340/157,656/3,844,272/0 at T32 and 546,156/501,736/12,389,488/0 at T100. Actual complete-solve values were 21,352/157,656/5,363,128/0 and 67,864/501,736/17,209,912/0. All four fields were identical when outer cap independently doubled 200 to 400 or linear cap independently doubled 128 to 256 in the prescribed program variants. This is bounded compiled storage at these shapes; logical alias repetitions, XLA estimates and physical process memory are distinct. Child lifetime high-water in the solve phase was 22.09/22.71 GiB, with both executables retained; storage-child process peaks are recorded separately.

Cost and exact call accounting: each horizon lowered and compiled its initializer once and its explicit-coordinate complete solve once, retained all four objects, then made six distinct authorized calls to each. Totals are 12 initializer calls and 12 solve calls. Storage made ten lowers and ten compiles with zero executable calls; the separate abstract inventory and preflight made zero complete-program calls. No warmup or timing repeat was added. Initializer compilation took 53.85/52.30 seconds; complete-solve compilation 195.57/191.45 seconds; synchronized authorized solves 0.110-0.138/0.221-0.272 seconds, with exports timed separately. Both solve StableHLO inventories reported 145,990 f32 value-type occurrences; this is not an IR byte-size measurement. Lowered IR bytes are unavailable. The raw backend generated-code byte field is zero, which supplies no meaningful executable-size measurement. Persistent and XLA auxiliary caches were disabled. Historical PR147 uses a full-solve entry including initialization and different arguments, so differences cannot be attributed solely to source repairs. The two moderate horizons establish no scaling result for registered T6400/train5120. No registered dataset, recovery fit, fit warmup or restart was touched. These moderate cases also add no fallback execution or changing-parameter callable-reuse evidence.

Validation and handoff: the reviewed validate_report independently rechecked both complete saved reports; the reviewed NumPy evaluate adapter was used only for fixed-coordinate arithmetic summaries, with no optimizer calls. Canonical ARCHITECTURE, DERIVATIONS and LIMITATIONS now state the measured boundary. Final artifact checks verify every archived file, source/spec hashes, summary field identities, tk dep cycle and git diff --check. This is evidence/documentation/ticket-only work; accepted focused tests and the unchanged 2,832-pass short suite are reused, with no new full-suite claim. Root explicitly owns independent artifact review, closure and publication. After that review, compiler activation, parameter-update reuse and registered recovery remain owned by the named downstream tickets. The author remains available for review feedback.

**2026-09-15T04:15:27Z**

Independent supervisor artifact review accepted at author commit ad98537c110b3bbb4017b49b10f15f8180dcc4a9; no findings. Review began only after the author committed and paused. Root notified the author after completion; no author repair was needed.

Root read and hashed every one of the 101 archived files, extracted a separate review copy, and revalidated both saved phases from that copy. The audit independently checked all 24 initial/terminal model-point energies, all twelve statuses/certificates/work records, all 30 initialization comparisons, the ten compile-only records and cap comparisons, all three logical inventories and nineteen output-group mappings, and the twelve numerical/storage process records. All summary claims matched the raw evidence. All 32 numerical/harness source hashes match the frozen executed revision 298790bf857753d4a9393015b299e9e90ca1ac84, current source, and the previously passing short-suite code revision 144fb21466894bfd91d2040ff2d5e1d003cc82e9. The specification, raw archive and numerical payloads are unchanged. Review commands/results live in /tmp/cff-a3wz-root-audit.py, /tmp/cff-a3wz-root-audit.log and /tmp/cff-a3wz-root-audit.json; the archived evidence remains reproducible through the committed runner validate command.

Decision: A3WZ passes its bounded pre-integration numerical and storage checkpoint. All required tiny/host/JIT/vmap and hostile-input evidence is inherited from the accepted source-matched prerequisites; the current moderate run supplies the missing native float32 convergence, independent initial/terminal arithmetic, initialization sensitivity and actual enclosing-program allocation evidence. No change to D12 mathematics, supported scope, approximation, acceptance thresholds or registered recovery gate is needed. This is evidence/documentation/ticket-only work, so the existing 2,832-pass final code gate (1,126.534 seconds under the 1,200-second cap) is reused; no new full-suite run is claimed. Root's artifact checks, git diff --check and tk dep cycle passed.

Handoff: close this prerequisite and publish the stacked evidence PR on PR149. The next fresh sequential author takes ready K6G5 declarations; compiler integration remains blocked on that ticket. The current D9 path still avoids unselected D12 compilation. The two moderate horizons do not establish registered long-horizon scaling, changing-parameter reuse, universal convergence or recovery. All three registered recovery fits remain unused. The separate affine-input audit cdh-kvy5 is nonblocking.

Dependency companion from tk dep tree --full cff-zov5, tk ready and tk blocked; TK records remain authoritative and include additional direct prerequisites. Legend: prerequisite --> dependent; joined inputs are all required.

[x] kernels/reference/solver/harness --> [x] A3WZ numerical checkpoint --+
[x] OCXL ---------------------------> [ ] K6G5 (ready; next author) ---+--> [ ] 7SW5
                                                                         |
                                                                    [ ] IDHK --> [ ] BRCF
                                                                                       |
                                                                                    [ ] F3JZ
                                                                                     /     \
                                                                               [ ] CZDQ   [ ] ULBZ
                                                                                     \     /
                                                                                    [ ] KUL2
                                                                                       |
                                                                                    [ ] X0IF --> [ ] ZOV5

K6G5 was structurally parallel to the checkpoint, but implementation remains sequential as Cooper requested. 7SW5 is immediately blocked by K6G5; downstream execution, learning and public composition remain required before X0IF's registered fits. A3WZ closure is not a recovery claim.
