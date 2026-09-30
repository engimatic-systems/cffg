---
id: cff-qu2j
status: closed
deps: [cff-vxza]
links: []
created: 2026-09-25T03:46:25Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Resolve float32 exact Gaussian-chain certification failures against an independent oracle

## Background in Plain English

Replaying the saved first-order controller with its normalized observation [-1,-0.6] gives budget_exhausted in float32 after one exact pass (residual 0.0006103515625), while float64 converges (about 9.1e-13). This is evidence for numerical investigation, not a reason to declare all Gaussian models float64-only or loosen every certificate.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The saved three-dimensional, past-4/horizon-15 problem is reduced to a self-contained numerical regression.

2. Posterior means, covariance and free energy are checked against an independent dense-Gaussian or Kalman reference in both dtypes.

3. The investigation distinguishes message recomputation error, conditioning and genuine certificate failure, and records the resulting supported precision contract.

4. A correction preserves protections against stale schedules and unrelated scale inflation in tests/test_schedule.py; a non-root cannot become converged by relaxing a global threshold.

5. Sparse/zero information, improper/vacuous messages and a horizon at least ten times larger are exercised with dtype-derived guards.

6. README and LIMITATIONS explain any remaining precision sensitivity and how to select precision before constructing arrays.

## Methodological Suggestions

Inspect exact_strategy.py and Gaussian information-form operations. Architecture: Runtime numerical or status invariant. Pairs: host/JIT checks, flooding/exact scheduled comparison, Engine/Colony parity; checkpoint N/A. Retain independent moment/energy checks before accepting a certificate change.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-d63w — Reduce avoidable float32 Gaussian message/readout accuracy loss without weakening certificates.

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Float32/float64: 32-/64-bit floating-point numbers. Certificate: a numerical check supporting convergence. Kalman reference: an independent linear-Gaussian filtering/smoothing calculation. Residual: the measured mismatch used to assess convergence.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-09-30T21:01:17Z**

Pre-implementation handoff: Runtime numerical or status invariant architecture row. Investigate exact_strategy certification and Gaussian information operations before selecting a repair; no global tolerance relaxation. Host/JIT state checks and map arithmetic, flood/exact scheduled executions, Engine/Colony apply; adaptive nonlinear certified-return code is unchanged unless new work is split. Current/legacy checkpoints N/A absent schema change. Hostile checklist: partial/vacuous zero-information Gaussian messages preserve their defined support/integrability semantics; invalid/nonfinite messages remain loud; f32/f64 posterior and FE checked independently; >=H150 chain scale retained as opt-in. One current ticket; source constants delegated as specific read-only research.

**2026-09-30T21:08:49Z**

Source research question—what exact saved first-order inputs reproduce the report? Answer: six retained controller candidates (runs v3a and runs_v1, seeds0..2), same3D[x,d,u],P4/H15 graph. Current e4=[-1],future e5..e19=[-.6,0],past e0..e3=None. F=[[a,1,B],[0,1,0],[0,0,1]],c=[b,0,0],Q=Qp=diag(qx,.000025,4),m0=0,P0=diag(1,.000625,1);past/goal projection[x,u],current[x];Rpast=diag(.0001,.000001),Rnow=.0001,Rgoal=diag(.0004,1),offset0. Original auto run4,tol1e-6,damp0. Source common/agent.py97-110,141-157,179-191,222-245 and common/graphs.py72-103. Static pickletools+struct extraction of only numeric bytes recovered qx=psi/nu+1e-8,nu99993; NO unpickling,class import/reducer execution or pickle dependency. Six[a,B,b,qx] tuples will be preserved in maintained fixture. Ticket residual seed/version not attributed; report only~3e-3, so do not silently call any chosen tuple the exact incident. v3a results records fce60bd,JAX0.11.2,NumPy2.5.3,Python3.12.13,arm64,x64enabled. Oracle should compare to dense conditioning using both original64-bit and f32-rounded input constants to separate input rounding/inference error. Confidence high on arrays/topology, uncertain on exact-residual provenance. H150 repeated goal is an explicitly declared scale variant, not preview-schedule replay.

**2026-09-30T21:24:57Z**

Controlled attribution: source-order fixture now inserts emission before transition, uses source float64 covariance inversion before runtime precision casting, and recovers all six saved candidate constants without executing pickle data. Current branch certifies all six in f32 with raw recomputation residual zero; detached historical fce60bd rejects all six, residuals .0003662109375 to .0006103515625. Candidate v3a2 also reproduces the exact reported .0006103515625 when ONLY current gauss.equality_messages is replaced by the historical total-minus-recipient equation; restoring exclusive cavities certifies with zero residual. This ablation leaves certificate code and thresholds identical. ed9b40d (cff-vxza, PR #170) already repairs the status mechanism: recipient rounding must not affect its outgoing cavity. Source research question -> answer -> sources: Gaussian equality changes, gauss.py251/_cavities.py6; both routes kernel_runtime.py126; unchanged exact_strategy.py104 and schedule_program.py324. Confidence high on causal mechanism, exact incident seed remains unattributed. Accuracy remains separate: runtime-rounded dense reference vs original inputs differs <1e-7 in mean/cov for v3a2, whereas inference differs ~1.17e-4 mean/~3.08e-5 cov; f64 is ~1e-12. No production certificate adjustment is warranted.

**2026-09-30T21:32:34Z**

Bounded accuracy contract chosen independently of maxima: 1% posterior-standard-deviation mean error and relative covariance spectral error, 0.001 nat per observed scalar; f64 comparisons are tighter. These targets are fixture-only, not generic error guarantees. Independent NumPy Joseph filtering/smoothing confirms additional avoidable f32 mean loss; precise follow-up cff-d63w depends on qu2j and is linked to cff-wt1c without blocking f64 benchmark work. Legend prerequisite --> dependent: [in_progress]qu2j --> [ ]d63w (immediate blocker qu2j); parallel ready roots ees0 and nd1q; parent roadmap xa6z and existing wt1c remain blocked by their recorded feature dependencies. tk dep tree --full, tk ready/tk blocked checked and cycle check passed. Reviewer confirmed source order/covariance conversion and oracle independence; added returned free-energy trace check as well as recomputed ledger check.

**2026-09-30T21:35:43Z**

Current validation: default retained precision/mechanism + README checks 8 passed,12 deselected (19.75s before helper split); equality/Gaussian/schedule/README focused checks 58 passed,3 deselected (25.43s). First opt-in hostile run found a real all-missing f32 energy error, not a certification error: .204666 absolute vs analyticalzero. It is explicitly excluded from conditioned energy guarantees and retained as strict expected failure in cff-d63w. The cff-qu2j PR will include the discovery ticket but not implement or close its independent accuracy work; one primary ticket/PR as just requested.

**2026-09-30T21:37:23Z**

After final helper/trace check: uv run pytest -q tests/test_first_order_precision.py tests/test_readme.py -> 8 passed,13 deselected in22.65s. Retained-source replay and opt-in ablation do not require the removed detached historical worktree anymore; fce60bd remains recoverable in Git. Atomic commits89a2837 (oracles/hostile regressions/ticket discovery),3bf2c22 (precision contract docs). No production message rule or solver threshold changed in this ticket.

**2026-09-30T22:05:02Z**

Final short gate at3bf2c22: uv run python tools/dashboard.py --run-suite ->2640 passed,185 deselected,0 failures; pytest1195.71s,wall1210.236s. Full configured quiet wait1500s honored; refreshed /Users/ravd/agent-share/cffg-tests.html. Required <=1200s budget exceeded10.236s. Per-case timings from passing dashboard: retained f64 replay9.078s; exact-Colony size1 zero/undamped/damped replays .160+6.061+1.651s; batch3 damped replay3.740s. Move those five lower-value replay cases to opt-in slow, retaining default f32 exact/controller oracle, batch3 zero/undamped Colony correctness, existing f64 Kalman/energy tests and hostile schedule protections. Removed measured20.690s; estimated remaining wall1189.546s <1200s. Marker-only changes; passing suite result reused per repository rule and explicit user instruction, no rerun. Final opt-in regression batch before these markers:12 passed,3 deselected,1 strict known xfailed in151.31s. Focused final default+README8 passed,13 deselected22.65s. Known energy accuracy issue remains cff-d63w; no certificate tolerance or production math changed.
