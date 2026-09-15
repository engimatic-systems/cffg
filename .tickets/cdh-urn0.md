---
id: cdh-urn0
status: in_progress
deps: []
links: [cdc-ztuf, cdh-kvy5]
created: 2026-09-14T22:56:19Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-zov5
tags: [hybrid, numerical, inference]
---
# Diagnose native weighted covariance projection discrepancy before moderate qualification

## Background in Plain English — WHY

The cdc-ztuf report repairs exposed a native-float32 projection discrepancy in unchanged hybrid_natural.marginal code. A single T2-shaped local pair belief with K=4 and d=2, nonuniform masses proportional to 1..16, means spaced by 1/16, covariance 100000*I and retained low words eps*high/4 produces source-label-0 off-diagonal covariance 0.05000000074505806. Replaying the prescribed mixture operations from the saved belief gives about 0.0625. The difference exceeds the frozen local allowance 512*4*eps32 (absolute and relative). A uniform rare source at mass scale 1e-12 also produced 0.076171875 versus 0.078125. Both native status flags were zero. This is local arithmetic evidence, not a moderate solve or qualification.

The repaired report validator rejects this discrepancy; weakening it would conceal an unresolved numerical problem. The implementation author must finish ordinary report repairs and pause, so diagnosis and any core numerical repair belong to this separate bounded ticket. No underlying numerical code was changed during discovery.

## Outcomes / Acceptance Criteria — WHAT

1. Identify whether compiler transformation, native local arithmetic, or the independent replay causes the covariance discrepancy, with a bounded reproducer and operation-level evidence.
2. Preserve the frozen D12 mathematical definition, approximation, dtype policy and allowances; do not make a discrepant result pass by increasing tolerances or choosing only favorable masses.
3. Make the preserved native projection regression pass normally, including nonuniform and rare-source controls; remove its explicit expected-failure marker only with supported repair evidence.
4. Preserve both retained covariance words for residuals and the once-rounded physical covariance consumed by learning. No universal exactness claim is required.
5. Before implementation, record the architecture change path and affected host/JIT/single/vmap/derivative surfaces. Sparse support, vacuous inputs, float32 and 10–100x domain size need explicit planning answers. Bound any larger audit prospectively in a dependent ticket.

## Methodological Suggestions — HOW

Optionally start with tools/d12_moderate_checks.py --projection-probe and the strict expected-failure regression in tests/test_d12_moderate_integrity.py. Compare the native marginal against saved-input replay and exposed intermediate local operations. During discovery, a separate compilation of mixture intermediates produced approximately 0.0625 while the marginal returned 0.05 from the same saved region; this suggests execution-context sensitivity but does not establish its cause. /tmp/cdc-ztuf-local-debug.py and its output contain supplementary diagnostics; the committed probe is the durable reproducer.

## Future work

This ticket is an explicit prerequisite of cff-a3wz moderate qualification. cdc-ztuf owns report validation and setup review; it does not own a new solver-arithmetic change. Root must assess this blocker before authorizing subsequent immutable source requalification. No moderate solve, storage compilation, recovery fit, historical-evidence rewrite, or extra timing solve is authorized by this ticket.

## Definitions

D12 is the complete hybrid message/auxiliary solve. T2 describes two time points; K is label count and d is continuous dimension. A covariance low word retains the rounding remainder alongside its primary float32 word. eps32 is float32 machine epsilon. A source-conditioned marginal is the projected Gaussian mixture for one source label. JIT means just-in-time compilation; vmap is JAX batching. An expected-failure regression keeps a known defect visible without claiming it passed.



## Notes

**2026-09-14T23:24:33Z**

Independent published-source scope audit at d22a43a, not an arithmetic diagnosis. Question: do existing public D9 models or fit_rslds reach the questioned D12 endpoint projection/covariance mixer? Answer: built-in public D9/fit_rslds paths do not. fit_rslds builds Engine(compile_template(...)) and calls engine.run; recurrence execution/readout uses recurrence.structured_recurrence_update and recurrence.weight_stats. Public hybrid_gfe/hybrid_learning modules are separate implementations. This establishes call-path isolation, not a claim about explicit external users of unbound D12 modules.

Affected D12 reachability: marginal -> collapse_label -> _mix -> hc.combine; gaussian_adapter also calls _mix. Local hybrid_separator.evaluate projects both endpoints for message targets and consistency. Joint evaluate/initialize therefore reach it, as do feasible initialization, trials, residual linearization/CGLS fallback and terminal certification. Storage buffers evaluates/linearizes, and exposed residual/iteration/certification/complete-solve programs inherit those paths; arrays alone inventories leaves. Eager/JIT and single/vmap are applicable verification pairs, not proof the discrepancy occurs in every mode/dtype. The independent NumPy reference has its own projection.

Repair-boundary caution: if hc.combine is changed, verify its broader D12 users in natural addition, transition precision assembly and retained inverse-word normalization/canonical conversion, not only mixtures. Standalone auxiliary_statistics uses a different suffix mixer and does not directly call _mix, although solver-returned learning statistics depend on the affected inference path.

Sources in /Users/ravd/projects/cffg-d12-scaling: models/rslds_fit.py:334,478; execution/cross_domain.py:72; readouts/recurrence.py:93; hybrid_natural.py:73,470,548; hybrid_separator.py:131,195,430; hybrid_joint.py:210,260; hybrid_joint_solver.py:142; hybrid_joint_storage.py:29; hybrid_covariance.py:169. Confidence high for published call-path scope. No active harness inspection, edits, tests, compilation or numerical calls ran, and this report does not independently establish the affected input range or arithmetic root cause. Distill into cdh-urn0 after the current author pauses, for a fresh bounded implementer's planning and honest CFFG impact statement.

**2026-09-14T23:44:22Z**

Independent read-only research at published d22a43a. Question: what operation-order assumptions do hybrid_natural._mix and hybrid_covariance.combine require, and which installed transformations preserve their protection? Answer: _mix preserves small covariance increments around the first positive-mass high-word base, averaging (H-base)+L+centered_mean_outer_product. Probability sums, division, mean subtraction and H-base subtraction are ordinary floating operations outside affine; compensation cannot recover already-lost input information. combine computes a rounded high word through affine, then reevaluates the sum with -high to obtain the remainder. TwoSum and this remainder require actual intermediate rounding; algebraic reassociation can invalidate the identity.

Current protections: hybrid_affine._checked_sum places lax.optimization_barrier around checked addition/subtraction results and inspects bit patterns for exact cancellation versus unsupported subnormal inputs/results. Products use integer significands, two limbs and explicit nearest-even rounding rather than a floating multiply/subtract expansion vulnerable to FMA contraction. affine accumulates words in lax.scan and adds offsets separately. A barrier orders only dependent uses through its returned value. Installed JAX lowers the barrier to OptimizationBarrierOp and registers batch, JVP and transpose rules; the nearby docstring's claim that differentiation/batching are absent is stale relative to these registrations. Nested jit/custom_jvp alone is not a documented backend rounding fence.

Differentiation: affine, compensated product and rounded_product provide ordinary real-expression JVPs; tangents use ordinary floating arithmetic. Integer bitcast JVPs are zero, making custom-JVP boundaries essential. inverse assigns zero tangent to its derived low word; combine has no separate custom rule and differentiates through both affine calls. Spectral certification stop_gradient controls differentiation, not reassociation. A zero status checks representation/properness, not the mixture identity independently.

Sources in /Users/ravd/projects/cffg-d12-scaling: src/cffg/hybrid_natural.py:470; hybrid_covariance.py:110,120,192; hybrid_affine.py:24,53,60,145,177. Installed .venv/lib/python3.12/site-packages/jax/_src/lax/lax.py:5561,9711; api.py:297; custom_derivatives.py:467. Confidence high for code/control-flow/registered transform facts. Compiler reassociation, pre-helper rounding, and representation/accumulation errors remain unconfirmed hypotheses. No edits, kernel execution, compilation or tests occurred. Root will distill this note into cdh-urn0 after the active author/reviewer pauses; it is guidance for fresh diagnosis, not a new numerical result.

**2026-09-14T23:45:51Z**

Fresh implementation plan (authorized after author pause and independent re-review). Work on the existing cdc-ztuf-moderate-harness branch; base/bootstrap already completed by root. Diagnose only local kernels, using preserved weighted/rare probes and independent mixture arithmetic. First locate the earliest differing operation and compare intermediate rounding in enclosing versus separate compilation. Compiler reassociation is a hypothesis until operation-level evidence establishes it. Make the smallest reliable arithmetic repair; preserve identities, both covariance words, approximation, dtype policy and frozen allowances. If bounded ordinary repair cannot satisfy these, commit exact evidence and pause for root.

Architecture change-path rows: Runtime numerical or status invariant, and Derived component coordinate (local unbound D12 arithmetic subset). Applicable: marginal -> collapse_label -> _mix -> hybrid_covariance.combine and gaussian_adapter; downstream separator/joint targets, residuals, certification, storage and learning views inherit this path. If affine/combine changes, inspect natural addition, transition precision assembly, inverse normalization and canonical conversion too. Verify host/JIT and single/vmap, local JVP and reverse derivatives. Flooding/scheduled and current/legacy checkpoint pairs: N/A, no public runtime integration or serialization change. IR/topology/packing/registered execution/ledger/public exports/estimator: N/A. Built-in D9 and fit_rslds do not invoke this path. Abstract source binding: check after final native edit; regenerate only the prospective abstract descriptor if invalidated, preserving revisions and immutable evidence.

Hostile inputs and evidence: retain weighted and rare-source regressions plus uniform controls; compare both retained words and once-rounded physical covariance against independent stored-input arithmetic. Exact zero masses must stay exactly absent, including an entirely absent source. Vacuous/all-zero probability or nonfinite represented values must retain nonzero status; positive-mass underflow must never silently become support deletion. Native f32 is required with x64 disabled; existing f64 primitive checks remain applicable. Bound scale to 40 labels (10x K=4; 100x pair-count) with small continuous dimension, local projection only. Include meaningful derivative and broader combine-user checks if shared arithmetic changes. No complete optimizer, registered fit, full short suite, historical runtime/evidence edits, publication, or review worker. Root owns final full gate after independent review and reference-adapter work. End with atomic cdh-urn0 commits, focused command/results handoff, clean tree and ticket in_progress, then pause.

**2026-09-14T23:50:24Z**

Bounded diagnosis before repair: original weighted probe reproduced status 0 and covariance[0,0,1]=0.05000000074505806. /tmp/cdh-urn0-boundaries.py isolated the first discrepancy to the weighted variation accumulation (before hc.combine); separate high/low combination of the saved correct increment agrees. Returning mean/centered/variation leaves yields 0.0625, while increment alone yields 0.05. Barriers at mean, centered difference, covariance difference and increment did not repair it. Smaller /tmp/cdh-urn0-decode.py returns variation 0.0625 and full bits 1031798784, yet separately returned decoded mantissa/exponent fields include mantissa 0 with exponent 122 (the 0.03125 binade) where exponent 123 is expected. This demonstrates native execution-context sensitivity in recomputed product inputs; independent mixture/replay is not the source. Optimized local HLO (/tmp/cdh-urn0-words-hlo.txt) replaces mean division with reciprocal multiplication and distributes centered-product decoding across fusions. The exact compiler pass/contraction responsible remains unproven; do not label this a confirmed compiler fault.

Repair candidate: compute the centered difference through existing compensated affine, supplying relative means and negative conditional mean as separate terms. The bounded candidate gives 0.0625 and the expected retained low words without changing mixture weighting, combine, affine primitives, thresholds or support semantics. Normalizing weights before accumulation also repairs this example but changes more operation order; pursue the smaller centered-difference change first. Test the preserved probes, independent mixture identities across powers-of-two and nearby inputs, native transformed surfaces and derivatives, sparse/nonfinite/underflow status and planned K=40 local projection. Broader arbitrary composed affine/bit-decoding certification is outside this bounded fix; if these checks expose additional failures, split evidence and pause before expanding.

**2026-09-14T23:57:12Z**

Repair and scope update: only _mix centering now uses the existing compensated affine helper; combine/affine primitive implementations are unchanged. Both committed weighted/rare probe commands pass with native=true, local_cases=2, complete_program_calls=0 and retained_low_word=0.0029802322387695312. New independent tests compare covariance increments after subtracting the common large base, so the diagonal cannot hide error behind 100000; they also check once-rounded physical covariance, six small controls (uniform/weighted/rare/adjacent binary inputs/unequal covariance), native host/JIT and single/vmap, analytic JVP/reverse derivatives including low words, and K40 weighted/rare cases. The old ordinary-centering expression fails this new oracle in /tmp/cdh-urn0-regression-control.txt. This control changes only the in-memory function, not source files.

The residual general composed-affine input-decoding audit is split into dependent cdh-kvy5 with a compact reproducible expression and exact observations. No shared primitive expansion is included here. Compiler-pass attribution is still unproven; the narrow repair is reviewed against its own numerical evidence.

Prospective descriptor regenerated after the final native edit, using tools/d12_moderate_layouts.py. Old hash a8a4205509c9cf19b6037338e5fda584258c1ea2c9f5665c1346c4be71b766ac (8a9c9a0) is preserved in the spec revision; new hash e2ab3a4ce574b67fa3243ecb08744971408efc35581a67802b18259dc797605c. Runtime binding is unchanged; only hybrid_natural.py source binding changed. The custom affine differential adds eight local derivative-capture leaves (773 -> 781), adding 16,384 logical bytes at T32 K4 d2, 51,200 at T100 K4 d2, and 6,400,000 at abstract T100 K20 d10. Every other inventory group is unchanged. These are abstract logical sizes, not compiled memory measurements. Complete-program lower/compile/execute counts remain zero; no historical evidence or runtime history changed.

**2026-09-15T00:03:03Z**

Implementation handoff — ready for root's independent review; leave in_progress.

Done: narrow compensated-centering repair in hybrid_natural._mix; preserved weighted/rare regressions now pass normally (strict xfail removed), with uniform controls unchanged. Added native float32 independent moment/retained-word/physical-rounding controls, host/JIT and single/vmap coverage, analytic JVP and reverse derivatives for probabilities, means and low words, exact absent-source support, empty/nonfinite/subnormal status, inherited normalization underflow, positive normal source masses whose weighted variance is unrepresentable, and K40 weighted/rare projections. Shared combine/affine/inverse primitives and mathematical/dtype/threshold contracts are unchanged. Gaussian-adapter and canonical/quotient coverage is included in the existing projection file. Architecture note and prospective descriptor revision accompany the source change; registered the new test file in dashboard EPIC_FILES without changing execution/selection policy.

Final focused results after the final native source edit:
- uv run pytest -q tests/test_hybrid_weighted_projection.py tests/test_d12_moderate_integrity.py tests/test_d12_moderate.py: 104 passed, 2 slow tests deselected, 91.94 s. Log /tmp/cdh-urn0-native-harness-tests.txt.
- After adding the final positive-normal-mass/unrepresentable-variance assertion, uv run pytest -q tests/test_hybrid_weighted_projection.py: 1 passed, 40.37 s. Log /tmp/cdh-urn0-weighted-final.txt. This is the final run of the new native test.
- uv run pytest -q tests/test_hybrid_projection.py: 31 passed, 408.05 s. Log /tmp/cdh-urn0-projection-tests.txt. Covers existing f32/f64 projection, ordinary adapters, canonical conversion, quotient and covariance derivatives, represented-word consumers and status cases.
- Both preserved CLI probes --projection-probe weighted and --projection-probe rare passed; unchanged --arithmetic uniform controls passed inside the integrity file.
- The in-memory original-centering control fails the new independent oracle with maximum covariance-increment error 0.0125; /tmp/cdh-urn0-regression-control.txt. No source edit was used for the control.
- Prospective descriptor generation reports lower=0, compile=0, execute=0. Verified unchanged runtime binding and all non-derivative inventory groups; only native source binding and the eight additional local derivative captures differ. git diff --check and tk dep cycle pass.

Next: root starts a fresh independent reviewer after this committed pause. Root owns the final short-suite gate after reviewed kernel and cdh-m75a reference-adapter work; cdc-ztuf owns final harness integration. No full short suite, complete D12 optimizer, registered fit, tiny checkpoint, moderate/storage compilation, push/PR, merge or external message was performed. Historical tiny attempts/spec, compile archive/report and runtime history remain untouched.

Gotchas: the evidence supports this bounded repair, not universal composed-affine exactness or a confirmed compiler-pass attribution. The separate reproducible decoding audit is cdh-kvy5. Abstract derivative captures grow by the recorded local sizes; actual compiled-memory/timing qualification belongs to root's later gates. No user decision is currently required.

**2026-09-15T00:04:10Z**

Independent read-only source research at published d22a43a. Question: are compensated float32 product sign/significand/exponent decoded from one bit pattern, or independently evaluated floating exponent operations? Answer: hybrid_affine._product_words.decode uses one bitcast_convert_type(value,uint32) expression for all input fields. Raw exponent E=(bits>>23)&255; significand M=(bits&(2**23-1))+[E>0]*2**23; decomposition power P=max(E,1)-150; sign S=bits>>31. The finite represented value is (-1)**S*M*2**P. There is no input frexp/logarithm/floating exponent calculation. JAX lowers the bitcast to hlo.bitcast_convert. No explicit optimization barrier is present in decode itself. Common source expression does not prove how the optimized executable shares or materializes an upstream floating value.

Integer limb multiplication, nearest-even rounding and remainder extraction precede jnp.ldexp reconstruction of the rounded product and remainder. Significand-to-float conversions are intended exact. _two_product rejects nonfinite or nonzero subnormal words; rounded_product deliberately permits backend subnormal scaling. Installed ldexp calls frexp on these converted output significands (not original operands), adds the supplied exponent after float conversion, and reconstructs through floating multiplication and power. That frexp uses one normalized bit pattern, with floating subnormal normalization; power lowers to hlo.power rather than a dedicated ldexp operation.

Exponent conventions must be explicit:0.0625 has raw exponent123, decomposition power-27 and frexp exponent-3. The author's ticket/probe reports raw122 versus123, so this warning does not itself invalidate the observed discrepancy. Sources in scaling worktree: src/cffg/hybrid_affine.py:81,91; installed jax/_src/lax/lax.py:5569,4791; numpy/ufuncs.py:2957,3004,3065. Confidence high for source/lowering account only; no cause for the observed mismatch or output-sensitive behavior is established. No kernels, compilation, tests or edits ran. This research neither rejects nor qualifies the proposed narrow compensated-centering repair; root will retain it with the numerical ticket after author pause for independent review.
