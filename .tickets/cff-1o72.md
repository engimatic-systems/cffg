---
id: cff-1o72
status: closed
deps: [cff-o7hs, cff-n60h, cff-v53k, cff-ll6y, cff-3zzo, cff-domd]
links: [cff-fxmj, cff-o7hs, cff-ri8m, cff-l6i5, cff-209l, cff-cz6v, cff-u06e]
created: 2026-07-08T16:50:10Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Neural Gaussian factors: learned means, free-energy training, amortized recognition

## Background in Plain English — WHY

CFFG needs neural factors for nonlinear relationships that are too large or flexible for a table or a hand-written function. This ticket adds a Gaussian factor whose mean is a deterministic neural network: `y | x, theta ~ N(f_theta(x), V)`, with strictly positive-definite noise covariance `V`. Given the input and weights, the network computes the same mean; the modeled output can vary around it. The noise describes output variation remaining when the input and weights are known, separately from uncertainty propagated from the input or uncertainty in the weights.

The ticket trains the network parameters from a free-energy objective and lets a trained network propose better starting messages. It builds on cff-o7hs, which owns nonlinear Gaussian messages and their approximate ledger terms. It preserves the project rule that serialized templates remain the source of truth and that `cffg.core` can load and edit them without importing JAX or Equinox.

The neural node does not make an approximate objective exact. Its ledger entry remains a named surrogate, while analytic factors elsewhere in the graph keep their Bethe or Generalised free-energy meaning. Solver gradients describe an inference fixed point only after the runtime has certified the complete fixed-point equations; an exhausted, diverged, or infeasible run does not silently become training data.

## Outcomes / Acceptance Criteria — WHAT

1. cff-o7hs is closed with its nonlinear Gaussian message, fixed-point/site, damping, and approximate-ledger contracts satisfied before this ticket closes.
2. The serialized template schema declares a neural Gaussian factor `y | x, theta ~ N(f_theta(x), V)` between Gaussian edges, with a deterministic network mean and an explicitly specified noise covariance `V`.
   2.1. The template records a namespaced implementation identity and compatible version, the network architecture data needed to reconstruct the factor, and references to external blobs for network parameters and noise covariance. It does not store an Equinox module or a large parameter array inline.
   2.2. Core template loading, validation, editing, hashing, and bundle inventory do not import JAX or Equinox. Compilation fails with a specific missing-implementation or incompatible-version error when the required neural kernel is unavailable.
   2.3. A template and its blobs round-trip through JSON and `.cffgz` without changing the factor identity, network parameter values, noise covariance, structure hash, or compiled neural-factor behavior.
   2.4. The noise covariance is finite, symmetric positive definite, and has dimensions matching the network output.
   2.5. Invalid noise, including zero, singular, nonsymmetric, indefinite, non-finite, or shape-inconsistent covariance, fails compilation with a specific covariance error through the cff-o7hs contract.
3. The compiled neural factor evaluates the network and sends Gaussian messages through the nonlinear rule contract delivered by cff-o7hs.
   3.1. Forward values and Jacobians agree with an independent analytic network fixture within stated float64 and float32 tolerances.
   3.2. From an identical initial state, synchronous and scheduled execution produce messages, marginals, run status, and ledger values that agree within the stated tolerance.
   3.3. Each neural node records which cff-o7hs nonlinear constraint mode it uses. Its ledger term is labeled with that mode's named Laplace, linearization, or moment-matching surrogate, and analytic terms elsewhere in the graph retain their analytic labels.
4. The training API optimizes neural parameters against BFE and against GFE only when the complete graph already lies within the documented GFE scope.
   4.1. The training API differentiates a certified inference fixed point with respect to neural parameters through the complete coupled equations, including the dependence of linearization or sigma-point coefficients on cavities and weights. Gauge freedoms are removed or projected before the implicit derivative is solved; the reported approximate energy uses its total derivative, without an envelope or stop-gradient shortcut.
   4.2. On a reproducible fixture with declared positive noise variance, one scalar input, one scalar output, and no more than 20 neural parameters, parameter gradients agree with independent central finite differences at a tolerance justified by the finite-difference step, objective conditioning, and active dtype.
   4.3. The training API reports and refuses to apply a fixed-point gradient when inference is budget-exhausted, diverged, infeasible, non-finite, or outside its declared surrogate domain.
   4.4. On a seeded nonlinear-regression fixture with a held-out input grid, free-energy training lowers held-out error below both the untrained network and a linear-Gaussian baseline. The fixture records its generating map, observation noise, split, seeds, stopping rule, and numeric acceptance threshold before the final comparison is run.
   4.5. Saving and restoring the neural parameters and training state preserves the next parameter update and objective value within the checkpoint tolerance; loading a legacy checkpoint either migrates it explicitly or fails with a specific incompatibility error.
5. A trained amortizer proposes initial messages for a neural-factor graph without changing the model or its certified inference fixed point.
   5.1. On a paired held-out benchmark, amortized and neutral initialization reach fixed-point beliefs and ledger values that agree within the benchmark tolerance. A reported 95% confidence interval for the paired sweep-count reduction excludes zero.
   5.2. An amortized proposal never restores a state excluded by exact support and never converts an improper belief into a proper one through a numeric floor.
   5.3. If the amortizer supplies an importance-sampling proposal, the evaluator applies the corresponding importance weights, uses an explicit JAX PRNG key, reports effective sample size, and reproduces particles and results when the key and inputs are replayed.
6. The neural rule satisfies the numerical robustness contract.
   6.1. Sparse discrete support is N/A for a factor whose declared edges are nondegenerate Gaussian variables with support on `R^d`. Zero-noise and singular-noise conditionals are rejected; no covariance floor or precision clamp substitutes for an exact neural relationship.
   6.2. A vacuous Gaussian input follows the cff-o7hs neutral-message rule. Finite indefinite site precisions are permitted under its site contract; every accepted combined belief and tilted covariance must be finite and symmetric positive definite. Improper required beliefs and non-finite quantities follow its explicit failure-status and non-finite-ledger contract, never a finite converged result.
   6.3. Float32 execution derives floors, damping guards, positive-definiteness checks, finite-difference steps, and solver tolerances from the active dtype.
   6.4. A benchmark with at least ten times the basic fixture's input dimension, output dimension, parameter count, or neural-factor count completes within a documented runtime and memory envelope or returns a machine-readable non-converged status. The resource estimator includes neural parameters and declared per-sweep work.
7. The implementation covers compiler grouping and parameter packing, Gaussian arena addressing, synchronous steps, scheduled stages, BFE/GFE dispatch within the declared scope, run-status propagation, readouts, serialization manifests, resource estimation, and documented exports. It also covers `member_of`, or records why `member_of` is structurally irrelevant when neural parameters reside in a separately typed pytree.
8. `README.md`, `docs/ARCHITECTURE.md`, `docs/DERIVATIONS.md`, and `docs/LIMITATIONS.md` document construction with explicit positive noise covariance, training, checkpoint and blob behavior, fixed-point-gradient requirements, the named surrogate, and the boundaries against exact zero-noise factors and arbitrary nonlinear GFE.

## Methodological Suggestions — HOW

These suggestions are optional. Another method is acceptable if it satisfies the numbered outcomes with equivalent artifacts and executable evidence.

- Use the **New factor or rule family** row in `docs/ARCHITECTURE.md` as the primary change path. Also use the **Learnable or checkpoint feature** row for parameter updates and persistence, and the **Runtime numerical or status invariant** row for fixed-point-gradient gating.
- Build on the Gaussian `FactorKernel` binding supplied through cff-209l and cff-o7hs. Keep network callables and Equinox modules on the compiler/runtime side of the core-IR boundary. Store portable architecture data in the template and parameter arrays in `.npy` or `.safetensors` blobs.
- Audit compiler grouping, slot addressing, parameter packing, arenas, `member_of`, synchronous `_make_step`, scheduled `_batch_stage_fn` and `_scan_stage_fn`, `_make_bfe`, schedule classification, readouts, serialization manifests, estimator accounting, and package exports. Explicitly check host/JIT, flooding/scheduled, single-instance/`vmap`, and latest-schema/legacy checkpoint pairs; mark a pair structurally irrelevant only after stating why.
- Consider `jax.jacfwd` for the local network Jacobian and a custom JVP/VJP for the fixed-point solve. Attach the implicit derivative only to a solution whose full fixed-point residual meets the runtime certificate, following the gauge-reduced patterns already used by the GFE inner and mixed outer solvers.
- Derive the Laplace or other named neural-node message and ledger surrogate before implementation. Put formal equations and the implicit-gradient argument in rule docstrings and `docs/DERIVATIONS.md`; record the approximation boundary in `docs/LIMITATIONS.md`.
- Use independent oracles: an analytic affine or shallow network for values and Jacobians, central finite differences for parameter gradients, a dense-grid posterior for a nonlinear factor with at most two input dimensions or a Monte Carlo posterior with reported sampling error, and paired held-out models for amortized initialization. Record seeds, tolerances, benchmark hardware where timing matters, and confidence-interval construction.
- Test literal support preservation, vacuous and improper Gaussian inputs, rejection of invalid noise covariance, float32, uncomfortable network/factor scale, non-finite parameters, exhausted inference, replayed PRNG keys, serialization failure modes, and host/JIT plus `vmap` agreement.
- Per Cooper's 2026-07-08 delegation decision, keep message and ledger derivations plus the fixed-point-gradient argument under top-level review. Well-specified Equinox plumbing, training-loop scaffolding, analytic golden fixtures, and narrowly framed literature questions may be delegated and reviewed at top level.
- During implementation, run the focused neural rule, kernel, compiler, schedule, runtime-status, serialization, bundle, learning, estimator, and engine-oracle tests. After the final code change, run the dashboard-backed short suite once under the `AGENTS.md` protocol; run marked slow behavioral or scale tests only when this ticket's final evidence requires them.

## Future work

- cff-ahps uses this ticket to close the neural-factor branch of the universal factor-library roadmap after every other epic prerequisite is satisfied.
- cff-u06e owns the mathematical contract for exact zero-noise neural relationships `y = f_theta(x)`. It is independent of this ticket's positive-noise Gaussian contract and is not a prerequisite for this ticket or cff-ahps.
- Arbitrary Gaussian-only or nonlinear GFE messages remain excluded. cff-dfpi and cff-izjd selected the bounded discrete-regime epistemic route; expanding nonlinear GFE requires a separate ticket and a new derivation.
- Neural structure search, architecture selection, and automatic network growth are not part of this ticket. No downstream ticket for that work is identified.

## Definitions

- **Amortized recognition:** A trained function that predicts an inference starting point or proposal from model inputs so repeated inference can use less work.
- **Bethe free energy (BFE):** The graph objective formed from factor-local energy and entropy terms plus edge-entropy overcounting corrections.
- **Central finite difference:** A numerical derivative computed from objective values at equal positive and negative parameter perturbations.
- **Conditional noise covariance:** The matrix `V` describing output variation when the input `x` and network weights `theta` are fixed.
- **Deterministic neural mean:** The network prediction `f_theta(x)`, which is repeatable for fixed input and weights and supplies the Gaussian conditional's mean.
- **Effective sample size:** A diagnostic that summarizes how concentrated a set of importance weights is.
- **Equinox:** The JAX-compatible neural-network and pytree library already used by CFFG runtime state.
- **FactorKernel:** The executable registration record that binds a factor's structure, parameters, messages, energy, robustness declarations, and serialized identity.
- **Gauge freedom:** A parameter direction that changes a message representation without changing the represented belief.
- **Generalised free energy (GFE):** The active-inference objective used for graph regions with a supported P-substitution construction.
- **Implicit differentiation:** Computing how an inference fixed point changes by differentiating its complete defining equations instead of differentiating every solver iteration.
- **Importance sampling:** Estimating a target expectation with samples from another distribution and weights that correct for that proposal distribution.
- **JAX PRNG key:** Explicit state that makes JAX random sampling reproducible and traceable.
- **Laplace approximation:** A Gaussian approximation derived from local curvature around a mode.
- **Positive-definite covariance:** A symmetric covariance matrix with strictly positive variance in every nonzero direction; a singular covariance has a direction with zero variance.
- **Site / tilted distribution:** An unnormalized local factor used by expectation propagation / the normalized product of the cavity and local factor before moment matching. A cavity is the belief with the old site removed; an indefinite site precision can be valid when the combined belief is proper.
- **Pytree:** A nested container of arrays that JAX transformations traverse as one parameter object.
- **Inference fixed point:** A solver result whose complete message and approximation-coefficient equations satisfy the residual certificate; it need not be a stationary point of the total adaptive energy.
- **Support:** The inputs or states to which a probability model assigns nonzero density or probability.
- **Surrogate:** A named approximate objective term used in place of an unavailable analytic free-energy term.
- **Vacuous message:** A neutral incoming message that contributes no information.

## Notes

**2026-08-08T04:18:34Z**

Retained wiggle-audit candidates and intended meanings after the bounded current scan: n_a = sparse discrete support is structurally irrelevant to the declared nondegenerate Gaussian factor; preserve = a stated semantic or checkpoint-value invariant; register = executable FactorKernel registration; implement = the planned code and integration work owned by this ticket; support = literal probability support or the documented mathematical objective scope, as stated locally; complete = the locally named whole graph or execution termination; deliver = the prerequisite contract supplied by cff-o7hs; remaining = a preserved invariant or an explicitly excluded future capability; approximate = a named Laplace, linearization, or moment-matching method; close = acceptance-based ticket closure; certify = the runtime full fixed-point residual certificate; check = a numeric guard or an explicit duplicated-surface review; converge = the machine-readable converged/non-converged status; correct = importance weights mathematically compensate for the proposal distribution; equivalent = equal acceptance artifact and executable-evidence coverage; estimate = the named resource estimator or importance-sampling estimation; evidence = named automated checks and reference artifacts; exact = analytic mathematical meaning or literal zero support; match = the technical phrase moment matching; oracle = an independent analytic, finite-difference, grid, or Monte Carlo reference; reproduce = equal particles and results under replayed keys and inputs; robust = the AGENTS.md numerical robustness contract; test = automated test execution or named test artifacts; validate = core input/schema validation; infer = fixed-parameter state inference; learn = the integrated parameter-training test surface; learnable = the named persistent-parameter change-path category; scale = increased network dimensions, parameter count, or factor count; can = demonstrated core loading/editing capability or possibility in the amortization definition; may = explicit permission to delegate bounded work; explicit_owner = cff-o7hs owns prerequisite nonlinear rules while this ticket owns its numbered neural outcomes; named_ticket_handoff = the concrete cff-o7hs prerequisite owner; at_least = the inclusive ten-times lower bound; every = each cff-ahps prerequisite or each solver iteration, as stated locally; only = a temporal gate, structural irrelevance gate, test-selection condition, or explicit scope boundary. Intelligible-ticket review: complete.

Intelligible-ticket review: complete.

**2026-09-15T17:58:09Z**

Implementation handoff from bounded GPT-6 Astra xhigh code research: reuse gaussian_nonlinear plus a neural MeanImplementation and portable FunctionSpec configuration. Factor parameters have canonical blob/role ownership in member_of; build a JAX-only injection plan that scatters one trainable vector per tied blob to every packed reader. set_params is host validation, not differentiable. Extract the existing chart unpack, exact-forest reconstruction, full adaptive update and raw ledger composition from OuterKernelProvider into one shared root-program owner. For R_u transpose lambda = E_u, use a matrix-free adjoint solve and explicitly certify its original residual; JAX GMRES info alone does not certify convergence. Total gradient is E_theta minus R_theta transpose lambda, including all adaptive coefficient and reconstruction dependence. Gate on a matching certified result and finite energy; changing observations/weights invalidates a prior certificate. Existing Learnables persistence is conjugate-family-specific, so optimizer state needs a separate versioned array checkpoint with structure, fixed-parameter and target-layout identity, atomic restoration, and explicit PRNG/data-position state if owned. Amortization can be a deterministic learned proposal from certified targets; importance sampling is then N/A. Preserve exact support, reject improper proposals without floors, and verify held-out paired final-root agreement plus the requested95% sweep-reduction interval. Sources: params.py, compiled member addresses, outer_kernels.py, outer.py, runtime_types.py, learning_schema.py, _agent_checkpoint.py, and installed JAX GMRES. Confidence high for code ownership and observed API behavior; training/benchmark evidence remains to be implemented.

**2026-09-15T18:35:51Z**

Bounded implementation plan (tk dependencies are authoritative). Legend: prerequisite --> dependent.

[x] cff-o7hs (PR156)
  +--> [x] cff-n60h (portable neural mean)
         +--> [ ] cff-v53k (ready: complete fixed-point derivative)
                +--> [ ] cff-ll6y (optimizer and continuation)
                +--> [ ] cff-3zzo (deterministic amortized initialization)
                       |
        completed children --> [ ] cff-1o72 --> [ ] cff-ahps

Optimizer and amortizer work can proceed independently once the shared derivative/root boundary is available. Exact-zero-noise derivation cff-u06e remains independently ready and is outside this implementation.

**2026-09-15T19:16:17Z**

Implementation is complete across the bounded child tickets; the final short-suite gate remains pending. Architecture change paths: New factor or rule family, Runtime numerical or status invariant, and Learnable or checkpoint feature. Duplicated-surface audit: core JSON/hash/bundle and lazy import; parameter tying/injection; mixed/flood/exact-island reconstruction; local JIT/vmap and host run control; raw complete energy; dtype and exact support; checkpoint save/load and public facade are covered. Existing Colony uses the same neural rule and host adaptive solve. Training/amortizer own no RollingAgent cadence, conjugate posterior, particles, PRNG continuation or minibatch cursor; those pairs are N/A.

Legend: prerequisite --> dependent (only this ticket's immediate work shown)

[x] cff-o7hs --> [x] cff-n60h --> [x] cff-v53k
                                    |       |
                                    |       +--> [x] cff-3zzo --+
                                    +--> [x] cff-ll6y ---------+--> [ ] cff-1o72 (final gate)
[x] cff-6kwo ---------------------> cff-ll6y                         |
                                                                   +--> [ ] cff-ahps (other blockers remain)

cff-u06e remains independent future work for exact zero noise. cff-1et5 tracks derivatives of an optional entropy-bound diagnostic outside the training energy. tk ready confirms cff-1o72 has no open prerequisites; tk dep cycle passes. Frozen regression and paired-recognition benchmarks passed; both code and validation details are in the child tickets and D14.


**2026-09-16T00:26:38Z**

Completed neural Gaussian means, total certified fixed-point gradients, full-batch Adam continuation, deterministic learned message initialization, and sparse GFE reverse-mode repairs. All child implementation tickets are closed. The exact zero-noise factor remains the independent future ticket cff-u06e; optional diagnostic-bound derivatives remain cff-1et5.

Final validation at production code head 10e0615: uv run python tools/dashboard.py --run-suite passed 2,404 tests with 58 slow deselections. Pytest time 1,265.83 seconds; measured whole-suite wall 1,279.975 seconds (21m19.975s). Log: /tmp/cffg-neural-short.log; dashboard: /Users/ravd/agent-share/cffg-tests.html. The frozen regression and paired initializer benchmarks passed separately, as recorded in cff-ll6y and cff-3zzo.

Following Cooper's explicit protocol, cff-domd moved nine already-passing lower-priority cases to slow, totaling 105.936 measured seconds. Estimated remaining wall time is 1,174.039 seconds (19m34.039s), not a new measured result. Collection confirms exactly those nine removals, no additions, and 2,395 selected tests. Production code is unchanged after the passing gate; no suite rerun was performed for selection-only pruning. Runtime history and dashboard retain the measured result. All required neural gradients, exact-support, float32 and scale checks remain short. git diff --check and tk dep cycle pass after ticket whitespace cleanup.

Landing: cff-1o72 branches from cff-o7hs / PR #156; publish as its own stacked PR. No implementation work remains for this ticket.

**2026-09-16T00:27:21Z**

Published PR #157: https://github.com/engimatic-systems/cffg/pull/157 . Base is cff-o7hs / PR #156. Both requested implementations are complete in separate reviewable PRs; all implementation and budget tickets are closed. The passing suite and post-run selection estimate are recorded above. No required implementation or validation work remains.
