---
id: cff-o7hs
status: open
deps: [cff-4e5z, cff-ri8m, cff-209l]
links: [cff-1o72, cff-fxmj, cff-ri8m, cff-l6i5, cff-pksr, cff-lquu, cff-k9ws, cff-209l]
created: 2026-07-08T16:49:34Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Nonlinear Gaussian factors: linearization + sigma-point moment matching (EP)

## Background in Plain English

CFFG has exact Gaussian messages for linear relationships, but many dynamics and sensors are nonlinear. This ticket adds a Gaussian factor whose mean is a deterministic function `g(x)` and whose output has nondegenerate Gaussian noise. It supplies a local linearization and a sigma-point method that better captures curvature. Both methods keep messages Gaussian, so they are approximations whose ledger terms and failure modes must be labeled honestly.

The work builds on the Gaussian arena from cff-4e5z, executable factor registration from cff-ri8m and cff-209l, and the dtype-scaled Gaussian propriety contract from cff-3hgi. It activates the reserved `MOMENT_MATCHING` node constraint for expectation-propagation updates. It does not cover zero-noise continuous delta factors, continuous-to-discrete links, or epistemic GFE scoring through nonlinear nodes.

## Outcomes / Acceptance Criteria

1. The dependency contracts are satisfied before this ticket closes.
   1.1. cff-4e5z is closed with its Gaussian arena contract satisfied.
   1.2. cff-ri8m is closed with its executable-registration contract satisfied.
   1.3. cff-209l is closed with its Gaussian `FactorKernel` binding satisfied.
2. The public template schema represents the nonlinear Gaussian factor.
   2.1. The factor has the form `y ~ N(g(x), V)` with Gaussian `in` and `out` edges.
   2.2. The factor records a namespaced implementation identity and compatible version for `g`.
   2.3. The factor requires a symmetric positive-definite noise covariance `V`.
   2.4. Loading or compiling without the recorded implementation fails with a specific missing-implementation error.
   2.5. Loading or compiling with an incompatible implementation version fails with a specific version error.
   2.6. Compiling a shape-inconsistent function fails with a specific shape error.
   2.7. Compiling invalid noise fails with a specific covariance error.
   2.8. A JSON round-trip retains identical function identity, parameter-blob values, constraint details, and structure hash.
   2.9. A `.cffgz` round-trip retains identical function identity, parameter-blob values, constraint details, and structure hash.
   2.10. Recompilation after either round-trip produces messages and ledger values that agree with the pre-round-trip engine within the stated tolerance.
   2.11. A zero-noise deterministic delta is rejected instead of being approximated by a large precision clamp.
3. The factor supplies declared linearization and sigma-point message methods.
   3.1. The linearization method evaluates `g` at the incoming cavity mean.
   3.2. The linearization method evaluates the Jacobian of `g` at that mean.
   3.3. The linearization method propagates Gaussian natural parameters through the resulting local affine model.
   3.4. The sigma-point method declares its point set.
   3.5. The sigma-point method declares its mean and covariance weights.
   3.6. The sigma-point method declares its covariance correction.
   3.7. The sigma-point method declares the admissible range of each scaling parameter.
   3.8. The sigma-point method matches the transformed mean and covariance without sampling randomness.
   3.9. For an affine `g`, linearization messages agree with `gaussian_linear` within stated float64 and float32 tolerances.
   3.10. For an affine `g`, linearization marginals agree with `gaussian_linear` within stated float64 and float32 tolerances.
   3.11. For an affine `g`, linearization node-energy values agree with `gaussian_linear` within stated float64 and float32 tolerances.
   3.12. For an affine `g`, sigma-point messages agree with `gaussian_linear` within stated float64 and float32 tolerances.
   3.13. For an affine `g`, sigma-point marginals agree with `gaussian_linear` within stated float64 and float32 tolerances.
   3.14. For an affine `g`, sigma-point node-energy values agree with `gaussian_linear` within stated float64 and float32 tolerances.
   3.15. The nonlinear comparison fixture is range-bearing or comparably curved and has at most two latent dimensions.
   3.16. The fixture records its domain and dense-grid reference resolution before the final comparison.
   3.17. The fixture records its posterior-error metric and numeric threshold before the final comparison.
   3.18. The linearization method reports its posterior error against the dense-grid reference.
   3.19. The sigma-point method reports its posterior error against the dense-grid reference.
   3.20. The sigma-point error does not exceed the linearization error plus the measured quadrature discretization error.
4. A `MOMENT_MATCHING` constraint compiles to an expectation-propagation site update.
   4.1. The site update runs through synchronous inference.
   4.2. The site update runs through scheduled inference.
   4.3. Each update removes the old site from the local belief to form a cavity.
   4.4. Each update computes the declared approximate tilted moments.
   4.5. Each update obtains the replacement site by subtracting cavity natural parameters from the matched posterior natural parameters.
   4.6. A site precision may be indefinite because a site is an unnormalized factor rather than a belief.
   4.7. Every accepted cavity-plus-site belief is symmetric positive definite and finite.
   4.8. Every accepted tilted covariance is symmetric positive definite and finite.
   4.9. Damping, backtracking, or another declared globalization rule accepts an update only when the combined belief remains proper.
   4.10. The globalization derivation names and justifies its acceptance metric, such as decreasing stationarity residual or another scalar merit function tied to the EP stationary equations.
   4.11. The globalization rule does not add an absolute ridge.
   4.12. The globalization rule does not clamp the site precision to positive semidefinite.
   4.13. A rejected site update is not reported as convergence.
   4.14. The runtime distinguishes converged, budget-exhausted, diverged, and infeasible EP termination.
   4.15. A non-finite moment, residual, or ledger value produces divergence.
   4.16. Failure to find an acceptable globalized step returns a machine-readable budget-exhausted or infeasible reason chosen by the derived status contract.
5. The nonlinear node reports a named approximate free-energy surrogate.
   5.1. The surrogate uses the same cavity, tilted-moment, and site convention as the message derivation.
   5.2. The ledger labels the nonlinear node term as approximate.
   5.3. The ledger labels the global objective as approximate when it contains the nonlinear node.
   5.4. The ledger does not claim the exact linear-Gaussian tree identity for a graph containing the nonlinear node.
   5.5. Linear-Gaussian terms elsewhere in the graph retain their analytic labels.
   5.6. The derivation states the surrogate's mathematical target.
   5.7. The derivation states the surrogate's normalization convention.
   5.8. The derivation states the surrogate's stationary equations.
   5.9. If the derivation proves a bound direction, the ledger reports that direction.
   5.10. If no bound direction is proved, the ledger and limitations documentation state that fact.
   5.11. The surrogate is finite only when the required beliefs and covariances are proper.
   5.12. Convergence requires a finite dtype-scaled surrogate change below its declared tolerance.
   5.13. Convergence requires a finite stationarity residual below its declared tolerance.
6. The rule satisfies the numerical robustness contract.
   6.1. Sparse discrete support is structurally irrelevant because this factor has only nondegenerate Gaussian edges.
   6.2. Singular noise and deterministic limits follow criterion 2.11 rather than a numeric floor.
   6.3. When a message method requires a cavity mean and that cavity is vacuous, the dependent outgoing update returns the exact neutral Gaussian message `(eta, Lambda) = (0, 0)`.
   6.4. The vacuous-cavity path does not invent a linearization point.
   6.5. A final vacuous or improper local belief makes the ledger non-finite.
   6.6. A run with a final vacuous or improper local belief does not report convergence.
   6.7. An update with a nonsymmetric, indefinite, singular, or non-finite belief or tilted covariance exposes a machine-readable failure reason.
   6.8. An energy term with an improper required belief returns `+inf`.
   6.9. A marginal readout with an improper belief returns shape-preserving NaNs.
   6.10. A sufficient-statistic readout with an improper belief reports `proper=False`.
   6.11. Float32 derives rank, symmetry, positive-definiteness, sigma-point, damping, and residual guards from the active dtype and matrix scale.
   6.12. Accepted proper solves have no deliberate regularization bias.
   6.13. The uncomfortable-scale case uses input dimension at least 32 or at least 128 nonlinear factors.
   6.14. The scale case executes compiled code without constructing a dense whole-graph Jacobian.
   6.15. The scale case returns finite outputs or a truthful machine-readable non-converged status.
   6.16. Persistent arena bytes equal the estimator's named byte counts.
   6.17. Persistent arena bytes grow no faster than factor count times the square of the largest incident Gaussian dimension.
7. Each affected execution and transport surface recognizes the factor and `MOMENT_MATCHING` constraint.
   7.1. Compiler dispatch recognizes the factor and constraint without a central type-string branch.
   7.2. Gaussian arena addressing covers the factor.
   7.3. Parameter packing covers the factor.
   7.4. Synchronous steps cover the factor and constraint.
   7.5. Scheduled batch and scan stages cover the factor and constraint.
   7.6. Surrogate-ledger evaluation covers the factor and constraint.
   7.7. Schedule classification covers the factor and constraint.
   7.8. Readouts cover the factor and constraint.
   7.9. Serialization manifests cover the factor and constraint.
   7.10. Resource estimates cover the factor and constraint.
   7.11. Public exports cover the factor and constraint.
   7.12. Host and JIT execution agree within the stated tolerance.
   7.13. Flooding and scheduled execution agree within the stated tolerance.
   7.14. Single-instance and `vmap` execution agree within the stated tolerance.
   7.15. Latest-schema and legacy decoding are covered or marked structurally irrelevant with a reason.
8. Durable documentation describes the delivered boundary.
   8.1. `README.md` documents public construction and both message methods.
   8.2. `docs/ARCHITECTURE.md` documents EP site semantics and execution integration.
   8.3. `docs/DERIVATIONS.md` records the message, globalization, stationary, and surrogate equations.
   8.4. `docs/LIMITATIONS.md` records the approximate ledger meaning, numerical failure behavior, and excluded continuous-to-discrete and nonlinear-GFE cases.

## Methodological Suggestions

These suggestions are optional. Another method is acceptable if it satisfies the numbered outcomes with equivalent artifacts and executable evidence.

- Use the **New factor or rule family** and **Runtime numerical or status invariant** rows in the `docs/ARCHITECTURE.md` change-path index. Start from the Gaussian `FactorKernel` binding delivered through cff-209l instead of adding a central type-string branch.
- Keep the serialized core representation free of JAX callables. Resolve the namespaced nonlinear implementation at compile time, and keep function parameters in external `.npy` or `.safetensors` blobs when they are too large for ordinary JSON metadata.
- Consider `jax.jacfwd` for Jacobians. For sigma points, use a square-root construction whose tolerances and admissible scaling parameters derive from the active dtype and matrix dimension.
- Derive the linearization, sigma-point, cavity, site, damping/globalization, residual, and surrogate equations before implementation. Put the formal results in rule docstrings and `docs/DERIVATIONS.md`; identify every approximation and unproved bound in `docs/LIMITATIONS.md`.
- Surrogate-monotonic backtracking is a candidate globalization method, not a requirement established by the repository. Use it only if the new derivation proves that its surrogate is a valid merit function for the selected EP update; otherwise derive and use a stationarity-residual or other scalar merit rule while still reporting the surrogate as the free-energy convergence signal.
- Audit the `FactorKernel`/`FactorType`, compiler grouping, Gaussian slots, parameter packing, `Arena`, synchronous `_make_step`, scheduled `_batch_stage_fn` and `_scan_stage_fn`, `_make_bfe`, schedule classification, readouts, serialization manifest, estimator, and public exports. Check host/JIT, flooding/scheduled, single-instance/`vmap`, and latest-schema/legacy pairs explicitly.
- Use independent references: `gaussian_linear` for affine-limit identities, dense-grid quadrature for a low-dimensional curved posterior, and direct moment calculations or a seeded high-particle Monte Carlo estimate with reported sampling error for sigma-point moments.
- Test weak and strong precisions across coordinate scales, vacuous cavities, improper and indefinite combined beliefs, admissible negative site precision, rejected EP steps, float32, dimension 32 or larger, 128-factor batching, non-finite function values and Jacobians, serialization failures, and schedule/`vmap` agreement.
- Per Cooper's 2026-07-08 delegation decision, keep message, EP, and surrogate-ledger derivations under top-level review. Sigma-point utilities, focused engine plumbing, analytic golden fixtures, and narrowly framed literature questions may be delegated and then reviewed at top level.
- Run focused nonlinear-rule, Gaussian, kernel, compiler, schedule, runtime-status, serialization, estimator, and ledger tests during implementation. After the final code change, run the dashboard-backed short suite once under the `AGENTS.md` protocol; run marked slow scale or globalization tests only when the final evidence requires them.

## Future work

- cff-1o72 builds deterministic neural factors, free-energy-gradient training, and amortized message initialization on this ticket's nonlinear Gaussian and surrogate-ledger contracts.
- cff-ahps uses this ticket and cff-1o72 to close the nonlinear and neural branches of the universal factor-library roadmap after its other prerequisites are satisfied.
- Continuous-to-discrete nonlinear links use declared hybrid subclasses with proved variational bounds. cff-k9ws delivered the recurrent stick-breaking link; this ticket does not replace or generalize it.
- Nonlinear or Gaussian-only GFE messages remain excluded. cff-dfpi and cff-izjd delivered the bounded discrete-regime epistemic route; expanding that scope requires a separate ticket and a new derivation.

## Definitions

- **Cavity:** The local belief formed after removing the previous contribution from the site being updated.
- **Dense-grid quadrature:** Numerical integration over a low-dimensional grid used here as an independent posterior reference.
- **Expectation propagation (EP):** An iterative method that replaces a difficult local factor with a tractable site whose moments match an approximate tilted distribution.
- **Gaussian natural parameters:** Information vector `eta` and precision matrix `Lambda`; Gaussian message products add these values.
- **Globalization:** Damping, backtracking, or another method that rejects or shrinks an iterative update when the full proposed step is not acceptable.
- **Indefinite matrix:** A symmetric matrix with both positive and negative quadratic directions.
- **Linearization:** Replacing `g` locally by its value and Jacobian at a selected point.
- **Moment matching:** Choosing a Gaussian with the same mean and covariance as the distribution being approximated.
- **Nondegenerate Gaussian:** A Gaussian distribution with a symmetric positive-definite covariance.
- **Propriety:** The condition that a belief is finite and normalizable.
- **Sigma points:** A deterministic weighted point set chosen to reproduce selected moments of a Gaussian input.
- **Site:** The tractable unnormalized local factor that EP updates.
- **Surrogate:** A named approximate objective term used when the analytic free-energy term is unavailable.
- **Tilted distribution:** The normalized product of an EP cavity and the exact or approximated local factor before moment matching.
- **Vacuous Gaussian message:** The neutral natural-parameter message `(eta, Lambda) = (0, 0)`.

## Notes

**2026-07-15T01:45:14Z**

Scope boundary (2026-07-14, from the recurrence-link design discussion): this ticket remains valid for its own domain — deterministic nonlinear maps between GAUSSIAN edges (nonlinear dynamics/emissions, EKF/UKF perception), with the approximate FE term labeled in the ledger as already specified. It is explicitly NOT the route for continuous→discrete links (e.g. a recurrence factor p(z'|z,x) with a softmax/stick-breaking link): those get a dedicated hybrid factor with a provable variational bound (Pólya-Gamma / Jaakkola-Jordan), per the cff-fxmj audit directive to prefer declared subclasses with bounds over universal approximate machinery. Epistemic (GFE) scoring through nonlinear nodes stays out of scope here (cff-dfpi owns that question).

**2026-08-08T04:24:35Z**

Retained wiggle-audit candidates and intended meanings after the bounded current scan: integrate = numerical integration in the dense-grid definition; preserve = exact round-trip value identity, semantic identity, or shape preservation as stated locally; public = the documented template contract or exported package symbols; register = executable compiler/runtime FactorKernel coverage; same = the identical cavity/site convention or value-equal matched moments; expose = machine-readable failure information or readout metadata; implement = this ticket's planned code and integration work; provide = the two documented message methods or an existing readout contract; support = literal probability support; deliver = a repository contract or closed hybrid capability supplied by the named ticket; remaining = continued propriety/vacuity state or an explicit excluded capability; approximate = the named linearization, sigma-point, moment-matching, or EP surrogate; close = acceptance-based ticket closure; check = a numeric propriety guard or duplicated-surface review; converge = explicit runtime termination status; equivalent = equal artifact and executable-evidence coverage; estimate = the named resource estimator or a Monte Carlo numerical estimate; evidence = named executable checks and independent references; exact = analytic mathematical meaning or the literal neutral Gaussian message; match = moment matching or numeric agreement within a stated tolerance; reproduce = sigma-point moment equality; robust = the AGENTS.md numerical robustness contract; test = automated execution or named test artifacts; verify = a future required executable check; infer = fixed-parameter Gaussian posterior inference; scale = dtype/matrix magnitude, dimension, factor count, or runtime test load as stated locally; speed = the marked slow-test selection, not an unspecified performance claim; may = a mathematically admissible indefinite site or explicit delegation permission; explicit_owner = this ticket's historical nonlinear-Gaussian scope and cff-dfpi's historical GFE decision; named_ticket_handoff = the historical cff-dfpi authority; scope_exclusion = continuous-to-discrete and nonlinear-GFE work assigned to named closed decisions; all = every member of the locally enumerated integration-surface list; at_least = inclusive dimension-32 or factor-count-128 lower bounds; every = each accepted tilted covariance or each named approximation; only = an acceptance gate, mathematical finiteness gate, structural scope boundary, or test-selection condition. Intelligible-ticket review: complete.

**2026-08-08T04:30:08Z**

Targeted correction review (2026-08-07 local): a bounded search of `docs/DERIVATIONS.md`, `docs/ARCHITECTURE.md`, `docs/LIMITATIONS.md`, `src/`, and `tests/` found no nonlinear-EP derivation or executable rule proving that each accepted EP step decreases the reported surrogate. `src/cffg/core/ir.py` still describes `MOMENT_MATCHING` as reserved with no registered rules. Criteria 4 and 5 therefore require a derived globalization/merit rule, proper accepted beliefs, finite surrogate-change and stationarity certificates at convergence, and truthful statuses, but do not require monotonic surrogate descent. Surrogate-monotonic backtracking remains a methodological option only if the new derivation proves the surrogate is a valid merit function. Composite outcomes were split into decimal leaves so each leaf presents one reviewer decision. The rerun retained the previously reported candidates with their recorded meanings and added only note-derived or technical uses: fixed = fixed-parameter inference in the earlier audit note; complete = the exact review-completion marker; sufficient = the established term sufficient-statistic readout; performance, intended, current, and existing = literal words inside the earlier timestamped audit explanation, not new capability claims. Intelligible-ticket review: complete.
