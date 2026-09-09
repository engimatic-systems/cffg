---
id: cff-209l
status: open
deps: [cff-ri8m, cff-l6i5]
links: [cff-1o72, cff-o7hs, cff-l6i5, cff-cfqg, cff-edoz, cff-ws3c]
created: 2026-07-30T16:09:57Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-cfqg
---
# Bind discrete and Gaussian arenas to the FactorKernel execution contract

## Background in Plain English

CFFG already lets exponential-family factors register their executable messages, energy, parameters, observations, and readouts in one `FactorKernel`. Ordinary discrete and Gaussian factors still rely on type-name branches spread across the compiler and inference runtime. This ticket extends the executable contract to those two dedicated arenas and migrates their ordinary built-in rules without changing serialized models or numerical behavior. The completed workflow lets a new factor using one of the EF, discrete, or Gaussian arenas compile, run, report its objective term, and serialize without adding its type name to central dispatch code.

The migration is not a promise that one callback shape fits every factor in the project. Structural equality, Dirichlet parameter-edge composites, hard constraints, GFE composites, and factors that span more than one arena keep their specialized plans. cff-o7hs depends on this ticket because its nonlinear Gaussian factor needs the new same-arena Gaussian extension path.

## Outcomes / Acceptance Criteria

1. The prerequisite contracts are satisfied before this ticket closes.
   1.1. cff-ri8m is closed with its executable EF-arena `FactorKernel` contract satisfied.
   1.2. cff-l6i5 is closed with its shared natural-parameter algebra and bit-identical Cat/Dir/Gaussian parity evidence satisfied.
2. `FactorKernel` represents executable factors in the EF, discrete, and Gaussian arenas.
   2.1. Each registered kernel declares exactly one arena binding.
   2.2. A discrete binding declares its role cardinalities.
   2.3. A discrete binding uses the discrete arena's local padded-message convention.
   2.4. A Gaussian binding declares its role dimensions.
   2.5. A Gaussian binding uses the Gaussian arena's `(eta, Lambda)` message convention.
   2.6. Each parameterized binding declares its parameter keys.
   2.7. Each parameterized binding declares one parameter packer.
   2.8. A discrete binding declares a BP message callback when it accepts ordinary BP.
   2.9. A discrete binding declares a VMP message callback when it accepts a mean-field constraint.
   2.10. A discrete BP callback receives the selected sum-product or max-product reduction operation.
   2.11. A Gaussian binding declares a message callback that returns one information-form message per declared role.
   2.12. Each binding declares the node-energy callback used by its declared objective mode.
   2.13. Each binding declares observation conditioning when it accepts a DATA port.
   2.14. Each observation-conditioned binding declares an observation validator.
   2.15. Each binding declares optional factor readouts when it exposes them.
   2.16. Each binding carries a nonempty `HostileInputs` declaration.
   2.17. Registration rejects a callback set that is incomplete for the declared arena.
   2.18. Registration rejects a callback set that is incomplete for the declared constraint modes.
3. Test-only factors prove the same-arena extension workflow.
   3.1. A test-only discrete factor registers without editing `compiler.py`, `engine.py`, `core/validate.py`, or the built-in rule tables.
   3.2. The test-only discrete factor compiles into the discrete arena.
   3.3. The test-only discrete factor executes in sum-product mode against an independent finite-enumeration oracle.
   3.4. The test-only discrete kernel declares max-product support.
   3.5. The test-only discrete factor executes in max-product mode against an independent finite-enumeration oracle.
   3.6. The test-only discrete kernel declares mean-field support.
   3.7. The test-only discrete factor's VMP messages agree with an independent finite-enumeration oracle.
   3.8. The test-only discrete factor contributes its registered BP node energy to BFE in sum-product mode.
   3.9. The test-only discrete factor contributes its registered VMP node energy to BFE in mean-field mode.
   3.10. A test-only Gaussian factor registers without editing `compiler.py`, `engine.py`, `core/validate.py`, or `gauss.py`.
   3.11. The test-only Gaussian factor compiles into the Gaussian arena.
   3.12. The test-only Gaussian factor's messages agree with an independent analytic Gaussian oracle.
   3.13. The test-only Gaussian factor contributes its registered node energy to BFE.
   3.14. The test-only Gaussian kernel declares observation conditioning on one role.
   3.15. The test-only Gaussian kernel rejects an invalid observation through its registered validator.
   3.16. Conditioned messages from the test-only Gaussian factor agree with an independent analytic oracle.
   3.17. Conditioned node energy from the test-only Gaussian factor agrees with an independent analytic oracle.
   3.18. A source check confirms that neither test-only factor's type name appears in compiler dispatch.
   3.19. A source check confirms that neither test-only factor's type name appears in engine dispatch.
   3.20. A JSON round-trip records and checks each external kernel's namespaced implementation identity and version.
   3.21. A `.cffgz` round-trip records and checks each external kernel's namespaced implementation identity and version.
   3.22. Loading a template without its required external kernel fails and names the missing implementation identity and version.
   3.23. Loading a template with an incompatible external-kernel major version fails and names the recorded and installed versions.
4. Ordinary built-in discrete and Gaussian factors dispatch through registered kernels.
   4.1. Ordinary BP `categorical` groups dispatch their message update through their registered kernel.
   4.2. Ordinary BP `categorical` groups dispatch their node energy through their registered kernel.
   4.3. Ordinary BP `matrix_cat` groups dispatch their message update through their registered kernel.
   4.4. Ordinary BP `matrix_cat` groups dispatch their node energy through their registered kernel.
   4.5. Ordinary BP `transition_mixture` groups dispatch their message update through their registered kernel.
   4.6. Ordinary BP `transition_mixture` groups dispatch their node energy through their registered kernel.
   4.7. Mean-field `categorical` groups dispatch their VMP message update through their registered kernel.
   4.8. Mean-field `categorical` groups dispatch their VMP node energy through their registered kernel.
   4.9. Mean-field `matrix_cat` groups dispatch their VMP message update through their registered kernel.
   4.10. Mean-field `matrix_cat` groups dispatch their VMP node energy through their registered kernel.
   4.11. Mean-field `transition_mixture` groups dispatch their VMP message update through their registered kernel.
   4.12. Mean-field `transition_mixture` groups dispatch their VMP node energy through their registered kernel.
   4.13. `gaussian` prior groups dispatch their message update through their registered kernel.
   4.14. `gaussian` prior groups dispatch their node energy through their registered kernel.
   4.15. Unobserved `gaussian_linear` groups dispatch their message update through their registered kernel.
   4.16. Unobserved `gaussian_linear` groups dispatch their node energy through their registered kernel.
   4.17. Conditioned `gaussian_linear` groups dispatch observation validation through their registered kernel.
   4.18. Conditioned `gaussian_linear` groups dispatch messages through their registered kernel.
   4.19. Conditioned `gaussian_linear` groups dispatch node energy through their registered kernel.
   4.20. The built-in `gamma` kernel retains bitwise-identical message outputs.
   4.21. The built-in `gamma` kernel retains bitwise-identical energy outputs.
   4.22. The built-in `beta` kernel retains bitwise-identical message outputs.
   4.23. The built-in `beta` kernel retains bitwise-identical energy outputs.
   4.24. The built-in `poisson_obs` kernel retains bitwise-identical message outputs.
   4.25. The built-in `poisson_obs` kernel retains bitwise-identical energy outputs.
   4.26. The built-in `bernoulli_obs` kernel retains bitwise-identical message outputs.
   4.27. The built-in `bernoulli_obs` kernel retains bitwise-identical energy outputs.
   4.28. Family-polymorphic `equality` remains structural infrastructure rather than a public same-arena extension example.
   4.29. Dirichlet parameter-edge composites retain their specialized compiler and runtime plan.
   4.30. Hard-constraint plans retain their specialized combinatorial dispatch.
   4.31. P-substitution and other GFE composites retain their specialized plans.
   4.32. Gaussian/EF multi-arena factors retain their specialized plans.
   4.33. Discrete/Gaussian multi-arena factors retain their specialized plans.
5. The migration preserves built-in model structure and values.
   5.1. Serialized built-in factor names remain unchanged.
   5.2. Built-in templates omit `cffg.` implementation identities from the external-kernel manifest as before.
   5.3. The HMM reference template's structure hash remains bitwise identical to its pre-migration value.
   5.4. The Kalman reference template's structure hash remains bitwise identical to its pre-migration value.
   5.5. The T-maze reference template's structure hash remains bitwise identical to its pre-migration value.
   5.6. Discrete message-slot offsets remain bitwise identical for the HMM reference template.
   5.7. Gaussian eta-slot offsets remain bitwise identical for the Kalman reference template.
   5.8. Gaussian precision-slot offsets remain bitwise identical for the Kalman reference template.
   5.9. Parameter stacking order remains bitwise identical for each migrated built-in group.
   5.10. Group member ordering remains bitwise identical for each migrated built-in group.
   5.11. HMM messages remain bitwise identical on the deterministic parity fixture.
   5.12. HMM marginals remain bitwise identical on the deterministic parity fixture.
   5.13. HMM BFE remains bitwise identical on the deterministic parity fixture.
   5.14. Kalman messages remain bitwise identical on the deterministic parity fixture.
   5.15. Kalman smoothed means remain bitwise identical on the deterministic parity fixture.
   5.16. Kalman smoothed covariances remain bitwise identical on the deterministic parity fixture.
   5.17. Kalman BFE remains bitwise identical on the deterministic parity fixture.
   5.18. T-maze messages remain bitwise identical on the deterministic parity fixture.
   5.19. T-maze BFE remains bitwise identical on the deterministic parity fixture.
   5.20. HMM public readouts return bitwise-identical arrays on the deterministic parity fixture.
   5.21. Kalman public readouts return bitwise-identical arrays on the deterministic parity fixture.
   5.22. T-maze public readouts return bitwise-identical arrays on the deterministic parity fixture.
   5.23. HMM resource-estimator named byte counts remain bitwise identical on the deterministic parity fixture.
   5.24. Kalman resource-estimator named byte counts remain bitwise identical on the deterministic parity fixture.
   5.25. T-maze resource-estimator named byte counts remain bitwise identical on the deterministic parity fixture.
   5.26. The migration adds no runtime dependency.
   5.27. The migration adds no development dependency.
6. The generalized registration and dispatch obey the numerical robustness contract.
   6.1. A discrete kernel preserves the exact `NEG` sentinel on every state excluded by its incoming support and factor potential.
   6.2. An empty discrete support intersection produces an infeasible result rather than a finite normalized message.
   6.3. A Gaussian kernel propagates the exact vacuous message `(eta, Lambda) = (0, 0)` according to the registered rule's neutral-message contract.
   6.4. A Gaussian kernel reports nonsymmetric, non-positive-definite, or non-finite required beliefs through the established improper-belief outputs.
   6.5. A non-finite registered message or energy output produces divergence rather than a finite converged result.
   6.6. Float32 registered rules derive guards from the active dtype and matrix or message scale.
   6.7. A test-only discrete kernel uses cardinality at least 128 in its uncomfortable-scale fixture.
   6.8. The cardinality-128 discrete fixture compiles without model-wide cardinality padding.
   6.9. The cardinality-128 discrete fixture executes without model-wide cardinality padding.
   6.10. A test-only Gaussian kernel uses dimension at least 100 in its uncomfortable-scale fixture.
   6.11. The dimension-100 Gaussian fixture compiles without model-wide dimension padding.
   6.12. The dimension-100 Gaussian fixture executes without model-wide dimension padding.
   6.13. Persistent arena bytes for the cardinality-128 discrete fixture equal the resource estimator's named byte counts.
   6.14. Persistent arena bytes for the dimension-100 Gaussian fixture equal the resource estimator's named byte counts.
   6.15. Compilation rejects an outgoing-message shape that does not match the declared roles and arena representation.
   6.16. Compilation rejects undeclared parameter output from a parameter packer.
   6.17. Compilation rejects missing declared parameter output from a parameter packer.
   6.18. Registration rejects blank sparse-support, improper-message, float32, or scale declarations.
7. Each affected execution surface dispatches the migrated kernels.
   7.1. Compiler grouping selects migrated kernels without a built-in factor-type dispatch branch.
   7.2. Compiler parameter packing calls the migrated kernel's declared packer.
   7.3. Synchronous flooding calls the migrated message callbacks.
   7.4. Scheduled batch stages call the migrated message callbacks.
   7.5. Registerized scan stages call the migrated message callbacks.
   7.6. BFE calls the migrated node-energy callbacks.
   7.7. Schedule classification uses registered arena and constraint metadata.
   7.8. Public factor readouts use registered readout callbacks when declared.
   7.9. Serialization manifests use registered implementation identity and version metadata.
   7.10. The resource estimator uses registered arena metadata.
   7.11. The resource estimator uses registered parameter metadata.
   7.12. The resource estimator uses registered callback metadata for operation counts.
   7.13. Package exports expose the documented registration surface.
   7.14. Host and JIT execution agree within the stated dtype tolerance for both test-only kernels.
   7.15. Flooding and scheduled execution agree within the stated dtype tolerance for migrated ordinary BP groups.
   7.16. Single-instance and `vmap` execution agree within the stated dtype tolerance for both test-only kernels.
   7.17. Latest-schema and legacy checkpoint decoding are unchanged or marked structurally irrelevant with a reason.
8. Durable documentation distinguishes stable serialization from the pre-1.0 Python extension API.
   8.1. `README.md` documents one discrete custom-kernel example.
   8.2. `README.md` documents one Gaussian custom-kernel example.
   8.3. `docs/ARCHITECTURE.md` documents the three single-arena kernel bindings.
   8.4. `docs/ARCHITECTURE.md` documents the specialized-plan exclusions.
   8.5. `docs/LIMITATIONS.md` removes the EF-only registration limit.
   8.6. `docs/LIMITATIONS.md` records the remaining multi-arena and structural exclusions.
   8.7. Documentation states that external `(impl, version)` manifest semantics are stable.
   8.8. Documentation states that a missing required external implementation fails loudly.
   8.9. Documentation states that `FactorKernel` Python fields remain internal-unstable before 1.0.
9. Validation evidence covers the completed migration.
   9.1. The focused kernel, compiler, discrete, Gaussian, schedule, serialization, estimator, runtime-status, and hostile-input test selection passes.
   9.2. The dashboard-backed short suite passes after the final code change.

## Methodological Suggestions

These suggestions are optional. Another method is acceptable if it satisfies the numbered outcomes with equivalent artifacts and executable evidence.

- Use the **New factor or rule family** row in `docs/ARCHITECTURE.md`. Walk `FactorType`, `FactorKernel`, compiler grouping and packing, arena plans, synchronous and scheduled dispatch, `_make_bfe`, schedule classification, readouts, serialization manifests, estimator accounting, and public exports.
- Prefer an arena-specific typed binding or adapter inside `FactorKernel` over one callback signature filled with irrelevant arguments. Keep dispatch static at JAX trace time so compiled sweeps contain no runtime type-name switch.
- Reuse the discrete callbacks in `rules.py` and Gaussian callbacks in `gauss.py` as behavior-preserving adapters before considering code movement. Keep the family natural-parameter algebra from cff-l6i5 authoritative.
- Treat `gaussian_linear_obs` as observation-conditioned lowering of the registered `gaussian_linear` factor, not as a new serialized factor identity.
- Extend the executable custom-kernel pattern in `tests/test_kernels.py` with one discrete and one Gaussian oracle. Retain the source-inspection test that proves the custom type names require no central edits.
- Capture pre-change HMM, Kalman, and T-maze structure hashes, arena indices, messages, marginals, readouts, estimator bytes, and BFE values from the same base revision and compare them with the proposed head.
- Check the deliberate execution pairs: host/JIT, flooding/scheduled, single-instance/`vmap`, and latest-schema/legacy checkpoint decoding. Record a concrete reason for every structurally irrelevant pair.
- Run focused kernel, compiler, discrete-rule, Gaussian, HMM, Kalman, GFE, schedule, serialization, bundle, estimator, runtime-status, and hostile-input tests during implementation. After the final code change, run the dashboard-backed short suite once under the `AGENTS.md` protocol.

## Future work

- cff-o7hs uses the Gaussian same-arena kernel binding to add nonlinear Gaussian factors without central compiler or engine branches.
- cff-1o72 inherits this prerequisite through cff-o7hs and uses the same binding for neural Gaussian factors.
- cff-ahps uses this ticket to complete the executable-registration foundation for its nonlinear and neural roadmap branches.
- cff-6tek owns `member_of`, named parameter writes, and learnable-family registration outside the discrete arena. This ticket preserves those surfaces but does not expand them.
- A unified public registration path for hybrid or other multi-arena factors remains outside this ticket. No downstream ticket for that generalization is identified.

## Definitions

- **Arena:** Flat, shape-segmented JAX arrays and static index plans for one message representation.
- **Bethe free energy (BFE):** The sum of registered factor-local terms plus edge-entropy overcounting corrections.
- **Belief propagation (BP):** Message passing that marginalizes other factor variables with sum-product or max-product operations.
- **Central dispatch branch:** Compiler or engine code that selects behavior by comparing a factor's type-name string.
- **Conditioning:** Supplying an observed value to its likelihood rule without representing the observation as a high-precision message.
- **Exact support:** States whose log probability is not the reserved `NEG` zero-probability sentinel.
- **FactorKernel:** The executable registration record for factor structure, arena binding, parameters, messages, energy, observations, readouts, robustness declarations, and serialized identity.
- **HostileInputs:** Required declarations for sparse support, improper messages, float32, and uncomfortable scale.
- **Implementation manifest:** Serialized external-kernel identities and versions required to reconstruct executable behavior.
- **Internal-unstable:** A Python interface that may change incompatibly before version 1.0 even though its serialized contract is stable.
- **Same-arena factor:** A factor whose message-bearing edges use one arena representation.
- **Structure hash:** A deterministic identifier for serialized model structure, excluding runtime message values.
- **Variational message passing (VMP):** Mean-field message updates computed from expectations of a factor's log potential.
- **`vmap`:** JAX's transformation for applying one compiled function across a batch.

## Notes

**2026-07-30T16:51:43Z**

Dependency audit (2026-07-30): cff-o7hs now depends on this ticket because nonlinear Gaussian nodes are new same-arena Gaussian factors and should register without adding central compiler or engine branches. cff-1o72 inherits the prerequisite through its existing dependency on cff-o7hs. Hybrid multi-arena work (including cff-k9ws and cff-izjd) is not blocked here because this ticket explicitly does not promise generic hybrid registration; cff-6tek registers learnable families rather than executable factor kernels.

**2026-08-08T04:40:45Z**

Intelligible-ticket review retained all 42 current-scan candidates because their local meanings are bounded: `fixed` means supplied parameters remain unchanged during inference; `generic` means protocol-general hybrid registration in the preserved historical note; `preserve` means byte/value identity or the explicitly named unchanged surface; `public` means exported/documented consumer surface; `register` means the executable metadata, compiler, runtime, or external-extension layer named by the sentence; `reusable` means internal callback reuse; `same` means same-arena or same-base-revision identity; `expose` means package export; `implement` means the named implementation identity or delivery action; `support` means declared inference-mode coverage or exact probability support; `complete` means the numbered migration contract or named downstream foundation; `deliver` means satisfying that numbered contract; `remaining` means the listed out-of-scope exclusions; `close` means ticket closure after named prerequisites; `check` means the named source, manifest, artifact, or automated comparison; `converge` means the solver status; `equivalent` means equal review evidence; `estimate` means the resource estimator; `evidence` means named tests, snapshots, or artifacts; `exact` means literal sentinel/vacuous values or exact identity; `match` means declared shape/representation agreement; `oracle` means an independent finite-enumeration or analytic reference; `parity` means comparison with the named deterministic baseline; `pass` means test success or message-passing terminology; `robust` means the repository numerical-robustness contract; `stable` means the documented serialized contract; `test` means the named automated fixture or selection; `validate` means observation/input or complete acceptance validation as stated; `fit` means structural callback compatibility; `infer` means fixed-parameter inference runtime; `learnable` means the registered parameter-family scope owned by cff-6tek; `scale` means named numeric magnitude or domain size; `may` means permitted pre-1.0 API change; `should` means design intent in the preserved historical note; `explicit_owner` and `named_ticket_handoff` point to cff-6tek; `all` refers exactly to the 42 listed scan candidates; `at_least` is an inclusive cardinality/dimension lower bound; `every` distributes over the locally named states, factors, or execution pairs; `only` marks test fixtures or the stated arena-scope boundary; `current` identifies the post-rewrite scan recorded by this note; and `existing` means the dependency that predates this rewrite in the preserved historical note. Capability owners were also checked separately: this ticket owns EF/discrete/Gaussian same-arena executable dispatch and parity, cff-o7hs and cff-1o72 own the nonlinear and neural consumers, cff-ahps owns the roadmap foundation, cff-6tek owns learnable-family registration, and no ticket currently owns generalized hybrid registration.

Intelligible-ticket review: complete.
