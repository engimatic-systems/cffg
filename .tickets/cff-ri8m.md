---
id: cff-ri8m
status: closed
deps: [cff-3hgi]
links: [cff-1o72, cff-fxmj, cff-o7hs, cff-l6i5]
created: 2026-07-10T18:05:00Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ahps
tags: [adversarial-reviewer, architecture, modularity]
---
# Make factor registration executable across compiler, VM, ledger, and serialization

ADVERSARIAL REVIEWER FINDING (2026-07-10). core.ir.register_factor_type registers only a structural signature. Executable extension remains centralized: compiler.py branches by family/type and owns multiple plan dataclasses/parameter stackers; engine.py owns family-specific sweeps/readouts/ledger functions; rule tables are separate; validation has global family maps; serialization has no factor implementation identity/version. Evidence from recent additions: gaussian_mixture plumbing changed compiler +120 and engine +87; gaussian_prec +78/+72; constraints +132/+87 plus IR/validation. compiler.py is 1,146 lines and engine.py 892 after only a small factor vocabulary. A new same-family discrete factor is tolerable; a new edge family, mixed factor, observation form, or constraint requires central surgery. The documented registry abstraction therefore does not yet exist, and neural/nonlinear/hybrid roadmap items will compound this.

## Acceptance Criteria

1. One typed executable FactorKernel/RuleSpec contract declares structural roles/domains, parameter schema/packing, applicable constraint modes, message update(s), node energy, observation conditioning, and optional readouts. 2. Compiler and VM group/dispatch generically from that contract, including multi-arena factors; no factor-type string branches in central compiler/engine for a representative new factor. 3. A test-only custom factor and energy can be registered without editing cffg compiler/engine/validation source and survives template JSON/.cffgz round-trip via namespaced implementation id + version; missing implementation fails loudly. 4. Built-ins migrate incrementally with parity and central module size/churn decreases. 5. Rule registration requires hostile-input declarations and an independent energy/oracle test. 6. Public vs internal extension compatibility policy is documented before 1.0.

## Notes

**2026-07-27T03:27:04Z**

PR: https://github.com/engimatic-systems/cffg/pull/33 (stacked on cff-ey4p PR #32; base branch cff-ey4p-belief-log-floor; awaiting Cooper's review — do not merge from a session). Delivered: cffg/kernels.py FactorKernel contract + registry; EF built-ins migrated; serialization manifest with loud missing-implementation failure; tests/test_kernels.py exponential_obs worked example with independent evidence oracle. Remaining scope deliberately deferred: discrete/Gaussian/hybrid migration is cff-l6i5/cff-x0sv (recorded in LIMITATIONS S1i). Full pass on branch: 1184 passed, 0 failing.

**2026-07-27T04:04:28Z**

Review regression added in tests/test_kernels.py: loading a serialized external kernel must fail even when its FactorType was structurally pre-registered and remains after unregister_kernel. The new test intentionally fails on the current PR head with DID NOT RAISE; it should turn green when _check_kernel_requirements checks the executable kernel manifest independently of FACTOR_TYPES.

**2026-07-27T04:23:50Z**

Review fix (2026-07-26): the load-time gate could be bypassed when a required external kernel's FactorType was structurally pre-registered — load passed and failure surfaced later at compile. _check_kernel_requirements now requires a registered executable kernel for every manifest entry a factor uses, naming the recorded (impl, version). Reviewer regression test_missing_implementation_fails_when_factor_type_was_pre_registered passes; full pass 1185/0 failing. Pushed to PR #33.
