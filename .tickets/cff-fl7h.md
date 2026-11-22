---
id: cff-fl7h
status: in_progress
deps: [cff-nd1q]
links: []
created: 2026-09-25T03:46:26Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Implement bound-aware continuous planning from the approved derivation

## Background in Plain English

Once the bounded-decision mathematics is verified, model authors need to express actuator limits inside the model and receive feasible actions without relying on post-hoc clipping. Unsupported multivariate cases must remain explicit.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The public factor/constraint and decision API implements the exact scope approved by the derivation ticket.

2. Messages, normalizers and free energy agree with the independent mathematical oracles within declared dtype-aware tolerances.

3. Accepted decisions satisfy bounds and the constrained reference case; infeasible bounds cannot return a plausible successful action.

4. Sparse/zero support, vacuous/improper inputs, float32, extreme tails and larger supported dimensions meet the numerical contract.

5. Template validation, serialization, compilation, readouts and status behavior agree across host/JIT, flooding/scheduled and single/vmap execution.

6. A runnable example and matched ablation document where bound-aware planning improves on clipping without claiming every reward gap is solved.

## Methodological Suggestions

Use the New factor or rule family and New IR field change-path rows; review all derived packing, lifecycle, ledger and readout owners. Current/legacy serialization must reject or migrate unsupported identities explicitly. Keep equation derivations separate from execution glue.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Feasible action: one satisfying the declared limits. Extreme tail: a region with very small probability before truncation. Ledger: the assembled free-energy accounting.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-22T03:42:58Z**

D15 in docs/DERIVATIONS.md supplies the v1 handoff. Represent the finite hard box once in a closed frozen Gaussian region; proper full-rank cavity, finite positive widths, and certified marginal action blocks of size 1 or 2 (any number of structurally independent blocks and nuisance states). Use the first bounded joint mean as the command. Check state/action cross-covariances and F_G minus the log box mass. Reject larger correlated blocks, unrepresentable arithmetic, unsupported external non-Gaussian messages and singular/invalid cavities with explicit status. A finite flat local cavity is a separate uniform unit-message case without an absolute baseline evidence. Mode is a separate oracle. Implement the D15.6 matched short-horizon ablation and hostile cases before publishing model capabilities; any fully bounded fifteen-step reactor needs a method for dense correlated boxes. Inspect current/legacy template and checkpoint, host/JIT, single/vmap and flood/scheduled execution when a persistent factor is added.

**2026-11-22T19:47:10Z**

Change paths: New factor or rule family / New IR field checked. Closed standalone frozen-region model with explicit independent Gaussian priors and conditional state model. Arena packing, template/checkpoint changes, engine flooding/scheduled, external separators N/A. Host/JIT/vmap use the same pure solve. Empty box, improper/nonfinite Gaussian, unresolvable tails and float32 moments fail closed. Test independent structural blocks at horizon 15, dense >2 rejected; parent owns full suite and directed integration.

**2026-11-22T19:51:00Z**

Implemented closed D15 region in d6b13b4; matched coolant example and README in 48944bc. BoundedGaussianPlan.solve_joint accepts one dense <=2D bounded action block first and arbitrary trailing states. solve accepts structural product action marginals plus shared affine state conditional. Bind either method for JIT/vmap. F_G is required from the caller. No inference schedule or adaptive-site integration. Focused validation on feature/bounded-actions: UV_PROJECT_ENVIRONMENT=/Users/ravd/projects/cffg/.venv PYTHONPATH=$PWD/src uv run --no-sync pytest -q tests/test_bounded_action_{numerics,derivation}.py: 60 passed in 7.79s. Same environment, uv run --no-sync python examples/bounded_coolant.py: clipped/bounded first 298.5/297.38750 K, tracking 1.0/0.682142, penalty 2.630977 nats. git diff --check passed and tk dep cycle found no cycles. Parent owns integration and final short suite. Leave in progress until parent records gate and closes.

**2026-11-22T19:53:38Z**

Graph integration: portable wrapper owns a source template plus bounded edge/coordinate selections. Compile to an identified auto-scheduled engine; validate complete directed topology, nonterminal state leg identity, unique coordinates and maximum block width before solving. Reconstruct the frozen full joint and normalized evidence from live DATA and registered Taylor bindings. Host evaluator wraps staged D15 kernel; ordinary sweeps, ledger and checkpoint fields N/A. H2 and H15 reactor checks plus foreign result, alias, dtype and scoping cases.

**2026-11-22T19:58:59Z**

Graph path delivered on feature/bounded-actions in ae84866, with directed tip a343a98 merged in 7b3a08d. Dtype-scale covariance symmetry guard 5588a88; reactor horizon fd8f8d1; shared graph oracle fixture 3a05aae; selected-edge first/warm timing and regression 709d642. README graph example and docs/ARCHITECTURE, LIMITATIONS and DERIVATIONS D15 updated. Reproduce from this worktree with UV_PROJECT_ENVIRONMENT=/Users/ravd/projects/cffg/.venv and PYTHONPATH=$PWD/src: uv run --no-sync pytest -q tests/test_bounded*.py tests/test_cstr_runtime.py tests/test_directed_chain_{plan,run}.py -> 83 passed, 3 opt-in slow deselected, 21.85s. Includes original hostile-input guards, dense precision independent oracle, transport/scoping, changed goals, mixed dtype/mutated parameters, foreign and nonterminal certificates, alias/terminal-edge/width checks and real H15 declaration rejection. uv run --no-sync python tools/cstr_bounded_comparison.py -> directed_chain_certified. Clipped vs bounded first command 295.177973 vs 295.849702 K, objective 32.249295 vs 32.824832 nats, one-step concentration tracking error 0.027459 vs 0.028521. Adjacent action covariance 0.0352643. Single local first/warm readout times 0.134/0.106 ms selected-edge clipped, 332.86/5.05 ms bounded, after shared solve 176.92 ms. Full suite and held-out validation are parent-owned; leave ticket in progress for integration gate.
