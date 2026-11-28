---
id: cff-34gi
status: in_progress
deps: [cff-bfr9]
links: [cff-sf29, cff-ddsq]
created: 2026-09-09T23:41:24Z
type: bug
priority: 2
assignee: coopwilliams
tags: [compiler, numerical, validation]
---
# Reject negative fixed probabilities in specialized GFE packing

## Background in Plain English — WHY

Fixed GFE likelihood and goal arrays can currently contain negative entries as
long as each checked probability vector sums to one. The specialized packer
checks that sum, then maps negative values to the log-zero sentinel. A vector
[-0.2, 1.2] therefore compiles into probability mass 1.2, not a valid normalized
factor. This can invalidate free-energy interpretation for malformed input.

A bounded probe reproduced acceptance for both a column of A and the goal c in
the existing two-state GFE model, on source 707c31b and pre-extraction compiler
0bcda82. This predates the compiler refactor. It does not change valid-input
parity or block engine characterization; it is a separate input-validation fix.

## Outcomes / Acceptance Criteria — WHAT

1. Fixed probability arrays entering categorical GFE and transition GFE reject
   non-finite or negative entries at their existing family validation checkpoint.
   Errors identify the factor and parameter role or blob.
2. Include the reproduced negative likelihood column and negative fixed goal.
   Check the analogous transition-GFE fixed-table path before implementation;
   add its mechanism regression if it shares the same omission.
3. Valid normalized arrays retain identical packed values and free energy.
   Literal zero support stays valid and retains the existing log-zero encoding.
   Do not silently clip or renormalize malformed input.
4. Learned Dirichlet arrays remain positive concentrations, not probabilities;
   this fix does not impose unit sums on them or change their interpretation.
5. The planning checklist covers sparse/zero support, float32, and a probability
   vector at least ten times the two-state fixture. Incoming vacuous/improper
   messages are N/A because this validation runs on host model parameters.
6. Change-path row: New factor or rule family, numeric validation only.
   Flooding/scheduled and direct/vmap consumers share the validated arena.
   Host/JIT duplication is N/A for this compile-time check. JSON and bundle
   reconstruction must reject the same invalid model data.
7. Focused source/packing/GFE regressions and one final default suite pass.
   Correct the corresponding scope limitation when the fix lands.

## Methodological Suggestions — HOW

Keep the rule in specialized/packing.py beside its family-owned probability
meaning. Inspect _stochastic and its callers, including pack_gfe and
pack_transition_gfe. Reuse an appropriate declared probability validator if its
error timing, zero support and source dtype behavior agree. Generic resolution
and assembly should not gain factor-name branches.

## Future work

No runtime algorithm, new inference family, learning cadence or broad numerical
audit is included. Existing scientific oracle tests remain authoritative.

## Definitions

- **GFE:** Generalised free energy, the objective used by P-substituted factors.
- **Log-zero sentinel:** The engine's representation of exact zero probability.
- **Concentration:** A positive Dirichlet parameter; it need not sum to one.



## Notes

**2026-09-09T23:42:33Z**

Reproduction: tests/test_compiler_contracts.py:model("gfe"); replace arrays["A"][:, 0] or arrays["c"] with [-0.2, 1.2], then compile_template. Both inputs are accepted. Exponentiating the affected packed values yields first A-column mass 1.2 or goal mass 1.2. The same two inputs compile under the pre-extraction compiler loaded from 0bcda82. Source: specialized/packing.py:pack_gfe, _stochastic, _log_of at 707c31b. Confidence high for these two confirmed paths; transition-GFE shares the helper but awaits its own mechanism regression. This ticket depends on the completed array-resolution extraction for its named owner path and is linked to assembly/engine characterization for traceability, not as a blocker of the architecture epic.

**2026-11-28T03:56:48Z**

Pre-implementation trace. Architecture row “New factor or rule family” applies at the numeric packing checkpoint. Both fixed paths call _stochastic before _log_of. A fixed transition checks each selector slice of its matrices; the epistemic_b builder normally emits Dirichlet counts, so the regression will explicitly declare a fixed transition. Registered categorical packing also calls this helper. Flooding/scheduled and direct/vmap consume the same compiled arrays. Host/JIT and incoming messages are N/A. Check source float32/64, signed malformed mass summing to one, NaN/infinities, exact zeros, a 32-outcome vector, and JSON+NPY/CFFGZ reconstruction. Keep the existing sum tolerance and learned concentration semantics.

**2026-11-28T03:59:41Z**

Handoff for the integration owner. Fixed probabilities now reject each negative or non-finite entry before log packing. Diagnostics name the factor and parameter role, including the transition selector slice. Tests cover in-memory compilation, JSON with physical NPY files, and validation by the bundle reader. Float32 and float64 inputs with 32 outcomes retain the expected sparse log tables, accept rounded thirds and reject even the smallest negative subnormal. The existing concentration and column-sum paths retain their original meaning.

Compared against an unmodified a774a54 source archive. That compiler wrote and reopened three validly hashed but numerically invalid CFFGZ bundles, each with packed mass 1.2, for a likelihood column, goal and transition column at selector 1. Public load_bundle on the fixed compiler rejected all three with the expected factor and role. All packed arrays of the valid rollout matched exactly. The converged valid categorical GFE objective was bit-identical (0x1.079ab2e207e00p-1). Probe and baseline archives are in /tmp/cff-34gi-baseline.PekEfm.

Focused validation: UV_PROJECT_ENVIRONMENT=/Users/ravd/projects/cffg/.venv PYTHONPATH=src uv run --frozen --no-sync pytest -q tests/test_specialized.py tests/test_resolution.py tests/test_engine_tree.py::test_compile_rejects_non_stochastic_params tests/test_b_epistemics.py::test_two_composite_chain_matches_constrained_objective tests/test_gfe.py::test_bfe_equals_local_objective_minimum tests/test_shared_goals.py::test_off_coordinate_energy_charges_one_shared_parameter_term passed all 68 tests. tk dep cycle and git diff --check passed. Ticket stays in progress for the combined default-suite gate, PR and closure by the integration owner.
