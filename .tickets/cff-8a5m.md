---
id: cff-8a5m
status: closed
deps: []
links: [cff-o7hs]
created: 2026-09-01T02:20:18Z
type: task
priority: 1
assignee: coopwilliams
external-ref: gh-82
---
# Clean-port paper inventory and Figure 5 correctness

## Background in Plain English — WHY

The discontinued PR #82 found two production defects that remain on current
`main`. First, the compiler accepts the reserved `MOMENT_MATCHING` constraint
and silently runs a different rule. Second, functional `mu3(A)` turns an
incoherent exact-zero support case into a large finite number instead of a
diverged update. The PR also contains a useful, source-anchored inventory of
the papers' constraint constructions and direct evidence for each Part II
Figure 5 message.

This ticket ports those meanings onto current `main` without preserving the old
branch ancestry or overwriting documentation that has evolved since the old
PR. It is the first layer of a new PR chain.

## Outcomes / Acceptance Criteria — WHAT

1. Every compiler path rejects a serialized `MOMENT_MATCHING` constraint with
   a specific error before arena construction because no executable rule owns
   that reserved syntax.
2. The legacy factor-only `MOMENT_MATCHING` placeholder remains backward
   compatible but is not presented as the paper's complete syntax.
   2.1. Serialization and structure hashing preserve the placeholder.
   2.2. Documentation states that it cannot identify the incident edge
   endpoint or sufficient-statistic projection required by Part I §5.4.
   2.3. Future executable moment matching requires a new schema and migration;
   it must not guess the missing endpoint or projection from the factor name.
3. Functional `mu3(A)` preserves the support cases in `docs/DERIVATIONS.md`:
   zero sampled predictive mass contributes zero, positive mass outside exact
   goal support contributes negative infinity, and positive mass over zero
   marginal predictive support produces a non-finite divergent update.
4. `mu3_importance_update` and `Engine.run` report every incoherent or non-finite
   functional score as divergence; they do not floor it, replace it with
   uniform weights, or report convergence or budget exhaustion.
5. `docs/ARCHITECTURE.md` contains compact, source-ordered inventories of the
   constraint constructions in Parts I and II and all Part II Figure 5
   messages, with bounded execution classifications, production owners,
   objective-ledger owners, and direct test authorities.
6. Direct tests inspect the raw executable Figure 5 messages and their local
   energy or normalizer before end-to-end normalization can hide an error.
7. Runtime tests cover exact-zero and vacuous support, float32, and a domain at
   least ten times the smallest analytic fixture where the mathematics defines
   those cases. The compile-time rejection is independent of dtype, messages,
   and scale because it runs before numeric inputs exist.
8. `README.md` and `docs/LIMITATIONS.md` distinguish exact execution, named
   approximations, reserved syntax, and unsupported combinations without
   claiming that the finite registry implements the papers' whole language.
9. The implementation is a semantic port from old PR #82 and its three source
   commits, not a merge or ancestry replay, and it remains compatible with all
   changes already on current `main`.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Compare old commits `4c9b6c68`, `ce5efe25`, and `55d5ef25` with the current
  compiler, rule, engine, derivation, and documentation authorities.
- Put the structural guard beside compiler constraint classification and keep
  the exact support masks in the functional rule rather than introducing a
  numeric floor.
- Use the **Runtime numerical or status invariant** change-path row for the
  `mu3` correction. Flooding is applicable; scheduled execution is structurally
  inapplicable because GFE groups run outside schedule stages. Single-engine
  execution is applicable; `Colony`/`vmap` is inapplicable because `Colony`
  rejects `mu3`. `RollingAgent` and checkpoint propagation belong to
  `cff-lpsn`.
- The reserved IR enum, JSON/hash round trip, compiler, and direct serialization
  test are applicable to the compiler guard. Plates, compose/edit behavior,
  bundles, host/JIT, schedules, batching, and checkpoints are unchanged.

## Future work

`cff-lpsn` clean-ports functional `mu3(A)` through `RollingAgent` after this
ticket. `cff-7ozr`, `cff-cfhi`, and `cff-9tvt` carry the remaining old PR
capabilities in later layers of the new stack. `cff-o7hs` owns the new
endpoint- and projection-aware moment-matching schema, migration from the
rejected legacy placeholder, factor-specific EP rules, and objective semantics.

## Definitions

- **Figure 5:** The Part II table of messages for the discrete goal-observation
  composite.
- **Functional message:** A message evaluated as a function of a parameter
  instead of represented by a fixed conjugate family.
- **Incoherent support:** Incoming and sampled beliefs assign incompatible
  exact support, making the functional ratio undefined or infinite.
- **Legacy placeholder:** A serializable construct retained for backward
  compatibility even though it lacks fields required for faithful execution.
- **Sufficient-statistic projection:** The selected moments that replace full
  node-to-edge marginal density equality in a moment-matching constraint.

## Notes

**2026-09-01T02:23:06Z**

Planning audit, 2026-08-31. Capability owners are compiler constraint classification, rules.mu3_log_message, rules.mu3_importance_update, Engine.run, and the named documentation and direct-test files. Wiggle scan reviewed in context; retained candidates mean: end_to_end=component-chain normalization; fixed=conjugate-family representation; preserve=ancestry exclusion, serialized identity, or probability semantics as stated; register=compiler/runtime dispatch; implement=finite registry coverage or this port; support=probability support; remaining=defects present at main 5c3b06a or named downstream layers; approximate=named mathematical method; converge=RunStatus criterion; correct=defect class; evidence=direct automated or formal evidence; exact=literal zero/support or non-approximated registry class; sound=optional method quality; test=automated artifacts; scale=domain size; small=fixture domain; can=error-masking possibility; may=optional method or future scope; explicit_owner=the named rule or ticket; all/every=the explicitly named compiler paths, scores, paper rows, and current-main changes; current=main 5c3b06a or active runtime value; later=named downstream ticket.

**2026-09-01T02:38:32Z**

Completion handoff, 2026-08-31. Clean-ported old PR #82 from current main 5c3b06a without its ancestry. The compiler now rejects reserved MOMENT_MATCHING syntax before arena construction while JSON and hashing retain it. Functional mu3 preserves exact support and Engine.run reports incoherent marginal-predictive support as divergence. ARCHITECTURE contains the paper constraint and Part II Figure 5 ledgers; README and LIMITATIONS use the same bounded claims. Direct raw mu1 and mu2 plus isolated-energy checks landed. Focused results: test_serialize 9 passed; non-slow mu3 8 passed and 3 deselected; GFE, goal, and hygiene 26 passed. Final unchanged-head short suite: 1477 passed, 39 deselected, 0 failed in 566.68 seconds; dashboard refreshed at /Users/ravd/agent-share/cffg-tests.html. No slow run was needed because the changed hostile mu3 mechanism is covered directly in the default suite. Next ready layer: cff-lpsn.

**2026-09-01T14:30:25Z**

Independent-review correction, 2026-09-01. Part I section 5.4 binds one selected factor endpoint to its incident edge marginal through named sufficient statistics; the legacy factor-only payload cannot identify that endpoint or projection. Criterion 2 and the future-work handoff now require cff-o7hs to add and migrate to a faithful schema. Capability owners for this correction are core.ir ConstraintKind and ConstraintSpec, core serialization and structure hashing, compiler rejection, viz.diagram rejection, ARCHITECTURE inventory, README, LIMITATIONS, and cff-o7hs. The wiggle scan was rerun after the rewrite. New or changed retained matches mean: preserve = serialized/hash value identity; sufficient = the mathematical sufficient-statistic projection; can/cannot = representational capability of the named schema; explicit_owner and named_ticket_handoff = cff-o7hs owns the future schema and migration; every = each compiler path named by criterion 1. Earlier retained matches keep the meanings recorded in the planning audit.
