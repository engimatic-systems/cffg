---
id: cff-u06e
status: closed
deps: []
links: [cff-1o72, cff-o7hs]
created: 2026-09-15T15:31:47Z
type: task
priority: 3
assignee: coopwilliams
---
# Specify exact zero-noise neural factors

## Background in Plain English — WHY

Some models need the output to equal a neural network prediction exactly: `y = f_theta(x)`. This conditional relationship has zero noise, even when uncertainty about the input makes the output uncertain. It needs its own message and free-energy rules because an ordinary Gaussian conditional requires an invertible noise covariance. This ticket establishes a bounded mathematical contract before production implementation.

## Outcomes / Acceptance Criteria — WHAT

1. A named class of neural maps has an explicit exact-equality contract, including input/output dimensions and any smoothness, invertibility, or rank restrictions; unsupported maps have stated rejection conditions.
2. `docs/DERIVATIONS.md` and adjacent oracle docstrings derive messages in both directions and the node-energy/edge-entropy convention for that class using a Dirac delta conditional. Any approximation in the messages or objective is named separately from the exact model relationship. No positive variance or precision clamp substitutes for equality.
3. An independent analytic fixture verifies the derived messages and objective, including uncertainty propagated from the input and the declared treatment of impossible evidence.
4. The derivation and executable evidence address exact constrained support, vacuous or improper incoming messages, rank-deficient maps, float32, and a case with 10–100 times the basic fixture dimension or factor count; any structurally irrelevant case has a reason. Numerical guards use the active dtype and scale.
5. A linked implementation ticket records the proved domain, required compiler/runtime/serialization changes, and explicit rejection boundaries before this derivation ticket closes.

## Methodological Suggestions — HOW

- Choose a small map class with an independent analytic reference; the method is optional if it satisfies the outcomes. Keep formal equations and a plain-English explanation in `docs/DERIVATIONS.md`.
- Use the **New factor or rule family** change path in `docs/ARCHITECTURE.md`; the production handoff also names **New IR field** and **Runtime numerical or status invariant** where applicable. Before implementation, enumerate affected surfaces and explicitly check host/JIT, flooding/scheduled, single-instance/vmap, and current/legacy decoding, with reasons for N/A cases.
- Compare the proposed rules with existing exact affine cases. Follow the AGENTS.md numerical robustness and short-suite protocols for any executable oracle changes.

## Future work

- cff-1o72 owns neural means with strictly positive Gaussian noise and proceeds independently of this derivation. Neither cff-o7hs nor the cff-ahps roadmap requires exact zero-noise factors.
- Production exact-factor integration and training are deferred to the implementation ticket required by outcome 5; that ticket has not yet been created.

## Definitions

- **Dirac delta conditional:** A probability distribution that puts all output probability at the exact network result for a given input.
- **Neural map:** A network computation `f_theta`, where `theta` denotes its weights.
- **Rank restriction:** A condition on the number of independent directions a map can transmit.
- **Free energy:** The graph objective combining local energy and entropy contributions.
- **Oracle:** An independently computed reference used to check a rule.
- **Vacuous message:** An incoming message that supplies no information.
- **Improper belief:** A proposed probability distribution that cannot be normalized.
- **Exact support:** The states allowed by the model, including its equality constraints.
- **float32 / dtype:** A 32-bit floating-point number format / the active numerical representation.
- **Host/JIT, flooding/scheduled, single-instance/vmap:** Ordinary/compiled execution, simultaneous/ordered message updates, and individual/batched model execution.

## Notes

**2026-11-28T04:45:13Z**

New factor or rule family change path. Use an invertible, coordinatewise two-slope neural activation in equal input and output dimension. Positive slopes permit exact push-forward and likelihood pullback. Integrate finite interval densities on the graph of the map and compare the complete ordinary tree ledger to analytic event evidence. Include impossible support, vacuous and non-normalizable beliefs, rank loss, float32 and 32 dimensions. This is a derivation with an independent host oracle. Compiled/host, flood/exact, single/vmap and current/legacy codec obligations go in the linked production ticket; no new schema or factor in this PR.

**2026-11-28T04:47:00Z**

Research question: which sign and reference measure should an exact deterministic conditional use? The input-chart graph law dx δ_f(x)(dy) gives −H(q_x), independently of a chart rewrite in y including E log |det Df|. Part I Eq. 6 (notes/pt1.txt lines 220–237) and Part II Eq. 1 (notes/pt2.txt lines 295–305) give +H for each degree-two separator. The displayed sign in Part I Eq. 8 conflicts with its factorization; the complete two-unary oracle verifies the sign independently. Confidence high within the separable positive-slope interval family.

**2026-11-28T05:42:07Z**

D13a and its independent interval oracle establish forward density transport, backward likelihood composition, and the tree energy for separable invertible two-slope activations. Five focused examples cover both precisions, 32-dimensional product evidence and impossible support. The canonical four-worker suite on 0a1d307 passed all 2,843 selected tests in 436.343 seconds. The archived runtime receipt and the worktree record match byte for byte (SHA-256 113d9497902cb8c3ab3007bf02ac290538b167ae306025ad7ef952373555d276). cenc-luez tracks the production interval representation, compiler, energy and persistence checks.
