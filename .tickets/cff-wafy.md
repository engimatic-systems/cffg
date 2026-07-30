---
id: cff-wafy
status: in_progress
deps: [cff-d56d, cff-j4mu]
links: []
created: 2026-07-30T17:58:57Z
type: task
priority: 1
assignee: coopwilliams
tags: [gfe, solver, formalism, numerical-robustness]
---
# Formalize mixed-GFE outer inference and choose a proof-backed globalization

## Plain English

Mixed-GFE inference will receive a mathematically explicit outer-solver contract
before its runtime is changed. The work will prove how the coupled outer
residual represents the original constrained-GFE stationarity equations, then
determine whether a monotone double-loop majorization proof closes or whether
the implementation should use a safeguarded trust-region root solve. This keeps
the papers' local updates and free-energy objective intact.

## Motivation

Part II Sections 4.4--4.5 prove that stationary messages correspond to fixed
points of the local Lagrangian, then explicitly warn that iterating those
fixed-point equations need not converge. Corollary 1 replaces an unstable local
message iteration with a Newton solve for the local belief, but the paper's
conclusion still permits a suboptimal local minimum or limit cycle. CFFG already
globalizes that *inner* Newton solve in `rules._gfe_solve_result`; the remaining
defect is the coupled *outer* iteration in `Engine.run`, which applies one
synchronous mixed sweep with a caller-chosen constant damping value.

The retained closed PR #43 exposed that distinction after one learned
transition update: every local Newton solve reported converged while the outer
messages and free energy entered a finite cycle. Fixed damping values from 0.5
through 0.99 and budgets up to 15,000 sweeps did not produce an outer fixed
point.

Closed PRs #42 and #43 preserve an earlier annealed-novelty proposal and its
invalid run. Annealing changed the scientific objective by scaling a transition
novelty channel according to accumulated evidence; it was not an inference
globalization method, and the run stopped in the learn-only arm before
annealing could activate. It therefore supplied no comparative evidence and
cannot repair this solver defect. The replacement research ticket cff-2gev now
requires an explicit generative account of nonstationarity rather than a
one-way objective schedule. This ticket keeps the Bayesian GFE objective at
full strength and asks how to solve its original stationarity conditions.

## Formal problem

Define `T` as one complete **undamped** outer iteration for a mixed-GFE plan:

1. solve every local P-substituted composite to its Corollary-1/D1 stationary
   belief;
2. execute the other iterative factor updates;
3. refresh scheduled BP islands according to the existing mixed schedule; and
4. put every message in a canonical gauge.

Let `C` pack the free coordinates of a feasible fixed-support message state,
with one categorical gauge direction removed per message. In reduced
coordinates `y`, define

    R(y) = C(T(C^-1(y))) - y.

The formalism must prove, under stated support and propriety assumptions, that
`R(y) = 0` is equivalent to simultaneous satisfaction of:

- every local Part II, D1, and ordinary factor stationarity equation;
- every edge/node marginal-consistency constraint represented by the messages;
  and
- the first-order conditions of the complete constrained Bethe/generalised
  free-energy Lagrangian.

The proof must state how categorical gauge freedom, exact-zero support,
Gaussian/EF natural coordinates, scheduled BP elimination, and active
constraints enter the equivalence. It must distinguish a stationary point,
local minimum, and global minimum. No global-minimum claim is permitted for a
non-convex objective without a separate proof.

## Proof decision

Evaluate these routes in order.

### Route A: tight double-loop majorization

Derive whether the supported mixed-GFE objective admits a tight convex
majorizer analogous to convergent Bethe/Kikuchi double-loop methods. A valid
construction must cover the actual P-substituted local terms and edge-entropy
corrections rather than citing the ordinary Bethe result by analogy. Prove that
the bound:

- upper-bounds the unchanged complete GFE for every feasible inner iterate;
- touches it with matching first variation at the current outer iterate;
- has a well-defined convex constrained inner problem; and
- yields non-increasing original GFE with limit points stationary for the
  original constrained objective.

If this proof closes with a practical factor-local implementation, Route A is
the selected implementation contract.

### Route B: trust-region solve of the KKT/fixed-point residual

If Route A does not close, record the exact failed assumption or unsupported
P-substituted term and specify a gauge-reduced trust-region
Gauss--Newton/Levenberg--Marquardt solve of `R(y) = 0`. Define the numerical
merit

    phi(y) = 0.5 * ||D^-1 R(y)||^2

with a dimensionless, dtype-aware scale `D`. Prove that `phi = 0` has exactly
the required stationarity solutions and state the standard assumptions under
which the trust-region iteration globalizes from an arbitrary initial point.
The merit is solver machinery only: it must never replace `Engine.bfe`, enter a
node ledger term, or score a policy. A nonzero stationary point of `phi` is
failure/budget exhaustion, never convergence.

Bare Anderson acceleration is not a proof-backed primary route. Published
safeguarded guarantees assume a nonexpansive or related fixed-point map, which
has not been established here and must not be inferred from damping. Anderson
may only appear as an optional trial step if acceptance and convergence remain
governed by the selected proof-backed method.

The selected route, rejected route, assumptions, and confidence must be
recorded in this ticket and in `docs/DERIVATIONS.md`.

## Reproducer and independent oracles

- Extract a deterministic, minimal state from the retained factored-corridor
  lifecycle after one conjugate transition update. It must reproduce the
  finite outer cycle without importing the abandoned annealing experiment or
  analyzing its behavioral outcomes.
- For a small coupled discrete GFE graph, write the complete constrained
  objective independently of engine messages. Enumerate or directly optimize
  the simplex, verify the KKT equations, and classify the selected solution
  with the projected Hessian.
- Show that the packed residual is gauge-invariant and vanishes on the existing
  analytically checked T-maze and D1 fixed points.
- Make the reproducer reusable by cff-weff so the implementation PR changes
  the solver outcome, not the fixture or tolerance.

## Change-path handoff

Applicable row: **Runtime numerical or status invariant**.

- Authority: Part II Sections 4.4--4.5, the constrained-GFE derivation added to
  `docs/DERIVATIONS.md`, and the numerical contract in `AGENTS.md`.
- Formalism surfaces: `rules._gfe_solve_result`, `Engine.run`, `_make_step`,
  the mixed schedule, `_make_bfe`, and `RunResult` must be mapped even though
  this ticket does not change their runtime behavior.
- Duplicated surfaces: host/JIT and flooding/mixed must be named explicitly;
  exact two-pass schedules are N/A; single-instance/`vmap` determines whether
  the selected solver can feed cff-wbhu; checkpoint decoding is N/A.
- Canonical tests: `tests/test_gfe.py`, `tests/test_b_epistemics.py`,
  `tests/test_schedule.py`, and `tests/test_runtime_status.py`.

## Hostile-input proof obligations

- **Sparse/exact-zero support:** define fixed-support charts, feasible support
  changes, and active-set restarts without probability floors.
- **Vacuous or improper input:** state when `T`, `R`, and the ledger are
  defined, and map undefined/non-finite evaluations to a truthful status.
- **Float32:** every norm scale, stopping test, trust/majorization guard, and
  rank test must derive from the active dtype and problem scale.
- **Uncomfortable scale:** the selected method must admit a matrix-free or
  factor-local implementation for the horizon-8, 24-logit GFE-TM members and
  the existing 32-position by 4-selector rule case; a dense global Hessian or
  Jacobian is not an acceptable production contract.

## Non-goals

- No novelty annealing, objective reweighting, temperature schedule, FE
  surrogate used for policy scoring, or relaxation of the convergence
  tolerance.
- No change to the local Part II/D1 message equations, inner Newton solution,
  node-local GFE terms, or Bethe edge correction.
- No claim that globalization finds the global minimum of every non-convex
  CFFG.
- No implementation of the production outer solver; that is cff-weff.

## Primary references

- van de Laar, Koudahl, de Vries, *Realising Synthetic Active Inference
  Agents, Part II*, Sections 4.4--4.5 and Appendix B:
  https://arxiv.org/abs/2306.02733
- Heskes, Albers, Kappen, *Approximate Inference and Constrained
  Optimization*: https://arxiv.org/abs/1212.2480
- Heskes, Zoeter, *Generalized belief propagation for approximate inference in
  hybrid Bayesian networks*:
  https://proceedings.mlr.press/r4/heskes03a.html
- Conn, Gould, Toint, *Trust Region Methods*, nonlinear equations and
  nonlinear least squares: https://doi.org/10.1137/1.9780898719857
- Zhang, O'Donoghue, Boyd, *Globally Convergent Type-I Anderson Acceleration
  for Nonsmooth Fixed-Point Iterations*:
  https://doi.org/10.1137/18M1232772

## Acceptance Criteria

1. `docs/DERIVATIONS.md` defines `T`, the canonical coordinate map, `R`, the
   complete constrained stationarity system, and proves the zero-set
   equivalence without changing the model objective.
2. Route A either closes with a tight-majorizer proof and an implementable
   double-loop contract, or its blocking assumption is identified precisely
   and Route B receives the complete trust-region residual contract. There is
   no choose-at-implementation-time ambiguity.
3. The chosen contract states its convergence theorem and assumptions, its
   weaker failure outcomes, and why it cannot be read as a global-minimum
   guarantee.
4. The deterministic one-update cycle reproducer and independent small-model
   constrained oracle are executable and pin fixed-point, ledger, and
   projected-curvature claims.
5. The implementation handoff specifies packing, gauge/support handling,
   matrix-free scaling, step acceptance, public statuses/diagnostics, and
   JIT/`vmap` requirements sufficiently for cff-weff to proceed without
   re-deriving the method.
6. Existing local GFE equations and ledger values remain unchanged; the full
   suite passes; `tk dep cycle` and `git diff --check` pass.

## Notes

**2026-07-30T18:01:42Z**

Planning handoff (2026-07-30). The ticket contract and dependency chain are prepared on branch agent/cff-weff-formalism for the proof author to continue in the same draft PR.

Legend: prerequisite --> dependent; ... = linked sequencing only

[x] cff-d56d schedule contract --+
[x] cff-j4mu truthful statuses --+--> [>] cff-wafy formalism (active)
                                          |
                                          +--> [ ] cff-weff implementation
                                                    |
                                                    +--> [ ] cff-izjd hybrid GFE
                                                    ... [ ] cff-wbhu compiled colony loop

Ready root: cff-wafy. Immediate blocker: cff-weff waits for the proof-backed route selection; cff-izjd now waits for cff-weff. cff-wbhu remains formally parallel but is linked because its compiled loop must consume, not reinvent, the selected solver contract.

Next work: derive the complete coupled stationarity/KKT residual in DERIVATIONS; extract the one-update fixed-cycle reproducer from the retained closed experiment branch without importing its annealing mechanism or outcome claims; test whether a tight P-substituted GFE majorizer closes; and record either the double-loop proof or the precise obstruction plus trust-region residual contract. Do not implement production globalization in this ticket. Closed PRs #42/#43 are historical negative evidence only: the run stopped in learn-only before annealing activated, so it supports neither annealing nor a behavioral comparison.

Branch/stack handoff: continue formalism work on agent/cff-weff-formalism and keep its PR based on main. After formalism acceptance, cut the cff-weff implementation branch from that head and base its draft PR on the formalism branch until the first PR merges.

**2026-07-30T18:03:14Z**

Draft PR handoff: https://github.com/engimatic-systems/cffg/pull/47 targets main from agent/cff-weff-formalism. Continue the proof, DERIVATIONS update, and deterministic reproducer on this same branch; keep the PR in draft until cff-wafy acceptance criteria are met.
