---
id: cff-kfo7
status: in_progress
deps: [cff-8t4d, cff-n183, cff-gvch]
links: [cff-bfr9, cff-n183, cff-gvch]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, refactor]
---
# Extract explicit specialized compiler lowering

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/compiler.py:898-1479` is the source-
region anchor for parameter-edge VMP, P-substitution/GFE composites, hybrid
mixtures, functional evaluators, recurrence, and Gaussian-precision
relationships; lines 1743-1818 anchor hard constraints. The implementation
starts from the exact `cff-ecw0` baseline, and those old line ranges are not
promised future maps. These constructs cross arenas or change structural/
objective meaning and cannot use one ordinary same-arena callback.

This ticket gives those plan classes explicit lowering owners behind one typed
specialized-stage result. It does not change their mathematics or invent a
universal solver interface.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-8t4d` is closed with raw topology and logical slots characterized.
2. The specialized owner exposes two ordered operations:
   2.1. Discovery accepts `Template`, `TopologyLayout`, and semiring and returns
        immutable `SpecializedClaims`: plan kind, owned factors, absorbed
        goals/priors, and inferred parameter-edge shape/axis tags.
   2.2. Completion accepts those claims plus ordinary registered membership
        and returns `SpecializedPlanDrafts` and typed `ArrayRequirement`
        values. It can construct recurrence components without classifying
        factors again. Until cff-ejib lands, a compatibility delegate supplies
        the ordinary membership from the existing compiler path.
3. Named lowerers cover the plan classes present at the `cff-ecw0` baseline:
   parameter-edge Dirichlet VMP,
   mixture/hybrid, recurrence, Gaussian precision, P-substitution and GFE,
   hard constraints, and functional evaluator metadata.
   Structural equality is also owned here: discovery chooses its representation
   from declared edge domains, and lowering may retain the existing runtime
   group shape. Edge DELTA applicability and semantic validation belong here;
   assembly receives validated delta-pair facts without choosing their meaning.
4. Each plan class has one classification/lowering owner. Shared orchestration
   invokes those named lowerers without passing unrelated arena state through a
   universal callback.
   4.1. Neither topology, registered lowering, data resolution, nor assembly
        chooses specialized plan meaning; this ticket owns those choices.
   4.2. Claims partition factor ownership before ordinary grouping. Conflicting
        claims fail with the affected factor or absorbed member. Shared
        parameter terms have one declared owner rather than one owner per user.
   4.3. Drafts describe blob-derived hard-constraint k/table payloads through
        declared requirements and family-owned validators/transforms. They do
        not pretend the numeric payload exists before data resolution.
   4.4. A compact mapping connects each composite's original factors and
        constraints to its runtime plan, local objective terms, absorbed
        factor/edge exclusions, and shared-parameter owner. Existing GFE,
        hybrid, and multi-slice mu3 authority tests check the mapping.
5. The affected rows are **New factor or rule family**, **New IR field**, and
   **Learnable or checkpoint feature**. **Runtime numerical or status
   invariant** is N/A except that emitted plan/status metadata must stay
   identical for runtime consumers.
6. Flooding/scheduled, direct/`vmap`, and latest-schema/supported-legacy
   consumers receive the same specialized drafts. Host/JIT is N/A because the
   stage is host-side.
7. Sparse/vacuous/improper messages are runtime N/A. Float32 preserves plan
   identity and address order. The uncomfortable compiler fixture exercises at
   least one specialized plan in a 32-group template.
8. Existing hybrid, GFE, recurrence, mu3, constraint, learning, evaluator, and
   compile-failure tests protect old/new observable behavior. Existing
   mathematical authorities remain in their family tests; no new recovery or
   scientific benchmark is required for this move.
9. Unsupported substituted roles, constraints, evaluator metadata, shapes,
   and plan combinations retain a loud factor-specific `CompileError`.
10. The specialized stage changes no approximation label, factor behavior,
    objective term, schema, public API, or supported combination.
    Correctness prerequisites cff-n183 and cff-gvch explicitly settle their
    behavior changes first; record their resulting baseline here before moving
    code. Do not silently resolve either issue during extraction.
11. Each named lowerer and its stage module is below 1,000 physical lines. The
    ticket fits one context because it moves structural classification only;
    arrays and final assembly remain separate.
12. Focused specialized-family/compiler tests and the final short suite pass.
13. `docs/ARCHITECTURE.md` names every specialized plan owner in the same
    commit as the extraction.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Return a tagged collection of explicit draft types rather than one object
  with irrelevant optional fields.
- Keep source-specific evaluator identity and GFE/P-substitution details on
  their plan drafts.
- Emit typed array requirements for the next stage instead of reading blobs.

## Future work

`cff-ejib` extracts ordinary registered lowering between discovery and
completion. `cff-bfr9` then resolves arrays, constraint payloads, and learning
addresses from both sets of declared requirements.

## Definitions

- **Specialized plan:** A plan whose multi-arena, structural, evaluator, or
  GFE behavior cannot use the ordinary same-arena contract.
- **SpecializedPlanDrafts:** The typed collection of explicit specialized plan
  values before array resolution.
- **SpecializedClaims:** Discovered ownership and semantic edge tags used to
  exclude composite members from ordinary grouping.
- **Absorbed member:** A goal or prior represented inside a composite rather
  than executed and charged to the objective again as an ordinary factor.
- **Universal callback:** One callable forced to accept unrelated state for
  every plan class.

## Notes

**2026-09-09 preparation at 8cd8919 — decision required before implementation**

Question: can the existing implementation satisfy one shared-parameter owner
without changing behavior? Answer: not for learned goals shared across GFE
shape groups. Ties, count pooling and parameter energy are group-local;
goal_prior_updates can reject two posteriors for the same prior blob. Reproduced
with two observation composites sharing Cc but state cardinalities 2 and 3.
cff-n183 records the reproducer and asks Cooper to choose correct global sharing
(recommended) or explicit temporary rejection. No runtime fix is underway.
Sources: compiler.py:608-645; engine.py:1900-1925,2821-2842,3807-3833;
tests/test_goal_hyperprior.py:103-166,245-252. Confidence: high, formalism-agent
inspection and primary-agent execution.

Question: can discovery lose a conflicting declaration? Answer: yes, the
target-keyed P-substitution dictionary silently retains the last declaration.
cff-gvch separates that validation correction from extraction. Sources:
compiler.py:161-165; core/validate.py:452 onward; test_mu3.py's model with
opposite substituted-role declarations. Confidence: high, bounded inspection
and reproduction. Both correctness tickets are explicit prerequisites.

Question: which decisions require blob contents? Answer: ownership and plan
kind are declaration-only. Dirichlet/GFE/hybrid/recurrence/precision arrays
validate and pack a previously selected kind; hard-constraint k/table contents
also become plan payloads. Suggested ArrayRequirement fields are owner,
ordered (factor, role, blob) references, declared initializer shapes,
family-owned validation/packing operation and a typed destination. Resolve a
blob source once while preserving each role's requirement: the same blob can
have distinct interpretations. The generic resolver must not choose
probabilities versus counts, absorption, evaluators or constraint meaning.
Keep diagnostic bookkeeping separate from ownership claims: invalid Dirichlet
counts currently produce both concentration and orphan-prior diagnostics, and
early claims must not erase either or preempt existing blob-load failures.
Source: bounded modularity-agent inspection of compiler.py at 8cd8919.
Confidence: high. Array resolution remains cff-bfr9, not this ticket.

Question: what is the factor-to-objective ownership map? Answer:

- DirichletGroupPlan absorbs likelihood plus prior; parameter-edge counts are
  excluded from categorical normalization/support/entropy and its dedicated
  energy function owns the parameter term.
- GfePlan absorbs an observation and categorical goal; the substituted output
  has no ordinary interior-edge entropy correction. A learned goal also absorbs
  its Dirichlet prior. Fixed goals are absorbed even though existing compiled
  goal_members is empty; discovery must retain original identity without
  changing that public field merely for convenience.
- Mu3Plan intentionally overlays the same observation members as GfePlan for
  evaluator/shared-parameter energy ownership. This is not a second execution
  claim to reject as a collision.
- GfeTmPlan consumes only the transition-mixture factor and does not put its
  output in gfe_x_edges; downstream factor and interior-edge correction remain.
- HybridGfePlan absorbs the mixture and Gaussian goal, removes the substituted
  output entropy correction, retains selector/input edges, and owns goal_m and
  goal_W without a separate goal execution or energy charge.

Sources: compiler.py:330,462,608,1477; engine.py:3780 onward;
compiled.py:365 onward at 8cd8919. Confidence: high, bounded formalism-agent
review against existing family tests. Preserve these distinctions in claims
and the compact mapping required by acceptance criterion 4.4.

Dependency companion, checked with tk dep tree --full cff-kfo7, tk ready,
tk blocked and tk dep cycle. The two correctness roots have no tk prerequisites;
cff-n183 nevertheless awaits Cooper's scope choice. They can be separate PRs
and do not depend on each other. Extraction waits for both and topology.

```text
Legend: prerequisite --> dependent
[x] cff-544x --> [x] cff-8t4d (1,676 tests pass) ---+
[ ] cff-n183 (ready in tk; user decision pending) --+--> [ ] cff-kfo7
[ ] cff-gvch (ready; independent correctness fix) -+         |
                                                           v
                                              [ ] cff-ejib registered
                                                           |
                                                           v
                                              [ ] cff-bfr9 arrays
                                                           |
                                                           v
                                              [ ] cff-ddsq assembly
                                                           |
                                                           v
                                              [ ] cff-edoz compiler track
                                                           |
                                                           v
                                              engine track --> cff-cfqg epic
```

**2026-09-09T03:29:39Z**

PR #108 research — question: can ordinary lowering precede specialized ownership? Answer: no; GFE absorbs ordinary-looking goals/priors before ordinary grouping, and recurrence completion later needs ordinary BP membership. One specialized owner now discovers claims first and completes drafts after registered membership. Raw slots remain topology-owned; semantic parameter-edge tags and blob-derived constraint payloads have explicit later owners. Sources at 7885bf5: compiler.py:760-823, 877-947, 1389-1392, 1547-1569, 1776-1815, 2080-2130. Confidence: high from direct code inspection. Assembly checks execution ownership and objective exclusions without rediscovering semantics.

**2026-09-09T20:03:17Z**

Correctness baseline update: Cooper approved full cross-group learned-goal sharing; cff-n183 is now closed at code head cdbad68 with 1,690 default-suite tests passing. GoalParameterPlan is the immutable global owner by serialized prior blob, independent of GfePlan shape buckets. goal_parameters.py owns count pooling and the complete parameter FE contribution; OuterChart ties trial-coordinate aliases. Existing GfePlan.goal_* fields are compatibility views, not independent ownership. Preserve this corrected behavior during extraction; it is an explicit authorized correctness prerequisite, not an unexplained parity exception. cff-gvch remains the only open immediate prerequisite before specialized extraction. Current dependency companion: [x] topology + [x] cff-n183 + [ ] cff-gvch (ready) --> cff-kfo7 --> registered --> arrays --> assembly --> compiler track --> engine track --> cff-cfqg; arrows mean prerequisite to dependent.

**2026-09-09T20:23:43Z**

Bounded extraction review after cff-gvch implementation at ad028f0: structural discovery and draft completion can remain one ticket if numerical validation, packing, addresses and assembly stay in the compiler compatibility path. Proposed interfaces: discover_specialized(template, topology, semiring) -> immutable SpecializedClaims; complete_specialized(claims, OrdinaryMembership(bp_members)) -> SpecializedPlanDrafts. Claims carry explicit family identities/group signatures, parameter-edge tags, execution ownership separately from objective/absorbed-edge facts, domain-based equality routing, deferred structural diagnostics, and immutable topology references. Reuse compiled records when entirely structural; hard constraints instead declare a payload requirement, never a placeholder k/table. ArrayRequirement names owner, ordered factor/role/blob references, initializer shapes, family recipe and destination keys or payload. Keep current diagnostic checkpoints: admission, parameter layout, numeric validation, recurrence/late constraints and DELTA. In particular, do not let early discovery preempt blob/shape failures, and retain invalid-Dirichlet-count to orphan-prior diagnostic bookkeeping outside semantic ownership. Ordinary-membership compatibility computation supplies recurrence completion; classify_recurrence_components only needs variables/endpoints and membership sets, not mutable Template. Preserve fixed categorical goal identity even where GfePlan.goal_members stays empty; mu3 and global GoalParameterPlan are objective/parameter owners, not duplicate execution claims. Runtime goal pooling/energy remain in goal_parameters.py; its host plan builder can delegate to specialized lowering. Sources: bounded read-only modularity-agent inspection of compiler.py:154-415,868-1045,1220-1336 and recurrence_topology.py:16 at ad028f0. Confidence high. Do not move array loading or packing mathematics early into this ticket.

**2026-09-09T20:27:25Z**

Ownership-map test preparation: reuse compile-only fixtures from test_compiler_contracts.model/compiled. parameter_edge: dirvmp executes lik and absorbs theta_prior; prior_in stays ordinary; theta is excluded via param_shape, not gfe_x_edges. gfe: gfe:2x2 executes obs and absorbs fixed goal, excludes x, retains prior/z; public goal_members remains empty. test_shared_goals.mixed_goals(states=(2,3,2)): gfeC:3x2 has obs0/obs2, gfeC:3x3 has obs1; absorb matching goal/cprior, retain state factors; one GoalParameterPlan for Cc with group_rows ((gfeC:3x2,(0,1)),(gfeC:3x3,(0,))). mu3 fixture: gfeLM executes obs0/obs1 and absorbs goals; one mu3:gfeLM:2x2:Ac evaluator overlay owns the shared functional parameter coordinate/KL, not duplicate factor execution. test_b_epistemics.build_curiosity_engine(True): gfeTML:4x4x2 executes tm1/tm2 only; absorbs no facing factors, leaves ordinary substituted-output entropy, and Bc does not create mu3/goal/Dirichlet-carrier owners. hybrid_gfe: gfeH:gaussian_mixture_linear:1x1:2 executes emission, absorbs Gaussian goal, excludes x entropy, retains state/selector and z/s corrections. Across fixtures require ordinary ownership plus specialized claimed-factor sets to be disjoint and exhaustive; overlays are separate referenced objective/parameter owners. Sources: bounded formalism-agent inspection of test_compiler_contracts.py:536-718, test_shared_goals.py helper, test_b_epistemics.py:530 and test_mu3.py:131 at ad028f0. Confidence high; no new math, existing numerical authority tests stay unchanged.

**2026-09-09T20:29:06Z**

Pre-extraction comparison source: detached worktree /tmp/cffg-specialized-baseline.QWbM09/repo at ad028f0, containing the approved shared-goal fix and declaration-conflict correction. It is a read-only parity/diagnostic oracle, not a second implementation branch. Existing characterization fixtures remain the primary contract; use targeted cross-worktree comparisons only for new extraction boundaries.

**2026-09-09T20:33:32Z**

Corrected extraction baseline is now complete: cff-n183 at cdbad68 and cff-gvch at ad028f0, with 1,709 default-suite tests passing at ad028f0 (1015.217 seconds wall, 41 slow deselections, four existing Gym warnings). Both correctness prerequisites are closed. The detached ad028f0 comparison worktree contains the exact numerical, declaration-validation, encoding and diagnostic behavior to preserve. Extraction is ready; no pending user decision.

**2026-09-09T20:34:36Z**

Implementation starts on agent/cff-kfo7-specialized-lowering from updated agent/cff-gvch-psub-validation at 69864ab (PR #117); corrected code baseline ad028f0. Applicable change-path rows: New factor or rule family, New IR field, Learnable or checkpoint feature. Emitted immutable plan metadata remains the shared input for flooding/scheduled and direct/vmap consumers; numerical runtime/status changes N/A, host/JIT compiler pair N/A. JSON/current/legacy bundles reconstruct the same declarations; agent checkpoint envelopes N/A. Scope is explicit immutable discovery and completion, named family lowerers, typed array requirements, equality/DELTA meaning and factor/objective ownership. A compatibility ordinary-membership product feeds recurrence completion; array resolution/validation order, packer mathematics, observation/member addresses and final Arena assembly remain in compiler until their own tickets. Named deferred checkpoints preserve existing error precedence, including blob-load failures and invalid concentration plus orphan-prior diagnostics. Sparse/improper incoming messages N/A, float32 plans identical, 32-group specialized fixture required. No approximation, source schema, supported-combination, message, energy or public-plan field change. Source modules stay below 1,000 physical lines. Focused family/compiler tests throughout, then one final dashboard-backed short suite after the final code change.
