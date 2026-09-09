---
id: cff-ty0v
status: closed
deps: []
links: [cff-cfqg]
created: 2026-09-09T03:23:33Z
type: task
priority: 1
assignee: coopwilliams
external-ref: gh-109
tags: [architecture, review]
---
# Review PR 108 architecture plan against engine goals

## Background in Plain English — WHY

Cooper wants to know whether PR #108 builds the architecture needed for lower
warm execution latency across many graphs, faithful CFFG mathematics, and understandable
factor extensions. The PR defines work tickets; this review changes that plan
before engine implementation starts.

## Outcomes / Acceptance Criteria — WHAT

1. The review records the PR head and evaluates its ticket changes against
   the implementation and the papers using three focused research agents.
2. Findings name their code or paper sources, affected ticket owners, and
   confidence; ticket refinements resolve the findings selected for this plan.
3. A PR based on PR #108 contains the ticket refinements and a dependency
   diagram derived from `tk` records.
4. `/Users/ravd/agent-share/CFFG architecture overhaul/index.html` explains
   the original proposal and review refinements in separate columns, with
   links to evidence, ticket changes, and the PR.
5. Ticket parsing, dependency-cycle detection, whitespace checks, and the
   required wording scans pass review; the HTML loads through the file server.
6. The final handoff states what the epic delivers and which throughput or
   production claims still require downstream work.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method may satisfy the outcomes.

- Use the four rows of the architecture change-path index to audit owners.
- Audit flooding/scheduled, host/JIT, direct/batched, and checkpoint pairs in
  the proposed plan. Runtime validation is N/A for this ticket-only review.
- Keep engine implementation and benchmark execution outside this review.
- Keep the HTML readable on a phone and usable without external assets.

## Future work

`cff-cfqg` owns implementation of the revised architecture plan. `cff-o7hs`
owns new nonlinear factor behavior, `cff-wbhu` owns compiled-loop throughput,
`cff-rqqc` owns stochastic Colony isolation, and `cff-j7mm` owns cost reporting.

## Definitions

- **CFFG:** Constrained Forney-style factor graph, the model formalism used here.
- **JIT:** Just-in-time compilation of array operations by JAX.
- **Batched execution:** Running several model instances through one array program.
- **Checkpoint:** Saved agent state used to resume execution.

## Notes

**2026-09-09T03:31:46Z**

Wording and capability audit (2026-09-08): reviewed all scan matches in their ticket context for cff-209l, cff-3ola, cff-5xdy, cff-8t4d, cff-ahps, cff-bfr9, cff-cfqg, cff-ddsq, cff-edoz, cff-ejib, cff-gl89, cff-ii9t, cff-jrbw, cff-kfo7, cff-ny7f, cff-nyad, cff-ofv1, cff-sf29, cff-ws3c, cff-ty0v, cff-wera, cff-zq25. Three research agents reviewed their compiler, formalism, and runtime subsets; the primary reviewer checked the umbrella, roadmap, lifecycle/readout/strategy, and review tickets. Scans were rerun after refinement. Retained candidate meanings follow; historical audit counts in older notes remain historical.

all, any, every, only: the locally enumerated factors, interfaces, fixtures or explicit scope restriction.
at_least: inclusive numeric lower bounds on batch size, cardinality, dimension, or scale.
approximate: named mathematical approximations; in source-region language, an approximate location.
available, can: the named supported path, data availability, or specifically described capability/consequence.
canonical: the single internal readout owner.
check, test, validate, verify: the named executable or artifact checks; input/result validators where explicitly identified; scientific validation is explicitly deferred.
close, complete, deliver, land: acceptance-satisfied ticket closure, fully populated declared values, or ordered implementation/artifact delivery.
converge: the numerical stopping contract and its explicitly named downstream compiled loop.
correct: entropy correction, named estimator repair, or scientific correctness distinguished from characterization.
current, existing: the specified prerequisite baseline or a dated historical audit.
equivalent, parity, preserve, same: the identified baseline values, applicable execution pairs, matching snapshots, shared owner, or one message representation; equal beliefs alone was refined to require equal parameter/observation/evaluator snapshots.
estimate, performance, speed: named byte/cost reporting or warm latency/throughput work; this architecture protects execution structure without claiming a speedup.
evidence, oracle: named code/paper provenance, characterization artifacts, or independent mathematical references.
exact: a specified commit, literal zero/vacuous value, exactly-once ownership, or exact two-pass inference.
explicit_owner, named_ticket_handoff: the named stage, migration, result-policy, estimator, learning, or downstream throughput owner; endpoint 'own' is the emitting factor's message slot.
expose: documented exports/readouts or typed internal operations as stated.
fit: callback compatibility or the bounded extraction responsibility.
fixed: fixed numerical shapes, fixed parameters, or the fixed-A family.
generic: the generalized hybrid-registration boundary in a preserved historical note.
implement, integrate, work: future code delivery across the named compiler/runtime/reconstruction path; implementation identity means the serialized kernel identity.
infer: posterior computation, initializer metadata derivation, or the expressly disallowed inference of a speedup, according to the sentence.
intended: the meaning recorded in the historical wording-audit note.
later, remaining: the explicitly ordered downstream work or retained compatibility property.
learn, learnable, sufficient: declared parameter/statistic/address/cadence surfaces; sufficient statistics is the mathematical term.
match: mathematical moment matching, or stated shape/snapshot agreement.
may, should, sound, will: permitted alternate methods, dated design intent, a method satisfying acceptance, or the planned extraction.
n_a: runtime probes inapplicable to host topology compilation or explicitly unchanged schema/decoding boundaries; applicable deferred work has a named ticket.
near: the compiler source-region anchor, not a promised new line number.
pass: test/artifact success, forwarding declared values, or an algorithmic traversal/message-passing term.
public: documented/exported consumer types and call forms.
read, register: array/register transport versus executable factor registration, as locally distinguished.
ready: dependency readiness or a fully initialized/prepared value ready for its stated consumer.
recover: explicitly deferred scientific/source recovery.
reusable: shared rule owners/fixtures or repeated execution without another trace at fixed static controls.
robust, scale, small: the repository numerical contract, its stated domain/horizon/dtype probes, or the narrow local block/fixture.
stable: declared serialization compatibility or the named compiler contract; no new stable Python extension API is promised.
support: literal probability support, declared constraint/semiring coverage, or supported execution/checkpoint paths.

Capability ownership was separately checked. Specialized classification includes equality and edge DELTA; blob-derived payloads are resolved through declared family operations. Engine and Colony share status interpretation. cff-wera/cff-zq25 own the two arena migrations; cff-j7mm, cff-wbhu, and cff-rqqc retain cost analysis, compiled throughput, and stochastic batching. No omitted owner was treated as a successful wording scan.

**2026-09-09T03:35:40Z**

Review complete at PR #108 head 7885bf5b954307ffb671ced9c92c6ecb7489ea8b. PR #109 is stacked on agent/cff-cfqg-architecture-salvage: https://github.com/engimatic-systems/cffg/pull/109.

Delivered: 19 refined original tickets, two migration tickets (cff-wera and cff-zq25), and this handoff. Three other PR #108 tickets retain their text. The refinements specify BP/VMP and assignment-score contracts, specialized discovery/completion and equality/DELTA ownership, resolved constraint payloads, local solver facts and registerized transport, a shared Engine/Colony result policy, actual batch isolation, and bounded execution-structure checks.

The plain-English report is /Users/ravd/agent-share/CFFG architecture overhaul/index.html, served at http://100.115.170.73:8080/CFFG%20architecture%20overhaul/index.html. It includes eight proposal/refinement comparisons and the original/revised text for all 25 review tickets.

Validation: all 25 tickets parse through tk show; selected metadata also parsed through tk query; dependency trees, ready roots, and blocked edges were inspected; tk dep cycle found none; git diff --check passed. Chromium checked desktop and 390px phone layouts, no page-width overflow, 25 embedded tickets, 22 refined/new versus three kept, filtering, deep links, raw comparisons, and no JavaScript errors. No production code changed and no inference suite or benchmark ran.

Next: review/merge the stacked planning changes, then begin cff-wera. Implementation tickets remain open. cff-wbhu, cff-rqqc, and cff-j7mm remain separate throughput, stochastic-batching, and analysis work; architecture completion does not certify production readiness. At Cooper's request no further wiggle-audit runs were made after the completed review pass.
