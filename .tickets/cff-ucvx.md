---
id: cff-ucvx
status: open
deps: [cff-rbsy]
links: []
created: 2026-09-22T19:18:13Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-3c6f
---
# Rewrite README around a complete newcomer capability map

## Background in Plain English — WHY

The README proves many individual features but does not give a newcomer a clear map of the library. It is more than 1,400 lines, starts with contributor test policy, names only part of the public facade, and mixes high-level explanation with long model-specific walkthroughs, derivation history, migration notes, and benchmark postmortems. Important capabilities such as linear Gaussian inference, exponential-family factors, hard constraints, and MAP inference are hard to find. The current experiment-harness section also shows only a tool.uv.sources fragment, not the complete uv command that adds the dependency.

This ticket rewrites the README around the decisions an experiment agent must make: what CFFG is for, how to install and import it, what it can do, how the API layers connect, how to run and interpret a result, where to find a runnable example, and which limitations control the claim.

## Outcomes / Acceptance Criteria — WHAT

1. The README opens with a concise purpose statement, suitability boundary, Python requirement, and complete local uv add command for a sibling experiment project.
2. The installation path includes an import smoke test and explains the x64 choice without presenting x64 as required for ordinary use.
3. One capability table covers every supported importable experiment feature delivered on the branch.
   3.1. Each row names the task, supported import path, model or factor family, inference or learning mode, primary readout, objective or approximation meaning, runnable example when applicable, and main limitation.
   3.2. Hidden but delivered capabilities such as LGSSM/Kalman inference, Gamma/Beta/Poisson/Bernoulli inference, hard constraints, and max-product/MAP are included.
   3.3. Open, experimental, internal-unstable, and unsupported capabilities are labeled rather than implied.
4. The high-level explanation uses the same concepts and names as the capability table. Every table row connects to a later concise section or an example; no feature appears as an unexplained island.
5. A second editorial pass removes repetition, ticket-history narration, contributor-only test policy, benchmark postmortems, and long model-specific programs from the main flow.
6. Model-specific workflows are replaced by short API patterns and links to tested files under examples/.
7. The README accurately documents the complete supported top-level facade and the cffg.core, cffg.models, and cffg.viz package facades without calling unstable internals stable.
8. Result interpretation is centralized: status gating, readout families, proper or ESS checks, objective_contract and objective_metadata, exact-tree evidence, loopy Bethe, GFE, bounds, and nonlinear surrogates are distinguished in one compact guide.
9. Serialization, learning cadence, compile-once parameter swapping, checkpoints versus bundles, and experiment-harness metadata remain discoverable without repeating family-specific internals.
10. Contributor validation policy is reduced to a short pointer to AGENTS.md.
11. README links and marked executable snippets are validated automatically. The migrated quickstart or example still runs in float32 and x64.
12. README, package facades, examples, and docs/LIMITATIONS.md agree at the proposed PR head.

## Methodological Suggestions — HOW

These suggestions are optional. Another structure is acceptable if it meets the outcomes.

- Suggested order: purpose and fit; install; five-minute workflow; capability map; mental model; build/compile/run/read; learn/act/persist; extension boundary; experiment checklist; examples and reference links.
- Use progressive disclosure. Keep one minimal workflow inline and move model-specific detail to examples or focused API reference paragraphs.
- Generate or test the top-level symbol list against cffg.__all__ so it cannot drift silently.
- Show uv init and uv add --editable ../../cffg, or the current equivalent verified from uv help, rather than a hand-written pyproject fragment.
- Keep the bounded paper claim and exactness warnings close to the first inference result.
- Applicable architecture change path: public user guide and executable documentation; no runtime row applies. The upstream facade and examples tickets are authorities for imports and runnable code. Host/JIT, flooding/scheduled, single/vmap, and current/legacy pairs are N/A except for the retained precision-parameterized quickstart check.

## Future work

- Published package installation, a real vendoring process, hosted API documentation, notebooks, and a 1.0 import-compatibility policy are deferred.
- CFFGZ migration remains cff-1xig.
- Mathematical and performance gaps remain in their existing tickets.
- No other future work is identified.

## Definitions

- Capability: A code feature that can be imported and used in an experiment on the current branch.
- Capability table: The README map connecting each capability to imports, execution, outputs, examples, objective meaning, and limitations.
- ESS: Effective sample size, a diagnostic for importance-weight concentration.
- Facade: A supported package import surface.
- GFE: Generalised free energy, the active-inference objective for supported P-substitution constructions.
- LGSSM: Linear Gaussian state-space model.
- MAP: Maximum a posteriori assignment.
- Objective contract: Runtime metadata that labels the meaning and approximation status of the reported energy.
- Status gating: Refusing to use a result unless its status satisfies the API contract.
- uv add: The uv command that records a project dependency and its source.

