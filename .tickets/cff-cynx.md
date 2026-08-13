---
id: cff-cynx
status: open
deps: [cff-izjd, cff-du6m]
links: []
created: 2026-07-13T03:51:42Z
type: epic
priority: 2
assignee: coopwilliams
---
# Epic: structure learning module (the outer loop)

## Background in Plain English

The engine can infer hidden variables and update supported parameters, but it does not change a model's graph, state count, or factorization. `docs/LIMITATIONS.md` H3 and S6 place that responsibility in an outer model-comparison loop. This epic owns that outer loop: it proposes candidate structures, fits them on one data partition, compares them on a separate partition, and records accepted fragments for later proposals.

The engine continues to own inference, free-energy accounting, and run status. The outer loop must not add a structure-changing gradient or alter an engine ledger value. A repeated poor evaluation score is a configurable reason to request proposals, not a proof that the current model class is wrong; this distinction preserves the warning in `docs/LIMITATIONS.md` H7.

The repository already supplies the graph-building and execution foundations: `Template`, serialized edit actions, typed-port composition, structural hashes, conjugate parameter updates, and `Colony` groups. Closed ticket cff-izjd also supplies the D5 hybrid regime-epistemics composite. That composite may choose data-gathering actions, while this epic's outer loop compares structures using held-out model evidence.

Latent labels can be permuted without changing a model, as `docs/LIMITATIONS.md` H9 explains. Candidate comparison must therefore align labels or use quantities that do not depend on label names. `docs/READING.md` sections 2 and 3 provide the curated rationale for add, split, prune, merge, typed fragments, and later proposal sources.

## Outcomes / Acceptance Criteria

1. The documented structure-search module runs outside the engine objective.
2. A structure-search run leaves each engine-produced free-energy ledger value unchanged.
3. A candidate hypothesis consists of a source `Template` plus serialized edit actions and typed-port composition operations that produce its candidate `Template`.
4. Each candidate record identifies its source template hash, edit payload, resulting template hash, proposal source, and parent candidate.
5. The module fits candidate parameters without accessing observations assigned to the evaluation partition.
6. The module evaluates each candidate on one identical set of held-out rollout artifacts for a comparison round.
7. Structure ranking uses Bethe free energy from an engine evaluation without P-substitution.
8. Generalized free energy is excluded from the model-evidence score.
9. A comparison round ranks a lower finite held-out Bethe free energy ahead of a higher one.
10. A candidate whose engine run reports budget exhaustion receives no finite rank.
11. A candidate whose engine run reports divergence receives no finite rank.
12. A candidate whose engine run reports infeasibility receives no finite rank.
13. A comparison result records each candidate's score, engine run status, fit-partition identifier, evaluation-partition identifier, and evaluation-template hash.
14. The result warns that Bethe free energy uses the named Bethe approximation when an evaluated candidate contains loops, as required by `docs/LIMITATIONS.md` H1.
15. A caller can configure a positive held-out Bethe-free-energy threshold and a consecutive-evaluation count for the proposal tripwire.
16. The tripwire requests proposals when the active candidate's held-out Bethe free energy exceeds the configured threshold for the configured number of consecutive evaluations.
17. The built-in structural operators succeed on permitted inputs and reject invalid inputs:
17.1. The add-regime operator produces a valid candidate with one additional mixture regime for a fixture permitted by the accepted cff-du6m equation.
17.2. The add-regime operator uses the parameter convention accepted by cff-du6m.
17.3. The add-regime operator returns a machine-readable rejection for an invalid input.
17.4. The split-state operator produces a valid candidate that replaces one discrete latent state with two states.
17.5. The split-state operator returns a machine-readable rejection for an invalid input.
17.6. The prune operator removes one Dirichlet state for a fixture permitted by the accepted cff-du6m equation.
17.7. The prune operator removes one Gaussian edge for a fixture permitted by the accepted cff-du6m equation.
17.8. The prune operator removes one Gaussian regime for a fixture permitted by the accepted cff-du6m equation.
17.9. Each prune operation uses the parameter convention accepted by cff-du6m.
17.10. The prune operator returns a machine-readable rejection for an invalid input.
17.11. The merge operator combines two Dirichlet states for a fixture permitted by the accepted cff-du6m equation.
17.12. The merge operator uses the parameter convention accepted by cff-du6m.
17.13. The merge operator returns a machine-readable rejection for an invalid input.
18. Permuting latent-state labels without changing a candidate's probability model does not change its rank.
19. The module races candidates through `Colony` groups while retaining a separate engine run status for each candidate.
20. The module retains a separate score for each candidate raced through `Colony` groups.
21. A serialized fragment-library entry contains a `Template` fragment, its typed ports, provenance, and an identifier that is unchanged by a serialize-deserialize round trip.
22. The shipped library contains one validated object-persistence fragment.
23. The shipped library contains one validated contact-causality fragment.
24. The shipped library contains one validated agency fragment.
25. The proposal interface accepts built-in operators and caller-provided proposal sources through one candidate-record contract.
26. The shipped built-in operators complete a proposal-and-score round without an external proposal source.
27. An integration fixture lets the cff-izjd D5 regime-epistemics composite choose observations for candidate switching models and then gives those observations to the outer evaluator.
28. An impossible observed event retains the engine's infeasible status in the comparison result.
29. A candidate with an impossible observed event receives no finite rank.
30. Vacuous observations retain the engine's reported status in the comparison result.
31. The outer loop does not add an evidence term for vacuous observations.
32. Improper incoming messages retain the engine's divergent or infeasible status in the comparison result.
33. A candidate with improper incoming messages receives no finite rank.
34. A float32 comparison fixture produces finite scores for valid candidates.
35. A float32 comparison fixture selects the unique winner selected by its float64 reference.
36. A scale fixture evaluates at least ten times as many candidates as the smallest proposal-and-score fixture at the proposed epic implementation commit without retaining pairwise candidate state.
37. A scale fixture evaluates a discrete cardinality at least ten times the smallest operator fixture's cardinality at the proposed epic implementation commit.
38. The cardinality scale fixture retains the engine's run status for each candidate.
39. `README.md` documents the module's construction and use.
40. `docs/ARCHITECTURE.md` documents the module's component boundary.
41. `docs/LIMITATIONS.md` documents the module's accepted approximations.

## Methodological Suggestions

The implementer may choose another sound method if it satisfies the outcomes. Prefer the contracts in `cffg.core.edits`, `cffg.core.compose`, and `cffg.core.serialize` for hypothesis construction and provenance. Prefer `Learnables` in `src/cffg/learn.py` for candidate fitting and `Colony` for grouping identical structures.

Use a non-P-substituted evaluation copy when an acting model contains a D5 generalized-free-energy composite. Keep fit and evaluation partition identifiers in the comparison artifact. Use label alignment only when a downstream report needs aligned semantics; otherwise prefer a label-invariant score.

Before implementation, split this epic into review-sized tickets after cff-du6m reports its formulas. Each implementation handoff names the applicable `docs/ARCHITECTURE.md` change-path row. The likely rows are **New IR field** for a fragment or proposal schema added to `Template`, **Learnable or checkpoint feature** for candidate fitting or stored parameters, and **Runtime numerical or status invariant** if orchestration changes how engine statuses are transported. For each row, inspect host/JIT, flooding/scheduled, single-instance/`vmap`, and current/legacy checkpoint pairs; mark a pair N/A when that pair is structurally irrelevant to the planned change.

Use operator-specific tests for edit validity and hand-checkable score oracles. Also test the numerical-robustness cases in criteria 28-38. Derive any numerical guard from the active dtype and state the invariant it protects.

## Future work

Ticket cff-du6m owns the literature pass that extracts the AXIOM growth and merge criteria, the supervised-structure-selection prior, and Bayesian model reduction formulas for the repository's Dirichlet, normal-inverse-Wishart, and matrix-normal-inverse-Wishart parameterizations. Its accepted equations unblock criteria 17.1-17.2, 17.6-17.9, and 17.11-17.12 of this epic.

As of 2026-08-07, no ticket owns a large-language-model proposer, library compression in the style of DreamCoder, or a broader grammar of structural forms. Those proposal sources remain outside this epic's acceptance criteria; criterion 25 keeps the interface open to a future ticket.

This epic does not block another open roadmap item. Closed ticket cff-izjd is a prerequisite because it owns the D5 experiment-selection capability used by criterion 27.

## Definitions

- **Agency fragment:** A graph fragment that represents an actor whose actions can affect modeled outcomes.
- **AXIOM:** The expanding object-centered model described in the 2025 paper listed in `docs/READING.md` section 2.
- **Bayesian model reduction:** A method for scoring a reduced model from an already fitted full model without fitting the reduced model from scratch.
- **Bethe free energy:** The engine's model-fitting objective and model-evidence approximation. Lower values are better within one controlled comparison.
- **Candidate:** One proposed model structure under evaluation. A valid candidate passes `Template` validation and compiles with its declared data shapes.
- **Candidate record:** The provenance artifact that describes how one candidate was produced.
- **Checkpoint:** Serialized runtime parameters and beliefs that a later process can restore.
- **Colony:** The repository runtime that groups model instances by structural hash and keeps one state and run result per instance.
- **Conjugate parameter update:** A closed-form Bayesian update in which the posterior belongs to the prior's distribution family.
- **Constrained Forney-style Factor Graph (CFFG):** The project's graph representation for factors, variables, and local variational constraints.
- **Contact-causality fragment:** A graph fragment that represents effects caused by modeled physical contact.
- **D5 hybrid regime-epistemics composite:** The cff-izjd engine component that lets uncertainty about a discrete regime influence data-gathering actions in a discrete-continuous model.
- **Dirichlet:** A probability distribution used as a conjugate belief over categorical probabilities.
- **DreamCoder:** The fragment-library compression system listed in `docs/READING.md` section 3.
- **dtype:** The numeric storage type used by an array, such as float32 or float64.
- **Edit action:** A serializable instruction that changes a `Template`.
- **Epistemic term:** An objective contribution that rewards an action for reducing modeled uncertainty.
- **Evaluation partition:** Observations reserved for comparing fitted candidates and excluded from parameter fitting.
- **Fit partition:** Observations that candidate parameter updates may use and that structure ranking does not score.
- **Fragment library:** A persisted collection of graph fragments that proposal sources may compose into candidates.
- **Free-energy ledger:** The engine's sum of mathematically defined local energy and entropy terms.
- **Gaussian:** A normal probability distribution over continuous values.
- **Generalized free energy:** The planning objective produced by an engine model with P-substitution; it includes goal and epistemic terms and is not the held-out model-evidence score in this epic.
- **Held-out rollout:** A time-ordered observation sequence assigned to the evaluation partition.
- **Host/JIT pair:** The ordinary host-language path and the just-in-time-compiled path that must transport equivalent semantics.
- **Intermediate representation (IR):** The typed in-memory model structure consumed by validation, serialization, and compilation.
- **Label alignment:** A mapping between latent-state names in two candidates that accounts for arbitrary permutations.
- **Label-invariant score:** A score whose value does not depend on latent-state names.
- **Latent state:** A model variable that is inferred rather than directly observed.
- **Large language model:** A text-generating model that a later integration may use as a proposal source.
- **Machine-readable rejection:** A structured failure code and context that software can inspect without parsing prose.
- **Matrix-normal-inverse-Wishart:** A conjugate belief over a linear map and its covariance.
- **Model evidence:** The probability that a model assigns to observed data after accounting for uncertain latent variables and parameters.
- **Normal-inverse-Wishart:** A conjugate belief over a Gaussian mean and covariance.
- **Object-persistence fragment:** A graph fragment that represents an object's continued existence across time.
- **Outer loop:** Orchestration that proposes, fits, and compares whole engine models without becoming part of any candidate's inference objective.
- **P-substitution:** The CFFG constraint that changes a local Bethe objective into a generalized-free-energy objective for planning.
- **Posterior:** A probability distribution after conditioning on observations.
- **Provenance:** The recorded source and transformations of an artifact.
- **Proposal-and-score round:** One call that obtains candidates, fits them, evaluates them, and returns their comparison records.
- **Proposal source:** Code that returns candidate records, such as a built-in edit operator or a caller-provided generator.
- **Proposal tripwire:** A configured rule that asks proposal sources for candidates after the active candidate exceeds a held-out score threshold for a declared number of evaluations.
- **Regime:** One discrete mode in a switching model.
- **Run status:** The engine's categorical result: converged, budget exhausted, diverged, or infeasible.
- **Scheduled execution:** Message updates ordered by a compiled schedule instead of updating all eligible messages together.
- **Structural hash:** A stable digest of graph structure that excludes fitted data values and blob locations.
- **Template:** The repository's serializable graph representation and composition unit.
- **Typed port:** A declared fragment boundary whose type constrains valid composition.
- **Validated fragment:** A fragment that passes `Template` validation, compiles after its declared ports are bound, and has an executable fixture for the behavior named by its library entry.
- **Vacuous observations:** An evaluation input that supplies no observation information to the candidate.
- **`vmap`:** JAX's transformation for applying one function across a batch axis.

## Notes

**2026-08-07 — downstream-first intelligibility review**

Source review: the historical ambiguity slice for cff-cynx; direct prerequisites cff-du6m and closed cff-izjd; `docs/LIMITATIONS.md` H1, H3, H7, H9, and S6; `docs/READING.md` sections 2 and 3; the change-path index and Colony boundary in `docs/ARCHITECTURE.md`; and the graph-edit, composition, serialization, Colony, and engine-ledger source paths cited above. No conclusion from the unfinished cff-du6m research was treated as settled.

Legend: prerequisite --> dependent

```text
[x] cff-izjd (D5 capability delivered) --------+
                                                  +--> [ ] cff-cynx
[ ] cff-du6m (ready; immediate blocker) --------+

cff-cynx is blocked and has no downstream roadmap dependent.
```

Current ambiguity scan, candidate -> intended meaning:

- `end_to_end` -> component validation across port binding, compilation, and the named behavior fixture.
- `fixed` -> fixed-dataset parameter fitting for one declared fit partition.
- `integrate` -> criterion 27 joins the D5 observation output to the evaluator in one fixture; the definition describes a deferred external proposal connection.
- `n_a` -> structurally irrelevant to the planned change, after inspection.
- `preserve` -> the meaning of the H7 warning remains intact.
- `register` -> concrete parameter-family registration in the engine.
- `implement` -> planned code work or a planned implementation handoff.
- `provide` -> the cited reading document supplies design rationale.
- `support` -> parameter families registered for engine updates.
- `complete` -> an execution reaches the end of one proposal-and-score round; the review marker below means this ticket rewrite is finished.
- `deliver` -> the component-completion state recorded for cff-izjd.
- `ready` -> the `tk` dependency state of cff-du6m because it has no open prerequisite.
- `remaining` -> named work is deliberately excluded from this epic.
- `appropriate` -> reasonable engineering judgment about an alternative method.
- `approximate` -> the named Bethe mathematical approximation, not an unspecified tolerance.
- `close` -> the administrative status of ticket cff-izjd.
- `converge` -> the engine's successful runtime-status category.
- `equivalent` -> behavioral equivalence between host and JIT status transport.
- `estimate` -> updating candidate parameters from the fit partition.
- `evidence` -> probabilistic model evidence in each occurrence.
- `oracle` -> a hand-checkable numeric test reference.
- `pass` -> a focused literature review for cff-du6m, or successful `Template` validation in the definitions.
- `robust` -> survival of the hostile numerical cases in criteria 28-38.
- `sound` -> a reasonable alternative method that still satisfies all outcomes.
- `stable` -> the structural-hash contract produces an unchanged digest for unchanged graph structure.
- `test` -> automated execution against a fixture.
- `validate` -> `Template` input validation or end-to-end fragment component validation, as identified by the adjacent noun.
- `fit` -> parameter estimation from the fixed fit partition; hyphenated field names and definitions refer to that process.
- `infer` -> engine posterior inference for hidden state or regime variables.
- `learn` -> the title means model-structure search; `Learnables` names the repository's integrated parameter-fitting surface.
- `learnable` -> the declared `Learnables` parameter artifact or its integrated fitting surface.
- `scale` -> candidate-count throughput load or discrete-cardinality domain size, as named by the criterion.
- `small` -> the candidate count or cardinality of the named baseline fixture at the proposed implementation commit.
- `can` -> a demonstrated capability in acceptance text; background and definitions describe a possibility.
- `intended` -> the heading labels the selected sense for each scan candidate.
- `may` -> possibility in background and definitions, or an optional method in Suggestions.
- `explicit_owner` -> cff-cynx owns the outer workflow, the engine owns inference and ledgers, cff-du6m and cff-izjd own their named prerequisites, and the future-work sentence explicitly denies an owner.
- `named_ticket_handoff` -> cff-du6m is the concrete future research owner and cff-izjd is the concrete delivered-capability owner.
- `all` -> a universal quantifier over outcomes, guards, or candidate objectives in the adjacent phrase.
- `any` -> all numerical guards are covered; the negated definition excludes the outer loop from all candidate objectives.
- `at_least` -> an inclusive numeric lower bound of ten times the named baseline.
- `only` -> label alignment is restricted to reports that need aligned semantics.
- `current` -> the active candidate model, the active checkpoint format paired with its legacy format, or the scan recorded in this dated note.
- `deferred` -> work postponed beyond this epic.
- `later` -> work postponed beyond this epic.
- `read` -> `docs/READING.md` identifies the document whose content supplies context.

Intelligible-ticket review: complete.
