---
id: cff-cj5h
status: open
deps: []
links: [cff-i45j, cff-gul8, cff-fxmj, cff-hp3y, cff-4oq8]
created: 2026-07-06T00:37:22Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-cfqg
---
# Ergonomics: epistemic lint, MI-term diagnostic, MAP tie-breaking

## Background in Plain English

Three valid model states can mislead a user even though inference runs successfully. First, a future observation can face a goal factor without P-substitution. That graph performs ordinary KL-control and does not receive the observation-epistemic term the user may have wanted. Second, `Engine.bfe` returns one scalar, so a user cannot see whether a decision came from a goal term, an epistemic term, another node, or an edge correction. Third, `numpy.argmax` selects index zero when several actions share the maximum value, so an indifferent MAP policy can repeatedly execute the first action.

These are usability problems, not reasons to reject a valid model. This ticket owns three remedies: opt-in lint results that name suspicious declarations or runtime choices, an engine free-energy breakdown that reconciles with the scalar ledger, and seeded uniform selection among exactly tied MAP actions. The default compiler remains quiet. Structural validation remains an error boundary.

The missing-P-substitution case has a strong behavioral anchor. At repository commit `63c7a51`, `README.md` and `tests/test_tmaze_learning.py` record that adding one P-substitution bead to the same T-maze graph separates a roughly 90-percent learning agent from a 0-percent ablation. Closed ticket cff-i45j later showed that the repaired observation-GFE engine ranked sensing actions in the same order as their independently computed mutual information: 0.0693 nats for each of two informative actions and 0.0092 nats for the weaker action. Those results justify making the objective visible; they do not make KL-control invalid.

This ticket's objective diagnostic is distinct from cff-gul8. The diagnostic explains terms already scored by a compiled GFE model at the supplied state. Ticket cff-gul8 owns expected information gain for missing DATA ports and answers which candidate observation is most informative to acquire. Neither value may be labeled as the other.

## Outcomes / Acceptance Criteria

1. The package exposes a JAX-free `cffg.core.lint(template)` call that returns lint findings without changing the template.

2. Each lint finding has a stable machine-readable code.

3. Each lint finding names the model object that caused it.

4. Repeated lint calls on value-equal templates return value-equal findings in the same order.

5. The epistemic lint identifies each future observation likelihood that faces a categorical goal factor without a P-substitution constraint on that likelihood.

6. The epistemic lint message states that the named goal-observation pair contributes no P-substitution epistemic term.

7. The epistemic lint message states that ordinary KL-control remains valid for the named pair.

8. A future observation likelihood with the required P-substitution constraint produces no missing-P-substitution finding.

9. A perception likelihood whose observation edge is bound through a DATA port produces no missing-P-substitution finding.

10. A template with no goal-facing observation likelihood produces no missing-P-substitution finding.

11. The mean-field lint identifies a factor whose declared mean-field split has one non-parameter random-variable edge and therefore equals its unsplit belief.

12. A factor with two or more non-parameter random-variable edges produces no single-edge mean-field finding.

13. `compile_template` remains silent about lint findings unless its lint option is enabled.

14. Enabling the `compile_template` lint option emits each structural lint finding through Python's warning mechanism.

15. Enabling the `compile_template` lint option leaves the compiled `Arena` value-equivalent to compilation with lint disabled.

16. A structural validation failure still raises the existing model or compile error when lint is enabled.

17. `Engine.run(..., harden_last=0)` emits a lint warning when the arena contains at least one DELTA constraint.

18. The DELTA warning states that the named run disables hardening for the arena's DELTA-constrained edges.

19. `Engine.run` emits no disabled-hardening warning when the run executes a positive hardening phase.

20. A run on an arena without a DELTA constraint emits no disabled-hardening warning.

21. `Engine.fe_breakdown(state)` is a documented readout for sum-product and GFE engines.

22. Each breakdown term has a stable owner kind.

23. Each breakdown term has a stable owner name.

24. Each decomposed breakdown entry has stable component labels.

25. Each breakdown entry has one numeric owner total.

   25.1. Each labeled component has one numeric value.

26. The breakdown ordering is deterministic for a given arena.

27. Reading a breakdown does not mutate the supplied `State`.

28. The breakdown includes each ordinary factor-node contribution present in the scalar ledger.

29. The breakdown includes each constraint-node contribution present in the scalar ledger.

30. The breakdown includes each GFE-composite contribution present in the scalar ledger.

31. The breakdown includes each interior-edge entropy correction present in the scalar ledger.

32. The breakdown includes each shared parameter contribution present in the scalar ledger.

33. The sum of all breakdown owner totals agrees with `Engine.bfe(state)` within `256 * finfo(dtype).eps * max(1, number_of_owners) * max(1, abs(Engine.bfe(state)))` when the scalar ledger is finite.

34. The breakdown and `Engine.bfe` use one shared implementation of each mathematical ledger term.

35. `Engine.fe_breakdown` raises `RuntimeError` on a max-semiring engine.

36. `Engine.fe_breakdown` raises `RuntimeError` before evaluating an unprepared mu3 ledger.

37. A proven empty exact support produces a positive-infinity breakdown total.

38. A non-finite local ledger contribution remains non-finite in its named breakdown term.

39. A non-finite local ledger contribution is not replaced by a finite number in the breakdown total.

40. Every fixed-goal discrete observation-GFE composite reports an `extrinsic` component equal to `-sum_x q(x) log c(x)`.

41. Every fixed-goal discrete observation-GFE composite reports an `epistemic` component equal to `sum_x q(x) log q(x) + sum_z q(z) h_A(z)`.

42. Every fixed-goal discrete observation-GFE composite reports a `state_negative_entropy` component equal to `sum_z q(z) log q(z)`.

43. The three components in criteria 40 through 42 sum to that composite entry's owner total.

44. For a fixed observation matrix, the `epistemic` component equals the negative mutual information `-I_q(X;Z)` under an independently computed joint distribution.

45. A learned-observation-matrix composite labels its corresponding channel `parameter_aware_epistemic` rather than exact fixed-channel mutual information.

46. A learned-goal composite reports its shared goal-Dirichlet ledger contribution separately from its structural composite contribution.

47. A mu3 composite reports its shared importance-sampled parameter KL contribution separately from its structural composite contribution.

48. A GFE-TM composite reports its direct node-local total without including the downstream message coupling repaired by cff-hp3y.

49. A GFE-TM epistemic component is labeled as an optimistic transition-information bound.

50. A hybrid GFE composite retains the `reported_gfe_lower_bounds_exact` direction from `Engine.hybrid_gfe_diagnostics`.

51. `Engine.fe_breakdown` does not accept a query edge or a candidate DATA-port list.

52. The GFE objective diagnostic does not claim to rank missing DATA ports.

53. Sparse categorical support has explicit diagnostic behavior.

   53.1. A literal probability zero contributes zero to an entropy or mutual-information summand.

   53.2. A supported deterministic fixed observation channel produces a finite GFE component breakdown.

   53.3. A zero-probability separator or outcome does not create `NaN` through a zero-times-log-zero expression.

   53.4. The diagnostic introduces no positive probability floor for a literal probability zero.

54. Vacuous and improper states have explicit diagnostic behavior.

   54.1. Vacuous incoming messages act as ledger identities when the resulting local belief is proper.

   54.2. An improper local belief produces a non-finite breakdown total whenever `Engine.bfe` is non-finite for that state.

55. Float32 diagnostic behavior follows the numerical robustness contract.

   55.1. A proper float32 GFE fixture produces finite component values.

   55.2. The float32 component sum satisfies criterion 33 using float32 epsilon.

   55.3. Any diagnostic floor, guard, or tolerance derives from `finfo(dtype)` and the number of summed cells or terms.

56. An uncomfortable-scale breakdown fixture contains 80 latent states.

57. Every breakdown component in the uncomfortable-scale fixture is finite.

58. The uncomfortable-scale breakdown satisfies criterion 33.

59. MAP action selection chooses uniformly among indices whose probabilities equal the exact maximum.

60. MAP action selection excludes an index whose probability is below the exact maximum by one representable floating-point step.

61. The exact-tie comparison introduces no absolute or relative near-tie threshold.

62. Two agents with the same seed and the same sequence of action posteriors produce value-equal MAP action sequences.

63. Two agents with different seeds can produce different MAP actions from the same exact tie.

64. Restoring a checkpoint written by the implementation's `RollingAgent.checkpoint` resumes the exact seeded tie-breaking sequence.

65. A legacy checkpoint without an `rng_state` field retains the seed-initialized fallback behavior present at repository commit `63c7a51`.

66. A deterministic uniformity test draws at least 4,096 seeded choices from a four-way tie.

   66.1. In the test from criterion 66, each action's frequency differs from one quarter by no more than 0.03.

67. A float32 action posterior with 128 exactly tied maxima returns an index in the tied set.

68. Repeated choices from the 128-way float32 tie exercise more than one action index.

69. MAP action selection rejects a posterior containing a non-finite value.

70. MAP action selection rejects a posterior containing a negative probability.

71. MAP action selection rejects a posterior with zero total mass.

72. Posterior-sampling action selection keeps its existing normalized-probability behavior.

73. `RollingAgent.act` uses the seeded exact-tie rule when `select="map"`.

74. The T-maze learning harness uses the seeded exact-tie rule when `select="map"`.

75. The behavioral epistemics simulator uses the seeded exact-tie rule when it executes a MAP action.

76. The hybrid epistemics simulator uses the seeded exact-tie rule when it commits to a regime action.

77. `Engine.decode` remains a deterministic graph-assignment readout rather than an environment-action selector.

78. DELTA hardening remains deterministic for a fixed compiled state.

79. CSP decimation remains deterministic for a fixed compiled state.

80. Internal optimizer candidate selection remains deterministic for fixed inputs.

81. The lint test tier includes the missing-P-substitution positive case.

82. The lint test tier includes every no-warning case in criteria 8 through 10.

83. The lint test tier includes the single-edge mean-field case.

84. The lint test tier includes the DELTA-disabled-hardening case.

85. The lint test tier includes a template with at least 256 future goal-observation pairs.

86. The 256-pair lint result is deterministic.

87. The breakdown test tier independently checks the fixed-goal algebra in criteria 40 through 44.

88. The breakdown test tier covers every hostile-input criterion in items 53 through 58.

89. The breakdown test tier covers every ledger-owner category in criteria 28 through 32.

90. The tie-breaking test tier covers every selection criterion in items 59 through 80.

91. `README.md` documents how to request lint findings.

92. `README.md` documents how to inspect a free-energy breakdown.

93. `README.md` documents seeded MAP tie-breaking.

94. `docs/ARCHITECTURE.md` records the stable free-energy breakdown ownership model.

95. `docs/ARCHITECTURE.md` records that diagnostic terms and scalar scoring share mathematical implementations.

96. `docs/ARCHITECTURE.md` distinguishes objective epistemic components from cff-gul8 predictive DATA-port EIG.

97. `docs/LIMITATIONS.md` records each GFE diagnostic component that is a bound or approximation rather than exact mutual information.

98. `docs/todo-notes/ergonomics.md` is deleted in the implementation commit after its remaining work is represented by durable code, documentation, and tests.

## Methodological Suggestions

These are optional suggestions. An implementer may choose another sound method that satisfies every numbered outcome.

Add a JAX-free lint module under `src/cffg/core/` and export its public function from `cffg.core`. Represent findings with a small immutable record containing at least a code, owner identity, and message. Reuse validation's structural vocabulary, but do not make a lint finding a `ModelError`. Let `compile_template(..., lint=True)` expand and validate the template through its existing path, then emit the same findings that the standalone call returns. Keep the default value false.

Detect a goal-facing observation from topology and declared factor roles rather than factor names. The compiler's GFE-composite discovery near `src/cffg/compiler.py` already demonstrates the relevant `matrix_cat`-output-to-`categorical`-goal relationship. Detect the single-edge mean-field case from non-parameter structural roles. Detect disabled DELTA hardening where `Engine.run` has both the compiled DELTA groups and the resolved `harden_last` value.

Refactor ledger construction so scalar scoring and the breakdown consume the same per-owner term functions. A likely public shape is an immutable `FreeEnergyBreakdown(total, entries)` whose entries carry `(owner_kind, owner_name, total, components)`. An entry's owner total participates once in the global sum; its labeled components explain that total and are not added to the global sum again. Prefixing ownership by kind avoids collisions between factor and edge names. Preserve deterministic arena order rather than sorting numerical results after evaluation.

For a fixed-goal observation composite, derive the split directly from the stationary belief and the rule definitions in `src/cffg/rules.py`: `extrinsic = -q_x dot log(c)`, `epistemic = q_x dot log(q_x) + q_z dot h_A`, and `state_negative_entropy = q_z dot log(q_z)`. Use an independently normalized `q(x,z)=A[x,z]q(z)` oracle in tests. For learned parameters, shared goal coordinates, GFE-TM, and hybrid GFE, retain the ledger ownership and bound directions already documented in `rules.py`, `hybrid_gfe_diagnostics`, `docs/DERIVATIONS.md`, and `docs/LIMITATIONS.md`. Do not reinterpret a Lagrange message coupling as node energy.

Centralize seeded exact-tie action selection in one helper that accepts a one-dimensional probability array and an explicit NumPy generator. Find the exact maximum, collect entries value-equal to it, and call the generator's uniform choice on those indices. Route environment-executing MAP choices through that helper. Leave graph decoding, DELTA projection, CSP decimation, and optimizer-internal candidate selection deterministic.

The `docs/ARCHITECTURE.md` change-path index has no exact row for this combined ergonomics ticket. Treat the JAX-free core lint function as the authority for structural findings, `src/cffg/engine.py`'s ledger term functions as the authority for the breakdown, and the shared action-choice helper as the authority for exact MAP ties. Derived surfaces include the compiler lint option, runtime DELTA warning, public exports, GFE family ledger paths, `RollingAgent`, action-executing experiment harnesses, and documentation. No new serialized model field, factor family, message rule, free-energy formula, or checkpoint field is intended.

Review the duplicated execution surfaces as follows:

- Host/JIT: lint and action choice are host operations. The scalar and detailed ledgers may use JIT, but they must call the same mathematical term functions.
- Flooding/scheduled: both runtime routes feed the same breakdown. A converged supported model must reconcile under criterion 33 after either route.
- Single-instance/`vmap`: the public breakdown reads one `State`; a Colony-wide breakdown is outside this ticket. Batched internal energy functions must retain the same per-member terms before host labeling.
- Current/legacy checkpoint decoding: current checkpoints must preserve the random-generator continuation. Legacy checkpoints retain the existing seeded fallback because this ticket adds no checkpoint field.

Focused tests may live beside validation/compiler tests, ledger-family tests, agent tests, and the affected experiment tests. After the final code change, run the repository's required short suite once through `uv run python tools/dashboard.py --run-suite`. Delete `docs/todo-notes/ergonomics.md` in the same implementation commit that lands its durable replacements.

## Future work

Ticket cff-gul8 is the direct downstream feature. It consumes the naming distinction established here and owns predictive EIG across candidate missing DATA ports; it does not own GFE ledger decomposition. This ticket delegates no other work. Colony-wide breakdowns, user-configurable near-tie policies, and lint policy beyond the three named findings have no ticket owner in this handoff and require separate tickets before implementation.

## Definitions

- **Bethe free energy (BFE):** The sum of local factor terms and interior-edge entropy corrections used by the engine's ordinary sum-product ledger.
- **Breakdown term:** One named numeric contribution to the engine's scalar free-energy ledger.
- **Categorical goal factor:** A categorical factor that places a preference distribution on a future observation edge.
- **Constraint node:** A compiled hard or variational constraint with its own ledger contribution.
- **DATA port:** A boundary edge through which runtime observation evidence enters a template.
- **DELTA constraint:** A form constraint that hardens a categorical marginal to one selected value during a configured hardening phase.
- **Dtype:** A numeric storage type such as float32 or float64.
- **Epistemic component:** The part of a GFE objective that values information under that composite's documented exact or approximate semantics.
- **Exact tie:** Two or more finite action probabilities whose stored floating-point values equal the largest stored value.
- **Expected information gain (EIG):** The mathematical expectation of uncertainty reduction from a future observation. Ticket cff-gul8 applies this to candidate DATA ports.
- **Extrinsic component:** The fixed-goal cross-entropy contribution `-sum_x q(x) log c(x)` in a discrete observation-GFE composite.
- **Free-energy ledger:** The complete set of node, shared-parameter, and edge-correction terms whose sum is the engine's reported BFE or GFE.
- **Generalized free energy (GFE):** The active-inference objective produced by P-substitution and its matching local and edge ledger terms.
- **GFE-TM:** The transition-mixture P-substitution composite used for dynamics curiosity.
- **Goal-Dirichlet contribution:** The shared learned-goal prior, posterior, and predicted-count ledger term.
- **Harden:** Replace a soft categorical belief with a point mass during a DELTA phase.
- **Improper belief:** A local product that cannot be normalized into a valid finite belief.
- **JAX:** The array and compiled numerical computing library used by the engine runtime.
- **KL-control:** Ordinary inference and goal-directed control without the P-substitution epistemic term on the named goal-observation pair.
- **Lint finding:** A non-fatal, machine-readable notice about a valid but potentially surprising declaration or runtime choice.
- **Literal probability zero:** An assignment excluded from support and represented by the runtime's log-zero sentinel.
- **MAP:** Maximum a posteriori selection of an assignment or action with the largest posterior value.
- **Mean-field constraint:** A declared factorization that replaces a joint belief with a product of marginal beliefs.
- **Mu3:** The importance-sampled functional message from future observation-GFE slices to a shared uncertain observation matrix.
- **Mutual information (MI):** `I(X;Z) = sum q(x,z) log(q(x,z)/(q(x)q(z)))`.
- **Negative entropy:** `sum_i q(i) log q(i)`, the negative of Shannon entropy.
- **Node-local:** Belonging to one factor or composite without a downstream Lagrange message coupling.
- **Objective diagnostic:** A readout of terms the compiled model already scores at the supplied state.
- **P-substitution:** A variational constraint that replaces an ordinary future-variable belief and introduces the corresponding GFE objective channel.
- **Parameter-aware epistemic component:** The observation-GFE channel formed with expected observation-matrix moments under a parameter belief; it is not labeled as fixed-channel MI.
- **Predictive DATA-port EIG:** Mutual information used to compare missing observation ports before choosing one to acquire.
- **Python warning mechanism:** The standard `warnings` module interface that callers can capture, filter, or display.
- **Random-generator state:** The complete internal state of the seeded NumPy generator used to continue a reproducible random sequence.
- **State:** The engine's immutable collection of messages, observations, parameters, and solver auxiliaries for one compiled arena.
- **Structural validation:** Checks whose failure means a template cannot be compiled correctly; unlike lint, these checks raise errors.
- **Vacuous message:** The multiplicative identity message, which supplies no evidence.

## Notes

**2026-08-07**

Evidence and source preservation:

- The original ticket and `docs/todo-notes/ergonomics.md` identify the three user-facing failures and record the frozen-at-index-zero MAP symptom. No versioned raw episode trace for that symptom is present at repository commit `63c7a51`, so the ticket requires a direct symmetric-posterior regression.
- `README.md` and `tests/test_tmaze_learning.py` preserve the roughly 90-percent versus 0-percent P-substitution ablation result.
- Closed ticket cff-i45j preserves the post-repair 432-state sensing audit and its 0.0693, 0.0693, and 0.0092-nat ranking.
- Closed ticket cff-hp3y establishes that a GFE-TM downstream message coupling is not node-local free energy.
- Closed ticket cff-4oq8 supplies independent normalized-joint MI and behavioral evidence for the ordinary observation-GFE channel.
- Closed ticket cff-fxmj traces the fixed-goal discrete observation-GFE rule to the repository papers and records the limits of broader MI claims.
- Commit `63c7a51` records cff-gul8's reviewed ownership of predictive DATA-port EIG; this ticket retains the separate GFE objective-diagnostic boundary without reopening that ticket.

Retained wiggle candidates after the 2026-08-07 bounded scan (candidate -> intended meaning): fixed -> a supplied goal matrix or a pinned compiled input; preserve -> value identity, semantic identity, or checkpoint continuation; public -> a documented package contract; reusable -> shared validation vocabulary; same -> value equality, matched experimental controls, or one shared mathematical implementation; expose -> export the named lint call; implement -> shared code construction or planned work; support -> probability support or an explicitly accepted input boundary; complete -> completion of this ticket review in the final marker; land -> durable artifacts entering the repository in one implementation commit; remaining -> work still represented by the ergonomics todo note; approximate -> the named GFE family bound or approximation; close -> administrative completion of a named historical ticket; near -> a probability below the maximum by a stated amount or a configurable near-tie policy outside this ticket; check -> an automated assertion or structural validation; converge -> a `RunStatus.CONVERGED` runtime result; correct -> mathematical or structural validity; equivalent -> value equality of compiled arenas; evidence -> empirical results or observation input; exact -> literal floating-point equality, exact probability zero, exact tree or channel semantics, or absence of an applicable change-path row; match -> agreement under a stated numeric bound; oracle -> an independently implemented numerical reference; reproduce -> seeded sequence reproduction; robust -> compliance with the numerical-robustness contract; sound -> an alternative method satisfying the outcomes; stable -> an identifier or ordering promised by the documented contract; test -> an automated executable check; validate -> structural model validation; infer -> the named inference or control calculation; learn -> the multi-episode parameter-learning result; scale -> model cardinality or term-count growth; small -> the suggested lint record's code size; can -> user or engine capability described by the sentence; expected -> mathematical expectation or expected parameter moments; intended -> declared absence of specified repository changes; may -> permission, possibility, or prohibition as fixed by the sentence; will -> the map records that no normative ticket sentence retains this candidate; explicit_owner -> this ticket owns lint, ledger diagnostics, and exact-tie action choice, while cff-gul8 owns predictive port EIG; named_ticket_handoff -> cff-gul8 is the concrete downstream owner; scope_exclusion -> cff-gul8 is explicitly denied ownership of ledger decomposition; all -> every breakdown owner total; any -> every diagnostic guard in the named class; at_least -> an inclusive minimum count; every -> each member of the locally named factor, component, criterion, or action set; current -> the current/legacy checkpoint review pair; existing -> behavior present at commit `63c7a51`; later -> historical ordering of cff-i45j after the original finding; read -> machine access to a `State` or breakdown.

Intelligible-ticket review: complete.
