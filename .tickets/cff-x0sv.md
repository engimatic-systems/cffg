---
id: cff-x0sv
status: in_progress
deps: [cff-cd1i, cff-ri8m]
links: [cff-gjjk, cff-2gev]
created: 2026-07-08T22:00:30Z
type: feature
priority: 2
assignee: coopwilliams
---
# Generic conjugate-update API: declared learnables replace learn_a special-casing

GPT-5.5 feedback item 4 (agreed, but sequenced AFTER B-learning so the API generalizes from two concrete clients, not zero — repo culture is concrete-first). Each factor type declares its conjugate family and sufficient-statistic map: categorical/matrix_cat/transition_mixture -> Dirichlet counts from node beliefs; gaussian/gaussian_linear -> NIW stats (that half lands with cff-7c5o, design the contract to fit it). Template marks learnable parameter roles (constraint bead or FactorSpec flag — decide; serialization additive). RollingAgent.learn_a/learn_b collapse into a learnables dict {factor_or_group: counts}; checkpoint/restore generic. Includes d-learning (accumulate q(z0) across episodes) and e-learning (habits: accumulate chosen controls) as trivial instances — GPT item 2 folds in here EXCEPT learned goals c which stays cff-gjjk (hyperprior mu1(c)=Dir(c|Az+1) has its own message derivation). DELEGATION: Fable specifies the contract + which updates are exact conjugacy vs mean-field approximations (document per type); Opus refactors A/B onto it behind existing tests (zero regressions) and adds d/e instances.

## Acceptance Criteria

A- and B-learning re-expressed on the API with zero test regressions; d and e learnable with recovery tests; contract documented in ARCHITECTURE (rule-library section).


## Notes

**2026-07-09T21:23:55Z**

cff-7c5o landed the Gaussian half of what this API generalizes: Engine.gaussian_node_stats (NIW/MNIW sufficient stats per node belief, obs-masked) + cffg.conj (NIW/MNIW updates with rollover() returning VMP-correct E-parameters — mean and E[W]=nu*Psi^-1, per D4.1.3). The generic declared-learnables API should unify: discrete counts (agent.py learn_a/_elog), Gaussian NIW/MNIW (conj.py + stats readout), and EF-edge conjugates (in-loop, free — cff-8s6a). The trial-loop pattern to generalize is tests/test_conj.py::test_trial_loop_recovers_lgssm_parameters (compile once, swap params through state, accumulate stats, update, roll).

**2026-07-10T03:31:28Z**

Evaluation finding for this API: member_of covers discrete groups only — consumers needing per-member writes into Gaussian/EF/hybrid/plate-expanded groups currently write state.params[group] arrays by member order by hand. The generic learnables API should extend member_of across all group families and offer a bulk set_params(factor_name -> arrays) helper.

**2026-07-27T20:14:54Z**

New consumer (2026-07-27): cff-2gev now depends on this API.

The annealed-novelty experiment needs a factored corridor model — an
orientation chain selecting into a cell chain, with turn rules as one
shared parameter block — because the flat single-chain rollout gives
curiosity no structure to exploit. That model is expressible in the IR
today, but RollingAgent cannot learn it: the agent is hardcoded to the
build_rollout node names (member_of["prior"] line 97, "u1" line 142,
"act1"/"obs1" line 150, "zm1" line 152, "tm1" line 160) and to a single
[U, K', K] count array bound to that one transition node.

Two requirements this adds to the contract:

- learnables must address a named factor or group, so a model with two
  transition nodes on separate chains can learn both
- accumulation must work when several factor instances share one parameter
  blob, which is how the IR already ties parameters (build_rollout ties Bs
  across every slice, and plates tie by name across copies). Counts from
  every instance sharing a blob accumulate into that one block.

The second is the substantive one: parameter tying is what makes learning
transfer across states, and it is the reason the factored model is worth
building at all.

**2026-07-27T20:52:00Z**

Contract decided and implemented (branch cff-x0sv-declared-learnables).

The ticket left open where the declaration lives. It lives on the Template
as a LearnableSpec naming a conjugate family and a set of BLOBS — not a
FactorSpec flag and not a constraint bead.

Why blobs: the ticket's own second requirement is that accumulation works
when several factor instances share one parameter blob, because that is how
the IR ties parameters (build_rollout ties Bs across every slice, plates tie
by name across copies). Tying is the thing that makes learning transfer, so
the posterior has to be addressed at the thing that is tied. A FactorSpec
flag declares per instance and would then need a separate tying story; a
constraint bead would put a learning concern into the message-passing bead
vocabulary, where it does not belong.

A learnable spans a SET of blobs for two reasons, and both are load-bearing:
consumption form (A read as probabilities by the rules, Ac read as raw
counts by the GFE composite that differentiates them) and joint family (an
MNIW over a gaussian_linear owns matrix, offset and cov together). That is
why LearnPlan carries per-blob shapes rather than one shape — it is the hook
cffg.conj's NIW/MNIW needs, per "design the contract to fit it".

Three rules the implementation settled:

- GFE composites are write targets, never stat sources. The composite owns
  its update path and its node belief is not a plain factor belief. So a
  planning graph with parameter uncertainty has zero sources and writes raw
  counts; the perception graph supplies statistics and writes E[log theta].
  A graph with no sources is normal, not an error.
- Cadence is not part of the contract. accumulate() takes the graph, the
  state and an optional name filter; the agent decides that a transition
  updates per tick and an initial-state prior per episode. That is a
  modeling choice, and hardcoding it is what the old learn_a/learn_b did.
- LearnableSpec.sources restricts which instances supply statistics.
  Habits forced this: the control prior is tied across every horizon step
  (it applies at each), but only the executed control is evidence. Where
  tying and evidence differ it is a property of the model, so the model
  declares it — this is what keeps RollingAgent from naming "e1".

Also fixed en route: build_rollout ignored the gfe flag when choosing
between A and Ac, so learned_a with gfe=False compiled a counts array as if
it were a conditional and failed the stochasticity check. Learning A without
an epistemic channel now works.

Delivered against acceptance: A and B re-expressed on the API with the
existing tests unchanged; d and e added as instances with recovery tests;
contract in ARCHITECTURE section 3b, including which updates are exact
conjugacy (BP node beliefs) and which are mean-field (VMP groups).

Not done here, deliberately: registering NIW/MNIW as families. cff-7c5o
already landed that machinery (Engine.gaussian_node_stats + cffg.conj) and
the contract is shaped for it, but member_of covers the discrete families
only, so a Gaussian learnable currently raises a clear compile error naming
that limitation. Wiring it up is a follow-up.

Also still open from the 2026-07-10 evaluation note: extending member_of
across all group families and a bulk set_params helper. Same blocker, same
follow-up.
