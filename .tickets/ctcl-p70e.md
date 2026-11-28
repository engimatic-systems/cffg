---
id: ctcl-p70e
status: in_progress
deps: []
links: []
created: 2026-11-28T05:05:52Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-2gev
---
# Score transition novelty for named state blocks in a stationary factored rollout

Changing-corridor comparisons also need a stationary model that values information about named transition parameters. The factored builder currently rejects epistemic_b, and RollingAgent requires a learnable literally called B. The flat builder already supports the required native transition-mixture composite.

## Design

Extend the public factored builder and RollingAgent to route a named transition owner to the existing counts-form GFE-TM on future slices and the expected-parameter form on executed slices. Keep the controls on the same native ledger. Do not modify the inference equation, anneal novelty, or launch the behavioral comparison here.

## Acceptance Criteria

An arbitrary block name and a product selector compile and round-trip, with no future observation counted as executed evidence. Check a stationary analytic column, public rollover on one physical observation, zero observed mass, true f32 and a larger selector. Existing flat and no-epistemics cases keep their outputs. Document initialization and the distinction between positive Dirichlet prior support and fixed known impossible transitions.


## Notes

**2026-11-28T05:06:03Z**

Change paths: template builder and learned parameter declaration. Host/JIT, flooding/scheduled, single/Colony consume the same existing lowered rule and ledger, so no new kernel or cache path. Current and legacy templates with the option disabled retain their ordinary blobs. Test the declared plan/readout against the existing native conditional algebra, selective learned blocks, all-past windows, JSON/bundle round trips and both hard/latent inputs. No scored data or simulated change. Dependency direction: ctcl-p70e (stationary named model) and ctcp-6msb (reset derivation) --> ctcp-w305 (changing model) --> ctcp-qodi (experiment).

**2026-11-28T05:13:29Z**

Native D1 stationary implementation complete. With equal outcome probabilities and counts (1,1) vs (10,10), the action posterior matched the independent digamma expression (about .54209 on the less familiar action); planning added zero counts and the next applied action/observation added one count to the corresponding entry. Checked 30 product selectors in actual f32, separate owners, sparse fixed support and learned CFFGZ. All 37 tests in the focused factored_rollout, b_epistemics, agent and README files including slow passed. The two new cases passed again on final formatted code. Full suite pending. Research limitation: selected stationary block requires full support. ctcp-w305 must qualify declared sparse faces separately.
