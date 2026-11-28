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
