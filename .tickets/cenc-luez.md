---
id: cenc-luez
status: closed
deps: [cff-u06e, cei-k93f]
links: []
created: 2026-11-28T04:45:36Z
type: feature
priority: 3
assignee: coopwilliams
---
# Execute separable invertible zero-noise neural factors with exact interval messages

A deterministic neural activation can transmit uncertainty even though its conditional output has no noise. For the separable two-slope maps qualified in cff-u06e, a Gaussian-shaped approximation would lose the change in density across the bend. Implement the limited family for independent piecewise-constant interval potentials on each coordinate.

## Design

New IR field, new factor or rule family, and runtime numerical or status invariant change paths. Define compile-time interval capacity and behavior for insufficient capacity before introducing the kernel; do not truncate occupied intervals. Attribute node energy and separator entropy once. No arbitrary-network delta map, parameter learning or normal-density clamp is included.

## Acceptance Criteria

A serialized template declares positive left/right slopes, input and output knots, equal dimension, parameter blobs, and an explicit exact interval message domain. Forward messages divide by the active Jacobian, backward likelihoods compose with the map, and the local energy uses entropy in the input chart. Reject rank loss, unsupported couplings and unresolved numerical intervals explicitly. Preserve structural zero mass, distinguish improper from impossible evidence, and compare scalar/batch, f32/f64, 1D/32D, flooding/exact, and bundle reconstruction to D13a. Legacy positive-noise neural templates retain their current semantics.


## Notes

**2026-11-28T04:47:00Z**

Dependency tree (prerequisite --> dependent): cff-u06e [D13a derivation and oracle] --> cenc-luez [production support]. Ready after cff-u06e closes on a reviewed base. Positive-noise neural inference remains independent. Proposed consumer contract lives in D13a; the finite interval representation has to preserve complete supported cells, reject capacity overflow before committing updates, and name its support/energy behavior at every run status.

**2026-11-28T06:18:24Z**

D13a implementation requires a fourth domain: current VariableSpec chooses exactly one of cardinality, Gaussian dimension or scalar EF, and State/Arena allocate only those layouts. Split so the bounded numerical primitive is independently reviewable before generated schedules, state and codecs depend on it. Prerequisite --> dependent: [x] cff-u06e --> [ ] cei-6lc5 (complete finite-capacity messages) --> [ ] cei-k93f (declared domain, compiled graph, energy and bundle) --> [ ] cenc-luez. Existing stochastic neural inference uses Gaussian edges and can never reinterpret a missing or zero process covariance as an interval-domain model.

**2026-11-28T14:13:40Z**

D13a now executes exact separable two-slope activations on interval-only forests through public APIs. Complete support and signed node/separator BFE are checked on scalar and 32D cases in f32/f64. Focused compatibility checks: 183 passed, 7 deselected; 38 final serialization checks passed. The first canonical run identified two missing diagram glyphs; corrected in 94df674. Its fresh short suite passed 2,858 tests in 421.89 s. No mixed-domain, cyclic, point-evidence or learning extension is claimed.
