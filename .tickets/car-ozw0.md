---
id: car-ozw0
status: closed
deps: []
links: [car-8rgl]
created: 2026-11-24T05:32:46Z
type: task
priority: 3
assignee: coopwilliams
parent: car-zieb
tags: [axiom, provenance]
---
# Establish the provenance and permitted use of direct AXIOM reference outputs

## Background in Plain English — WHY

The paper gives ten-seed averages. The public agent has its own academic/nonprofit license. Our independent analytic fixtures can check individual formulas, but they cannot establish which beliefs the authors' agent produced at each frame or reconstruct the original ten runs.

## Outcomes / Acceptance Criteria — WHAT

1. Record the available route to a direct comparison. For executable access, document who may run the pinned release, for what purpose and what outputs may be shared. For author-provided data, record its source and permitted uses. An explicit unavailable disposition is a valid outcome.
2. For each permitted reference output, identify the source commit, configuration, input and seed identities, reward clock, dtype, model/checkpoint version and file digest. Mark any missing information.
3. For a matched comparison, use a consented or independently supplied input tape and report intermediate error, action agreement and failures. Classify a fresh reference run and a historical run separately.
4. State which historical questions remain unanswerable. Associate every paired claim with verified common inputs and seed provenance.

## Methodological Suggestions — HOW

The study owner can inventory already authorized outputs first. Any contact with an external party or new execution of the research agent requires its own authorization. Keep shareable metadata apart from materials with narrower terms.

## Future work

car-8rgl supplies independent reference fixtures; car-zieb owns the prospective CFFG campaign. Both can proceed with the rights-cleared evidence already available. A paired native comparison becomes possible if this ticket establishes an appropriate reference.

## Definitions

A native reference is an output of the authors' implementation. A historical reference is a record from the runs used in the paper. A digest identifies the exact bytes of a file.


## Notes

**2026-11-24T05:33:37Z**

Direct native parity and historical seed-level comparisons remain unverified as of the car-8rgl reference freeze. Start with the source and rights inventory when the study owner requests either comparison.

**2026-11-28T04:27:04Z**

Disposition at published main a774a54 on 2026-11-27: direct author output unavailable for a permitted matched comparison.

1. The frozen inventory identifies the authors' release b99f62599e73294639f1944827bf427909f6c698 and its academic/nonprofit license (Git blob bff389091d718f454c5bec3d5da4eb4451ffbf24). No documented authorization for a named operator, purpose and shareable output, or separately supplied author trace with terms, was found in the project evidence. Source inspection and independent fixtures are the recorded route. The separately MIT-licensed Gameworld at 85493c477004ecfd3299b0b053884f23d59a5ec5 supplies simulator observations.
2. The historical numerical reference is arXiv:2505.24784v1, Table 1, 60 mean/SD pairs over ten seeds and 10,000 native reward increments per run. The untuned, tuned and ablated configurations are named in the frozen protocol. Exact historical model and simulator seeds, CLI/configuration, dtype, initial checkpoint, action/reward/frame records and corresponding file digests are unavailable. The source tag of the comparison implementation is not a proven tag for each paper run. No fresh reference-agent run was supplied.
3. Verified the bytes under ~/agent-share/axiom-gameworld-reference/reference-v1: protocol.json SHA-256 ef5141f70e4ef6792a40e7ade7086109fc9e04b68afed19764294f9b53282caf, final-intermediates/checks.json 27755723842ab2910fc5b75584326975a055ce88902cef27b478a595e5033cd1, its four files and the 30 simulator NPZ hashes in tapes/manifest.json. The manifest names each game, condition, simulator and action seeds and dtype (uint8 RGB); 48 independent scalar checks and eight labeled synthetic frames supplement the MIT tapes. The simulator records the reward following the applied action. These files and the CFFG model checkpoints have identifiable producers, clocks and digests. None contains author-agent inference outputs. In particular explode-native-replay.json verifies Gameworld observations under the CFFG pilot's actions.
4. There are zero eligible pairs of author and CFFG trajectories. Per-frame belief error, action agreement, original per-seed deviations and paired reward uncertainty are unmeasured. The publication's aggregates can be quoted as historical aggregates. No external contact, agent execution or game campaign took place in this audit. Revisit if an appropriately documented file or separate execution authorization becomes available.
