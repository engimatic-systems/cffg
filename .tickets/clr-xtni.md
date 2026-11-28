---
id: clr-xtni
status: open
deps: []
links: []
created: 2026-11-28T13:52:20Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-5edo
---
# Stabilize the local GFE certificate for a concentrated categorical cavity

## Background in plain English

In the frozen 60-state Hallway, the current planner withholds the first action at horizon 25 in both precisions. The local GFE solver subtracts two independently centered logit vectors. Its 64- and 256-step replays stop at the same residual just above the published threshold. See the actual obs21 float64 and obs14 float32 cavities in /Users/ravd/agent-share/cffg-5edo-envelope. At horizon 20 the same failure can affect a shifted initialization even when the cold plan certifies.

## Design

Use the Runtime numerical or status invariant row in the architecture change-path index and the existing fixed-support simplex solver. An algebraically equivalent diagnostic writes the logits as normalized cavity plus correction, then centers the difference between correction and potential. This avoids subtracting large cavity terms to recover a small residual. Check whether the same chart is valid at the implicit derivative and shared hybrid/functional callers. Reuse the current support, Newton globalization, finite budget and 512*dtype_eps residual gate. Keep true outer root and free-energy checks.

## Definitions

The cavity is the incoming log message before applying the local objective factor. A correction is the additive logit change to that cavity, up to an irrelevant common constant. GFE means generalized free energy. The active face consists of labels with finite semantic support, even when a returned low-precision probability underflows.

## Future work

cff-5edo owns rollout budgeting, action-equivalent shifted initialization and the prospective continuation. Local qualification alone cannot discharge those outcomes.

## Acceptance Criteria

1. Replay both saved cavities at the original precision, threshold and 64-step cap. Verify stable certified residuals against an independent log-space stationary reference. Preserve literal support, with extreme finite logits represented separately from exact exclusion.
2. Exercise binary and 60-state cases, vacuous and empty support, both supported dtypes, current globalization and implicit derivatives. Keep categorical and hybrid local callers covered. Nonfinite, budget and support failures remain explicit.
3. Check the public Engine H25 witness and document any subsequent bottleneck as independent of the local fix. A larger outer cap, an unqualified action or a looser tolerance cannot replace the witness. Leave the long campaign to the parent ticket.
