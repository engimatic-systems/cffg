---
id: codg-0sy5
status: in_progress
deps: []
links: []
created: 2026-11-28T06:21:03Z
type: bug
priority: 2
assignee: coopwilliams
tags: [solver, numerics]
---
# Reject nonfinite diagnostics at a certified mixed-model fixed point

## Background in Plain English

The mixed solver can find a point that maps to itself yet have an undefined derivative there. It reports both the residual merit and its gradient norm to the caller. A scalar map sqrt(abs(x)) at zero currently returns root_converged with zero residual, zero merit and NaN gradient. A consumer reading the status can treat that incomplete numerical result as certified.

## Methodological Suggestions

Use the Runtime numerical or status invariant row. Inspect the final host decision used by single and grouped runs, and the adaptive proposal check. The merit is one half the squared norm of the scaled residual. Its gradient norm measures local stationarity. Timing optimization, new factors, serialization and checkpoint formats are outside this fix.

## Outcomes / Acceptance Criteria

1. If the terminal diagnostic returns a nonfinite merit or gradient norm, the call reports nonfinite evaluation and public divergence. Apply this to the initial candidate, trust iteration and shadow candidate. Retain the last point and the observed evidence.
2. Finite root certificates retain their merit, gradient, status and evaluation counts. A failed derivative during a discarded proposal keeps its existing fallback semantics.
3. Exercise a finite forward map with an undefined derivative in float32 and float64. Cover ordinary and compiled-controller routing, a 32-coordinate case, and the public status. Existing support, improper-input, root and budget checks continue to pass.

## Future work

cff-g7rc owns the separate cold-start performance investigation.

## Definitions

A fixed point is a state that one undamped solver update maps to itself. The residual is the difference between those two states. The merit summarizes the squared, scaled residual. Its gradient describes how the merit changes nearby. The trust-region controller chooses a bounded step and also checks an auxiliary candidate. An undefined derivative can produce the numeric marker NaN even when the forward calculation returns finite values.

## Notes

**2026-11-28T06:24:45Z**

On the original code a real differentiated scalar map returned root_converged with residual 0, merit 0 and gradient norm NaN. All three terminal checks now classify a nonfinite diagnostic as nonfinite_evaluation. Both public controller modes return DIVERGED. Six new regressions passed. The outer formalism, failure and strategy files passed 63 checks, including their opt-in cases, with the older adaptive chart-capture test replaced in a separate focused invocation by its corrected #210 version (one pass on these runtime bytes). No model data, inference tolerance or finite certificate changed. The required repository gate is pending.
