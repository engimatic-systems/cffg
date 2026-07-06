# M7: hierarchical couplings + shallow contingent planning

**Ticket:** cff-pyy9 (epic) · **Priority:** low · **Status:** staged in ARCHITECTURE.md

## Hierarchy — cleanly additive

The claim to cash in: templates-with-ports already compose *vertically*. A
higher level places priors on a lower level's parameters or goals. Concrete
first rungs, in order:

1. **Goal hyperprior** (see goal-hyperprior.md) is the depth-2 base case —
   do it first, it derisks the rest.
2. **Cat-parameters-from-a-parent**: a higher-level state selects the
   lower level's transition regime. Structurally this is just our
   `transition_mixture` with the selector edge owned by the parent level —
   the *node* exists; what's new is the schedule (parent ticks slower).
3. **Nested schedules**: run the lower level to coherence between single
   sweeps of the upper level (temporal thickness, de Vries & Friston 2017).
   Engine-wise: two Engines + a small driver that passes port messages up
   (posterior over regimes) and down (empirical priors); the buyer/seller
   pattern from Part II §7.3 is the two-timescale template to copy.
4. The Colony already handles "two models coupled through the host" — a
   hierarchy is that, with the host being the parent's port bindings.

## Contingent planning — shallow only

Policy trees ("if the cue says left, go left") can be expressed as a bigger
template: branch the window on a future observation via a mixture at the
branch point. Costs grow exponentially in branch depth; depth 2–3 is
practical. Do NOT attempt deep sophisticated inference through message
passing — that's tree search wearing a costume (LIMITATIONS H5/S4). The
receding-horizon replan in `RollingAgent` already recovers most contingency
in execution; build branch templates only when a test shows replanning
demonstrably insufficient (construct one: a task where committing at t=0
without contingency is provably worse).

## Acceptance sketch

Two-level T-maze variant (context drifts on a slow clock; upper level tracks
the drift regime, lower level plays trials): upper posterior tracks the true
regime; lower-level win rate under drift beats the flat agent's.
