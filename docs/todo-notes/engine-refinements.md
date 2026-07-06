# Engine refinements: known rough edges, none urgent

**Ticket:** none yet — open one per item when picked up · **Status:** catalogued

- **GFE composite generality.** Current composite = one z edge + fixed/
  Dirichlet A + categorical goal. Part II's general result allows multiple
  state/parameter edges on each side (their θ, w, φ sets). Needed for e.g.
  observation models with two parents. Mechanical extension of
  `GfePlan`/`gfe_update` (the Lemmas are already multi-edge); grow it when a
  model demands it, with a two-parent unit test vs. brute force.
- **Damping/schedule autotuning.** `damp=0.5` is folklore that works; the
  delta trace already detects limit cycles (period-2 oscillation in
  `deltas`), so `run()` could auto-engage damping on detection instead of
  requiring the caller to know the folklore. Cheap and kind.
- **Newton solver hardening.** Fixed 25 iterations, 1e-9 ridge. Fine at
  K ≤ 16; for larger state spaces add a residual-based early exit
  (`lax.while_loop`) and a fallback halving step. Watch for saddle-point
  convergence (LIMITATIONS H6) — log the residual into the FE breakdown when
  the ergonomics diagnostic lands.
- **Sparse/structured parameters.** T-maze B's are 0/1-sparse; arenas store
  them dense. Fine at 8×8; the glucose factored state will want factored
  transitions (kron-structured `transition_mixture` variant) before dense
  K² bites. Pair with the resource estimator to know when.
- **Colony restacking.** `Colony.run` re-stacks states every call —
  correct, measurable overhead for many tiny ticks. Keep batches resident
  (invalidate on spawn/kill) when a workload actually shows the cost.
- **δ + BFE semantics.** The ledger currently reports plain BFE over
  δ-hardened messages; a pointmass form constraint formally changes the
  entropy accounting (H[δ]=0 is handled, but the constrained-BFE
  interpretation deserves a written derivation + test like the tree
  identity, before anyone leans on FE values from δ-hardened runs).
