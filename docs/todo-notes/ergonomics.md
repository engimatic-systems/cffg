# Ergonomics: the P-bead footgun and friends

**Ticket:** cff-cj5h · **Remaining:** ledger decomposition (cml-6ba0) and exact MAP ties (cml-xdw1)

## MI-term diagnostic

The FE ledger is per-node; expose the decomposition instead of only the sum:
`engine.fe_breakdown(state) -> {node: term}` plus, for GFE composites, the
split of −z̄ᵀρ into extrinsic (Āᵀlog c part) vs epistemic (−log q_x − h̃
part). "How curious is my agent right now" becomes a printable number; the
dashboard could even chart it. This is mostly refactoring `_make_bfe` to
return a dict before summing.

## MAP tie-breaking

`np.argmax` resolves ties to index 0 — which turned "indifferent" into
"frozen at the start position" for the ablation agent. Fix in
`RollingAgent.act` and anywhere else argmax picks actions: break exact ties
uniformly at random with the agent's own rng (deterministic given seed).
Keep a regression test: an agent with a perfectly symmetric posterior should
not always pick action 0.
