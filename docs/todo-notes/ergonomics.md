# Ergonomics: the P-bead footgun and friends

**Ticket:** cff-cj5h · **Priority:** medium · **Status:** not started

Context (from the ablation discussion with Cooper): omitting a P-bead is a
*silent-but-well-defined* failure — you get a coherent KL-control agent that
never explores, and nothing errors. Same UX class as forgetting
`requires_grad`. Humans will hit this; rigorous agents won't. Three
mitigations, in effort order:

## 1. Epistemic lint (small)

At compile time, warn when the model has goal factors on future/unclamped
observations but zero `P_SUBSTITUTION` beads: "this agent will not explore —
intended?" Warning, not error (KL-control is legitimate). Implementation:
a `lint(template) -> list[str]` in `cffg/core/` (keep the compiler quiet by
default; `compile_template(..., lint=True)` or a standalone call for GUIs —
`check_edits` + lint is the editor's live-validation pair). Also lint:
δ-beads on edges that are never hardened (harden_last=0 pipelines), and
mean-field beads that provably equal BP (single free edge).

## 2. MI-term diagnostic (small, high value)

The FE ledger is per-node; expose the decomposition instead of only the sum:
`engine.fe_breakdown(state) -> {node: term}` plus, for GFE composites, the
split of −z̄ᵀρ into extrinsic (Āᵀlog c part) vs epistemic (−log q_x − h̃
part). "How curious is my agent right now" becomes a printable number; the
dashboard could even chart it. This is mostly refactoring `_make_bfe` to
return a dict before summing.

## 3. MAP tie-breaking (tiny, do first)

`np.argmax` resolves ties to index 0 — which turned "indifferent" into
"frozen at the start position" for the ablation agent. Fix in
`RollingAgent.act` and anywhere else argmax picks actions: break exact ties
uniformly at random with the agent's own rng (deterministic given seed).
Keep a regression test: an agent with a perfectly symmetric posterior should
not always pick action 0.
