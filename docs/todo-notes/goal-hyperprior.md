# Goal hyperprior: learning what to want (μ1)

**Ticket:** cff-gjjk · **Priority:** low · **Status:** not started

## What's missing

Part II §7.2: place `p_s(c_k) = Dir(c_k|c_{k,s−1})` over the goal statistic
and learn *preferences* across trials — the agent consolidates the outcomes
of epistemic policies into its goals ("I have learned to prefer visiting the
cue"). The composite's message toward c is closed-form: μ1(c) ∝ Dir(c|Āz̄+1)
(Part II Fig. 5, message 1). We currently treat c as a fixed blob.

## How to build it

1. Mirror the learned-A pattern exactly: a P-bead detail
   `goal_uncertainty=dirichlet`, GfePlan gains `goal_learned`, params key
   `:c_counts` instead of `:c`; inside the messages use E[log c] =
   ψ(γ) − ψ(γ₀) where the c-terms appear (that is the only change to ρ).
2. The outgoing update is conjugate — counts_c += Āz̄ per planning slice per
   trial (no sampling needed, unlike μ3).
3. Trial loop: posterior → next-trial prior recursion for c, same as A.
4. Reproduce §7.2: fix A = Â (delta), initialize c ~ mild reward preference,
   10 trials; expected result — c₁ learns to prefer the *cue observation*,
   c₂ the reward arms: an extrinsically-consolidated epistemic habit.
   Assert the reinforced statistics pattern (their Fig. 10).

## Watch out

- c enters ρ as log c; with learned c the ψ(small counts) penalty gives goal
  *novelty* — the agent will be drawn to observations whose desirability is
  uncertain. Decide whether to keep that (it is what the math says) or use
  E[c] in the extrinsic term; document whichever with a unit test like
  `test_learned_composite_novelty_direction`.
- Learned c + learned A simultaneously is untested territory; land them
  separately.
