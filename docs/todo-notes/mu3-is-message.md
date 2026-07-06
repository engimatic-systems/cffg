# μ3 IS message: future-slice contributions to q(A)

**Ticket:** cff-iwgl · **Priority:** low · **Status:** not started

## What's missing

The last unimplemented piece of Part II's machinery. Our learned-A GFE
composite sends messages *from* q(A) into planning (via Ā = E[A] and the
closed-form E[A log A] pseudo-entropy in `rules._dirichlet_stats`), but sends
nothing *back*: the observation-matrix posterior is updated only by
perception slices (conjugate `counts[x] += q(z)` in the trial loop). Part II
Fig. 5 message 3 is the planning-side backward message
`log μ3(A) = z̄ᵀ ξ(A)`, `ξ(A) = Aᵀ(log c − log q_x) − h(A)` — non-conjugate
in A, hence their importance sampling with the incoming Dirichlet as proposal.

## Why it was deferred (and when it matters)

Fig. 9 reproduces without it (see `tests/test_tmaze_learning.py`) because
observed trials dominate q(A) in the T-maze. It matters when an agent must
update parameter beliefs from *pure planning* — e.g. counterfactual "if I
went there, my model of there would sharpen" reasoning across long horizons,
or offline what-if analysis with no incoming data.

## How to build it

1. `rules.py`: `gfe_mu3_logweights(key, counts, logc, zbar, n_samples)` —
   sample A_s ~ column-Dirichlet(counts), weight w_s ∝ exp(z̄ᵀξ(A_s));
   explicit `jax.random` key threading (AGENTS invariant).
2. Represent the outgoing q(A) as weighted moments (Ê[A], Ê[log A]) rather
   than a sample list; moment-match back to a Dirichlet for the
   posterior→prior recursion (log-moment matching per column).
3. Burn-in averaging of the GFE trace (Part II uses ~10 iterations) before
   any policy readout — the stochastic message makes FE fluctuate.
4. Acceptance: (a) with huge counts the IS message is a no-op (weights
   uniform); (b) on a two-slice model, planning-only A-updates move counts
   toward the informative column; (c) T-maze learning curve unchanged or
   improved; (d) fixed seed ⇒ bit-reproducible.

## Pointers

`rules.py` GFE section · `compiler.GfePlan.learned` · the LAIF reference's
sampling path is `mean_h(::MatrixDirichlet)` (see cff-fdcl notes).
