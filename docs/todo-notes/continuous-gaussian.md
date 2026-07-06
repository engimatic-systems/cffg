# Continuous / Gaussian support

**Ticket:** none yet (open one before starting) · **Status:** research-grade — see LIMITATIONS S1.

## Where the wall is

Everything today is categorical/Dirichlet. Continuous support has two very
different halves:

1. **Engineering half (tractable):** Gaussian BP/VMP on linear-Gaussian
   factors is textbook (Loeliger 2007 tables). IR change: `VariableSpec`
   grows a domain variant (`cardinality: int` → `domain: Discrete(k) |
   Gaussian(dim)`); messages become (mean, precision) pairs, so the arena
   needs per-family slot layouts (a second message tensor, not padding
   tricks). Rules: `gaussian` factor types + node energies; FE terms are
   closed-form Gaussians. Test tier: Kalman smoother parity, mirroring the
   HMM tier exactly.

2. **Research half (genuinely open):** the *epistemic* part. Koudahl 2021
   showed EFE in linear-Gaussian systems yields no purposeful exploration —
   the continuous GFE story is unresolved in the literature, not merely
   underived here. Do not promise a "curious Kalman agent"; expect to
   experiment (nonlinear observation maps? state-dependent noise? hybrid
   discrete-continuous composites where the epistemic term lives on a
   discrete regime variable?).

## Recommended path

Hybrid first: discrete regimes with Gaussian emissions (switching KF). The
GFE composite stays discrete (regimes), Gaussians only carry perception —
sidesteps the research half while delivering most practical value
(continuous sensors!), and it is exactly what the glucose agent would want
(continuous BG, discrete regime/meal states). `distrax`/`tfp.substrates.jax`
adoption question reopens here (ARCHITECTURE dependency policy says
reconsider at this milestone).
