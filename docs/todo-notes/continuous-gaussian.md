# Continuous / Gaussian support — remaining threads

**Landed (2026-07-08, cff-4e5z):** the engineering half — domain families in
the IR, information-form Gaussian arena, linear-Gaussian BP with exact FE
(Kalman parity + continuous tree identity). See ARCHITECTURE §2b and
LIMITATIONS S1/S1b. What remains is ticketed under epic **cff-ahps**:

## Next up: hybrid first (cff-pksr)

Discrete regimes with Gaussian emissions (switching KF): conditional-Gaussian
factor, moment-matched mixture collapse, FE term becomes a documented *bound*
(first sanctioned approximation in the ledger). GFE composite stays discrete
(regimes); Gaussians only carry perception — delivers continuous sensors for
the glucose agent (cff-t2g3) while sidestepping the research half. Gaussian
VMP/conjugate learning (cff-7c5o) is independent and can land either side.

## Research half (cff-dfpi — genuinely open)

Koudahl 2021: EFE in linear-Gaussian systems yields no purposeful exploration;
the continuous GFE story is unresolved in the literature. Do not promise a
"curious Kalman agent". Working bet: the epistemic term lives on a discrete
regime variable inside a hybrid model; alternatives to experiment with are
nonlinear observation maps and state-dependent noise. Decision + experiment
is the deliverable, not shipped rules.

The `distrax`/`tfp.substrates.jax` adoption question was resolved with the
exponential-family abstraction (cff-8s6a, 2026-07-09): not adopted —
`jax.scipy.special` covers the math. See ARCHITECTURE §Dependency policy.
