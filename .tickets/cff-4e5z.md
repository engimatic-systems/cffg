---
id: cff-4e5z
status: in_progress
deps: []
links: []
created: 2026-07-08T16:49:01Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ahps
---
# Gaussian domain: IR variant + linear-Gaussian BP + Kalman parity

Engineering half of docs/todo-notes/continuous-gaussian.md. Add a Gaussian domain family end to end: (1) VariableSpec grows dim (Gaussian(dim)) alongside cardinality — serialization/edits/compose/validate updated, schema stays additive; (2) FactorType grows a family field; new types gaussian (prior: mean, cov) and gaussian_linear (out = A@in + b + noise(cov)) — covers Kalman transition and emission; equality becomes family-polymorphic; (3) compiler lays out a second arena — canonical/information-form messages (eta [n_gslots,d_max], Lambda [n_gslots,d_max,d_max]) with vacuous = zeros, groups keyed by dim signature; (4) Gaussian DATA ports are handled by CONDITIONING (observation substituted into the factor: a gaussian_linear with DATA-ported out edge compiles to an observed group whose clamp is a runtime input + missing mask) — NOT by huge-precision clamps, which diverge the differential-entropy ledger; (5) BP rules + node energies in closed form so bfe == -log p(y) exactly on trees (the continuous tree identity); (6) marginal readout (mean, cov). Derivations by Fable at top level, recorded in the rules module docstrings.

## Design

Messages in information form (eta, Lambda): products are additive, vacuous/improper messages are representable (Lambda=0), half-edges free. Loeliger 2007-style updates derived from first principles via Gaussian integration + Woodbury so no rule ever inverts an incoming Lambda (propagates vacuous exactly). Delta constraints, mean-field beads, and P-substitution on Gaussian targets are rejected at compile with pointers to follow-up tickets. Test: models/kalman.py LGSSM builder (equality nodes branch the state chain); parity vs numpy Kalman filter + RTS smoother at 1e-8 (x64); bfe vs innovation-form -log likelihood; existing discrete suite untouched.

## Acceptance Criteria

Kalman smoother parity (means, covs) at 1e-8; bfe == -log p(y) on the LGSSM tree; serialization round-trip with Gaussian variables; discrete tests all green; docs (ARCHITECTURE, LIMITATIONS S1, todo-note) updated in the landing commits.

