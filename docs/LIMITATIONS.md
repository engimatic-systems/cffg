# Known Limitations of the CFFG/LAIF Engine

Status: assessed 2026-07-05, before implementation. "Hard" = inherent to the
mathematical framework (constrained Bethe free energy + message passing + LAIF);
finishing the roadmap does not remove these. "Soft" = removable with additional
derivation or engineering work; each is a future research/engineering ticket.

## Hard limitations

**H1. Bethe is an approximation on loopy graphs.** Stationary points of the BFE
equal fixed points of belief propagation, and that is exact only on trees. On
loopy graphs marginals are approximate, the iteration may not converge, may
oscillate, or may converge to a wrong fixed point — and there is no general
certificate telling you which happened. Damping and good schedules mitigate;
nothing eliminates. Every interesting agent model (loops through time, shared
parameters) is loopy.

**H2. The posterior family is bounded by the constraint vocabulary.** Mean-field
and structured factorizations discard posterior correlations across beads by
construction. Strongly correlated or multimodal joint posteriors (explaining-away
between factorized variables, symmetric modes) cannot be represented, only
projected onto the chosen family. Richer constraints widen the family but the
principle stands: you choose the approximation when you draw the beads.

**H3. No structure learning inside the objective.** Continual learning updates
*parameters* (conjugate posterior → next prior). The graph topology, state-space
cardinalities, and factorization are fixed by the template. The framework has no
gradient that adds a latent variable, splits a state, or discovers a new factor.
(An outer loop scoring candidate structures by free energy is possible — see S6 —
but that is model *comparison* wrapped around the engine, not learning within it.)

**H4. Tabular representation → combinatorial state walls.** Discrete variables are
one-hot; joint states are Kronecker products. The T-maze is 4 positions × 2 reward
contexts = 8 states; every new orthogonal feature multiplies. A matrices are
|X|×|Z|, controlled transitions |Z|²×|U|. There is no representation learning and
no amortization: raw pixels, audio, or text cannot be consumed without a
hand-built (or externally learned) discretization/featurization front-end. This is
tabular RL's scaling wall, inherited intact. Neural hybrid nodes (S5) can push it
back but change the free-energy semantics at the boundary.

**H5. Planning is finite-horizon inference; policies are factorized.** LAIF's win
is linear-in-T policy inference, but T must be chosen, and a receding horizon is
myopic beyond it: payoffs more than T steps out exert zero pull. Further, the
control posterior factorizes per timestep (mean-field over u_t), so the agent
infers marginal action tendencies, not contingent plans ("if the cue says left,
go left" as a *policy tree*). Sophisticated-inference-style conditional planning
is outside this factorization; replanning every tick recovers much of it in
practice but is not the same object.

**H6. No convergence or optimality guarantees for GFE.** The GFE is non-convex;
Part II §4.5 documents that the naive fixed-point message diverges in cases, and
the Newton-solved variant still finds only local stationary points — Part II
Fig. 9's bimodal win histogram is the framework itself telling you agents
sometimes lock into bad optima depending on initialization. Expect and monitor
this; do not debug it as an implementation defect.

**H7. Conjugate continual learning has no principled forgetting.** Dirichlet
counts accumulate forever: the posterior sharpens monotonically, so a
nonstationary environment (the maze's reward statistics change at trial 500) is
absorbed ever more slowly. Exponential count decay / power priors fix this in
practice but are ad hoc — the "true Bayesian" semantics ends where the
environment stops being stationary. Relatedly, under model misspecification the
posterior converges confidently to the best wrong answer; the framework cannot
tell you your model class is wrong (low evidence hints at it; nothing repairs it).

**H8. Factors must have tractable local expectations.** Every node needs its
average energy and entropy contributions computable (closed form, quadrature, or
low-variance sampling) for the FE ledger to be exact. Factors without that —
arbitrary black-box likelihoods, implicit simulators — can only enter through
approximating surrogates, which reintroduces exactly the shortcut-taking the
project forbids.

**H9. Latent-state identifiability.** Learned latent categories are only defined
up to permutation (and sometimes merging) — the engine may learn a perfectly
predictive A matrix whose state labels mean nothing to a human. Interpretability
of learned structure is not provided by the math; it comes from priors and
constraints the modeler adds.

## Soft limitations

**S1. Message-rule coverage (the big one).** We have: BP/VMP for
Cat/Dirichlet/equality/δ/transition-mixture, GFE messages for the discrete
goal-observation composite, and BP for the linear-Gaussian family (gaussian,
gaussian_linear, Gaussian equality — cff-4e5z: Kalman-smoother parity, exact
continuous tree identity BFE = −log p(y)). Everything else is a derivation
ticket under the roadmap epic cff-ahps:
- *Gaussian VMP + conjugate learning (NIW/Gamma)* — cff-7c5o.
- *Hybrid discrete–continuous (switching KF, mixture collapse)* — cff-pksr;
  the first place exact FE meets a necessary approximation (mixture entropy).
- *Nonlinear continuous factors* — cff-o7hs: EP/moment-matching or sigma-point
  messages with controlled FE accounting.
- *Gaussian/continuous GFE messages* — genuinely open research (cff-dfpi):
  Koudahl 2021 showed EFE in linear-Gaussian systems yields no purposeful
  exploration, so the continuous epistemic story is unresolved, not merely
  underived.
The registry architecture exists precisely so each of these lands as an isolated
contribution.

**S1b. Gaussian numerics.** Rules never invert incoming precisions (improper/
vacuous messages propagate exactly), but potentially singular solves are ridged
with 1e-10·I so transiently underdetermined systems stay finite mid-flooding;
at a proper fixed point the ridge perturbs results at ~1e-10 relative — visible
only under extreme covariance scaling (condition numbers ≳ 1e8). Continuous
δ-clamps are rejected by design: a Gaussian observation is *conditioned into*
its likelihood node (a huge-precision clamp would diverge the differential-
entropy ledger). Free energies on models with improper beliefs (e.g. unbonded
Gaussian glue ports) come out ±inf — deliberate loudness, not a bug.

**S1c. Dynamics curiosity is an optimistic bound.** The GFE-TM composite
(cff-fk2u, `epistemic_b=True`; derivation in docs/DERIVATIONS.md D1) gives
transition learning an epistemic channel: the parameter-novelty term is
*exactly* the expected KL of the next Dirichlet update. Two honest limits:
(1) z' is latent, so the channel prices information as if the next state
were revealed — by data processing it upper-bounds the realizable gain
about B, tightly only when sensors pin the state down; (2) the fixed point
is solved over the joint (state, action) simplex, O((K·U)³) per composite
per sweep — fine at planning scale, a structure-exploitation ticket at
colony scale. Models containing the composite report GFE, not BFE (no
−log Z identity claimed).

**S1d. Hybrid mixture inference is assumed-density; its BFE is a lower
bound.** The `gaussian_mixture(_linear)` collapse rules (cff-pksr,
DERIVATIONS.md D3) moment-match every Gaussian-edge mixture message to one
Gaussian — the IMM/ADF family of approximations, and the reason exact
switching-KF inference (exponential in T) stays tractable. Honest
accounting: (1) marginals are approximate away from single-node models —
the switching-KF tier measures the gap against exact enumeration and pins
it with headroom rather than pretending parity; (2) selector messages and
node energies are exact *for the beliefs the engine holds*; the one
approximated ledger term is the interior Gaussian edge entropy, which
scores the collapsed Gaussian — the max-entropy density for its moments —
so each +H is an over-estimate and the reported BFE *lower-bounds* the
mixture-Bethe value at the same moments (bound direction pinned in
`tests/test_mixg.py`). Same optimistic-bound reporting stance as S1c.
(3) Observed mixture outputs (regime-dependent emissions with a DATA port)
are rejected at compile time in v1 — route observations through a plain
`gaussian_linear` emission node; (4) rank-deficient per-regime backward
messages have no well-defined collapse and fall back to vacuous — full-rank
A_s per regime is the supported envelope.

**S1e. Gaussian parameter learning is EM-rollover; flooding VMP is not
strictly monotone.** Continuous conjugate learning (cff-7c5o, DERIVATIONS.md
D4) has two honest edges. (1) NIW/MNIW updates run *between* trials from
node-belief sufficient statistics, rolling the expected natural parameters
(mean, E[W] = νΨ⁻¹) into the next trial — empirical-Bayes EM, the exact
mirror of the Dirichlet count loop. Messages built from E[W] are the VMP
geometric-mean messages, but the x-dependent trace term that would feed
*parameter* uncertainty into *state* inference (inflated effective noise) is
deferred; in-loop parameter uncertainty exists today only where it is exactly
tractable — `gaussian_prec`'s Gamma-precision edge. (2) "VMP monotonically
decreases free energy" holds for sequential coordinate updates; this engine
floods (Jacobi), which converges to the same fixed point but can take
bounded non-monotone steps on multi-node models (measured ≈4e-3 nats at
damp 0.5, ≈0.2 undamped; single-node flooding is monotone to machine
precision — both pinned in `tests/test_gauss_vmp.py`). The mean-field ledger
itself is exact: BFE at the `gaussian_prec` fixed point equals −log evidence
by quadrature to 1e-6.

**S2. Importance-sampling noise in parameter messages.** The message to A is a
non-conjugate log-message evaluated by importance sampling; variance grows with
column dimensionality and makes the GFE trace stochastic (hence burn-in
averaging). Better estimators (Laplace approximation of the message, quadrature,
reparameterized control variates) are incremental research that would directly
improve learning stability.

**S3. Deep temporal / hierarchical models.** The papers say extensions are
"straightforward — write the CFFG," which is true of the *notation*; the message
rules for inter-level couplings (Cat parameters generated by a higher level,
temporal thickness à la de Vries & Friston 2017) still need deriving and testing.
Engineering plus modest derivation.

**S4. Contingent planning.** H5's policy factorization can be partially lifted:
structured (non-mean-field) constraints over control sequences, or explicit
branching on future observations, trade compute for policy expressiveness.
Receding-horizon replanning already recovers most practical value cheaply.

**S5. Neural hybrid nodes.** Deterministic NN factors with learned amortized
messages (extended-VMP style) are the known escape hatch from H4's tabular wall —
JAX makes the plumbing natural. Research-grade: the FE contribution of such nodes
is approximate, so they must be quarantined behind explicit "surrogate" bead
types to keep the exact-FE guarantee for the rest of the graph.

**S6. Structure search as an outer loop.** Free energy is a model-comparison
score, so propose-and-score topology search (add state, split factor, compare FE
on held-out rollouts) is buildable on top of the engine. Expensive but not novel
math.

**S7. Scale engineering.** Sparse A/B matrices, factored transition tensors,
GPU-resident arenas, incremental recompilation, smarter (residual) schedules —
ordinary optimization work, unbounded appetite.

**S8. Goal-specification ergonomics.** Goals are prior distributions over future
observations; translating task intent into calibrated `c` vectors (the softmax
utility dance) is modeler burden today. Goal hyperpriors (learned goals, Part II
§7.2) and a small goal-DSL would soften it.

## The one-paragraph honest summary

Completed, this is a *transparent, compositional, continually-learning
discrete world-model engine* with principled curiosity — exact free-energy
accounting, models you can read, edit, checkpoint, and glue. It will not do
representation learning, will not scale past modest discrete state spaces without
hand-factored structure, will not plan contingently beyond its horizon, and
inherits every classical caveat of loopy variational inference. Its natural
habitat: problems where the state ontology is designable (devices, protocols,
games, control loops, cognitive-model experiments), not problems where the
ontology must be learned from raw sensor streams.
