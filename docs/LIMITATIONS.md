# Known Limitations of the CFFG/LAIF Engine

Status: reviewed against the implemented library on 2026-09-22. "Hard" means
inherent to the mathematical framework (constrained Bethe free energy + message
passing + LAIF); finishing the roadmap does not remove it. "Soft" means a
current implementation, modeling, numerical, or scaling boundary that later
work could remove.

A *named approximation* is usable only with the narrower claim stated here. An
*unsupported feature* has no executable contract. A *known defect* is accepted
by some current code path but must not be treated as supported input or output.
Ticket closure by itself is not delivery evidence: tickets closed as "not
pursuing" describe research history, not production capability. The README
capability table is the entry map; this document qualifies its claims.

Failure attribution is diagnostic, not a repair or an accuracy certificate.
Ordinary and exact runs retain the first observed local failure; global checks
can report a reason without a known factor. Adaptive outer solves report
terminal local evidence when available; derivative-kernel failures and budget
exhaustion can still have unknown ownership. Rejected trial/shadow failures
are discarded after recovery. A trust-region collapse attributes its selected
trial only when every trial at that step was invalid, not whenever a rejected
trial happened to fail. This is attribution, not a full evaluation history. Agent
checkpoints omit these run-local records, so a restored cached unsuccessful
result can have `failure=None`; its `status` remains authoritative.

## Hard limitations

**H1. Bethe is an approximation on loopy graphs.** Stationary points of the BFE
equal fixed points of belief propagation, and that is exact only on trees. On
loopy graphs marginals are approximate, the iteration may not converge, may
oscillate, or may converge to a wrong fixed point — and there is no general
certificate telling you which happened. Damping and good schedules mitigate;
nothing eliminates. Every interesting agent model (loops through time, shared
parameters) is loopy.

**H2. The posterior family is bounded by the constraint vocabulary.** The
papers' mean-field and structured factorizations discard posterior correlations
across beads by construction. This runtime executes registered naive
mean-field combinations; arbitrary structured mean-field partitions and
general edge/node form constraints are unsupported, not implicit. Strongly
correlated or multimodal joint posteriors (explaining-away between factorized
variables, symmetric modes) cannot be represented, only projected onto the
chosen executable family. Richer constraints widen the mathematical family,
but each needs its own syntax, update, and ledger before the runtime can use it.

**H3. No structure learning inside the objective.** Continual learning updates
*parameters* (conjugate posterior → next prior). The graph topology, state-space
cardinalities, and factorization are fixed by the template. The framework has no
gradient that adds a latent variable, splits a state, or discovers a new factor.
(An outer loop scoring candidate structures by free energy is possible — see S6 —
but that is model *comparison* wrapped around the engine, not learning within it.)

**H4. Tabular representation → combinatorial state walls.** Discrete variables are
one-hot; joint states are Kronecker products. The T-maze is 4 positions × 2 reward
contexts = 8 states; every new orthogonal feature multiplies. A matrices are
|X|×|Z|, controlled transitions |Z|²×|U|. The tabular path has no representation learning or amortization: raw pixels,
audio, or text cannot be consumed without a
hand-built (or externally learned) discretization/featurization front-end. This is
tabular RL's scaling wall, inherited intact. Neural hybrid nodes (S5) can push it
back but change the free-energy semantics at the boundary.

The factored rollout builder reduces a transition from a flat
`U(K₁⋯Kₙ)²` table to the sum of its declared block-conditionals, but does not
make Cartesian products free. A declared coupling still creates a categorical
product edge and deterministic selector tables proportional to that product;
a joint sensor still stores `|X|×(K₁⋯Kₙ)`. Separate block rollover is an
explicit mean-field family and therefore cannot preserve posterior
correlations between blocks. Use a joint `BeliefBinding` when those
correlations are load-bearing, accepting its product-size boundary belief.

**H5. Planning is finite-horizon inference; policies are factorized.** LAIF's win
is linear-in-T policy inference, but T must be chosen, and a receding horizon is
myopic beyond it: payoffs more than T steps out exert zero pull. Further, the
control posterior factorizes per timestep (mean-field over u_t), so the agent
infers marginal action tendencies, not contingent plans ("if the cue says left,
go left" as a *policy tree*). Sophisticated-inference-style conditional planning
is outside this factorization; replanning every tick recovers much of it in
practice but is not the same object. The costly-cue behavioral tier makes this
boundary executable: the open-loop second-action marginal is not treated as a
conditional policy, while replanning after the cue chooses the indicated arm
and improves realized return over the BFE ablation.

**H6. Globalized GFE is still non-convex.** Part II §4.5 documents that the
naive fixed-point iteration can diverge, and its local Newton solve does not
coordinate a whole mixed graph. CFFG now globalizes that coupled outer residual
with a fixed-support trust-region method and reports root, nonroot stationary
merit, exhaustion, divergence, and infeasibility separately. A certified root
establishes first-order constrained stationarity on the active support under D7's
assumptions; it does not prove a local minimum, a global optimum, or correctness
of the Bethe approximation on a loopy graph. Initialization can therefore still
select a poor stationary point, as Part II Fig. 9's bimodal win histogram warns.

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

The implemented surface includes discrete BP/VMP/GFE, linear Gaussian and
exponential-family inference, hard constraints and discrete MAP, assumed-density
hybrid and recurrent models, positive-noise nonlinear/neural Gaussian factors,
declared between-trial learning, current-schema persistence, rolling agents,
ordinary Colony batching, graph rendering, and a partial resource estimator.
The sections below state the exactness, objective, persistence, input, and scale
boundaries for those capabilities.

**S1. Message-rule coverage (the big one).** We have: BP/VMP for
Cat/Dirichlet/equality/δ/transition-mixture, GFE messages for the discrete
goal-observation composite, and BP for the linear-Gaussian family (gaussian,
gaussian_linear, Gaussian equality — cff-4e5z: Kalman-smoother parity, exact
continuous tree identity BFE = −log p(y)). Scalar exponential-family BP covers
Gamma–Poisson and Beta–Bernoulli conditioned-observation graphs. Between
converged runs, declared
NIW, MNIW, component-MNIW, and Pólya–Gamma weight learning covers `gaussian`,
`gaussian_linear`, `gaussian_mixture_linear`, and `recurrence_link`.
Endpoint-aware `moment_matching(factor, role, MomentProjection.GAUSSIAN)` is an
executable declaration for registered Gaussian factors, including the
positive-noise nonlinear and neural rules. It selects first and second moments
at one named factor endpoint; it is not a general EP constraint language. The
older factor-only `MOMENT_MATCHING` payload remains loadable so callers can
migrate it explicitly with a supplied endpoint and projection, but compilation
and diagram generation reject it rather than guessing. Supported declarations
are rendered and compiled; legacy declarations are neither.
Everything outside these named paths requires a separately selected derivation
or registration ticket. The following bullets mix delivered boundaries with
explicit gaps; present-tense executable claims, not ticket closure, distinguish
them. Epic cff-ahps records the finite integrated baseline and does not promise
that every gap below will be implemented:
- *Additional conjugate adapters* — Gamma precision and EF learning remain
  separate paths rather than `LearnableSpec` families. The rSLDS uses the
  generic hybrid pairings; other factor/family combinations need their own
  declared adapter and hostile-input contract.
- *Hybrid discrete–continuous (switching KF, mixture collapse)* — implemented
  by cff-pksr; the first place exact FE meets a necessary approximation
  (mixture entropy).
- *Nonlinear continuous factors* — `gaussian_nonlinear` supports positive-noise
  Gaussian conditionals with registered deterministic means, Taylor or fixed
  sigma-point approximations, and endpoint moment constraints. Its normalized
  frozen Gaussian Bethe energy has conditional stationarity only. The adaptive
  fixed point also enforces coefficient/cavity consistency; no stationarity of
  the total adaptive energy or bound on original nonlinear evidence is claimed.
  The full residual and finite energy change certify the returned point in a
  fixed reference chart, not its next update or a newly whitened chart. Difficult
  graphs may exhaust their budgets. Singular noise, invalid beliefs, and
  unrepresentable sigma-point spread fail explicitly; no ridge repairs them.
  Negative sites are permitted when combined beliefs remain proper. Full run
  control retains host authority; local updates, coordinate conversion, bounded
  Gaussian-only adaptive proposals and trust kernels support JIT. The proposal
  phase refreshes coefficients at every map and repeats the original host
  certificate before acceptance. Failed proposals consume the caller's budget
  before trust fallback; no global convergence or same-basin guarantee follows.
  Discrete/EF/functional plans do not use this proposal phase. Metrics are fixed
  within a solve but dynamic between solves; a changed layout/support/dtype or
  callback still needs a new kernel. Colony still controls adaptive runs per
  instance, not as one fully compiled batched control loop.
  Bounded startup can wait for numerically compatible partial inputs at
  neutral-at-entry, not-yet-admitted sites. Its sweeps consume `n_iters`; no
  approximation point, physical evaluation, finite energy or convergence is
  claimed while waiting. It cannot defer established sites or a loss of
  readiness after admission. Unresolved inputs exhaust truthfully with a
  pending factor diagnostic. Readiness has the stated D13 numerical-resolution
  band and conservatively refuses subnormal fallback entries; it is not an
  exact PSD/rank proof. Run-local admission does not change checkpoints.
  The reconstructed CSTR fixture in `tests/test_cstr_planning.py` distinguishes
  this missing-information failure from RK4 evaluations or sigma-point moments
  exceeding floating-point range. Its strict-update tests call sweeps without
  startup; fresh affine partial-goal chains now pass the dense Gaussian oracle.
  `examples/cstr_planning.py` supplies a separately declared supported candidate:
  past=0, horizon=2, nominal retained parameters, initial mean zero and
  P0=diag(.01,.001,1e-4,.1). Its normalized coordinates are concentration,
  temperature, additive concentration bias and coolant action. Current DATA
  is [-1/3,.2] (concentration .8 in raw model units, temperature 330 K).
  Zero concentration/action goals target .85 and 298.5 K, with variance
  diag(4e-4,1); current-observation variance is diag(1e-4,1e-4).
  Process covariance is Qp=diag(1e-4,4e-4,4e-6,.25). Both Taylor/sigma-point,
  f32/f64 and auto/flood have fresh and converged-warm checks at these defaults.
  The auto temperature-goal grid adds zero temperature targets (325 K), fixes
  other goal variances, and tests Qp multipliers {.01,1,100} crossed with
  temperature-goal variances {.01,1} in both methods/dtypes. These are enumerated
  cases, not an interval-wide guarantee. The 60-iteration budget uses
  tau=max(1e-6,64*eps) and actual finite FE-change/residual certificates in the
  solver's frozen chart. Independent affine conditioning and physical ODE
  comparisons are in `tests/test_cstr_envelope.py`; they do not make the
  nonlinear posterior exact.
  Physics evaluation clips initial concentration to [.1,1.6], temperature to
  [200,450] K and coolant to [275,322] K. Intermediate/final RK4 states are not
  clipped; the action belief is passed through unchanged. These are not hard
  [295,302] K actuator bounds. Larger priors/horizons, other observations or
  constants and closed-loop control performance require separate validation.
  The original failed CSTR invocation was not retained; wide-prior tests are
  mechanism reproductions, not incident replay or a proven noise threshold.
  The later retained native validation is separately reproduced by
  `tools/cstr_runtime_fixture.py` and `tools/benchmark_adaptive_runtime.py`:
  P4/H15, four RK4 substeps, true constants, P0=diag(1,1,1e-4,1),
  Q=Qp=diag(1e-5,1e-5,4e-6,.25), original emissions and tolerance 1e-6.
  The selected open-grid budget is 100; the source initial observation and
  labeled synthetic changes are not the missing historical closed-loop inputs.
  Linearization is the performance acceptance path. Sigma-point startup's
  `improper_tilt / tr1 / startup` failure remains separate and explicit.
  Cold compilation and compilation-warm, repeated-base plans must be reported
  separately, including changed-observation compilation behavior. The host
  controller's 7 ms uses a different numerical model and is not a parity oracle.
  Zero noise, nonlinear GFE, and continuous-to-discrete means remain excluded.
- *Gaussian/continuous GFE messages* — RESOLVED as a position, 2026-07-12
  (cff-dfpi, DERIVATIONS.md D5): none will be added, and that is a theorem,
  not a gap. Additive-control linear-Gaussian models provably have
  policy-independent epistemic terms (Koudahl 2021; re-derived and pinned in
  `test_additive_lgssm_epistemics_constant_across_policies`), so a
  Gaussian-only GFE message family has nothing to express. Continuous
  epistemics live on the discrete regime/selector variables of hybrid
  models instead — see S1h. The fixed-goal exterior hybrid composite is
  executable; arbitrary interior, nonlinear, and Gaussian-only forms are not.
The registry architecture exists precisely so each of these lands as an isolated
contribution.

**S1a. Executable registration covers three single-arena bindings.**
`FactorKernel` (`cffg.kernels`, ARCHITECTURE 2e) binds scalar EF and ordinary
discrete BP/VMP and Gaussian factors, including external implementations. The discrete
binding requires a log-potential accessor for support, joint beliefs and
assignment scoring; it is not a sparse/combinatorial factor interface.
Gaussian conditioning currently declares one DATA role per factor; Gaussian
mean-field and nonlinear rules are not delivered by this interface. Structural
equality, hard constraints, Dirichlet parameter-edge composites, GFE
composites and multi-arena factors remain separate plans. Registration alone
does not add a new conjugate-learning rule or inference method.
The registration API itself is internal-unstable pre-1.0; only the
serialized `(impl, version)` manifest and its loud-failure guarantee are
stable now.

The [extension matrix](ARCHITECTURE.md#bounded-factor-extension-paths) documents
these separate owners and their accepted modes. Ownership checks and the existing
external-factor oracles protect the extension boundary; they do not establish a
new rule's mathematical validity, performance or production readiness. Runtime
callbacks remain trusted local code, not serialized model or checkpoint payloads.

**S1b. Gaussian numerics.** Rules do not require incoming precisions to be
invertible: improper/vacuous messages propagate exactly. Potentially singular
Schur systems use a Moore-Penrose limit whose numerical-rank cutoff scales with
dtype and matrix dimension; accepted proper beliefs use unregularized Cholesky
solves, so there is no deliberate ridge bias or unit-dependent information
floor. At working-precision rank boundaries, directions below that relative
cutoff are treated as unconstrained. Continuous δ-clamps are rejected by design: a Gaussian
observation is *conditioned into* its likelihood node (a huge-precision clamp
would diverge the differential-entropy ledger). Nonsymmetric, non-PD, or
non-finite beliefs make node/edge energy terms `+inf`; marginal readouts return
NaNs, and sufficient-statistic readouts also carry `proper=False` — deliberate
loudness, not a plausible negative covariance.

Exact-tree certification checks a fresh message recomputation at each entry's
own scale. It is not an accuracy guarantee for posterior moments or evidence.
Gaussian equality cavities exclude the recipient before summing; the older
total-minus-recipient arithmetic caused the retained PCGym first-order chain
to fail certification in float32. Restoring only that old rule reproduces the
reported residual of `0.0006103515625`; the current rule certifies without
changing certificate tolerances.

The three-dimensional, past-4/horizon-15 retained fixture and its constant-goal
horizon-150 scale variant are checked against independent dense conditioning
in both precisions (`tests/test_first_order_precision.py`). The float32 fixture
contract permits a mean error below 1% of posterior standard deviation, a
covariance error below 1% in any posterior-scaled direction, and an evidence
error below 0.001 nat per observed scalar. These are bounded fixture checks,
not guarantees for arbitrary Gaussian graphs. Source covariance inversion
before runtime precision casting is included in the reference, so input
rounding is distinguished from inference error.

Float32 information-form and readout arithmetic still lose avoidable accuracy
on this controller (cff-d63w): the retained candidates have mean errors of
order `1e-4`, while float64 agrees to about `1e-12` on the original horizon.
Use `JAX_ENABLE_X64=1 uv run python experiment.py`, or configure
`jax.config.update("jax_enable_x64", True)` **before constructing arrays,
compiling or initializing** when tighter accuracy is required. Increasing the
iteration budget or enabling x64 after a float32 solve does not restore lost
precision. There is no blanket float32 ban or relaxed global certificate.
The fully unobserved variant also loses the zero-energy identity in float32
(about `-0.205` instead of zero); its moment checks pass, but its energy check
is a strict expected failure tracked by cff-d63w. The conditioned fixture's
evidence bound must not be extended to that variant.

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
−log Z identity claimed). Because its substituted next-state edge is
interior, its outgoing message is the cavity `log q_w - mu_down`, and its
ledger contribution is the direct node-local G_tm from D1 Proposition 1;
downstream message couplings are not node energy.
Do not transfer that caveat to the ordinary goal-observation composite: its
channel is the realizable observation itself. The costly-cue sensor-quality
tier checks that its posterior MI and sensing probability weaken together as
observation noise rises, reaching the random-distractor posture at chance.

**S1d. Hybrid mixture inference is assumed-density; its BFE is a lower
bound.** The `gaussian_mixture(_linear)` collapse rules (cff-pksr,
DERIVATIONS.md D3) moment-match every Gaussian-edge mixture message to one
Gaussian — the IMM/ADF family of approximations, and the reason exact
switching-KF inference (exponential in T) stays tractable. Honest
accounting: (1) marginals are approximate away from single-node models —
the switching-KF tier measures the gap against exact enumeration and pins
it with headroom rather than pretending parity; (2) selector messages and
node energies, including the public responsibility-weighted learning
statistics, are exact *for the beliefs the engine holds*; the one-step public
predictive exposes both the conditional mixture components and the same
moment-matched Gaussian used by the forward message, not an exact multi-step
switching posterior; (3) the approximated ledger term is the interior Gaussian
edge entropy, which
scores the collapsed Gaussian — the max-entropy density for its moments —
so each +H is an over-estimate and the reported BFE *lower-bounds* the
mixture-Bethe value at the same moments (bound direction pinned in
`tests/test_mixg.py`). Same optimistic-bound reporting stance as S1c.
(4) Observed mixture outputs (regime-dependent emissions with a DATA port)
are rejected at compile time in v1 — route observations through a plain
`gaussian_linear` emission node; (5) rank-deficient per-regime backward
messages have no well-defined normalized collapse and fall back to vacuous.
Full-rank A_s alone is not sufficient: each supported backward component must
inform every input direction. In the retained PCGym position/velocity model,
scalar position goals leave rank-one messages. The first action is exactly
unchanged between horizons 1 and 15 and when only the distant goal changes,
despite convergence and a changed terminal posterior/energy. This is a
planning limitation, not a remaining damping problem. The retained NumPy
moment/ledger oracle in `tests/hybrid_planning_fixture.py` and
`tests/test_hybrid_planning.py` make that distinction executable. The deferred
full-tilt EC work is not an available remedy; changing objectives to add
artificial full-state preferences would change the experiment.
Numerical moment collapse checks its computed covariances and output, not just
the input precisions. If an admitted component calculation overflows or loses
propriety, it emits nonfinite messages and the run fails with local attribution;
an internal collapse safe-inverse placeholder is not a valid approximation. This differs
from the vacuous fallback for components not yet normalizable during
initialization. Component producers retain Schur rank evidence before matrix
reconstruction and test backward rank before forming a Gram matrix; a rounded
null direction is not accepted as information. Proper-input covariance-form
branches check their computed intermediates without imposing a global
condition-number cutoff. Backward numerical-rank tests use equilibrated
coordinates and can still classify unresolved near-dependencies as vacuous
(`tests/test_mixture_rank.py`). The unchanged retained float64 model now reaches
the declared-rule fixed point in 16 undamped sweeps: the first finite energy
appears at sweep 15. Its agreement with the independent approximate-rule
oracle is not an exact-posterior or useful-planning claim. No automatic retry
or extra covariance floor is introduced.

Float32 convergence does not imply float64-level posterior accuracy. Selector
updates, node-energy weights and learning responsibilities now share centered
relative evidence for proper Gaussian inputs; the retained three-step test no
longer allows error from subtracting large common Gaussian constants. Improper
inputs whose combined joint is integrable retain the canonical joint formula
and its potential cancellation. No general condition-number or posterior-error
certificate is supplied. Extremely scaled but finite inputs can still overflow
the whitened evidence system; this is reported as invalid evidence, not repaired
by a floor or accepted as a finite posterior. These numerical changes do not
repair the partial-goal planning limitation above.

**S1e. Gaussian parameter learning is EM-rollover; flooding VMP is not
strictly monotone.** Continuous conjugate learning (cff-7c5o, DERIVATIONS.md
D4) has two honest edges. (1) NIW/MNIW updates run *between* trials from
node-belief sufficient statistics, rolling the expected natural parameters
(mean, E[W] = νΨ⁻¹) into the next trial — empirical-Bayes EM, the exact
mirror of the Dirichlet count loop. Messages built from E[W] are the VMP
geometric-mean messages, but the x-dependent trace term that would feed
*parameter* uncertainty into *state* inference (inflated effective noise) is
deferred; in-loop parameter uncertainty exists today only where it is exactly
tractable — `gaussian_prec`'s Gamma-precision edge. `accumulate` adds an
independent trial to the current posterior; `replace` recomputes one sequence
coordinate from the immutable initial prior. Neither operation turns point-
parameter inference into full joint VB. Generic registration also supports
component-MNIW/`gaussian_mixture_linear` and
PG-weights/`recurrence_link`. Exact-zero mixture responsibilities retain the
component prior; exact-zero structured recurrence exposure likewise retains
the source/stick prior. The recurrence posterior keeps source-regime, stick,
and augmented drive-coordinate axes. Gamma precision and external EF-kernel
adapters remain outside the declared-learning registry. (2) "VMP monotonically
decreases free energy" holds for sequential coordinate updates; this engine
floods (Jacobi), which converges to the same fixed point but can take
bounded non-monotone steps on multi-node models (measured ≈4e-3 nats at
damp 0.5, ≈0.2 undamped; single-node flooding is monotone to machine
precision — both pinned in `tests/test_gauss_vmp.py`). The mean-field ledger
itself is exact: BFE at the `gaussian_prec` fixed point equals −log evidence
by quadrature to 1e-6.

**S1f. Constraint factors and min-sum: exact but bounded envelope.** The
constraint library (cff-e8zk) computes all messages AND Bethe node energies
exactly by combinatorial DP (never materializing the K^n potential), so
sum-mode BFE = −log Z holds even through indicator factors. The envelope:
`alldifferent` is capped at n, K ≤ 10 at compile (the bitmask-permanent DP;
larger instances need Huang–Jebara approximate messages, arXiv:0908.1769 —
a follow-up, and it would make that node's ledger term approximate).
Min-sum (`semiring="max"`) is discrete-only in v1 (VMP/GFE/Gaussian/EF/
hybrid groups reject it) and reports `Engine.objective` — an exact
−log f(x̂) at the decoded assignment — while `bfe` deliberately raises: an
objective value is not a free energy. Max-product on loopy graphs keeps BP's
usual caveat (exact on trees — the pinned Viterbi identity; on loops, exact
for unique-MAP bipartite matching per Bayati–Shah–Sharma, otherwise a
heuristic that decimation, `cffg.csp.decimate`, drives to feasibility).

**S1g. Long receding-horizon plans: measured envelope (2026-07-09).** On a
60-state aliased-corridor planning stack, GFE planning beat a QMDP baseline
(0.79 vs 0.64 success, median 13 vs 19 steps) at horizons 5–20 with
per-env-step planning time growing roughly linearly (0.6 s at H=5 → 1.9 s
at H=20, sweep-dominated), but success *dropped* to 0.57 at H=25 (2.5 s per
step). The degradation is DIAGNOSED: pure under-convergence — messages
travel ~one slice per sweep, and a fixed 120-sweep budget leaves the H=25
control posterior at residual 5.5 (vs 4.9e-4 at 400 sweeps). Fixes are
mechanical, not research: depth-aware/tol-driven sweep budgets in the agent
loop plus warm-started replanning (cff-5edo). Related measured limits:
ordinary colony batching is dispatch-bound on CPU, 1.5× at 32 instances until
the sweep loop is a jitted scan (cff-wbhu). Exact-eligible colony members use
standalone propagation and certification per instance; this preserves exact
schedule semantics but does not yet provide vectorized exact execution.
Mixed-GFE colonies currently reuse
compiled trust kernels but run one controller per instance, so they have a
stronger host-control bottleneck until that same ticket moves the trust/status
carry into the scan. The GFE-TM curiosity composite
costs ~4× a plain act() in every slice and its novelty channel outcompetes
the goal channel while counts are near-uniform. After the cff-hp3y cavity and
ledger repair, the like-for-like 25-episode Hallway rerun scored 0.16 success
and 2.650 visitation-weighted transition KL versus 0.76 and 2.534 for the
learn-only/no-curiosity arm: the unannealed channel bought neither task return
nor faster early learning in that setting. Annealing or disabling it is
cff-2gev.
Until cff-5edo lands, require `RunStatus.CONVERGED` and inspect both FE and
message-delta traces before trusting a plan deeper than ~20 slices.
(Small-sample numbers: 14 episodes per horizon —
direction is reliable, third digits are not.)

**S1h. Continuous epistemics are discrete-channelled; the regime channel is
a declared optimistic bound.** The cff-dfpi position (DERIVATIONS.md D5,
2026-07-12): purposeful exploration in continuous/hybrid models comes from
P-substitution on *hybrid mixture* composites, never from Gaussian-only GFE
messages — additive-control linear-Gaussian epistemics are provably
policy-independent (Koudahl 2021, re-derived as D5 Prop. 5), so "make the
uncertainty discrete" (regimes, sensor choices, operating modes) is the
supported modelling recipe, matching Collis et al. 2024. Honest edges:
(1) the regime channel I(x;s) needs a Gaussian-mixture entropy and is scored
with the D3 moment-matched collapse — an upper bound on the entropy, so the
scored information gain over-states the exact channel (measured 1.86 nats
scored vs log 2 realizable on well-separated regimes) and the reported GFE
lower-bounds the exact substituted objective, the same optimism family as
S1c/S1d; the exact channel's H[q(s)] ≤ log K cap and a two-sided sandwich
are computed as diagnostics (bound directions pinned by quadrature in
`tests/test_hybrid_epistemics.py`). (2) The per-regime state channel
I(x;z|s) and the goal cross-entropy are exact. (3) Nonlinear observation
maps and heteroscedastic noise remain out of closed form by nature (H8);
regime mixtures are their tractable discretization. (4) v1 scope solves the
substituted fixed point over the regime simplex only — continuous edges keep
their D3 BP/collapse messages (full coupled stationarity would let the
epistemic term push per-regime state beliefs toward wishful separation;
excluded until derived and tested). The engine supports only an exterior
substituted output facing a fixed Gaussian goal. Interior predictive cavities,
learned Gaussian goals, and Prop. 6 parameter novelty remain follow-up work.

**S1i. Recurrent logistic transitions use a structured categorical belief and
a declared JJ bound.** A
`recurrence_link` represents `p(z_t | z_{t-1}, x_t)` with `K - 1`
source-specific stick-breaking logits for each previous regime. Its
Pólya–Gamma auxiliary is replaced by the coordinate-optimal mean-field
factor, equivalently the Jaakkola–Jordan lower bound on expected log
transition probability. The reported recurrence VFE therefore upper-bounds
the exact-transition variational VFE at the same Gaussian belief; dense
one-dimensional quadrature measures this gap in
`tests/test_recurrence_link.py`. D9 preserves the source/destination pair and
uses it for categorical and Gaussian messages plus the pair-region entropy.
On a compiler-classified acyclic ordinary categorical component, this is exact
sum-product for the bounded transition tables. On a categorical loop, the
same time-local rules are Bethe pseudomarginals and do not define one exact
path posterior. The remaining factorization separates the categorical path
from the Gaussian drive and PG auxiliaries, so a strongly coupled or
multimodal joint posterior can remain inaccurate even when the coordinate
residual converges. D11 evaluates a pair-conditioned Gaussian candidate with
explicit moment consistency and rejects it as a sufficient repair: separate
source and drive boundaries lose association inferred by external switching
factors even when the local JJ bound is exact. D12 derives hybrid regime/state
separators that repair that association loss in a NumPy research experiment,
not in the production engine. Its production chain was later closed as "not
pursuing"; the closed cff-ffsg, cff-f3jz, and cff-ulbz tickets did not land their
acceptance criteria on this branch. D12 keeps one conditional Gaussian per
regime at a boundary; histories sharing that regime can remain multimodal. Its
temporal reflected-mixture example has
exact JJ factors and zero EC residual yet opposite exact learning biases that
the approximation returns as zero. This is not an exact source-learning
guarantee. Destination-selected affine dynamics, declared hybrid ownership,
and stationary ordinary adapters define its accepted scope; cutting an
externally inferred regime/state association into two singleton ports is
unsupported. The candidate's JJ energy
upper-bounds the literal-factor EC energy at the same feasible region beliefs,
but neither has a general bound against true negative log evidence. It does
not establish source recovery, Gaussian trajectory exactness, or a
horizon-independent convergence budget. Its signed Gaussian messages would also
require compatible boundary consumers (cff-f3jz and cff-ulbz). The NumPy D12
experiment solves held auxiliaries and requires proper forward initialization;
it is derivation evidence, not an importable inference capability or current
production promise. The executable recurrence path remains D9.

The current v1 requires equal regime cardinalities
`K >= 2`, a proper Gaussian drive, finite fixed
`weights[K,K-1,d]`/`offsets[K,K-1]`, and sum-product semantics. Between-trial
PG learning accumulates the same structured pair-derived at-risk and
signed-success masses as the Gaussian drive message into independent Gaussian
posteriors for `[weight; offset]`. It rolls only posterior means into the next
inference state. Thus weight uncertainty does not enter recurrence messages or
the next PG coordinate; this is empirical-Bayes variational EM, not full joint
VB over weights and auxiliaries. The pair is derived from the current state
rather than independently serialized. Learned recurrence checkpoints identify
the D9 structured-pair statistic, factor/source ownership, dtype, shapes, and
exact-value digests.
Pre-manifest D8 and early unidentified D9 posterior arrays share the same keys
and shapes, so both are conservatively rejected instead of silently resumed.
Alpha CFFGZ loading accepts only the current schema. Older model-only bundles
are rejected, and missing learning history is never inferred from point values.
A compatibility and migration promise is deferred until the format stabilizes.
Flooding and mixed scheduling solve the same coordinates,
but neither guarantees monotone progress on every loopy hybrid graph.

All five declared-learning families supply versioned persistent-state
descriptors. The generic checkpoint inventory describes immutable priors and
current posteriors, but flat `Learnables` checkpoints restore only posteriors;
they retain the caller's destination priors. Schema-2 CFFGZ bundles store both
complete states and resume a later trial at a
committed learning boundary. Fit exports retain that coordinate's original
priors and current posteriors; they do not resume an outer fitting loop at its
interrupted iteration. They do not store inference messages, solver
progress, executing agent or hierarchy state, PRNG streams, or optimizer state.
Exact mid-run continuation and supported agent replay require their checkpoint
APIs; a portable learning bundle is not a runtime snapshot. The four unchanged
nonrecurrence families
have deterministic legacy posterior migration; unidentified recurrence
posteriors have none.
Scoping or merging a template with compact plates requires `expand_plates`
first; direct compact composition is rejected to prevent declaration loss
(`cff-b2fz`).

The recurrence learning readout carries the Gaussian covariance computed
directly from precision; it does not reconstruct covariance as
`second - mean @ mean.T`. After a scale-relative symmetry and PSD check it uses
an exactly symmetrized covariance for the PG statistics. Material asymmetry,
negative eigenvalues, non-finite moments, and a mean outer product that
overflows the dtype remain loud failures. Factor-local diagnostics report the
failed predicate and dtype-scaled tolerance, but they do not repair or relax an
invalid belief.

The end-to-end `cffg.models.rslds` helpers preserve these same approximation
boundaries. `RSLDSLearning.update` accumulates initial-regime, per-regime
dynamics, and recurrence evidence across distinct trials; calling it again on
one sequence would count that sequence again. `fit_rslds` is the separate
one-sequence cadence: each outer coordinate starts from immutable priors,
atomically replaces the declared initial-regime, component-MNIW dynamics, PG
recurrence, and tied-emission posteriors, and writes all point estimates into
a fresh state. The compatibility facade keeps emission fixed; applications
that learn it across distinct trials use the same four declarations through
`Learnables`. Neither cadence learns the continuous initial-state mean or
covariance, which remain initializer outputs. Neither cadence carries parameter
uncertainty into the next hidden-state solve, so both remain empirical-Bayes
variational EM rather than joint VB.

The PCA/transition-pair initializer fixes one observation-derived coordinate
system and regime order, but those choices are not scientifically identifiable:
another affine latent representation or label permutation can describe the
same observed process. Recovery numbers therefore require post-fit affine and
label alignment and do not establish uniquely recovered physical parameters.
The authoritative outer residual is a fixed-point diagnostic over the
normalized potentials and point parameters installed into the next solve, not
a derived scalar variational objective. The historical posterior/reporting
snapshot and an observation-derived transition-function residual remain
diagnostic only. A finite outer budget stays `budget_exhausted`, even when each
inner engine run converged and the returned point parameters are useful.

The learned NASCAR comparison reports held-out prediction, aligned recovery,
and bounded long-rollout periodicity against same-`K` and closest-budget SLDS
controls. On the single preregistered `T=10,000` seed, the rSLDS passed only
the held-out prediction claim against same-K SLDS (`+0.313296` versus a
`+0.005` margin). Its aligned latent RMSE and regime accuracy were worse, so
neither recovery advantage is claimed. No claim was registered against the
closest-budget K=5 SLDS, which had the best held-out likelihood and regime
accuracy. The rSLDS alone produced a bounded 10,000-step rollout, but that
observation was not preregistered as an advantage.

All fits exhausted their registered two-coordinate outer budget despite
converged inner and final inference. Under the former independent categorical
coordinate, the compact pilot's fixed `0.9` damping attempt after the third
replacement entered a real finite message cycle despite proper recurrence and
Gaussian beliefs and finite FE. The D9 structured recurrence path converges on
the first registered damping attempt for that pinned fixture. The fitter still
restarts any finite exhausted solve from the same parameter-bound state with a
bounded convex step-expansion schedule; a synthetic status-bearing regression
preserves that mechanism independently of the superseded fixture. Exhausting
every registered retry returns `budget_exhausted`; no monotone-progress
guarantee is claimed for arbitrary loopy hybrid graphs. This correction does
not remove the recurrence JJ/point-posterior bound, point-parameter
approximation, single-seed boundary, or sensitivity to the initializer and
outer budget. The historical learned command took 2,063 seconds and 5.16 GB
peak RSS.

The retry loop is deterministic-only. A graph with a functional `mu3` update
is rejected before fitting because this API does not accept an explicit key or
promise reuse of one fixed particle set across attempts. Fit coordination is
still rSLDS-specific: status-gated retries and snapshot comparison are
plausibly reusable, but no second production fitter yet proves the same
contract, so the repository exposes no general fitting framework.

cff-r6bl removed the quadratic degree/port rescans from cold validation and
grounding, but it did not create a variable-shape or rSLDS-private executor.
The graph remains statically shaped: changing horizon, regime count, latent or
observation dimension, dtype, or schedule selects a distinct JAX executable,
while same-shape observations and parameters remain runtime inputs. The
measured `30.81×` `T=10,000` setup speedup covers one rSLDS instance with
`K=4`, latent `d=2`, observed `dy=10`, auto/mixed scheduling, x64, and the CPU
backend on an Apple M2 Max. Other models, schedules, dtypes, devices, and
batched execution receive no performance claim from that result. The indexed
path still spends a `15.296`-second median in host CFFG lowering and `6.631`
seconds across state setup and JAX readiness at that horizon; broader schedule
fusion and device-loop work remain cff-yl0a and cff-wbhu. The 10,000-step
timing and direct-memory cases stay in the external experiment, not the default
test suite.

**S1j. Schedules: measured envelope (cff-d56d, revised 2026-07-27 after
review).** Two-pass schedules make tree and chain inference O(E) with
exact marginals and FE; loopy pure BP gets colored Gauss–Seidel sweeps;
models with iterative structures get block-phased mixed scheduling. Only δ
form constraints and models with nothing schedulable still flood (plus any
run with `damp>0` on an otherwise-exact model). Live matched-shape
reference (same machine, same timing protocol, M2 Max CPU/x64,
`uv sync --group bench` for the reference library; the benchmark asserts
posterior parity against the reference smoother, so both engines provably
solve the same model): hostile K=2 chains run status-bearing at 13-23x the
reference smoother through T=256 (kernel 9-13x; flooding was
1,400-6,500x) — the reference denominator is 14-40 µs, so a few µs of
run-to-run jitter swings that ratio several x; ordinary K=8 chains at
7-8x; LGSSM d=4/m=2 at 15-18x (kernel 13-17x). That straddles the
ticket's 20x envelope with no margin; per the operator's 2026-07-27 call
the remaining performance gap is deferred (cff-yl0a is the lever), and
correctness/scheduling work proceeds independently. The earlier recorded
minimal-output reference (0.0171-0.0198 ms at T=256) projected only the
requested smoother output inside JIT (and supplied precomputed HMM
likelihoods), allowing XLA to eliminate the rest of the posterior work; it
is ~1.5-2x faster and remains a useful strict kernel ceiling, but it is a
different output contract rather than machine variance. Honest residuals:
the exact certificate is elementwise,
`max(|new-old| / (1+|old|)) <= 64 eps`, so neither graph depth nor an
unrelated large-magnitude block can widen it; `gaussian_marginals` readout
is a Python-per-edge loop and dominates end-to-end LGSSM time at scale
(cff-3kdk); slice-fused composite kernels (cff-yl0a) remain the lever if
the envelope must tighten below ~10x.

**S2. Importance-sampling noise in parameter messages.** The message to A is a
non-conjugate log-message evaluated by importance sampling; variance grows with
column dimensionality. CFFG fixes particles within an inference attempt, so its
trace is deterministic conditional on the key; independently refreshed policy
score epochs still vary and therefore support burn-in averaging. Low ESS is
reported as budget exhaustion rather than silently accepted. Better estimators
(Laplace approximation of the message, quadrature, reparameterized control
variates) are incremental research that would directly improve learning
stability. Colony batching does not yet accept `mu3` models because it has no
per-instance explicit-key interface; `Colony.spawn` rejects them rather than
running the analytic base moments as though the functional update had fired.
`RollingAgent` supports one functional evaluator lifecycle with explicit keys,
fixed epochs, loud status, posterior rollover, and exact checkpoint replay.
Its default `learn_a` workflow remains an analytic Dirichlet-moment
approximation; only a planning template whose serialized constraints compile
to `cffg.mu3_is.v1` is evidence that `mu3(A)` ran. The agent does not average
multiple refreshed policy-score epochs automatically: callers may explicitly
refresh, while `Engine.score_mu3_epochs` remains the averaging API. Arbitrary
stochastic factor families and batched per-agent keys remain future work.

**S2a. Fixed mu3 particle validation has a known hole.** The low-level
importance rule can treat a NaN inside a caller-supplied fixed particle as
absent support and report success with a NaN returned moment. Public
`Engine.run` separately rejects non-finite initial state leaves, so no false
public convergence path has been demonstrated, but direct rule consumers must
prevalidate particles as finite, nonnegative probability tables. Ticket
cff-6n9c owns the direct and jitted rule-level repair. This is a defect, not a
supported sparse-support convention.

The agent's fixed-memory trial loop likewise uses separate immutable planning
and filtering templates. It does not mutate the paper's literal graph when an
observation changes from a planning constraint to data. A two-state, two-action,
fixed-A oracle proves the two-window transformation for two consecutive steps
with one joint categorical boundary: literal before/after snapshots,
independent action and filter enumeration, exact-zero rollover, float32, and
the derived past-evidence ledger offset are executable checks. This does not
generalize the equivalence to mean-field, continuous, hybrid, hierarchical,
learned-parameter, or arbitrary loopy boundaries; those retain only the named
two-window implementation claim until they receive their own oracle.

**S3. Deep temporal / hierarchical models.** The papers say extensions are
"straightforward — write the CFFG," which is true of the *notation*; the message
rules for inter-level couplings (Cat parameters generated by a higher level,
temporal thickness à la de Vries & Friston 2017) still need deriving and testing.
Engineering plus modest derivation.

**S4. Contingent planning.** H5's policy factorization can be partially lifted:
structured (non-mean-field) constraints over control sequences, or explicit
branching on future observations, trade compute for policy expressiveness.
Receding-horizon replanning already recovers most practical value cheaply.

**S5. Neural Gaussian means and recognition.** Dense neural means now enter
positive-noise Gaussian conditionals through Taylor linearization or fixed
sigma points (DERIVATIONS D13). Their labeled frozen Gaussian Bethe terms
claim conditional stationarity only. Rebuilding coefficients as cavities
change adds a fixed-point consistency condition; the resulting adaptive root
need not make the total reported energy stationary, and that energy has no
established nonlinear evidence bound. Exact zero noise remains cff-u06e;
arbitrary nonlinear P-substitution and Gaussian-only GFE are unsupported.

Neural training differentiates the complete adaptive root and approximate
ledger under a local isolated differentiable-root assumption (D14). Root and
original-adjoint residual checks detect failed or exhausted numerical solves;
they do not prove nonsingularity in every direction, global uniqueness or a
minimum. Stochastic mu3 and discrete point-mass decisions are rejected.
Adam continuation is deterministic and full-batch; it binds the original
training initialization and data. It does not provide weight uncertainty,
minibatch scheduling or conjugate posterior learning.

The learned initializer is an affine recognition output layer over declared
polynomial features of scalar Gaussian observations. It predicts Gaussian
messages only, requires proper proposed beliefs without clipping, and leaves
final certification to the normal solver. The bounded paired benchmark
checks matching roots and a positive work-reduction confidence interval;
there is no universal speedup or same-root theorem for multiple-root models.
This deterministic API has no particles or importance sampling.

**S6. Structure search is not a library capability.** Free energy can be used by
application code to compare explicitly constructed candidate models, but the
library exposes no topology-search algorithm, candidate generator, or validated
search workflow. Earlier roadmap discussion of propose-and-score loops is a
design possibility, not delivered structure learning.

**S7. Scale engineering.** Sparse A/B matrices, factored transition tensors,
GPU-resident arenas, incremental recompilation, smarter (residual) schedules —
ordinary optimization work, unbounded appetite.
Plate fragments may not themselves declare plates in phase 1; validation rejects
that unsupported grounding shape until recursive/index-table expansion in cff-6s31.
Scoping or merging a compact plate is also rejected; call `expand_plates` before
composition until cff-b2fz lands.

`cffg.estimate.estimate` is only a partial planning aid. Its named persistent
arena byte counts come from the initialized JAX state, and calibration is an
approximate extrapolation for the selected run. The XLA cost path still calls an
obsolete step signature and silently returns no FLOP or byte-traffic figures;
parameter detail, backend temporary storage, compilation, first execution, and
scheduled work units are not a trustworthy complete report. Ticket cff-j7mm
owns that repair. Do not use the current summary as a memory limit, process-RSS
measurement, or cross-model performance prediction.

**S8. Goal-specification ergonomics.** Goals are prior distributions over future
observations; translating task intent into calibrated `c` vectors (the softmax
utility dance) remains modeler burden. Goal hyperpriors now learn a Dirichlet
preference from explicitly rolled future predictive mass, but they do not
translate natural-language/task rewards into an initial prior; a small goal
DSL would still soften that boundary. Learned `c` and learned `A` coordinates
are currently compiled separately rather than as one jointly uncertain
composite.

**S9. Fixed GFE probability validation is incomplete.** The categorical GFE
packer currently accepts negative likelihood or goal entries when their checked
sum is one. Conversion to log space maps those entries to zero probability,
leaving invalid total mass; for example, `[-0.2, 1.2]` becomes mass `1.2`.
Callers must supply finite, nonnegative, normalized fixed probabilities.
This is an inherited input-validation defect, not a supported approximation.
P2 ticket cff-34gi owns rejecting it and checking the analogous transition-GFE
path. Positive Dirichlet concentrations have a different contract and need not
sum to one.

Graph rendering is deterministic documentation of the declared template and
supported constraint decorations. A successful SVG/PNG/scene render does not
compile the model, run inference, or certify that numerical inputs and results
fall inside the envelopes above.

## The one-paragraph honest summary

The implemented library is a transparent, compositional factor-graph and active-
inference engine spanning tabular discrete models, registered continuous and
exponential-family factors, bounded hard constraints, declared hybrid/nonlinear
approximations, between-trial learning, rolling agents, persistence, rendering,
and ordinary same-structure batching. Exact claims apply only to the registered
factor/constraint combinations named in `docs/ARCHITECTURE.md`; other delivered
rows carry the approximation and objective labels above. It does not learn graph
structure or raw-sensor representations, does not provide contingent policies
beyond finite-horizon replanning, and inherits the convergence and accuracy
caveats of loopy variational inference. Treat cff-34gi and cff-6n9c as input-
validation defects, the estimator and stochastic Colony surfaces as partial,
alpha bundles as current-schema artifacts, and administratively closed research
chains as non-capabilities. Its natural habitat remains experiments whose state
ontology and local factor semantics can be designed and audited explicitly.
