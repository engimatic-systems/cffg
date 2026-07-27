---
id: cff-2gev
status: open
deps: [cff-hp3y, cff-4oq8, cff-x0sv, cff-cxuy]
links: [cff-x0sv]
created: 2026-07-10T03:34:54Z
type: feature
priority: 3
assignee: coopwilliams
parent: cff-mwjo
---
# GFE-TM novelty weight: anneal with accumulated count mass

Corrected post-cff-hp3y Hallway rerun (L=8, H=10, 120 sweeps, seed 0,
25 episodes): with near-uniform Dirichlet counts the dynamics-novelty channel
dominates the goal channel — task success is 0.16 versus 0.76 for the learn-only
ablation, and visitation-weighted transition KL is also worse (2.650 versus
2.534). The earlier pre-fix 0.40/0.80 success comparison and claimed marginal
KL benefit are invalid and must not be cited. The GFE-TM composite also costs
~4x a plain GFE act() every step whether or not novelty remains. Add an
annealed novelty weight (decaying with accumulated count mass, or per-
(state,action) saturation) plus the option to drop back to the plain composite
once counts saturate. The controlled arm must recover goal performance without
claiming useful exploration unless it also improves model KL. Semantics stay
optimistic-bound (LIMITATIONS S1c); the weight is a Q-side constraint knob,
document its GFE interpretation.

## Acceptance Criteria

Annealed arm matches the learn-only ablation's task success within noise after
counts saturate; any claim of useful early exploration requires lower model KL
than that ablation at a preregistered window; per-step cost falls back to plain-
GFE level once annealed out; S1c updated.

## Notes

**2026-07-27T19:46:58Z**

PREREGISTERED PROTOCOL (agreed with Cooper 2026-07-27, before any run).

Fixed in advance. The reported claim is the primary test at the stated
window and nothing else. Other windows are description, never evidence.

MECHANISM UNDER TEST

A per-column novelty weight lambda in [0,1] interpolating the two existing
pseudo-entropy paths in rules.py:

  h_lambda = H[Bbar] - lambda * (H[Bbar] - h_dirichlet)

lambda=1 is current behaviour (_tm_dirichlet_stats), lambda=0 recovers the
plain composite (_tm_fixed_stats at the posterior mean). h is shaped [U,K],
so lambda is per (action, state) column:

  lambda[u,z] = 1 / (1 + n[u,z] / n_half)

where n[u,z] is accumulated observation mass in that column, excluding the
prior. Columns anneal independently: well-visited transitions lose the
bonus while unexplored ones keep it. n_half is the single tuning constant
and is fixed before the run.

GFE interpretation, to document in S1c: lambda scales the parameter-
information term I(z'; B | z,u) in the substituted objective. At lambda<1
the reported value is the GFE of a modified objective with a weighted
information channel, not the unmodified GFE. It stays in the optimistic-
bound family because lambda <= 1 shrinks an already-optimistic term, but it
is no longer a bound on the same objective and must not be reported as one.

METRIC CHANGE

Primary model-error metric is the reference-weighted KL: weight every arm's
per-column KL(true B(.|s,a) || E[learned B](.|s,a)) by the state-action
occupancy of the known-model policy, identical across arms.

Reason: the existing KL weights each arm by its OWN visitation, so each arm
is graded only on the columns it chose to visit. That structurally favours
the arm that beelines to the goal and penalises the arm that explores, which
makes "exploration pays off" untestable under it. The self-weighted KL stays
in the report as a secondary, for continuity with the S1g numbers.

MODEL SPEC CHANGE

Initial Dirichlet drops from 1.0 to 1/K per entry, so each column starts at
prior mass 1 instead of K. At the old prior the recorded 25-episode run
accumulated about 4 observations per column against a prior of 32 — the
prior outweighed the data 8 to 1 and the anneal would never engage. At 1/K
the data dominates after roughly 10 visits and saturation is reachable
within the episode budget. Consequence: the recorded 0.16/0.76 and
2.650/2.534 numbers are NOT comparable to this run and must not be presented
as a baseline for it.

DESIGN

Three arms: annealed, unannealed (lambda=1 throughout), learn-only
(no composite). 5 seeds, 200 episodes per arm, windows every 25 episodes.
Paired across seeds; noise band is one standard error of the paired
difference.

PRIMARY TEST, at episode 100

  Annealed reference-weighted KL strictly lower than learn-only, by more
  than one standard error of the paired difference.

This is the only test that can support a claim of useful early exploration.

SECONDARY TESTS

  S1. Task success: annealed within one standard error of learn-only at
      episode 200. This is the "recovers goal performance" criterion.
  S2. Cost: median per-step planning time of the annealed arm at episode
      200 within 10 percent of the learn-only arm, via the drop-back to the
      plain composite once max lambda falls below threshold.
  S3. The unannealed arm reproduces the qualitative failure — worse success
      than learn-only. If it does not, the premise has changed and the whole
      result needs re-reading before anything is claimed.

OUTCOMES, decided in advance

  Primary passes and S1 passes: annealing works. Claim useful early
  exploration, cite this window, update S1c.
  Primary fails and S1 passes: the honest result. The novelty channel buys
  no model-quality advantage even when annealed. Keep the anneal as a cost
  and safety knob, default it off, and state in S1c that the channel is not
  known to help on this class of problem.
  S1 fails: the anneal is mistuned or the mechanism is wrong. Diagnose
  before any re-run, and record the diagnosis here. Do not retune n_half
  and re-run silently.
  S2 fails: ship the correctness result, file the cost fallback separately.

No metric, window, threshold or arm definition changes after the first run
starts. If something must change, the run is void and the reason is recorded
here before the replacement run.

**2026-07-27T19:47:26Z**

n_half fixed at 10, before any run.

Set from the prior, not from results: n_half = 10 * prior column mass, and
the new spec puts prior column mass at 1. So the novelty bonus halves once a
column carries 10 times the prior's worth of evidence, and falls to about
0.09 at 100 observations. That matches the "data dominates after roughly 10
visits" reasoning already used to justify the 1/K prior, so the two choices
come from one argument rather than two.

Stated as a rule rather than a number so it stays correct if the prior
changes: n_half = 10 * sum(alpha_prior[u,:,z]).

Drop-back threshold for the plain composite: max lambda over all columns
below 0.05.

**2026-07-27T20:14:37Z**

PREREGISTRATION VOID (2026-07-27, before any run — no results exist).

The protocol recorded above, and its n_half amendment, are void. Neither
was ever executed. Voiding on reasoning alone, with nothing observed, is
the only clean point at which to do this.

WHY

Two findings, both about the environment rather than the mechanism.

1. Observation aliasing defeats the measurement. The corridor's 32 states
   produce only 10 distinct observation signatures, and two of those groups
   hold 12 mutually indistinguishable states each — three quarters of the
   state space sits inside a 12-way tie. Counts are accumulated from the
   smoothed pairwise belief (agent.py:161), so under that aliasing each
   step's increment smears across dozens of columns. The novelty channel is
   an optimistic bound that assumes z' is revealed (rules.py, S1c: "tight
   when observations pin the state down, loose otherwise"), and this
   environment is built so observations do not. The experiment would have
   measured the bound's looseness, not whether annealing works.

2. The model has no structure for curiosity to exploit. The flat 32-state
   rollout treats 160 (state, action) columns as independent when the true
   dynamics are about 5 rules. Nothing learned at one cell transfers to
   another, so directed exploration cannot beat undirected — it pays the
   full cost of leaving the goal and collects almost none of the benefit.

Finding 2 is a limitation of that model, NOT of the engine. The IR already
supports the structure needed: transition_mixture's selector role carries
one chain into another's transition, factors tie parameters by shared blob
name (build_rollout already ties Bs across every slice), and plates tie
parameters by name across copies. A factored corridor — an orientation
chain selecting into a cell chain, turn rules as one shared block — is
expressible today.

WHAT REPLACES IT

A factored corridor, where the parameter count is small enough to learn in
tens of episodes, sensor noise is a dial rather than a fixed confound, and
information gained at one cell transfers to others. Only then does directed
curiosity have something to beat undirected exploration at.

That experiment cannot be driven by RollingAgent as it stands: the agent is
hardcoded to the build_rollout node names (member_of["prior"] line 97,
"u1" line 142, "act1"/"obs1" line 150, "zm1" line 152, "tm1" line 160) and
to a single [U, K', K] count array. It needs the declared-learnables API.

RESEQUENCED: this ticket now depends on cff-x0sv and is no longer the first
piece of work. It resolves mid-chain instead of up front.

The mechanism design survives unchanged and is still the thing to build:
per-column lambda interpolating h between H[Bbar] and h_dirichlet, annealed
on accumulated count mass, with drop-back to the plain composite. Only the
environment it gets tested in has changed. The reference-weighted KL
argument also survives — self-weighted KL grades each arm on the columns it
chose to visit, which favours the arm that does not explore.

The recorded 0.16/0.76 and 2.650/2.534 numbers stay invalid as evidence
about the mechanism, now for a second and stronger reason.

**2026-07-27T20:19:22Z**

Environment for the replacement experiment is cff-cxuy (factored rollout builder), now a blocker. That ticket also makes sensor observability a declared parameter, which is what lets the aliasing be swept as a dial rather than sitting in the background as a confound.
