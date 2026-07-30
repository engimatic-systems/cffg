---
id: cff-2gev
status: in_progress
deps: [cff-hp3y, cff-4oq8, cff-5ipi, cff-qr92, cff-weff]
links: []
created: 2026-07-10T03:34:54Z
type: feature
priority: 3
assignee: coopwilliams
parent: cff-mwjo
---
# GFE-TM novelty weight: anneal with accumulated count mass

## Plain English

This ticket gives transition curiosity a lifecycle. Unfamiliar transition
parameters receive a stronger novelty incentive, accumulated evidence turns
that incentive down, and a familiar model returns to the cheaper ordinary
planning path. The preregistered experiment measures both task performance
and whether the agent actually learns a better transition model.

Corrected post-cff-hp3y Hallway rerun (L=8, H=10, 120 sweeps, seed 0,
25 episodes): with near-uniform Dirichlet counts the dynamics-novelty channel
dominates the goal channel — task success is 0.16 versus 0.76 for the learn-only
ablation, and visitation-weighted transition KL is also worse (2.650 versus
2.534). The earlier pre-fix 0.40/0.80 success comparison and claimed marginal
KL benefit are invalid and must not be cited. That flat model also has no
parameter tying for directed exploration to exploit and its aliased sensor
smears learned statistics across states, so it is negative evidence about that
model, not a protocol for this ticket.

After cff-b47g supplies a factored two-chain rollout and cff-5ipi records a
replacement protocol before any run, add the protocol's declared-learnable
novelty weights and the option to drop to the plain composite when its
reachable/active parameter blocks saturate. The GFE-TM composite currently
costs about 4x a plain GFE act() whether novelty remains. The controlled arm
must recover goal performance without claiming useful exploration unless it
also improves the preregistered independent model-error metric. The weight is
a Q-side constraint knob; document its modified-GFE interpretation and keep
the optimistic-bound limitation in S1c.

## Acceptance Criteria

The implementation follows the live cff-5ipi protocol without changing its
weights, masks, arms, windows, thresholds, or outcome language after a run
starts. Lambda=1 reproduces the current learned GFE-TM rule and lambda=0
reproduces its fixed-parameter objective to tolerance. Weight/count routing is
by declared learnable and tied parameter block, never hard-coded node names.
Drop-back reaches plain-GFE cost when all protocol-declared active blocks
saturate; unreachable columns cannot prevent it. Exact-zero counts/support,
vacuous initialization, float32, uncomfortable parameter scale, finite FE and
truthful RunStatus are tested. Report the preregistered task, model-error and
cost outcomes without a useful-exploration claim unless its primary test
passes; update S1c either way.

## Invalid preregistration: cff-5ipi/v1

**Status:** INVALID after its first planner call; no cell completed and no
task, learning, policy, or timing result was accepted. This protocol was fixed
on 2026-07-29 after the factored builder landed and before implementing or
running the replacement experiment. PR #35 and its void protocol are not
inputs. Mechanism unit tests are not experimental results.

### What this experiment tests, in plain English

We are testing whether transition curiosity can be useful briefly and then
get out of the agent's way. The agent starts with a poor model of how movement
works. Its persistent mode reverses left and right, so it must gather
transition evidence while still trying to reach the end of the corridor.
Full-strength transition curiosity previously overwhelmed that goal without
learning a better model.

The treatment gives each learned transition an evidence odometer. Curiosity
starts at full strength, halves as observations accumulate, and disappears
from the planning path after every task-relevant transition has enough
evidence. Unknown transitions that are unreachable or irrelevant to the
goal cannot hold the expensive curiosity path open forever.

The factored state space makes this a more efficient and informative test.
Position and mode are represented separately instead of learning one large
flat transition table over their Cartesian product. Persistent mode dynamics
are known; only position dynamics conditioned on mode and action are learned.
Every rollout slice and episode contributes to the same named, tied parameter
block, concentrating evidence on the uncertainty the task actually needs
instead of diluting it across unnecessary joint-state parameters.

Four arms separate the explanations: a known-model reference, ordinary
learning without transition curiosity, curiosity that never fades, and the
annealed/drop-back treatment. A useful-exploration claim requires the
treatment to learn the true task-relevant transitions faster, recover normal
goal performance, and return to approximately ordinary-planner cost.
Independent transition-model error prevents aimless wandering from being
called useful exploration; task success prevents lower model error from
excusing an ineffective agent.

The sensor sweep is an honesty check on the transition-curiosity bound. The
primary condition reveals position and mode, so the priced transition
information is realizable. Progressively hiding mode shows whether the result
survives partial observability or depends on treating the next state as if it
were directly revealed. Balanced modes, paired seeds, fixed early/late
windows, and locked thresholds keep luck and post-hoc storytelling out of the
conclusion.

In one sentence: the experiment asks whether focused curiosity buys faster
model learning early, then fades soon enough to preserve task performance and
remove its extra computational cost.

### Model, environment, and declared uncertainty

The experiment uses `build_factored_corridor` with `length=8`, `modes=2`,
planning `horizon=8`, `dtype=float64`, `gfe=True`, action prior
`[1/3, 1/3, 1/3]`, position prior `delta(0)`, and mode prior `[1/2, 1/2]`.
Planning uses at most 400 sweeps, `tol=1e-10`, and `damp=0.5`; the one-slice
filter uses its existing 60-sweep, `tol=1e-12` contract. An episode ends on
first arrival at position 7 or after 14 actions. At every episode boundary the
harness reanchors the serialized `position` and `mode` beliefs to those two
initial priors without resetting learned parameters.

The environment uses the builder's exact deterministic position transition:
action indices `(0, 1, 2)` mean `(left, stay, right)`, and odd mode reverses
left/right. Mode is sampled at reset and remains fixed for that episode. The
known mode transition is identity and is never learned. The only empirical
learnable is the builder-declared, time-tied `position_transition`; initial
state, `mode_transition`, observation, preference, and action parameters stay
fixed. Its Dirichlet prior has `alpha0[k,c] = 1/8` for every next-position
outcome `k` and flattened column `c=(mode * 3 + action, old_position)`, hence
one prior pseudo-observation per column. The agent starts from the prior mean,
not from the true transition.

Every observation reports position exactly and a binary mode label with
accuracy `r` in the preregistered sweep `r in {1.0, 0.75, 0.5}`:

```text
A_r[(reported_position, reported_mode), (position, mode)]
  = 1[reported_position = position]
    * (r if reported_mode = mode else 1-r).
```

Thus `r=1` reveals the factored state, `r=.75` is a noisy-mode sensor, and
`r=.5` observes position but no mode information while keeping graph shape
fixed. The normalized preference is `c(y) proportional to
exp(3 * 1[reported_position=7])`, independent of the reported mode. The
primary inference is at `r=1`, where S1c's transition-information bound is
tight for the learned position outcome. The other two levels are a
preregistered limitation/intervention sweep, not extra chances to pass the
primary test.

### Novelty weight and tied count mass

For a declared Dirichlet transition learnable `ell`, output `k`, and
selector/input column `c`, the accumulated evidence and weight are

```text
n[ell,c]      = sum_k(alpha[ell,k,c] - alpha0[ell,k,c])
lambda[ell,c] = 2 ** (-n[ell,c] / n_half),       n_half = 4.
```

`n` is computed once from the posterior owned by the declared learnable.
Statistics contributed by all tied time instances therefore accumulate into
that one value; planning readers do not count as evidence. A zero accumulated
count gives exactly `lambda=1`. Four observations beyond the one-column prior
mass halve the parameter-novelty channel, so the schedule is stated in units
of actual evidence rather than raw concentration or rollout length.

Let `Hbar[ell,c]` be the entropy of the posterior-mean transition column and
`hhat[ell,c]` the GFE-TM Dirichlet pseudo-entropy. The annealed composite uses

```text
h_lambda[ell,c]
  = Hbar[ell,c] - lambda[ell,c] * (Hbar[ell,c] - hhat[ell,c]).
```

Only the parameter-novelty gap is weighted. `lambda=1` is the current learned
GFE-TM rule; `lambda=0` is the current fixed-parameter GFE-TM objective
(`learned=False`) at the same posterior mean. It is not the ordinary BP
transition factor. This is a Q-side modified-GFE constraint knob, not a
generative transition, temperature, probability floor, or change to the
node/edge FE ledger. The later cost switch deliberately removes the whole
transition P-substitution after its active novelty has saturated; ordinary
observation GFE remains enabled.

The environment terminates before an action from position 7. Its reachable
column mask is therefore

```text
R = {(mode, action, position):
     mode in {0,1}, action in {0,1,2}, position in {0,...,6}}.
```

The active mask is fixed by the occupancy of the true, goal-directed reference
policy: `u_forward(0)=2`, `u_forward(1)=0`, with the reset mode mass split
equally. It is

```text
C_active = {(mode, u_forward(mode), position):
            mode in {0,1}, position in {0,...,6}},
omega[c] = 1/14 for c in C_active.
```

The annealed arm switches at an episode boundary to a precompiled ordinary-GFE
planner exactly when every `position_transition` column in `C_active` has
`lambda <= 1/16`, equivalently `n >= 16`. Position-7 columns, non-reference
actions, absent modes in other models, and columns outside a declared
learnable cannot delay the switch. Posterior learning continues after the
switch. Generic tests must also cover multiple arbitrarily named transition
learnables and a structurally unreachable column; the empirical fixture
contains just the one declared block above.

### Arms, seeds, budget, and recorded traces

Run these four arms at every observability level:

1. **known/plain:** true position transition, ordinary GFE, no transition
   learning;
2. **learn-only/plain:** the uniform Dirichlet prior and conjugate learning,
   but no transition P-substitution;
3. **unannealed:** the same prior and learning with `lambda=1` GFE-TM
   throughout; and
4. **annealed/drop-back:** the fixed schedule and active-mask switch above.

Actions are sampled from the first-control marginal, not selected by NumPy's
lowest-index MAP tie break. Use master seeds `0,...,15`, 64 episodes per
seed/arm/observability cell, and a seeded random permutation containing
exactly 32 episodes of each reset mode. Derive separate deterministic PCG64
substreams for mode order, sensor samples, and action uniforms. Reset the
action substream per `(master seed, observability, episode)` and reuse the
same stream across arms, so early termination cannot misalign later episodes.
Any JAX-side randomness takes an explicit derived PRNG key. Record every
action marginal, action, true and observed state, posterior block belief,
per-column `n` and `lambda`, active-mask status, transition posterior mean,
node/total FE, terminal reason, sweep count, final delta, and `RunStatus`.

The fixed windows are early episodes 1--16 and late episodes 49--64
(one-indexed, inclusive). Plot/report all 64 episodes, but do not move these
windows. The effective statistical unit is the master seed; all comparisons
are paired by seed. Use 10,000 paired percentile-bootstrap resamples of the
16 seeds with PCG64 seed `20260729`, and report point estimates and 95%
intervals. There is no seed selection or outlier removal.

### Metrics and timing

Task success is entry into position 7 within 14 actions. Realized task return
is `1[success] - actions_taken/14`; report success rate, return, and successful
steps separately. The known/plain arm is a reference, not a substitute for
comparison with learn-only/plain.

Independent model error is evaluated after every episode at the posterior
mean, whether or not a column was visited:

```text
KL_active = sum_c omega[c] *
            KL(B_true[:,c] || B_mean[:,c])
          = -sum_c omega[c] * log(B_mean[true_successor(c),c]).
```

The primary learning statistic is the mean `KL_active` over episodes 1--16
for each seed. Unvisited columns retain their prior error; they are never
dropped as missing data. Also report late-window error, the saturation
episode, total occupancy inside/outside `C_active`, and the full per-episode
curve.

Timing excludes compilation, filtering, environment work, logging, and
metric calculation. At each exact-sensor annealed arm's final checkpoint,
load that same belief and transition posterior into (a) an ordinary-GFE
planner, (b) the annealed dispatch after its switch, and (c) a GFE-TM planner
forced to remain active. Warm each structure once, then alternate 25
synchronized `act()` calls. Use `time.perf_counter_ns`, block on JAX results,
retain every sample, and report per-seed medians plus median/MAD and the
hardware/JAX/XLA fingerprint. The cost ratio is computed from paired per-seed
medians; a seed that has not legitimately reached the switch is not timed as
if it had.

### Locked decisions and outcome language

At the primary `r=1` level, the label **useful annealed transition
exploration** is allowed only if all of these preregistered gates pass:

1. the bootstrap upper bound for
   `early_KL(annealed) / early_KL(learn-only)` is at most `0.90`;
2. for late success and late mean return, the lower bounds of
   `annealed - learn-only` are each at least `-0.05`;
3. annealed late success exceeds unannealed by at least 0.10 in the point
   estimate and the lower bound of that paired difference is above zero;
4. at least 15 of 16 annealed seeds legitimately switch by episode 48; and
5. the upper bound of the switched-annealed/plain timing ratio is at most
   `1.15`.

The `r=.75` and `r=.5` effects are reported with the same estimates and gates
but cannot rescue a failed `r=1` primary test. Exact passes followed by
degradation as mode observability falls are reported as **bound-sensitive
useful exploration**, retaining S1c's optimistic-bound warning.

If task and cost gates pass but the independent KL gate does not, the result
is **annealing recovers goal performance/cost without evidence of useful
exploration**. If unannealed is already task-equivalent, report **no observed
task need for annealing** rather than manufacturing a recovery claim. Failure
of task non-inferiority, switch coverage, or cost recovery is **annealing
does not recover the required behavior**. A point estimate that crosses a
threshold while its interval does not is **ambiguous at the preregistered
budget**, never a pass.

Every plan and filter call must return `RunStatus.CONVERGED` with finite
messages, delta, node energies, and total FE. Budget exhaustion, divergence,
or any non-finite diagnostic invalidates the complete experiment rather than
becoming an excluded seed. Fewer than all 16 complete paired seeds, a broken
balanced-mode schedule, a changed builder/environment, or a missing trace
also makes the run **invalid** and requires a clean rerun. Low active-column
occupancy caused by an arm is a measured failure to learn, not grounds for
exclusion.

Before the experiment, unit tests must pin the two lambda endpoint identities,
the half-life and saturation equations, declared-learnable/tied-count routing,
active-mask switch, continued learning after switch, and genuine plain-planner
dispatch. The hostile tier covers zero accumulated evidence and exact-zero
transition support, vacuous initialization, float32, at least 32 positions by
4 modes, non-finite FE/delta classification, and irrelevant/unreachable
columns. Passing these tests establishes mechanism identity only; it is not
evidence for any empirical label above.

## Invalid preregistration: cff-5ipi/v2

**Status:** INVALID during its pre-run boundary-belief feasibility check; no
environment rollout, complete cell, or substantive treatment result occurred.
`cff-5ipi/v2` inherited every model, environment, arm, randomization, trace,
metric, timing, bootstrap, gate, and outcome decision from `cff-5ipi/v1`
except for the two numerical-execution corrections listed here.

The `v1` CLI silently executed JAX in its default float32 mode even though the
serialized arrays requested float64. `v2` enables and asserts
`jax_enable_x64` before constructing an agent. This enforces the original
dtype choice; it does not change it.

Under genuine float64, the learned comparative arms satisfy the original
`400`-sweep, `damp=0.5` numerical contract at their initial state, but the
exact deterministic known/plain reference limit-cycles. That arm is a
descriptive reference and does not enter any primary paired gate. In `v2`
only known/plain uses at most 800 sweeps and `damp=0.85`; learn-only/plain,
unannealed, and annealed/drop-back retain at most 400 sweeps and `damp=0.5`.
Tolerance remains `1e-10` for every planner and the filter contract is
unchanged.

This exception was selected using a convergence-only feasibility probe: no
action was sampled or applied, no observation or task outcome was produced,
and no treatment comparison was inspected. With each of the 16 exact
`(position, mode)` boundary beliefs supplied directly, the known/plain
planner converged within the new budget (357--470 sweeps). The complete `v2`
run still aborts on the first non-converged or non-finite call, and any further
protocol change requires a visibly new version and a clean output directory.

## Invalid preregistration: cff-5ipi/v3

**Status:** INVALID after all 16 exact-sensor known/plain reference cells and
before any comparative cell completed. The first learn-only cell exhausted
its outer inference budget at episode 1, step 2; its inner Newton solve
converged, but messages and FE remained in a large limit cycle. No partial
reference result is an experimental outcome. `cff-5ipi/v3` inherited every
substantive choice and every comparative-arm numerical setting from
`cff-5ipi/v2`, with only these two changes:

1. it requires the cff-qr92 solver certificate at commit `443c51a`, whose
   `1024 * finfo(dtype).eps` recomputation floor is covered by a generic
   exact-support regression; and
2. known/plain keeps `damp=0.85` but receives at most 2,000 rather than 800
   sweeps. This descriptive reference remains outside every primary paired
   gate. Learn-only/plain, unannealed, and annealed/drop-back remain at the
   original `damp=0.5`, 400-sweep cap.

The fixed feasibility grid supplied each of the eight position point masses
and mode beliefs `[0,1]`, `[.25,.75]`, `[.5,.5]`, `[.75,.25]`, and `[1,0]`
directly to known/plain. After cff-qr92, all 40 cases converged in 357--1,211
sweeps with no sampled action, filter call, observation, environment step,
task result, or arm comparison. No further protocol change is allowed under
`cff-5ipi/v3`; its invalid call requires cff-weff and a new version in a clean
output directory before any complete rerun.

## Notes

**2026-07-28T04:21:51Z**

2026-07-27 clean-slate resequencing: PR #35 was closed unmerged and no replacement run occurred. The flat Hallway numbers remain negative evidence only. Current path is cff-x0sv -> cff-nhxf -> cff-b47g -> cff-5ipi -> cff-2gev. cff-5ipi must install one live protocol after the builder exists and before any result is observed; do not recover the void PR #35 protocol from git history.

**2026-07-30T01:02:37Z**

cff-5ipi/v1 invalidation and v2 lock (2026-07-29): The first full-run call stopped at known/plain, r=1, seed 0 with RunStatus.BUDGET_EXHAUSTED (400 sweeps; residual 1.5845) before any trace/cell or substantive outcome existed. Diagnosis found the CLI had not enabled the declared float64; true float64 made the comparative initial plans converge at the original 400/.5, while exact known/plain still limit-cycled. A convergence-only probe over all 16 exact boundary beliefs found known/plain converged at 800 sweeps/damp=.85 (357--470 sweeps). Source: local RunResult diagnostics from committed harness 77ec70f; confidence high. v2 is now locked to x64 enforcement and that reference-only solver exception; every substantive choice remains v1.

**2026-07-30T01:12:43Z**

cff-5ipi/v2 invalidation and v3 lock: The expanded pre-run boundary grid showed v2's exact-known reference could still exhaust its inner solver at 1.28e-13--1.56e-13 and its outer loop at noisy .25/.75 mode beliefs. This produced no action, observation, complete cell, or treatment result. Linked bug cff-qr92 fixed the general recomputation certificate at commit 443c51a. With known/plain alone at damp=.85 and a 2,000-sweep cap, all 40 fixed position/mode boundary beliefs converged in 357--1,211 sweeps. Source: local RunResult grid; confidence high. v3 locks those numerical changes; comparative arms and all substantive/statistical choices are unchanged.

**2026-07-30T02:49:34Z**

cff-5ipi/v3 invalidation (2026-07-29): The clean run at commit 19caadf completed all 16 r=1 known/plain reference cells, then aborted before completing the first comparative cell. Learn-only seed 0 failed at episode 1 step 2 after one transition update: outer RunStatus.BUDGET_EXHAUSTED after 400 sweeps, residual 1.6901, oscillating FE/messages; inner Newton remained CONVERGED. Invalid partial traces are preserved at /Users/ravd/agent-share/cffg-annealed-novelty-cff-5ipi-v3 and must not be analyzed as outcomes. An excluded-seed 999 lifecycle failed at episode 2 under damp .8 and .85, and independent frozen-state restarts from .9 through .99 failed up to 15,000 sweeps while inner solves converged. Source: locked harness RunResults and convergence-only diagnostics; confidence high. cff-weff now blocks a new protocol/clean rerun; do not weaken truthful convergence or reuse partial cells.
