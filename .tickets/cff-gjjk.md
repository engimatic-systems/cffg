---
id: cff-gjjk
status: closed
deps: [cff-007q, cff-fr5c]
links: [cff-x0sv]
created: 2026-07-05T23:03:40Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-mwjo
---
# Goal hyperprior: mu1(c) = Dir(c|Az+1) (learned goals)

## Plain English

This ticket lets the agent learn its preferred outcomes across trials. A
trial's predicted observation mass updates a Dirichlet belief over the goal
vector, and that posterior becomes the preference prior used by the next
trial.

Part II section 7.2 goal-learning variant: expose the goal vector c as a
Dirichlet parameter edge on the observation-GFE composite. The composite
sends mu1(c) proportional to Dir(c | A z + 1), where A z is its predicted
observation mass, and the resulting q(c) becomes the next trial's prior.
This is the conjugate goal-parameter client of cff-fr5c; the message is
standard Dirichlet and is unrelated to the functional mu3(A) path.

## Design

Add the plain-English and formal goal-message derivation to DERIVATIONS,
including the constrained node/parameter-edge ledger terms and the exact
posterior-to-prior recursion boundary. A planning sweep may refine one trial's
q(c) but must not accumulate that message repeatedly as if sweeps were new
trials. Template serialization declares the parameter edge and prior; runtime
checkpointing carries the posterior hyperparameters.

Hostile-input plan: exact-zero predicted observation mass contributes zero
fractional count; a proper positive prior keeps q(c) proper; vacuous/improper
priors fail loudly; sparse float32 priors and 10-100x observation cardinality
remain finite without absolute guards.

## Acceptance Criteria

The mu1 Dirichlet message, q(c) posterior and complete constrained FE terms
match direct algebra on a tiny observation model, with a named executable
anchor in DERIVATIONS. One trial contributes A z exactly once regardless of
sweep count; posterior-to-prior recursion across trials and checkpoint/restore
are exact. Learned goals change the next trial's policy in the analytically
predicted direction. Serialization/hash round-trip, exact-zero mass,
improper-prior refusal, sparse float32 and uncomfortable observation scale
pass without fixed-goal regressions.

## Notes

**2026-07-28T18:13:07Z**

Source audit (question -> answer -> source -> confidence): What is mu1(c), which c moment enters rho, and when does rollover occur? For one unobserved GFE composite, n=Abar*zbar and log mu1(c)=n^T log c + const, so the message is proportional to Dir(c|n+1); multiplying by prior Dir(alpha0) gives q(c)=Dir(alpha0+n), with no extra +1. Multiple composites sharing c add n before one normalization. rho uses ell=E_q[log c]=digamma(alpha)-digamma(sum alpha), never log E[c]. The complete shared-c ledger is KL(Dir(alpha)||Dir(alpha0))-n^T ell; combined stably at alpha=alpha0+n it is logB(alpha0)-logB(alpha), while the remaining predictive entropy/A terms stay in the GFE composite. Hold alpha0 fixed through every sweep and replace alpha=alpha0+n each time; only a caller-declared completed trial may roll the converged posterior into the next prior. Sources: Part II Fig.5 printed p17; App C.1-C.3 pp32-33 (notes/pt2.txt 2046-53, 2082-99, 2125-39); Part II sec.7.2 printed p23 (notes lines 1533-36); LAIF commit 9453782 goal_observation.jl lines 182-89/191ff and goal_agent.jl 57-60. Confidence high. Nuance: the published sec.7.2 experiment rolls the final observed x message at trial end; the generic Fig.5 planning rule implemented here rolls predicted Abar*zbar only through an explicit API, never automatically per sweep.

**2026-07-28T18:32:29Z**

Implemented the Fig. 5 learned-goal coordinate end to end: serialized categorical statistic edges with proper Dirichlet priors; shared-slice mu1 count aggregation using E[log c]; stable complete FE ledger; explicit fresh-state trial rollover; RollingAgent staging/reset cadence and checkpoint restore; rollout/T-maze/gridworld APIs; and derivation, architecture, limitations, README, and hostile-input coverage. Deleted the completed todo-note. Validation at proposed PR head: focused affected suite 123 passed; final dashboard suite 1270 passed, 0 failed, 6 new; git diff --check, py_compile, and tk dep cycle passed. Remaining M3 child cff-2gev is blocked by cff-5ipi -> cff-b47g -> cff-nhxf, so this is the coherent stopping point for the stacked PR.
