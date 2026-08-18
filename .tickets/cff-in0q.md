---
id: cff-in0q
status: closed
deps: [cff-epdp, cff-iwgl]
links: []
created: 2026-08-13T04:26:16Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-uy48
tags: [paper-fidelity, agent, mu3]
---
# Run functional mu3 through the RollingAgent trial lifecycle

## Background in Plain English

The engine already evaluates the Part II functional `mu3(A)` message with
importance sampling, explicit JAX keys, fixed particles during one solve,
effective-sample-size diagnostics, and direct quadrature tests. The public
`RollingAgent` does not use that path. Its learned-observation-model workflow
plans with analytic moments from a Dirichlet posterior and learns only from
perceptual evidence.

That workflow is useful, but it is not the paper's complete parameter-message
cycle. This ticket makes functional `mu3(A)` a declared agent option and
carries its random and stochastic state through planning, learning, trial
boundaries, and checkpoint restore. It preserves the existing analytic path as
a separately named approximation.

## Outcomes / Acceptance Criteria

1. The prerequisite inventory `cff-epdp` identifies the exact Figure 5
   `mu3(A)` source, destination, functional density, required moments, and
   ledger contribution.
2. A planning template declares use of the registered `cffg.mu3_is.v1`
   evaluator through serialized constraint metadata. `RollingAgent` does not
   infer the evaluator from factor names or from `learn_a=True` alone.
3. `RollingAgent` accepts an explicit JAX random key when its planning arena
   contains a stochastic evaluator.
   3.1. A deterministic agent requires no JAX key.
   3.2. A stochastic agent without a key fails before its first planning run.
   3.3. The NumPy generator used for action selection remains a separate random
   stream.
4. Before each functional-message planning epoch, the agent writes the current
   proper observation-model posterior through `Learnables` and prepares the
   evaluator from that exact base posterior.
5. One prepared particle set remains fixed for an inference attempt.
   5.1. Repeated sweeps do not draw new particles.
   5.2. Replanning does not refresh particles unless the declared refresh
   policy says that a new scoring epoch begins.
   5.3. A changed base posterior invalidates the old prepared state before
   inference.
6. The agent records and exposes each evaluator's effective sample size, log
   normalizer, stochastic status, and next-key cursor with the planning
   `RunResult`.
7. An action is available only after the ordinary inference result and every
   active `mu3` evaluator report a successful converged result.
   7.1. Sampling-budget exhaustion remains distinguishable from inference-
   iteration exhaustion.
   7.2. Weight collapse or a non-finite estimator reports divergence and does
   not return an action.
8. After an executed action and observation, perceptual sufficient statistics
   update the declared A posterior exactly once. The next planning trial uses
   that posterior as the proper base distribution for the new functional
   message.
9. The agent checkpoint stores enough versioned state for bitwise continuation
   of a prepared planning epoch.
   9.1. It preserves the caller-visible next-key cursor.
   9.2. It preserves or exactly reconstructs the current particles, weights,
   normalizer, and stochastic status without consuming an extra key.
   9.3. It preserves the base-posterior identity needed to reject stale
   prepared state.
   9.4. A legacy checkpoint either restores the documented analytic path or
   fails with a specific migration error; it cannot silently resume a
   functional-message agent without stochastic state.
10. A compact end-to-end Part II trial-loop oracle executes perception,
    planning with `mu3`, action, observation-model learning, a trial boundary,
    and the next plan.
    10.1. The agent path agrees with direct `Engine.prepare_mu3`, `Engine.run`,
    `Learnables.accumulate`, and `Learnables.write` orchestration on all state,
    posterior, key, status, and diagnostic values.
    10.2. A binary-model quadrature oracle independently checks the functional
    message moments and parameter-ledger contribution in both plans.
    10.3. A qualitative learning curve alone cannot satisfy this criterion.
11. Hostile-input coverage includes literal zero likelihood or goal support,
    an improper or non-finite base posterior, low effective sample size,
    all-nonfinite weights, float32, and at least ten planning slices sharing one
    A parameter.
12. `README.md`, `docs/ARCHITECTURE.md`, `docs/DERIVATIONS.md`, and
    `docs/LIMITATIONS.md` describe the functional and analytic agent paths,
    key ownership, refresh boundary, checkpoint behavior, stochastic failure
    states, and remaining importance-sampling limits.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Use the **Learnable or checkpoint feature** and **Runtime numerical or
  status invariant** change paths. Walk current and legacy checkpoints,
  deterministic and stochastic plans, and direct Engine versus RollingAgent
  execution.
- Reuse `Engine.prepare_mu3` and `Engine.score_mu3_epochs`; do not duplicate the
  functional density or importance sampler in the agent module.
- Put random-state ownership in a small typed runtime value rather than adding
  unrelated arrays directly to `RollingAgent`.
- Keep the analytic learned-A route available under explicit terminology so
  existing consumers do not acquire stochastic behavior by surprise.

## Future work

`cff-iqqk` separately tests the literal changing-constraint model against the
two-window implementation. `cff-rqqc` owns per-instance stochastic state in
`Colony`, not the single-agent lifecycle in this ticket. Arbitrary stochastic
factor families and batched stochastic policy scoring remain outside this
ticket.

## Definitions

- **Base posterior:** The proper Dirichlet distribution that is multiplied by
  the functional `mu3(A)` message.
- **Effective sample size:** A diagnostic for how many importance samples
  meaningfully contribute after weighting.
- **Planning epoch:** One fixed-particle policy-scoring attempt.
- **Random-key cursor:** The next explicit JAX key value available after a
  sampling operation.
- **Trial boundary:** The point at which accumulated learning becomes the
  prior used by the next trial.
- **Weight collapse:** A sampling failure in which no finite normalized set of
  importance weights remains.

## Notes

**2026-08-18T20:47:02Z**

Implementation handoff, 2026-08-18. Applicable change paths: Learnable or checkpoint feature and Runtime numerical or status invariant. Authorities: serialized ConstraintSpec evaluator metadata, AgentSpec/Learnables for declared model state, Engine.prepare_mu3/_update_mu3 for stochastic evaluation, and RollingAgent for single-agent cadence. Required surfaces: validation and JSON/hash are applicable only if persisted agent schema changes; compiler/Arena evaluator discovery is applicable; current and legacy checkpoint decode are both applicable; direct Engine versus RollingAgent is applicable. Host/JAX execution shares Engine mu3 rules; flooding/trust-region status propagation is applicable while scheduled stages are structurally inapplicable for GFE; single instance is applicable and Colony/vmap is deferred to cff-rqqc. Hostile inputs are the ticket's exact-zero goal/likelihood, improper base, low ESS, all-nonfinite weights, float32, and ten shared slices.

**2026-08-18T21:13:35Z**

Research distillation, 2026-08-18. Question: where must RollingAgent own functional mu3 state, and what can be reused? Answer: serialized arena.mu3_groups is the sole stochastic-mode authority; plan-only forwarding avoids the learned_a=False filter guard; Engine State already owns particles, weights, base_alpha, log-normalizer, ESS, readiness and moments, so the agent needs only epoch/key/result/invalidation lifecycle and exact input/base checks. Source: src/cffg/{agent.py,engine.py,learn.py,models/rollout.py,compiler.py}; confidence: high, confirmed by direct two-plan oracle. Question: what is the smallest independent oracle? Answer: a binary one-step rollout compared agent-for-agent with direct Learnables.write/accumulate plus Engine.prepare_mu3/run across observation, reset and the second plan; Gauss-Legendre product-Beta quadrature separately checks Abar, hbar, expected phi, log Z and the one-time KL ledger slot in both plans. Source: notes/pt2.txt Fig. 5/App. C.6, tests/test_mu3_derivation.py, tests/test_agent_mu3.py; confidence: high.

**2026-08-18T21:39:00Z**

Implementation complete, 2026-08-18. RollingAgent now selects functional mode only from serialized evaluator metadata, owns a typed explicit-key/fixed-particle planning runtime, invalidates epochs on posterior or solver configuration changes, gates actions on ordinary and stochastic convergence, and checkpoints prepared epochs for bitwise continuation. Learned observation-model compatibility now follows semantic parameter role and shape across the perception/planning windows. Public wrappers expose functional mode where supported, and the user/design/derivation/limitation documents distinguish functional and analytic paths. Verification: focused lifecycle, corruption, ESS, exact-zero, float32, and direct-orchestration tests passed; opt-in slow quadrature and ten-slice float32 tests each passed; the final not-slow suite passed 1,381 tests with zero failures and 41 deselected in 967.23 seconds. The remaining bounded limitation is finite-sample importance-estimator quality, including low-ESS rejection; multi-instance Colony ownership remains cff-rqqc.
