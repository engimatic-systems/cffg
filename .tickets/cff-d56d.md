---
id: cff-d56d
status: in_progress
deps: [cff-j4mu, cff-ri8m]
links: [cff-fxmj, cff-5edo, cff-wbhu, cff-3kdk]
created: 2026-07-10T18:05:00Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ahps
tags: [adversarial-reviewer, performance, scheduling]
---
# Schedule-aware inference: eliminate quadratic flooding on tree and chain graphs

ADVERSARIAL REVIEWER FINDING (2026-07-10). Engine.step is synchronous whole-graph flooding and Engine.run dispatches it until messages traverse the diameter. A tree that admits exact two-pass BP therefore costs O(E*diameter), O(T^2) on chains, instead of O(E). Direct M2 Max CPU/x64 parity benchmark against Dynamax 1.0.2: ordinary K=8 HMMs are 315-399x slower warm; hostile identity-dynamics HMM with only the final state observed needs exactly 2T+1 sweeps and grows from 623x slower at T=32 to 4,206x at T=256 (83.24ms vs 0.0198ms), with identical posteriors. LGSSM/RTS is 74x slower at T=16 and 137x at T=64 before CFFG readout. cff-wbhu's lax.scan will remove Python dispatch but still executes the O(T^2) work. Deep planning, large plates, and streaming cannot be called scalable until schedules exploit topology.

## Acceptance Criteria

1. Compiler emits a schedule abstraction, not a hard-coded special-case: exact inward/outward directed passes on acyclic BP subgraphs; iterative residual or colored schedules for loops/composites; synchronous flooding remains an option. 2. Tree HMM and LGSSM inference updates each directed message O(1) times and stays O(E) as T grows; exact marginals and FE remain bit/tolerance compatible. 3. On the recorded CPU benchmark, T=256 long-range HMM is within 20x Dynamax including general-engine overhead and shows linear scaling; LGSSM gap is <=20x through T=256. 4. Mixed BP/VMP/GFE/hybrid graphs document schedule boundaries and convergence semantics; no stale message crosses a phase unnoticed. 5. Schedules serialize or are deterministically reconstructible from the template. 6. Benchmarks separate compile, cold JIT, warm kernel, convergence/status, and readout.
