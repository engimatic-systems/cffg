---
id: cff-fxmj
status: closed
deps: []
links: [cff-hp3y, cff-fk2u, cff-vamy, cff-cpxj, cff-5edo, cff-j4mu, cff-i45j, cff-wbhu, cff-d56d, cff-44we, cff-1o72, cff-o7hs, cff-ri8m, cff-l6i5, cff-4oq8, cff-gul8, cff-cj5h, cff-j7mm, cff-3hgi]
created: 2026-07-10T17:49:46Z
type: task
priority: 0
assignee: coopwilliams
tags: [adversarial-reviewer, audit]
---
# Adversarial project review: math, engine, performance, roadmap, workflow

Independent end-of-PR-chain assessment requested by Cooper on 2026-07-10. Review architecture/scalability/modularity, real performance against specialized Bayesian libraries, curiosity/GFE test adequacy and mathematical tractability, commercial roadmap, AI-assisted workflow, ticket priority, and mathematical correctness. Produce question-by-question reports under notes/ as explicitly requested, a consolidated browsable HTML report in ~/agent-share, and evidence-backed follow-up tickets labeled adversarial-reviewer. Do not turn speculative concerns into tickets.

## Acceptance Criteria

1. Current full test suite and representative workloads are measured. 2. Core derivations are traced to primary sources and implementation/tests. 3. Seven requested questions have evidence-backed reports plus an executive synthesis and second pass. 4. Concrete defects/gaps have adversarial-reviewer tickets with practical impact and acceptance criteria. 5. HTML report is published to ~/agent-share. 6. Dashboard refreshed, ticket handoff recorded, changes committed, tree clean.


## Notes

**2026-07-10T18:06:44Z**

SUB-AGENT PRIMARY-SOURCE AUDIT (paper_math_audit, 2026-07-10; confidence high). Q: Is discrete observation GFE faithful? A: Yes for the reduced fixed-goal/frozen-parameter composite: Part II p11 Eq5, p15 Eqs20-22, App C pp32-34 Eqs24-25 and Part I pp26-27 match rules._gfe_solve/gfe_update/gfe_energy. Full Fig.5 mu1(c) and mu3(A) remain unimplemented, already tracked by cff-gjjk/cff-iwgl. Q: Is novel transition P-sub math correct? A: No: rules.py:451 omits cavity division by mu_down, and :471 scores G_tm-E_qw[mu_down] rather than node-local G_tm; independent numeric reproductions agree. Filed cff-hp3y. Q: Do tests prove curiosity? A: observation T-maze tests establish canonical cue-seeking/learning, but transition tests mostly prove local novelty and decay; missing relevance, cost crossover, realizable sensor-quality, and satiation coverage, now cff-4oq8. Q: Are continuous/hybrid epistemics impossible? A: no universal exact rule; papers give general integral stationary equations but no continuous messages. Fixed linear-Gaussian local GFE is analytic (MI=.5(logdet(CSC'+R)-logdet R), Gaussian goal cross-entropy analytic); solve over Gaussian mean/SPD covariance is tractable. Purposeful exploration requires action-dependent sensing/noise/parameters/nonlinearity because standard linear-Gaussian MI may be action-invariant. Mixture entropy/arbitrary nonlinear MI has no general closed form, so exact universal hybrid GFE should be abandoned in favor of declared subclasses/bounds/MC. Sources: the two repository PDFs directly, not OCR.

**2026-07-10T18:10:58Z**

PERFORMANCE AUDIT (2026-07-10, Apple M2 Max CPU, JAX 0.10.2, x64; reproducible harness ../experiments/cffg-adversarial-bench commit a543c6b). Full suite: 1,091 passed in 130.23s. Against Dynamax 1.0.2 with identical params/data and posterior parity: ordinary K8 HMM warm CFFG 7.41-28.17ms vs 0.0186-0.0892ms (316-399x); long-range identity HMM needs 2T+1 flooding sweeps and grows 623x at T32 to 4,206x at T256 (83.24ms vs 0.0198ms); d4 LGSSM 74x at T16 to 137x at T64 before CFFG readout. CFFG posterior errors <=5.7e-9. Existing consumer results agree: active vision 4.1 seq/5.1 batched images/s misses 20 target; Colony32 1.5x; long-horizon planning .6-2.47s/action; hybrid streaming reaches 455 filtered steps/s on a compile-once W25 workload. Diagnosis: Python dispatch (cff-wbhu) plus algorithmic whole-graph flooding O(E*diameter), global padding, and non-specialized readouts. Filed cff-d56d and cff-44we. No GPU benchmark was possible on this host; no GPU/scalability claim is warranted.

**2026-07-10T18:26:31Z**

FINAL HANDOFF (adversarial review complete, 2026-07-10). Reports: notes/adversarial-review-2026-07-10/{00-executive-summary,01-architecture,02-performance,03-curiosity-and-tractability,04-commercial-value,05-workflow,06-ticket-priorities,07-math-audit}.txt with index.html. Published: http://100.115.170.73:8080/cffg-adversarial-review.html. Verification: 1,091 tests passed, 4 dependency warnings, 129.72s; dashboard refreshed; all published assets return HTTP 200. Immediate trust blockers: cff-hp3y transition GFE cavity/energy, cff-j4mu honest termination/infeasibility, cff-vamy plate/nested-blob bundle preservation. Next architecture/numerics sequence: cff-3hgi, cff-ri8m, cff-d56d, cff-44we, then cff-wbhu; add cff-4oq8 behavioral epistemic oracles. Treat cff-dfpi as a bounded P1 research spike after cff-hp3y: exact action-dependent linear-Gaussian sensing, a negative additive-control case, one switching-regime hybrid, and an explicit exact/approximate scope decision. Do not promise universal exact nonlinear or mixture-hybrid GFE. Benchmark source: /Users/ravd/projects/experiments/cffg-adversarial-bench commit a543c6b, clean. Scope correction from second pass: Hallway epistemic-B must be rerun after cff-hp3y; RockSample uses observation GFE and is not evidence for that defect.
