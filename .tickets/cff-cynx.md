---
id: cff-cynx
status: open
deps: [cff-izjd, cff-du6m]
links: []
created: 2026-07-13T03:51:42Z
type: epic
priority: 2
assignee: coopwilliams
---
# Epic: structure learning module (the outer loop)

Turn S6 from a paragraph into a module: an outer loop that proposes model-structure changes, scores them by free energy, and grows a reusable fragment library. Four pillars, three already built for other reasons: (1) hypothesis representation = Template + edit codec + port/glue composition; (2) score = FE as model evidence, evaluated on HELD-OUT rollouts (training-FE overfits structure; loopy-graph comparisons inherit Bethe error per H1); (3) cheap per-hypothesis fitting/racing = conjugate learning + the colony runtime; (4) proposal generation = the open-research pillar this epic builds. v1 scope: (a) the FE tripwire convention — persistent held-out FE elevation means the hypothesis class is wrong and triggers proposals (the only misspecification signal that leaks through, per H7); (b) three closed-form-scoreable operators: add-regime (grow a mixture component when evidence demands, AXIOM-style), split-state, and BMR prune/merge (Bayesian model reduction is near-free algebra for Dirichlet/NIW posteriors); (c) colony-raced evaluation with an identifiability-alignment step (H9: latent labels permute across candidate structures — align or use label-invariant scores); (d) a fragment library with typed ports as the proposal vocabulary, seeded with core-knowledge fragments (object persistence, contact causality, agency — Spelke systems as template fragments). LLM-as-proposer slots in later as just another proposal source feeding the same scorer; the module must be useful without one. Synergy with cff-dfpi/D5: candidate structures wrapped as regimes of a switching model make the inner agent's regime channel drive exactly the experiments that discriminate hypotheses — inner curiosity is the experiment-runner for outer model comparison. Reading list: docs/READING.md sections 2-3.

## Design

Design constraints inherited from H3/S6: structure learning stays OUTSIDE the objective — this is model comparison wrapped around the engine, never a gradient inside it. Do not let proposal scoring corrupt the FE ledger contract. Dependencies rationale: blocked on cff-izjd because the add-regime operator and hypothesis-discrimination probing are only meaningful once the D5 hybrid GFE composite exists in-engine. Deliberately NOT blocking anything: no existing roadmap item needs this epic.

