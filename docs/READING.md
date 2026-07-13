# Recommended Reading

A deliberately short list — cream of the crop only — curated 2026-07-12
around two threads Cooper is actively pulling on: (1) where epistemic
behaviour can and cannot live in continuous/hybrid models (the cff-dfpi /
DERIVATIONS.md D5 story), and (2) the structure-learning outer loop (the
"structure learning module" epic): how an agent or module proposes new model
structure, scores it by evidence, and grows a fragment library. One line of
"why this one" per entry; read in listed order within each section.

## 1. Hybrid epistemics — the context for D5

- **Koudahl, Kouw & de Vries (2021), "On Epistemics in Expected Free Energy
  for Linear Gaussian State Space Models", Entropy 23(12):1565.**
  <https://www.mdpi.com/1099-4300/23/12/1565>
  The negative result D5 Prop. 5 restates in CFFG form: additive-control
  linear-Gaussian epistemic terms are policy-independent, and
  multiplicative/switching controls are the explicit escape hatch. Short,
  precise, and the reason this library will never promise a curious Kalman
  agent.

- **Collis et al. (2024), "Learning in Hybrid Active Inference Models".**
  <https://arxiv.org/abs/2409.01066>
  The field's convergent answer to the same problem: learn a discrete
  (rSLDS) abstraction over continuous dynamics and lift information-seeking
  into the discrete layer. Validates the D5 regime-channel decision from the
  learned-abstraction side — and its rSLDS fitting is a candidate automated
  regime-discovery operator for the structure module.

## 2. Structure learning, the pragmatic thread (Friston-lab lineage)

- **Friston, Parr & Zeidman (2019), "Bayesian Model Reduction".**
  <https://arxiv.org/abs/1805.07092>
  The cheapest useful structure operator: score edge/parameter *removals* in
  closed form from an already-fitted posterior, no refit. For our conjugate
  (Dirichlet/NIW) beliefs this is near-free algebra — the natural v1
  prune/merge scorer for the epic.

- **Friston et al. (2023), "Supervised Structure Learning".**
  <https://arxiv.org/abs/2311.10300>
  Grow discrete generative models by fast, evidence-scored assimilation,
  with priors over model selection based on expected free energy (note the
  emphasis on *data ordering* — curricula matter). Koudahl is a co-author;
  the modelling idiom is close to ours (tabular, conjugate).

- **VERSES (2025), "AXIOM: Learning to Play Games in Minutes with Expanding
  Object-Centric Models".**
  <https://arxiv.org/abs/2505.24784>
  The panned-out running system: online mixture-component growth when
  evidence demands + BMR-based prune/merge, gradient-free, over
  object-centric switching linear models — i.e. "add-regime / prune-regime"
  as a working structure-learning loop on exactly the model family our
  hybrid factors implement. The epic's literature-pass ticket extracts its
  growth/merge criteria. (Its "core priors" are also a worked example of
  section 3's ideas.)

## 3. Core knowledge & theory learning — where proposals come from

- **Spelke & Kinzler (2007), "Core Knowledge", Developmental Science
  10(1):89–96.** <https://doi.org/10.1111/j.1467-7687.2007.00569.x>
  The canonical short statement of the core-knowledge systems (objects,
  agents, number, geometry). Read as a requirements document: each system is
  a candidate template *fragment* with typed ports and characteristic
  epistemic probes.

- **Kemp & Tenenbaum (2008), "The Discovery of Structural Form", PNAS
  105(31).** <https://www.pnas.org/doi/10.1073/pnas.0802631105>
  Structure learning as search over a *grammar of forms* (chains, trees,
  grids, rings) scored by Bayesian evidence — the cleanest existence proof
  that "hypothesis space = generative grammar over structures" works, and
  the template for what our proposal vocabulary should look like.

- **Ullman & Tenenbaum (2020), "Bayesian Models of Conceptual Development:
  Learning as Building Models of the World", Annual Review of Developmental
  Psychology 2:533–558.**
  <https://doi.org/10.1146/annurev-devpsych-121318-084833>
  The best single survey of the child-as-scientist program: theory learning
  as stochastic search over structured hypothesis spaces, seeded by core
  knowledge. Subsumes and cites the earlier "theory learning as stochastic
  search in a language of thought" work (Ullman, Goodman & Tenenbaum 2012)
  — chase that reference if you want the algorithmic details.

- **Ellis et al. (2020), "DreamCoder: Growing Generalizable, Interpretable
  Knowledge with Wake-Sleep Bayesian Program Learning".**
  <https://arxiv.org/abs/2006.08381>
  The strongest available answer to "bootstrap": alternate solving problems
  with the current fragment library and *compressing* recurring solution
  structure into new named fragments, so the library's own successes become
  its future hypothesis space. Nobody has done this for factor-graph world
  models; that gap is the ambitious end of the structure-learning epic.
