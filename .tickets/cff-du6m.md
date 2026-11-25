---
id: cff-du6m
status: closed
deps: []
links: []
created: 2026-07-13T03:51:58Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-cynx
---
# Structure-learning literature pass: AXIOM mixture growth/merge + BMR formulas

## Background in Plain English

The structure-search epic cff-cynx needs equations for deciding when to add, prune, or merge model components. This ticket owns the bounded literature review and the repository-specific derivations that make those decisions implementable. It does not implement a search operator.

Three primary sources answer different parts of the problem. AXIOM, arXiv:2505.24784, supplies an online mixture-growth rule and a scheduled greedy merge procedure. Supervised Structure Learning, arXiv:2311.10300, combines model evidence with an expected-free-energy prior and explains why sequential data order matters. Bayesian Model Reduction, arXiv:1805.07092, supplies the general reduced-prior identity and worked Gaussian and Dirichlet forms.

The papers do not use one shared sign convention, and the Bayesian Model Reduction paper does not give Normal-Inverse-Wishart or Matrix-Normal-Inverse-Wishart formulas in this repository's parameterization. The research note must therefore preserve each source equation, state the sign conversion to CFFG's lower-is-better free-energy ledger, and derive the missing conjugate-family forms instead of presenting them as quotations.

The final research record belongs in this ticket's Notes as question -> answer -> source -> confidence. Later implementation may copy durable formulas into code docstrings, but this ticket closes on a reviewable research record with hand-checkable examples.

## Outcomes / Acceptance Criteria

1. Each research answer in Notes names one question, one answer, one primary-source location, and one confidence level.
2. The AXIOM growth record includes the main-text posterior-predictive score `ell[t,c] = E_q(Theta_c)[log p(y_t | Theta_c)]` from section 2.1.
3. The AXIOM growth record includes the main-text new-component threshold `tau_t = log p_0(y_t) + log alpha`.
4. The AXIOM growth record states that the main-text rule creates a component when `max_c ell[t,c] < tau_t` and the component cap has not been reached.
5. The AXIOM growth record includes Algorithm 1's batch quantity `ell_i_max = max_k ell[i,k]`.
6. The AXIOM growth record states that Algorithm 1 stops when `min_i ell_i_max > tau`.
7. The AXIOM growth record states that Algorithm 1 otherwise seeds a component from `i_star = argmin_i ell_i_max`.
8. The research record decides whether the main-text `tau_t` and Algorithm 1's fixed `tau` are two supported policies or one formula with missing notation, and cites evidence for that decision.
9. The AXIOM merge record states the paper's pruning interval `Delta T_BMR = 500` frames.
10. The AXIOM merge record states how the paper forms at most `n_pair = 2000` candidate pairs from used recurrent-mixture components.
11. The AXIOM merge record includes `eta_merge = eta_1 + eta_2 - eta_prior` and explains which sufficient statistics each `eta` contains.
12. The AXIOM merge record states that the algorithm evaluates all proposals against one ancestrally sampled data set for that reduction run.
13. The AXIOM merge record states that a proposal is accepted when its evidence lower bound is not smaller than the evidence lower bound of the currently accepted model.
14. The AXIOM merge record states that accepted proposals update the current model before the next greedy proposal is evaluated.
15. The AXIOM merge record distinguishes its generated-data multinomial evidence-lower-bound test from the general reduced-prior identity in arXiv:1805.07092.
16. The Supervised Structure Learning record includes Equation 5: `log(P(m|o) / P(m_prime|o)) = Delta F + Delta G`.
17. The Supervised Structure Learning record defines `Delta F = log P(o|m) - log P(o|m_prime)`.
18. The Supervised Structure Learning record defines `Delta G = log P(m) - log P(m_prime) = G(a|m) - G(a_prime|m_prime)`.
19. The Supervised Structure Learning record transcribes Algorithm 1's separate evidence and expected-free-energy gates without replacing them by a single undocumented threshold.
20. The Supervised Structure Learning record includes Equation 7's finite-style prior `p = n/N` and `Delta H = log p - log(1-p)`.
21. The Supervised Structure Learning record defines the paper's `n`, `N`, parent model, and augmented model in Equation 7.
22. The Supervised Structure Learning record states that transition-learning observations retain their generated temporal order.
23. The Supervised Structure Learning record explains that earlier posterior counts become priors for later sequential comparisons.
24. The Supervised Structure Learning record states whether data order adds an algebraic term to Equations 5 or 7.
25. The general Bayesian model reduction record states the shared-likelihood requirement for full and reduced models.
26. The general Bayesian model reduction record includes `Delta log Z = log E_Q[P_reduced(theta) / P_full(theta)]` from Equation 9.
27. The general Bayesian model reduction record includes `Q_reduced(theta) proportional to Q(theta) P_reduced(theta) / P_full(theta)` from Equation 10.
28. The sign table states that positive `Delta log Z` favors the reduced model in the papers' evidence-lower-bound convention.
29. The sign table states the equivalent comparison for CFFG's lower-is-better Bethe free energy.
30. The Dirichlet derivation uses `a_0` for the full prior, `a_N` for the full posterior, and `a_0_reduced` for the reduced prior.
31. The Dirichlet derivation includes `a_N_reduced = a_N + a_0_reduced - a_0`.
32. The Dirichlet derivation includes `Delta log Z = log B(a_0) - log B(a_0_reduced) + log B(a_N_reduced) - log B(a_N)`.
33. The Dirichlet derivation defines `B` as the multivariate beta function and states the tensor axes over which it is evaluated.
34. The Dirichlet derivation states the positivity conditions that make the full prior, full posterior, reduced prior, and reduced posterior proper.
35. The Dirichlet derivation explains how a proper reduced prior scores a proposed state prune without inserting a literal zero concentration into `cffg.learn.Dirichlet`.
36. The Dirichlet derivation explains whether merging two states is one same-support reduced-prior comparison or requires an additional likelihood and state-space transformation.
37. A hand-checkable Dirichlet example computes the reduced posterior and log-evidence difference from concrete vectors.
38. An independently calculated Dirichlet value agrees with the formula in criterion 32.
39. The Normal-Inverse-Wishart derivation uses the repository tuple `(mu, kappa, psi, nu)` and the density convention `Sigma ~ IW(psi, nu)`, `mu_given_Sigma ~ N(mu, Sigma/kappa)`.
40. The Normal-Inverse-Wishart derivation preserves the repository validity conditions `kappa > 0`, `psi` symmetric positive definite, and `nu > d - 1`.
41. The Normal-Inverse-Wishart derivation writes the complete log normalizer in the convention from criterion 39.
42. The Normal-Inverse-Wishart derivation maps `(mu, kappa, psi, nu)` to natural parameters.
43. The Normal-Inverse-Wishart derivation computes reduced posterior natural parameters as `eta_N + eta_0_reduced - eta_0`.
44. The Normal-Inverse-Wishart derivation reconstructs a valid repository tuple from the reduced natural parameters.
45. The Normal-Inverse-Wishart derivation expresses `Delta log Z` as the four required log-normalizer terms.
46. A hand-checkable scalar Normal-Inverse-Wishart example agrees with an independent numerical integral or another independent analytic calculation.
47. The Matrix-Normal-Inverse-Wishart derivation uses the repository tuple `(m, lam, psi, nu)`, augmented input `x_tilde = [x; 1]`, row covariance `Sigma`, and column precision `lam`.
48. The Matrix-Normal-Inverse-Wishart derivation preserves the repository validity conditions that `lam` and `psi` are symmetric positive definite and `nu > d_y - 1`.
49. The Matrix-Normal-Inverse-Wishart derivation writes the complete log normalizer in the convention from criterion 47.
50. The Matrix-Normal-Inverse-Wishart derivation maps `(m, lam, psi, nu)` to natural parameters.
51. The Matrix-Normal-Inverse-Wishart derivation computes reduced posterior natural parameters as `eta_N + eta_0_reduced - eta_0`.
52. The Matrix-Normal-Inverse-Wishart derivation reconstructs a valid repository tuple from the reduced natural parameters.
53. The Matrix-Normal-Inverse-Wishart derivation expresses `Delta log Z` as the four required log-normalizer terms.
54. A hand-checkable scalar-output Matrix-Normal-Inverse-Wishart example agrees with an independent numerical integral or another independent analytic calculation.
55. The family-to-operator table identifies which distribution scores a Dirichlet state prune.
56. The family-to-operator table identifies which distributions and transformations score a Dirichlet state merge.
57. The family-to-operator table identifies which Matrix-Normal-Inverse-Wishart prior change scores a Gaussian-linear edge prune.
58. The family-to-operator table identifies whether pruning a Gaussian regime is scored through its mixing-weight Dirichlet belief, its component belief, or both.
59. The family-to-operator table separates an evidence-scoring prior change from the physical `Template` edit performed after acceptance.
60. The implementation handoff states that exact support removal happens after model selection rather than by evaluating an improper zero-concentration or singular-covariance density.
61. The implementation handoff states how a vacuous posterior affects each reduced-evidence formula.
62. The implementation handoff rejects a reduced natural-parameter combination that reconstructs an improper distribution.
63. A float32 check preserves the winning model selected by its float64 reference on a fixture with a nonzero score margin.
64. A Dirichlet scale example uses at least 64 categories and remains finite in log space.
65. A Gaussian scale example uses at least 32 dimensions and remains finite in log space.

## Methodological Suggestions

The researcher may choose another sound derivation path if it satisfies the outcomes. Use targeted excerpts from the three named primary sources. Record page, section, equation, algorithm, or table identifiers and the arXiv identifier for every paper-derived claim.

Treat AXIOM section 2.1, Appendix Algorithm 1, Appendix A.10, Algorithm 2, and Table 3 as separate evidence. Preserve their different threshold presentations until criterion 8 resolves them. Treat Supervised Structure Learning Equations 5 and 7, Algorithm 1, and its discussion of curricula as separate evidence.

Start the conjugate derivations from Bayesian Model Reduction Equations 9 and 10. For exponential-family distributions, use natural-parameter addition and log normalizers. Do not add the displayed NIW or MNIW tuple fields component by component unless the natural-parameter map proves that operation.

Use `src/cffg/conj.py` and `docs/DERIVATIONS.md` D4 as the repository authority for NIW and MNIW names, shapes, validity, and rollover. `NIW.rollover` and `MNIW.rollover` write `psi / nu`, the inverse expected precision; that rollover value is not the density normalizer. Use `src/cffg/learn.py` as the authority for Dirichlet concentration storage and expected-log rollover.

For each toy example, show inputs, intermediate reduced parameters, the score, the sign decision, and the independent calculation. Prefer small rational or diagonal parameters that a reviewer can recompute. Then repeat only the score evaluation at the scale and dtype boundaries in criteria 63-65. Evaluate normalization terms with `gammaln`, multivariate log-gamma, and signed log determinants instead of products of normalization constants.

This is a research-only change. The `docs/ARCHITECTURE.md` change-path rows and duplicated execution surfaces are N/A because this ticket changes no IR, compiler, runtime, serialization, or checkpoint behavior. Put the distilled result in Notes; do not create a new document. Durable formulas move into code docstrings when cff-cynx implements them.

## Future work

Epic cff-cynx owns the outer structure-search module that consumes this ticket's growth, prune, merge, sign, and family-to-operator decisions. It remains blocked until this ticket supplies the accepted research record.

Ticket cff-6tek owns registration of NIW and MNIW as declared learning families. It does not own Bayesian model reduction scoring, but its family adapters must use the same repository parameter conventions recorded here.

No other future work is identified. A paper result that cannot support one of cff-cynx's proposed operators must be reported as a limitation rather than deferred without an owner.

## Definitions

- **Ancestral sampling:** Generating variables in the dependency order of a probabilistic model.
- **AXIOM:** The expanding object-centered active-inference system in arXiv:2505.24784.
- **Bayesian model reduction (BMR):** Scoring a model with changed priors from a fitted full model's prior, posterior, and evidence.
- **Bethe free energy:** CFFG's lower-is-better inference objective and model-evidence approximation.
- **CFFG:** Constrained Forney-style Factor Graph.
- **Component cap:** The configured maximum number of mixture components.
- **Dirichlet distribution:** A probability distribution over categorical probabilities, parameterized by positive concentrations.
- **Evidence lower bound (ELBO):** A lower bound on log model evidence; higher values are better.
- **Expected free energy:** A planning quantity used as a prior over model choices in Supervised Structure Learning.
- **Full model:** The fitted model whose prior and posterior are inputs to Bayesian model reduction.
- **`gammaln`:** A numerical function that evaluates `log Gamma(x)` without first forming `Gamma(x)`.
- **Hard assignment:** Assigning one observation to one selected mixture component rather than retaining fractional responsibilities.
- **Inverse-Wishart distribution (IW):** A probability distribution over positive-definite covariance matrices.
- **Log normalizer:** The logarithm of the factor that makes a probability density integrate to one.
- **Matrix-Normal-Inverse-Wishart (MNIW):** The repository's conjugate belief over a linear map, offset, and covariance.
- **Model evidence:** The probability assigned to observations after latent variables and parameters are integrated out.
- **Multivariate log-gamma:** The log normalization function used by Wishart-family distributions.
- **Multivariate beta function:** The Dirichlet normalization function `B(a) = product_i Gamma(a_i) / Gamma(sum_i a_i)`.
- **Natural parameters:** Exponential-family coordinates in which independent sufficient-statistic contributions add.
- **Normal-Inverse-Wishart (NIW):** The repository's conjugate belief over a Gaussian mean and covariance.
- **Posterior predictive density:** A likelihood averaged over the fitted parameter posterior.
- **Prior predictive density:** A likelihood averaged over the parameter prior before the new observation is assigned.
- **Proper distribution:** A distribution whose density has a finite normalization integral equal to one.
- **Recurrent mixture model (rMM):** AXIOM's mixture over continuous and discrete variables that predicts reward and the next transition mode.
- **Reduced model:** A candidate model represented by a changed prior over the full model's parameter space.
- **Signed log determinant:** A decomposition that records the sign and log magnitude of a determinant without forming the determinant directly.
- **Symmetric positive definite:** A symmetric matrix with strictly positive eigenvalues.
- **Transition mixture model (tMM):** AXIOM's mixture of linear object-dynamics modes.
- **Vacuous posterior:** A posterior that contributes no information beyond its prior.

## Notes

**2026-08-07 — downstream-first intelligibility review**

Dependency map. Arrow direction is prerequisite --> dependent.

```text
[ ] cff-du6m (ready research task)
          |
          +--> [ ] cff-cynx (blocked structure-search epic)

cff-cynx has no downstream roadmap dependent.
```

Targeted source anchors used to bound this rewrite; these do not replace the complete acceptance research record:

- Question -> How does AXIOM grow a mixture? Answer -> Section 2.1 uses `ell[t,c]`, `tau_t = log p_0(y_t) + log alpha`, and a best-active-component comparison. Appendix Algorithm 1 instead uses the worst-explained token and a displayed fixed `tau`; criterion 8 requires an explicit reconciliation. Source -> arXiv:2505.24784, section 2.1 page 6 and Appendix Algorithm 1 page 26. Confidence -> high for the transcription; unresolved for the relationship between the two presentations.
- Question -> How does AXIOM merge components? Answer -> Every 500 frames, Algorithm 2 greedily evaluates overlap-selected recurrent-mixture pairs on one ancestrally sampled data set, uses `eta_merge = eta_1 + eta_2 - eta_prior`, and retains a merge when its ELBO is not smaller. Source -> arXiv:2505.24784, Appendix A.10 and Algorithm 2, pages 26-27; Table 3, page 28. Confidence -> high.
- Question -> What prior does Supervised Structure Learning place over expansions? Answer -> Equation 5 adds an expected-free-energy log prior to the evidence difference; Equation 7 adds `Delta H` from the finite-style prior `p = n/N`. Source -> arXiv:2311.10300, Equations 5 and 7 and Algorithm 1, pages 11-14. Confidence -> high for the displayed formulas; the meaning of `n` versus the prose symbol `H` needs explicit resolution.
- Question -> How does order enter? Answer -> The paper requires temporally generated order for learning paths and updates each sequential model comparison after earlier parameter assimilation. The targeted equations show no separate order correction term. Source -> arXiv:2311.10300, "The importance of order," pages 3-4; supervised-order discussion, page 22; Equations 5 and 7. Confidence -> medium-high pending the final full set of targeted excerpts.
- Question -> What identity supports repository specializations? Answer -> Reduced log evidence is `log E_Q[P_reduced/P_full]`; the reduced posterior is proportional to `Q P_reduced/P_full`; the paper gives Dirichlet and Gaussian forms but not NIW or MNIW forms in CFFG coordinates. Source -> arXiv:1805.07092, Equations 9-10, pages 6-7, and Equation 12/Table 1, pages 9-10. Confidence -> high.
- Question -> Which repository conventions constrain the missing derivations? Answer -> NIW is `(mu, kappa, psi, nu)` with `Sigma ~ IW(psi,nu)`; MNIW is `(m,lam,psi,nu)` with augmented input and column precision; both roll over with covariance `psi/nu`. Source -> `src/cffg/conj.py` and `docs/DERIVATIONS.md` D4. Confidence -> high.

Retained current-scan map, candidate -> selected meaning:

- `fixed` -> the appendix threshold is a configured scalar rather than the observation-dependent main-text expression.
- `integrate` -> a density's normalization integral or marginalization over latent quantities, as stated by the definition.
- `n_a` -> the architecture change path is structurally irrelevant because this ticket changes research text alone.
- `preserve` -> source notation, repository parameter validity, model-selection sign, or winner identity remains unchanged across the named boundary.
- `register` -> cff-6tek's concrete family-registration work.
- `same` -> the shared likelihood parameter space required by BMR, or one repository parameter convention used by two tickets.
- `implement` -> planned downstream code work; this research ticket explicitly does not perform it.
- `support` -> a source or derivation supplies formal support, or criterion 8 permits one of two threshold policies.
- `work` -> work assigned to the named downstream tickets in Future work.
- `complete` -> every normalization term or every acceptance research answer is present.
- `ready` -> cff-du6m has no open prerequisite in `tk ready`.
- `remaining` -> a finite score stays finite, or cff-cynx is unfinished while this prerequisite is open.
- `appropriate` -> evidence-based judgment that an alternative derivation satisfies the stated outcomes.
- `approximate` -> the named Bethe approximation.
- `close` -> an analytic closed-form expression in the title, or administrative ticket completion in Background.
- `sufficient` -> the statistical summaries required by a conjugate distribution.
- `check` -> the numeric float32 evaluation in criterion 63.
- `equivalent` -> the mathematically corresponding lower-is-better CFFG sign decision.
- `estimate` -> parameter estimation that produced the full posterior.
- `evidence` -> probabilistic model evidence unless the adjacent phrase names primary-source support.
- `exact` -> literal structural support removal after a proper-density score.
- `pass` -> this bounded literature-review task.
- `sound` -> a derivation route that is reasonable and still satisfies every outcome.
- `test` -> AXIOM's mathematical merge acceptance comparison.
- `fit` -> parameter estimation that produced the full posterior.
- `infer` -> posterior inference over variables or policies, as selected by the adjacent term.
- `learn` -> model-structure search in prose, Supervised Structure Learning in the paper title, or the named `cffg.learn` parameter module.
- `rollover` -> conversion from posterior hyperparameters to the point parameters consumed by a later inference run.
- `scale` -> category count or Gaussian dimension in criteria 64-65.
- `small` -> numeric ordering of evidence-lower-bound values, or deliberately low-dimensional toy parameters.
- `can` -> a reviewer has the practical ability to recompute the chosen toy values.
- `expected` -> a mathematical expectation, the expected-precision rollover, or the named expected-free-energy quantity.
- `may` -> permission for a later implementation or an optional derivation method.
- `explicit_owner` -> cff-du6m owns research, cff-cynx owns structure search, and cff-6tek owns family registration.
- `named_ticket_handoff` -> cff-cynx and cff-6tek are the concrete future owners stated above.
- `scope_exclusion` -> the architecture change path is N/A because no executable surface changes.
- `all` -> the statement covers the complete locally named set.
- `at_least` -> the inclusive lower bounds of 64 categories and 32 dimensions.
- `every` -> the requirement applies separately to each paper claim or research answer in the named set.
- `only` -> the text excludes unscored work or restricts the repeated scale step to score evaluation.
- `current` -> the model retained by AXIOM's active greedy reduction step.
- `deferred` -> ownerless postponement is prohibited; unsupported operator claims become limitations.
- `later` -> downstream implementation or sequential posterior use after this research step.

Intelligible-ticket review: complete.

**2026-11-25T18:07:43Z**

Navigation example for the structure-learning review

What does AIMAPP learn? De Tinguy et al. propose unvisited places from unobstructed headings beyond an existing place radius. Equation 7 compares the pose likelihood before and after expansion. Experienced transitions update directed connection strengths. Position estimates, obstacle sensing and low-level movement come from separate components. Section 3.5 gives a worked example of suppressing incoming transitions to a blocked place. Confidence is high for these mechanisms. [AIMAPP v2, §§3.1–3.6](https://arxiv.org/html/2510.09574v2#S3.SS5)

What did the benchmark establish? The goal-reaching comparison starts at a known position on a previously explored graph. The median executed distance was 1.24 times the full-map A* distance over 40 trials. The scope of that result is route finding. The authors discuss symmetric views, drift and finite planning horizons separately. Confidence is high for the protocol and reported statistic. [Table 5](https://arxiv.org/html/2510.09574v2#S4.SS4) and [§5.1](https://arxiv.org/html/2510.09574v2#S5.SS1)

Which part would transfer? A proposed first test keeps exact local motion, presents two similar junctions with different exits, and includes a return route. Compare a fixed map model and a model allowed to propose distinct places or reconnect a revisit. Give candidates identical ordered fit data and score their predictions on the same reserved continuation. Compare route success and steps against a tile-map planner with the same observations. Keep predicted connections distinguishable from traversed connections. This is a proposed test. The paper's local expansion score and the epic's held-out model score have different meanings. Scale measurements should cover graph growth, parameter transfer, recompilation time and peak memory before choosing sparse storage.

Existing tracker dependencies, prerequisite --> dependent
    [ready] cff-du6m review ---------+
                                     +--> [blocked] cff-cynx outer model comparison
    [closed] cff-izjd hybrid GFE ----+

The navigation example supplies a possible consumer. The remaining review of growth, merge, and reduced-prior evidence stays on cff-du6m. The implementation split belongs after that review.

**2026-11-25T18:27:26Z**

What does a renormalising generative model contribute to learning places? In From pixels to planning, a chosen partition and temporal stride define which observations are grouped. Fast structure learning assigns distinct recurring patterns to states, records which states follow under latent transition modes called paths, and passes initial states and paths to the next level. A path in this model describes how states change. It need not correspond to a compass command or a route on a map. Confidence is high for the construction. [Friston et al., Figure 3 and Algorithm 1](https://arxiv.org/pdf/2407.20292#page=26)

How does this differ from an online map learner? The Pong demonstration selects successful event sequences from random play for training. That selection matters when interpreting what the learned higher-level sequences predict. The study gives a way to propose abstractions over ordered experience. It does not validate a rule for recognizing a physical junction on revisiting it. Confidence is high for the reported protocol. [Pong experiment, Figures 25–27](https://arxiv.org/pdf/2407.20292#page=50)

How could we test a transfer? Fit a dictionary of two-step events on the exploration prefix. Freeze it, then compare next-observation prediction with a one-scale model on the same ordered continuation. Include identical views reached after a successful and a blocked movement. Prefer cumulative conditional log loss and report dictionary size. This is a proposed temporal extension. The current navigation pilot tests distinct spatial places and loop returns.

**2026-11-25T18:29:10Z**

**When should a new component be proposed (criteria 1–8)?** AXIOM §2.1 assigns y_t to an existing component when max_c ell[t,c] reaches tau_t. Here ell[t,c]=E_q(Theta_c)[log p(y_t|Theta_c)] and tau_t=log p0(y_t)+log alpha. A strict shortfall seeds a component if capacity remains. Appendix Algorithm 1 instead forms b_i=max_k ell[i,k] for each batch token, stops when min_i b_i>tau, and otherwise seeds from i*=argmin_i b_i. Its tau is a configured scalar (Table 3). Even equality is handled differently. Support these as separately named proposal rules. Confidence is high for the two formulas. An identification of their thresholds would need further assumptions. [AXIOM §2.1](https://arxiv.org/html/2505.24784v1#S2.SS1) · [Algorithm 1](https://arxiv.org/html/2505.24784v1#alg1)

**How are AXIOM merges proposed and judged (9–15)?** Every 500 frames select at most 2,000 pairs from used recurrent-mixture components by feature compatibility. The release samples up to min(2 n_pair, K_used−1) occupied sources, takes their highest-scoring nonself neighbours, filters by modal identity/reward/switch, and caps the survivors. Merge natural parameters as eta_recipient + eta_donor − eta_donor_prior. This counts donor evidence once across mixing counts, continuous effective counts/first and second moments, and categorical head counts. Restore the donor slot to its prior. Freeze a common ancestrally generated dataset for the pass. Accept each live proposal at ELBO_try >= ELBO_current, updating the incumbent and its score after acceptance. The release retains a generated-record buffer and scores mean reward/switch expected log likelihood minus mixing-weight Dirichlet KL. This compares a specific pair of models on simulated data. It supplies neither a universal Bethe ledger nor a prior-ratio identity for a dimensional merge. Confidence is high. [Algorithm 2 and Appendix A.10](https://arxiv.org/html/2505.24784v1#A1.SS10) · [Released pair/score code](https://github.com/VersesTech/axiom/blob/b99f62599e73294639f1944827bf427909f6c698/axiom/models/rmm.py#L789-L920)

**Which way do the model odds run (16–24)?** Supervised Structure Learning Eq. 5 prints parent-over-augmentation odds log[P(m|o)/P(m′|o)]=ΔF+ΔG with ΔF=log P(o|m)−log P(o|m′) and ΔG=log P(m)−log P(m′)=G(a|m)−G(a′|m′). Algorithm 1 first requires ΔF<best (initially zero), then printed ΔG>0. This can accept even when the sum favors the parent. Eq. 3's prior proportional to exp(−G) gives G_aug−G_parent for parent-over-augmentation log-prior odds. Preserve the separate printed rules. For coherent posterior odds use explicit log priors; for a two-gate proposal policy using lower EFE, name G_parent−G_aug its cost improvement. Eq. 7 sets p=n/N, n distinct previously learned styles/likelihood columns in this class, N possible styles. ΔH=log p−log(1−p) is the parent's seen-style log-prior advantage over a new style. Preserve temporal adjacency and episode boundaries. Earlier counts become priors for later comparisons. There is no extra order summand. Confidence is high in the transcription and algebra, moderate for interpreting author intent in the conflicting signs. [Eq. 5](https://arxiv.org/html/2311.10300v1#S3.E5) · [Algorithm 1](https://arxiv.org/html/2311.10300v1#alg1) · [Eq. 7](https://arxiv.org/html/2311.10300v1#S4.E7)

**Which evidence identities can the engine reuse (25–38)?** Hold the observations, likelihood and parameter space fixed. Let priors be p0, pr with pr absolutely continuous with respect to p0. For a true full posterior Q, Δlog Z=log E_Q[pr/p0] and Qr∝Q pr/p0, with a finite positive normalizer. Substituting a variational Q gives an approximation. Positive Δlog Z favors reduction at equal model odds; for evidence-equivalent BFE the signed difference F_reduced−F_full is −Δlog Z. A zero score ties, a negative score favors the full model. Planning GFE and loopy approximations need their own interpretation. For Dirichlet, h=aN+ar−a0; Δlog Z=log B(a0)−log B(ar)+log B(h)−log B(aN). Here log B(a)=sum_i logΓ(a_i)−logΓ(sum_i a_i); use the categorical axis 0 for a vector or output×input matrix, axis 1 for action×next×previous, and sum over independent columns. Validate matching shapes and strictly positive finite concentrations including h. Example a0=(1,1), ar=(1,½), two first-category observations give aN=(3,1), h=(3,½). Independent sequential predictions give Z0=1/3 and Zr=8/15, hence BF=8/5. Confidence is high from Bayes' rule and exact rational arithmetic. [BMR Eqs. 9–12](https://arxiv.org/pdf/1805.07092#page=7) · src/cffg/learn.py, Dirichlet.

**How do NIW and MNIW use the repository's parameters (39–54)?** Write B with d output rows and p coefficient columns. MNIW(m,L,Ψ,ν) uses augmented input [x;1], Σ~IW(Ψ,ν), B|Σ~MN(m,Σ,L⁻¹). NIW(μ,κ,Ψ,ν) is the p=1 case with m=μ and L=[κ]. Require symmetric positive definite Ψ,L, finite tuples and ν>d−1. With W=Σ⁻¹, sufficient statistics (log|Σ|,W,WB,BᵀWB) pair with natural blocks (−(ν+d+p+1)/2,−C/2,H,−L/2), where H=mL and C=Ψ+mLmᵀ. The log integral of the unnormalized kernel on dB dΣ is A=(dp/2)log(2π)−(d/2)log|L|+(νd/2)log 2+logΓ_d(ν/2)−(ν/2)log|Ψ|. Thus NIW uses d/2 log(2π/κ) for the first two terms. For each family, form ηh=ηN+ηr−η0. Equivalently combine ν,L,H,C with these signs. Solve mh Lh=Hh, reconstruct Ψh=Ch−mh Lh mhᵀ, check propriety, then Δlog Z=Ah−AN+A0−Ar. Confidence is high from expanding and integrating the matrix density, checking the repository tuple, and independent density and quadrature checks. These are local derivations. [BMR Eqs. 9–10](https://arxiv.org/pdf/1805.07092#page=7) · src/cffg/conj.py · docs/DERIVATIONS.md D4.

**Do the derived normalizers agree with predictions?** NIW prior (0,1,2,4), reduced (0,4,2,4), y=1 give full posterior (½,2,5/2,5) and reduced (1/5,5,14/5,5). Independent predictive Student t densities give Δ=½log(8/5)+(5/2)log(25/28)=−0.048319899 nats. MNIW with m=[0,1], L=I, Ψ=2, ν=4 shrinks only the slope by choosing Lr=diag(4,1). For [2;1], y=2, full posterior has mN=[1/3,7/6], LN=[[5,2],[2,2]], ΨN=13/6, νN=5. Reduced has mh=[1/6,4/3], Lh=[[8,2],[2,2]], Ψh=7/3, νh=5. The independent predictive log ratio is ½log 2+(5/2)log(13/14)=+0.161303660. Quadrature agrees within 2e−14. Confidence is high. Reproduction is in ~/agent-share/navigation-structure/niw_bmr_check.py and niw-bmr.json.

**What gets edited after selection (55–62)?** Soft-prune a categorical state by suppressing its occupancy through proper positive concentrations in affected selectors. A merge additionally transforms incoming and outgoing rows, emissions and priors. Exact aggregation needs compatible emissions and per-action transition mass into every destination block. Otherwise refit and score the changed likelihood on identical observations. Example: mixing weights Dir(1,1,1) and Bernoulli rates (1,0,½) give P(two successes)=7/24. Merge the first two with rate ½ and Dir(2,1) weights and the same observation probability is ¼, BF=6/7. For a Gaussian-linear predictor column use a finite increase in column precision around zero, preserving a proper MNIW prior. For a Gaussian regime, a mixing-prior change uses its Dirichlet term. A simultaneous emission/dynamics-prior change also involves NIW/MNIW. Sum separate log scores only when the necessary posterior factorization holds. After acceptance, migrate affected axes and counts, remove exact support in the edited template and validate the changed likelihood. Vacuous data give Q=p0, Qr=pr, Δ=0 in all three families. Reject a reconstructed nonpositive concentration, nonsymmetric or nonpositive-definite scale/precision, or invalid degrees of freedom. Confidence is high for the distributional algebra and example. Implementation must declare each edit's transformation. [BMR §§3–5](https://arxiv.org/pdf/1805.07092#page=7) · ~/agent-share/navigation-structure/dirichlet_bmr.py.

**How do precision and scale affect the sign (63–65)?** At K=64 an exact sequential Dirichlet control gives +0.830855026 nats; float32 +0.830871582, float64 +0.830855026. At d=32 both NIW (+6.96973/+6.96964) and MNIW (+5.72046/+5.72027) kept the same sign across float32/64 and matched float64 predictive controls. A separate d=32 MNIW slope suppression gives 16 log(4/3) independently. Confidence is high for these nonzero-margin fixtures. Running a float32 score here does not claim that all repository fitting arithmetic is float32. Sources are the two named scripts and their JSON receipts. These results qualify the research handoff; production operators and practical map scaling belong to the outer-loop epic.
