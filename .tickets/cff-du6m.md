---
id: cff-du6m
status: open
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
