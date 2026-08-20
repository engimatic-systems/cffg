# Independent rSLDS Debug Review

This workspace holds two independent diagnoses of the unresolved rSLDS
experiment results. Reviewers should use the same questions and evidence, but
should not read each other's draft conclusions until both reports are marked
ready for comparison.

## Investigation map

Legend: prerequisite --> dependent

```text
[ ] cff-l5ag reference-adapter mixing
      |
      +--> fresh classic comparison

[ ] cff-q1au heterogeneous coverage --+
[ ] cff-qulu heterogeneous convergence +--> [ ] cff-0oqo source recovery
                                                     |
                                                     +--> fresh heterogeneous comparison

[ ] cff-3v7q classic periodicity robustness --> fresh classic comparison
```

The two main diagnostic tracks are:

1. **Reference-adapter mixing (`cff-l5ag`).** Determine why all ten external
   adapter arms failed their R-hat and effective-sample-size requirements.
   Separate regime-label or decision-list symmetry, poor initialization,
   inadequate chain movement, diagnostic mistakes, and sampler defects.
2. **Heterogeneous CFFG inference and recovery.** Determine why coverage failed
   for one dataset (`cff-q1au`), why primary fits and controls exhausted their
   outer convergence budgets (`cff-qulu`), and why the converged fits exceeded
   the source-function recovery caps (`cff-0oqo`). Separate data eligibility,
   outer learning, recurrence learning, alignment, and identifiability.

`cff-3v7q` owns the smaller classic periodicity miss. It is relevant to a fresh
classic run but is not evidence about the adapter's mixing failure.

## Shared questions

Each reviewer should answer these questions with concrete evidence:

### Reference adapter

1. Which monitored variables dominate R-hat and effective-sample-size failure?
2. Do the chains represent equivalent label-switched or decision-list-switched
   modes, or genuinely different posterior regions?
3. Are the registered diagnostics valid for the representation being sampled?
4. If the sampler is moving poorly, which update or parameter family causes it?
5. What is the smallest development-seed experiment that can distinguish the
   leading explanations?
6. What evidence would justify a fresh frozen comparison?

### Heterogeneous CFFG

1. Can source-and-gate coverage be made reliable without selecting favorable
   final seeds?
2. Where does outer convergence stall: belief inference, parameter replacement,
   free energy, damping, or the convergence criterion?
3. How much source-function error comes from recurrence learning, latent affine
   alignment, regime labels, or insufficient data?
4. Are shared and native initialization failing for the same reason?
5. What engine behavior must change, and what belongs only to experiment design
   or evaluation?
6. What development-seed evidence would justify a fresh frozen run?

## Evidence and independence rules

- Treat the completed `cff-oyie` result and artifact as immutable evidence.
- Retained final-chain traces may be inspected to diagnose the recorded failure,
  but final seeds 2001–2003 and 2101–2103 must not tune a repair, budget, scale,
  threshold, or generator.
- Use new development or calibration seeds for proposed changes.
- Distinguish probability-model correctness, inference behavior, reference
  implementation behavior, experiment design, and scoring.
- Record exact commands, commits, artifact identities, and generated evidence.
- State competing explanations, disconfirming evidence, recommendation, and
  confidence. Do not report a favored explanation without saying what would
  falsify it.
- A reviewer marks both of their reports `Ready for comparison: yes` only after
  their investigation is complete. Until then, reviewers should not open the
  other reviewer's folder.

## Comparison procedure

After both reviews are ready, compare them question by question. Classify each
conclusion as agreement, compatible difference, or contradiction. Resolve
contradictions with a discriminating test rather than by averaging confidence.
The resulting decisions belong in the affected `tk` tickets and, if they change
the architecture, in `docs/ARCHITECTURE.md` with the implementation.
