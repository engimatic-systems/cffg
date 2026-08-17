---
id: cff-dqk2
status: in_progress
deps: [cff-6tek, cff-lquu, cff-4apw]
links: [cff-ahps]
created: 2026-08-13T03:59:30Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-cfqg
tags: [trustworthiness, learning, hybrid]
---
# Register hybrid conjugate learning through LearnableSpec

## Background in Plain English

The generic `Learnables` runtime supports Dirichlet learning for categorical
factors, NIW learning for Gaussian priors, and MNIW learning for ordinary
Gaussian-linear factors. The rSLDS graph also learns switching
Gaussian-linear dynamics and recurrent stick-breaking weights. Those two
factor types cannot declare their learning families today. `RSLDSLearning`
therefore reads engine statistics itself, groups factors by name, updates
posterior objects, and writes point parameters through a separate path.

Closed ticket `cff-6tek` deliberately left these two adapters for later work.
Closed tickets `cff-lquu` and `cff-4apw` already supply the factor-local soft
statistics and posterior algebra. This ticket connects those existing pieces
to `LearnableSpec` and `Learnables`. It is model-independent: its tests use
neutral factor names and do not import the rSLDS builder or fitter.

## Outcomes / Acceptance Criteria

1. The prerequisite capabilities are present before this ticket closes.
   1.1. `cff-6tek` is closed with role-bearing parameter addresses, atomic
   named writes, immutable priors, and `accumulate` and `replace` cadences.
   1.2. `cff-lquu` is closed with proper responsibility-weighted switching
   Gaussian statistics.
   1.3. `cff-4apw` is closed with proper Pólya–Gamma recurrence statistics and
   posterior rollover algebra.
2. `Template.learnables` accepts one documented declared-learning pairing for
   `gaussian_mixture_linear` dynamics.
   2.1. One learnable owns the tied `matrix`, `offset`, and `cov` blobs.
   2.2. Its posterior has one MNIW component for each selector state.
   2.3. Compilation derives the component count and parameter roles from the
   serialized factor and blob shapes rather than from a model name.
   2.4. Unsupported factor, family, role, or posterior-axis combinations fail
   during compilation and name the learnable and incompatible pair.
3. The switching-dynamics adapter consumes each source factor's
   responsibility-weighted `Syy`, `Syx`, `Sxx`, and occurrence count from
   `Engine.gaussian_node_stats`.
   3.1. Sources tied to the same blobs pool into one posterior per selector
   component.
   3.2. A component with exactly zero responsibility adds exactly zero
   statistics and retains its prior coordinate.
   3.3. A direct tuple of `cffg.conj.MNIW` updates is an independent oracle for
   the adapter's posterior hyperparameters and rollover values.
4. `Template.learnables` accepts one documented declared-learning pairing for
   `recurrence_link` weights and offsets.
   4.1. The pairing consumes additive linear and quadratic natural statistics
   from `Engine.recurrence_weight_stats`.
   4.2. It preserves the source-regime, stick, and augmented drive-coordinate
   axes defined by `conj.PGWeights`.
   4.3. Sources tied to the same blobs pool without using factor-name prefixes.
   4.4. A direct `conj.PGWeights` update is an independent oracle for posterior
   natural parameters and rollover values.
   4.5. Unsupported factor, family, role, or axis combinations fail during
   compilation and name the learnable and incompatible pair.
5. Both new pairings use the existing `Learnables` contract.
   5.1. `accumulate` starts from the current posterior and adds evidence from a
   new, converged result.
   5.2. `replace` starts from the immutable prior and cannot count one
   sequence twice.
   5.3. An `only` selection and explicit `LearnableSpec.sources` selection have
   the same meaning as for existing families.
   5.4. A failed source, posterior, or rollover validation changes neither the
   selected posteriors nor the destination parameter mapping.
6. `Learnables.write` installs all roles through the existing parameter-address
   contract and preserves template-level blob tying.
   6.1. Switching dynamics write one value per selector component with the
   compiled array shapes and destination dtype.
   6.2. Recurrence learning writes the point weights and offsets returned by
   `conj.PGWeights.rollover()` with the compiled source-regime and stick
   ordering.
   6.3. Arbitrary unscoped and plate-scoped factor names produce the same
   posterior and write behavior as simple names.
7. Posterior reporting and checkpointing cover every new hyperparameter.
   7.1. Posterior-mean and rollover reporting use public `Learnables` methods;
   callers do not reach into the global family registry.
   7.2. The latest checkpoint schema at the implementation base round-trips
   both new families exactly.
   7.3. A legacy checkpoint either restores without ambiguity or fails with a
   specific migration error; it cannot silently drop a new posterior.
8. Hostile-input tests state and enforce the numerical behavior.
   8.1. Literal zero selector support and zero component responsibility remain
   exact zeros.
   8.2. Vacuous or improper Gaussian beliefs, non-finite statistics, invalid
   Pólya–Gamma precision, incompatible shapes, and non-positive-definite
   posterior values fail before mutation.
   8.3. Float32 updates, rollover, and writes remain finite, preserve float32
   destinations, and use guards derived from the active dtype and value scale.
   8.4. A model-neutral fixture with at least 32 selector states and 16 drive
   dimensions completes one posterior update and point-parameter write. It does
   not allocate a dense state-dependent transition tensor.
9. The new adapters do not change inference messages, schedule selection,
   factor energies, Bethe free energy, serialized factor identities, or the
   structure hashes of existing rSLDS templates.
10. Focused tests compare flooding and scheduled source statistics before the
    host-side posterior update. Host/JIT learning duplication is not
    introduced, and any single-instance/`vmap` or legacy-checkpoint surface
    that is not applicable is recorded with a concrete reason.
11. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` document the
    two supported pairings, posterior axes, accumulate-versus-replace meaning,
    rollover values, hostile-input behavior, and remaining unsupported hybrid
    learning combinations.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Use the **Learnable or checkpoint feature** row in the architecture
  change-path index. Treat `cffg.learn` as the family and cadence authority,
  `cffg.conj` as the posterior-algebra authority, the existing engine readouts
  as the statistic authority, and compiled parameter addresses as the write
  authority.
- Extend the family contract with an explicit posterior batch or component axis
  rather than teaching a model helper to loop over regime names.
- Put hybrid adapters in a focused learning module if adding them to
  `gaussian_learning.py` would blur ordinary Gaussian and selector-conditioned
  responsibilities.
- Build neutral templates named `source_a`, `source_b`, and scoped variants so
  a passing test cannot depend on `dynamics*` or `switch*` naming.
- Capture the existing compact rSLDS inference and ledger artifacts before the
  change and confirm that adding declarations alone leaves them unchanged.

## Future work

`cff-11jj` declares the `cff-garu` rSLDS initial-regime, dynamics, recurrence,
and emission posteriors through these generic families. It also removes manual
statistic discovery from `RSLDSLearning` and the observed-only fitter.

Generic learning for non-linear, neural, precision, synthetic-GFE, or other
hybrid factor types remains outside this ticket. Those combinations require
separate tickets with their own statistic, posterior, objective, and hostile-
input contracts.

## Definitions

- **Declared learning:** A serialized template names a learnable parameter
  group and conjugate family; the compiled runtime then knows its statistic
  sources, posterior state, parameter targets, and checkpoint fields.
- **MNIW (Matrix-Normal-Inverse-Wishart):** A conjugate posterior for a linear
  matrix, offset, and covariance.
- **Pólya–Gamma recurrence:** The augmentation used to turn the recurrent
  stick-breaking logistic transition into Gaussian natural statistics for its
  weights and offsets.
- **Posterior axis:** An array dimension whose entries hold separate conjugate
  posterior coordinates, such as one dynamics posterior per selector state.
- **Responsibility:** The soft posterior probability that one selector state
  generated a switching-factor observation.
- **Rollover:** Conversion of posterior hyperparameters into point parameters
  used by the next inference run.
- **Source factor:** A factor instance whose local belief contributes sufficient
  statistics to a declared learnable.
- **Tied blobs:** Serialized parameter arrays shared by several factor
  instances so all instances denote one parameter group.

## Notes

**2026-08-13T04:07:32Z**

Planning and ambiguity handoff, 2026-08-12. Applicable change path: Learnable or checkpoint feature. cffg.learn owns family and cadence behavior; cffg.conj owns posterior algebra; Engine.gaussian_node_stats and Engine.recurrence_weight_stats own source records; compiler addresses and set_params own writes. Inference messages, schedules, objective terms, structure hashes, and model-specific fit cadence are explicit N/A surfaces except for parity checks. Flooding and scheduled statistic readouts are compared; learning remains host-side, so a second JIT learning implementation is N/A; current and legacy checkpoint behavior is required; single-instance/vmap learning is N/A unless the implementation adds a batched public contract.
