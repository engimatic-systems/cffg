---
id: cff-7zq1
status: in_progress
deps: [cff-9cx5]
links: []
created: 2026-08-18T17:46:33Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, preregistration]
---
# Build the deterministic schema-v5 dual freeze

## Background in Plain English

`cff-huxt` must freeze the measured calibration before any final dataset is
generated. The calibration is still running, but the code that validates its
artifact and builds the registration does not need the measured values in
advance. It can be implemented and tested against registered schema-v4
fixtures now.

The existing `src/cffg_nascar_validation/freeze.py` owns the superseded
one-family registration and must remain byte-for-byte unchanged. This ticket
adds a separate dual-family freeze path. It
turns one validated schema-v4 calibration artifact into the exact schema-v5
fixed-dataset protocol prepared by `cff-9cx5`, records every required input and
code identity, and writes a deterministic registration plus manifest. It does
not run the live calibration, choose values, write the real registration hash,
or generate a final dataset.

## Outcomes / Acceptance Criteria

1. A separate dual-freeze command consumes a complete manifested schema-v4
   calibration artifact and produces one schema-v5 registration document.
2. The command reconstructs the registered schema-v4 protocol from raw
   calibration inputs and rejects a changed, missing, incomplete, non-finite,
   unmanifested, or not-ready input before writing output.
3. The final protocol is derived only through the schema-v5 derivation and
   validation functions from `cff-9cx5`.
   3.1. No generator, scale, split, final seed, threshold, budget, arm, control,
   grid, metric, or stochastic coordinate changes during the freeze.
   3.2. The final decision remains three fixed datasets per family with no
   interval, alpha level, family unlock, population inference, or joint pass.
4. The registration records and validates the authorized experiment and CFFG
   commits, both family generator descriptors, mapping version, adapter
   provenance, main and legacy lockfiles, calibration artifact, scale artifact,
   source-design artifact, and every evaluation-grid digest.
5. The registration records the exact hashes of
   `artifacts/rslds-preregistration/preregistration.json` and its manifest as
   superseded and unexecuted without modifying either historical file.
6. Equal scientific inputs, explicit registration date, and authorized commits
   produce byte-identical registration content and content hashes.
7. The generated manifest covers every file written by the dual freeze and
   validates from its artifact directory.
8. The registration names `cff-oyie` as the only final owner, preserves
   append-only family/seed/replicate checkpoint paths and retained posterior
   locations, and points to a manifested `final-command.txt`.
   8.1. `final-command.txt` contains the exact final command with the completed
   registration file's SHA-256. Keeping that command outside the registration
   avoids asking a file to contain its own final hash.
9. Focused tests reject tampered calibration, source-design, mapping, lock,
   registration, and manifest inputs, and prove that neither a final generator
   nor a fitting arm is called.
10. The historical one-family freeze module and committed artifact retain byte
    identity relative to the `cff-9cx5` base commit `361199e`, and their checks
    in `tests/test_rslds_preregistration.py` pass.
11. No registered final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the numbered outcomes.

- Add `dual_freeze.py` and a new command instead of changing historical
  `freeze.py`.
- Accept the registration date explicitly so repeated builds are deterministic.
- Store repository-relative paths for committed inputs and pair each path with
  its SHA-256.
- Build the document in memory, validate it, then write the registration and
  manifest as the last two operations.
- Inject repository state in tests and use only calibration seeds in fixtures.

## Future work

`cff-huxt` runs this command on the complete `cff-s6rs` artifact, commits the
real registration and manifest, records their hashes in its ticket and PR, and
stops before every final seed. `cff-oyie` then runs the exact authorized final
command. No other future work is deferred here.

## Definitions

- **Authorized commit:** The exact experiment or CFFG Git revision permitted
  to execute the frozen protocol.
- **Canonical registration content:** UTF-8 JSON encoded with sorted keys and
  compact separators, independent of display indentation.
- **Content hash:** A SHA-256 over the canonical registration content.
- **Deterministic freeze:** The same explicit inputs produce identical bytes
  and hashes.
- **Registration document:** The metadata artifact that fixes the experiment
  protocol and provenance. It is not a compiler or runtime dispatch registry.
- **Resolved final command:** A manifested command file written only after the
  registration bytes and their SHA-256 are fixed.
- **Schema v4:** The immutable measured calibration contract.
- **Schema v5:** The final-only fixed-dataset protocol derived from schema v4.

## Notes

Pre-implementation handoff: the `docs/ARCHITECTURE.md` change-path index is
N/A. This ticket changes the sibling validation harness only and consumes the
unchanged serialized `LearnableSpec` fitting path. The experiment README may
gain the dual-freeze command; the CFFG rSLDS result sections in `README.md`,
`docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` remain `cff-nkbj` work.

Duplicated execution surfaces checked before implementation:

- schema-v4 calibration validation and schema-v5 final derivation;
- both family descriptors, scales, grids, final identities, streams, tiers,
  controls, diagnostics, primary effects, and independent decisions;
- calibration, scale, source-design, adapter, mapping, lock, historical-freeze,
  registration, and manifest provenance;
- final command, checkpoint, and retained-posterior path templates;
- CFFG compiler, VM, schedule, kernels, ledger, factors, and serialization,
  which are N/A because this command performs no inference.

Hostile-input decisions:

- Any missing, non-finite, incomplete, not-ready, or unmanifested calibration
  input fails before output is written.
- A changed path, hash, commit, mapping, lockfile, family identity, or stream
  fails validation rather than receiving a fallback value.
- Sparse categorical support, improper beliefs, float32, and uncomfortable
  inference scales are N/A because the freeze performs no numerical inference;
  it preserves their already validated protocol settings byte for byte.
- An existing output is never overwritten. A different freeze requires a new
  empty artifact directory.

Legend: prerequisite --> dependent

```text
[x] cff-9cx5 schema-v5 runner
          |
          +--> [ ] cff-7zq1 deterministic freeze builder (ready)
                         |
[ ] cff-s6rs calibration +--> [ ] cff-huxt measured immutable freeze
                                          |
                                          +--> [ ] cff-oyie final run
                                                       |
                                                       +--> [ ] cff-nkbj
```
