---
id: cff-akmp
status: open
deps: []
links: []
created: 2026-08-10T02:58:08Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, reference, reproducibility]
---
# Lock the recurrent-slds decision-list reference runtime

## Background in Plain English

The comparison in cff-nkbj needs a runnable copy of the authors' legacy
`recurrent-slds` code. The selected upstream commit does not install unchanged
on this machine because its dependencies are not locked and use APIs removed
from the Python 3.11/macOS arm64 stack used by this project. The same checkout also generates
NASCAR data with a decision-list tree while its active learner silently uses a
balanced tree by default. That mismatch must be removed before the learner can
serve as a reference.

This ticket creates a reproducible legacy runtime and a compact decision-list
baseline. It does not add source-specific slopes or change the legacy learning
equations. Keeping environment repair separate makes later semantic changes
easy to identify and review.

## Outcomes / Acceptance Criteria

1. The sibling NASCAR validation project records the exact `recurrent-slds`
   commit, Python version, package versions, source commits, build settings,
   platform, and architecture used by the reference runtime.
2. `uv` creates and runs the complete environment from committed inputs
   without calling `pip` directly or relying on an unrecorded global package.
3. Every compatibility shim is isolated, listed, and shown not to change a
   transition probability, conditional distribution, prior, or stopping rule.
4. A compact observed-only smoke run executes the pinned upstream
   recurrence-only learner without receiving generating states, regimes, or
   parameters.
5. The smoke run passes `decision_list(K)` explicitly to both generator and
   learner and verifies their leaf order and categorical normalization.
6. A regression test demonstrates that omitting the learner tree at the pinned
   commit selects the different balanced-tree default for `K=4`.
7. The smoke artifact records its command, seed, configuration, dependency
   lock hash, upstream and shim commits, input hashes, output hashes, runtime
   status, wall time, and peak resident memory.
8. The ticket changes no CFFG inference code and implements no source-specific
   recurrence adapter.
9. Sparse-message support, CFFG belief propriety, float32 message guards, and
   large CFFG factor shapes are not applicable because this ticket establishes
   an external runtime without adding a CFFG numerical rule. The smoke run
   still rejects non-finite reference output and records the reference dtype.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Put build and API compatibility patches in a separately versioned adapter
  layer rather than editing a downloaded checkout without a patch record.
- Keep an unmodified-default tree probe beside the forced-decision-list smoke
  so the historical mismatch remains executable evidence.
- Prefer a small deterministic dataset for the smoke test. Canonical-scale
  fitting belongs to cff-oyie.

## Future work

cff-64ri depends on this ticket and owns the source-specific paper-equation
Gibbs adapter. cff-0l31 owns the preregistered comparison protocol. Broad
modernization of the legacy project remains outside this chain.

## Definitions

- **Compatibility shim:** A small adaptation that replaces a removed build or
  library API without changing the statistical model.
- **Decision-list tree:** A sequence of binary gates that selects the first
  accepted destination and sends all rejections to the final destination.
- **Observed-only:** A fit that receives observations but not generating
  latent states, regimes, or parameters.
- **Reference runtime:** The complete pinned software environment used to run
  the comparison implementation.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: canonical=the
cff-oyie `T=10,000` scale; same=the one pinned checkout; implement=the external
runtime or the explicitly excluded adapter; support=literal probability
support; complete=every recorded environment component; remaining and
later=deliberately excluded modernization or ordered semantic work;
evidence=the executable tree-mismatch regression; exact=the named commit and
environment identifiers; pass=data flow of the explicit tree argument;
reproduce=rerunning the locked environment; sound=an alternative method that
satisfies the outcomes; test, validate, and verify=the named automated smoke,
regression, normalization, or provenance checks; fit and learn=observed-only
parameter estimation by the external learner; infer=the explicitly unchanged
CFFG inference layer; scale and small=canonical versus compact smoke model
size; can=the learner's ability to serve only after the mismatch is removed;
may=optional method; explicit_owner and named_ticket_handoff=the cff-64ri,
cff-0l31, and cff-oyie boundaries; all, every, and only=the decision-list
rejections, compatibility-shim set, or stated truth/scope exclusions; and
current was removed in favor of a pinned platform boundary. Capability owner:
cff-akmp owns only environment reproducibility and the forced-decision-list
baseline. Intelligible-ticket review: complete.

Classification-note additions: check=the named normalization or provenance
verification; estimate=external observed-only parameter estimation;
intended=the contextual meaning assignments in this note; current=the term
identified here as replaced by the pinned Python/platform boundary.
