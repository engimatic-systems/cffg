---
id: car-k6oi
status: open
deps: [car-9ccf, cff-jilz]
links: [cff-ees0]
created: 2026-11-23T21:48:38Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom]
---
# Grow, merge and reuse mixture components in a fixed-capacity bank

## Background in Plain English — WHY

AXIOM reserves a fixed number of alternatives and activates them as observations arrive. A discarded interaction rule can release its row for later use. CFFG needs an explicit distinction between an unused slot, a learned component and a component with small probability so reuse never inherits stale evidence.

## Outcomes / Acceptance Criteria — WHAT

1. Offer a supported fixed-capacity bank with active mask, stable component references and per-family initialization. Activate a free slot at fixed compiled shape. A dormant slot contributes exactly zero assignment mass. Define results for empty/full banks and invalid covariance; return a caller-visible capacity outcome that preserves active components. Track if the reference implementation’s full-bank behavior would differ.
2. Permit a caller to propose a merge, evaluate the proposed snapshot, accept or discard it. Combine all affected statistics with the common prior counted once, reset the donor to the prior and update active references atomically. A rejected merge changes nothing. Recycling a donor cannot make an old reference alias an unrelated new component.
3. Compare birth, rejected/accepted merge and reuse against independent natural-parameter arithmetic with unequal component counts. Check continuous and categorical heads, exact zeros, deterministic tie order, float32/64 and banks of 32, 500 and 5,000 slots. Demonstrate that fixed-capacity operations preserve compilation structure.
4. Save and restore component identity, masks, sufficient statistics and the ordering/random state needed for replay. Verify one additional birth and merge after restart match uninterrupted operation. Exercise host/JIT and batched paths where supported and return explicit status for unsupported combinations.

## Methodological Suggestions — HOW

Reuse cff-enhs and cff-jilz. Use cff-du6m AC 2–15 and notes for the paper’s birth criterion and prior-subtracted merge arithmetic. Expose numerical primitives and transactions; the model supplies novelty triggers, candidate pairs and acceptance score. Review “Learnable or checkpoint feature” and any “New IR field” changes across validation, structure hash, bundles and legacy defaults. Message scheduling is N/A for a between-observation bank transaction.

## Future work

Searching arbitrary graph topologies and ranking rewritten models by held-out Bethe free energy belongs to cff-cynx. A fixed bank of prespecified parameter hypotheses is covered separately in cff-ees0.

## Definitions

Capacity is the maximum number of components that fit without resizing. Natural parameters encode a distribution so independent evidence can be added. A donor is the component whose evidence is moved and whose slot is then released.
