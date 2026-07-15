---
id: cff-4apw
status: open
deps: [cff-k9ws, cff-lquu]
links: []
created: 2026-07-15T01:45:57Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [hybrid, learning, rslds]
---
# Recurrence-weight conjugate learning via Pólya-Gamma augmentation

The continual-learning half of the recurrence link (cff-k9ws): given PG auxiliary variables, the posterior over stick-breaking weights (R, r) is Gaussian, so learning slots into the existing between-trial conjugate pattern (cffg.conj: accumulate node-belief sufficient statistics, fold into the conjugate posterior, roll expected natural parameters back as next-trial params — the D4/NIW/MNIW loop, never inside jit). Needs soft regime responsibilities and drive-edge moments from the hybrid node readouts (cff-lquu). Deliverable: a PGWeights posterior dataclass in cffg.conj + Engine stats readout for the recurrence factor, mirroring gaussian_node_stats.

## Acceptance Criteria

1. DERIVATIONS.md D-entry for the PG-augmented weight update (expected omega via tanh closed form; Gaussian posterior over R, r). 2. Between-trial loop recovers known recurrence weights on synthetic regime-switching data within documented tolerance (the analog of the 0.009-Frobenius switching-dynamics recovery). 3. Vague-prior constructor + improper-posterior loudness matching cffg.conj conventions. 4. Uses soft responsibilities (not hard assignment) once cff-lquu lands; hard-assignment fallback documented if sequenced first.

