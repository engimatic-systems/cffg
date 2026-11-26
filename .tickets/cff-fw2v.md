---
id: cff-fw2v
status: open
deps: [cff-djod]
links: []
created: 2026-11-26T19:44:49Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xikb
tags: [studio]
---
# Generate, inspect and replay observation simulators for a model

## Background in Plain English — WHY

Random values test the plumbing. A simulator lets Cooper test a graph against a specified stochastic process and replay the same observations after an edit.

## Outcomes / Acceptance Criteria — WHAT

1. A prompt plus a revision-pinned observation contract produces inspectable Python source and a machine-readable manifest through a configured language-model provider.
2. The runner checks port names, dimensions, dtypes and domains against actual engine binding. It separately retains optional ground truth. Running a source version twice with one seed reproduces its observation tape.
3. Generated code runs with limits inside a network-isolated worker. An invalid simulator or model binding reports a structured error and preserves the current graph.
4. A three-step noisy switching example can be generated, edited, run and compared between two model revisions. Agents can use the same job and event interface.

## Methodological Suggestions — HOW

Use a pinned Python/NumPy worker, serialized arrays, explicit random seeds and a simulator hash. Check float32, unsupported or missing data, sparse discrete support and long runs. Model execution delegates to the run protocol. Changes to inference schedule, vmap, JIT and checkpoint decoding are N/A for the generator.

## Future work

No further simulator work is required by this proposal.

## Definitions

An observation contract defines the data a model accepts. Ground truth means the simulator's hidden state, retained separately for evaluation.
