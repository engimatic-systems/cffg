---
id: cff-fw2v
status: closed
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

## Notes

**2026-11-26T21:13:28Z**

Shipped structured model-conditioned simulator generation, editable source and a pinned no-network, read-only Docker/Colima NumPy runner. The live provider authored switching sensor dynamics against DATA ports y0,y1,y2. Two isolated runs at seed 7 produced equal observations plus hidden position and regime, and subsequent profile and replay had identical posterior beliefs. Production job e784def0b24042488f2c67895872ff83 preserves source and contract. Tape 8c9c534f246149e19f93989c90eeec81 can be reused from Runs. The model remains unchanged when generation, binding or execution fails.
