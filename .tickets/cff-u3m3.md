---
id: cff-u3m3
status: open
deps: [cff-epdp]
links: []
created: 2026-08-13T04:26:16Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-uy48
tags: [usability, public-api, release]
---
# Stabilize one newcomer path from model build through checkpoint restore

## Background in Plain English

The repository has real model builders, serialization, compilation, inference,
objective diagnostics, a rolling agent, and checkpoint restore. A newcomer must
still learn deep module paths to combine them. The top-level `cffg` package
exports almost nothing and retains a scaffold `hello()` function. README
examples therefore demonstrate internal capability without defining one small
supported facade.

This ticket establishes the minimum public path required by the release gate.
It does not freeze every pre-1.0 Python object or redesign the modeling API.

## Outcomes / Acceptance Criteria

1. `src/cffg/__init__.py` removes the scaffold `hello()` function and its stale
   export.
2. The top-level package exposes a deliberately small documented facade for
   the quickstart's model construction, compilation, inference, status,
   inspection, agent, and checkpoint operations.
3. Each top-level export has one canonical implementation owner. The facade
   re-exports it and does not add a second behavior implementation.
4. README contains one newcomer path that uses supported imports and performs
   these operations in order:
   4.1. build a serialized model template and parameter arrays;
   4.2. compile the template;
   4.3. initialize and bind data;
   4.4. run inference and require a successful status;
   4.5. inspect at least one posterior and the correctly named objective;
   4.6. construct the corresponding rolling agent;
   4.7. perform one environment step;
   4.8. write a checkpoint;
   4.9. restore into a fresh agent; and
   4.10. continue with the same belief, learned state, cadence, and random
   sequence.
5. An executable documentation test runs the complete path in a temporary
   directory using only the imports shown to the newcomer.
6. The example states whether its objective is exact BFE, exact GFE for a named
   discrete composite, or a declared bound. It does not label every
   `Engine.bfe` result exact.
7. The example fails loudly on a nonconverged result instead of inspecting or
   acting on it.
8. Checkpoint restoration either reproduces the documented continuation or
   raises a specific version, structure, or shape error. It cannot silently
   discard a known field.
9. The public facade policy states which names are supported before 1.0 and
   which deeper extension interfaces remain internal-unstable.
10. README and architecture descriptions of edit operations, bundle APIs, and
    checkpoint formats name the APIs that actually exist. Planned high-level
    edit bundles are not written as shipped functions.
11. The example runs in float32-capable JAX configuration without relying on
    the test suite's forced x64 setting.
12. The default short suite and the executable newcomer test pass after the
    final code change.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Prefer re-exports of stable existing types over a new facade class.
- Use a small discrete builder so the quickstart is fast and its objective has
  an independent exact interpretation.
- Keep the executable example in an existing test or README-snippet authority;
  do not create a second tutorial document.
- Use the **New IR field** and **Learnable or checkpoint feature** checklists
  only for surfaces the facade actually exposes. The ticket should not change
  serialized structure merely to shorten imports.

## Future work

GUI selection-to-edit workflows, a general sensor transport, arbitrary online
optimization, and a frozen external-kernel Python API remain outside this
minimum path. Their existing tickets and documented limitations remain the
owners.

## Definitions

- **Facade:** A small supported set of imports that delegates to existing
  implementation owners.
- **Fresh agent:** A newly constructed agent with the same model declaration as
  the checkpoint source and no copied in-memory runtime state.
- **Public API:** A documented callable or type that the project intends users
  to import directly.
- **Scaffold function:** Placeholder code created by a project template and not
  part of the intended product.
