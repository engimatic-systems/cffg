---
id: ccpr-ojza
status: in_progress
deps: [ccpr-rl0k, cff-d56d, cff-9xmq, cff-b2fz]
links: []
created: 2026-11-28T05:57:47Z
type: feature
priority: 3
assignee: coopwilliams
parent: cff-6s31
tags: [plates, compiler]
---
# Compile one-level plates into the existing inference arrays

## Background in Plain English

Today a 64-by-64 grid makes the compiler construct thousands of temporary Python models before inference can start. The existing expansion is a reference for what each connection means. The compiler can write the same factor groups and message addresses from the repeated fragment.

## Outcomes / Acceptance Criteria

1. Compile one-level plates through the normal compiler and Colony entry points without materializing a flat Template or allocating copied variable, factor, or port records. Check both cold construction and parameter-only reuse.
2. Match the public expansion for Chain, Expose and the balanced Broadcast tree shipped in cff-9xmq. Preserve scoped tied and per-copy parameters, paths, constraints, learned ownership and collision diagnostics.
3. Match the parent's named slot tables, group order, schedule, arrays, observations, energy traces and decisions for its grid, mixture, sensor, resize and chain fixtures. Cover sum and max execution where supported, single and batched runs, source precision and exact support.
4. Preserve the runtime distinction between an impossible observation, a divergent numerical calculation, an exhausted iteration budget and a converged result.
5. Publish phase timings and allocation counts for the 64-by-64 fixture. The parent retains the five-run 1.5-second median target and cross-feature release checks.

## Methodological Suggestions

Generate the same logical names and slot ordering that independent expansion produces. Reuse existing inference rules and run the live model through the public API.

## Future work

Recursive compilation and the parent's resource qualification consume this path.

## Definitions

A slot is one allocated message address. Broadcast connects copied ports to one shared hub. Expose keeps a copied port available to the caller.

## Notes

**2026-11-28T06:47:05Z**

Base is the frozen row-major index plus the published Colony parameter-reuse handoff. Architecture row: compiler and parameter packing. Normal compile and cold Colony construction need the same topology. Cached spawns rebind numeric owners, hard constraints and schedule execution consume the same compiled addresses, and stored models remain the compact source. Use the existing flat expansion as an independent semantic oracle, including aliases, collision handling and balanced equality trees. Measure the full 64-by-64 public compile and allocations. Phase-1 source semantics, exact support and old bundles are release checks.
