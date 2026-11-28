---
id: cei-k93f
status: open
deps: [cei-6lc5]
links: []
created: 2026-11-28T06:17:51Z
type: feature
priority: 2
assignee: coopwilliams
parent: cenc-luez
tags: [serialization, neural, inference]
---
# Bind the exact interval domain to serialized neural factors and graph inference

The exact interval algebra needs an ordinary model entry point with an explicit domain. Gaussian dimensions currently imply Gaussian messages; the new template must bind a separate interval domain without changing positive-noise neural semantics.

## Design

New IR field, new factor or rule family, and runtime numerical or status invariant. Carry independent coordinate dimension and static interval capacity through model declarations, compile, sources and bundles. Bind parameter vectors for input/output knots and positive slopes. Preserve complete cells and propagate typed failures through scalar/batch and flooding/exact routes.

## Acceptance Criteria

Round-trip a template and bundle and run the reconstructed model against the independent D13a tree oracle in f32/f64 at 1D/32D. Attribute -H(input) at the graph factor and both positive separator entropy corrections once. Check capacity failure, impossible/improper/numerical diagnostics, stable model/shape identity, and no aliasing to Gaussian inference. Document the supported graph boundaries and unchanged positive-noise behavior.

