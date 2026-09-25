---
id: cff-0kyl
status: in_progress
deps: []
links: []
created: 2026-09-25T03:46:25Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Render every disconnected graph component and clarify SVG entry points

## Background in Plain English

The supplied visualization script renders one Gaussian component but raises KeyError for two disconnected components because placement visits only one component. A separate newcomer mistake passes Template to to_svg, which expects an already laid-out Scene and currently fails with an unrelated attribute error.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. Every factor in a disconnected valid template is placed exactly once and rendering does not raise a missing-placement error.

2. One, two and many mixed-size components produce deterministic non-overlapping component layouts while connected-graph behavior remains covered.

3. Empty and isolated valid diagram cases have defined behavior.

4. render_svg(Template) and to_svg(Scene) are explained in the public example; wrong input types fail with an actionable error.

5. Focused layout/rendering tests cover the supplied minimal reproduction without depending on external files.

## Methodological Suggestions

Inspect viz/layout.py's rail traversal and pack component bounding boxes. No exact change-path row: visualization layout/render API is authority; IR/inference/readouts/persistence unchanged. Pairs host/JIT, flooding/scheduled, single/vmap, checkpoint all N/A. Sparse support/numeric dtype N/A because this changes scene layout, not inference; large graph means many components.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

No additional future work is identified within this ticket; the completed outcome contributes to epic cff-xa6z.

## Definitions

SVG: scalable vector graphics. Template: serialized model structure. Scene: positioned drawing objects. Connected component: a maximal group of nodes joined by wires.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-09-25T06:53:12Z**

Pre-implementation handoff: layout.py visits only one rail-connected component; every missing factor later fails side assignment. No exact architecture change-path row exists: use Diagram rendering section and viz layout/SVG as authority. Inference, template persistence, readouts and numerical semantics unchanged; host/JIT, flood/scheduled, single/vmap and current/legacy checkpoint pairs all N/A. Plan deterministic component extraction, per-component existing rail layout, and vertical packing of full framed component scenes; preserve single-component goldens. Empty direct layout returns an empty framed scene. Validate multi-size/many components and entry-point type errors, then one full gate.

**2026-09-25T06:54:48Z**

Implemented independent component scenes ordered by natural factor name, packed vertically by complete framed extents; connected component rail/sides/geometry are unchanged. Empty direct layout yields the defined empty frame. Template/Scene SVG entry points now raise actionable TypeErrors and README shows both routes. Focused final check: 52 visualization tests passed, including all connected SVG goldens, 1/2/30 Gaussian components at levels 0/4, mixed-size GFE composite fragments, empty layout, and wrong-input messages. Full gate next.
