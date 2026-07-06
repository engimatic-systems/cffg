---
id: cff-q3k7
status: closed
deps: []
links: []
created: 2026-07-06T19:48:28Z
type: task
priority: 3
assignee: coopwilliams
parent: cff-21ta
---
# Visualization: deterministic CFFG/model diagrams for experiment artifacts

Add a reusable library component that renders CFFG-style diagrams with deterministic layout, plus a script that walks experiments/ and creates a model diagram for each experiment artifact that lacks one.


## Notes

**2026-07-06T19:55:46Z**

Implemented deterministic SVG diagramming for CFFG templates in cffg.diagram, plus scripts/add_experiment_model_diagrams.py. The renderer draws blobs/weights, factors, variables, constraints, and ports with stable semantic lanes, natural ordering, fixed 1280x720 output, and an expanded 16:9 viewBox for larger models. Added simglucose policy-template adapter so controller experiment configs can be visualized as CFFG-style policy graphs. Generated model_diagram.svg for all 17 existing experiment artifact folders; rerunning the script skips existing diagrams unless --overwrite is passed. Verification: uv run pytest => 114 passed; uv run python tools/dashboard.py => 114 tests, 0 failing, 3 new.
