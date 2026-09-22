---
id: cff-rtuf
status: in_progress
deps: [cff-odag]
links: []
created: 2026-09-22T19:16:54Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-3c6f
---
# Move model workflows into tested standalone examples

## Background in Plain English — WHY

The README contains many long, model-specific code blocks. They prove that advanced features exist, but they make the main guide hard to scan and most are not complete standalone programs. The repository has no examples directory, so an experiment agent must copy fragments and invent the missing arrays, observations, and setup.

This ticket creates small runnable Python examples and moves the detailed model workflows out of the README. The examples become executable evidence for the capability table and leave the README free to explain how the pieces fit together.

## Outcomes / Acceptance Criteria — WHAT

1. A new examples/ directory contains self-contained Python programs for the major experiment families advertised by the current library.
2. At minimum, the directory covers the newcomer discrete agent, generic discrete rollout, linear Gaussian inference, exponential-family inference, hard-constraint or MAP inference, nonlinear or neural Gaussian inference, declared learning and bundle continuation, and hybrid or rSLDS usage.
3. Each example uses supported public imports, defines its own small inputs, rejects unfinished inference statuses, and prints or asserts the result an experiment should inspect.
4. Examples state their objective or approximation boundary in a short module docstring when the returned quantity is not exact evidence.
5. Runtime-heavy workflows use deliberately small fixtures and remain suitable for the default test budget. Long scientific benchmarks are linked or described but are not run as examples.
6. Automated tests execute the examples or their main functions and fail on stale imports, undefined names, unfinished status handling, or missing expected outputs.
7. The existing executable README quickstart test is migrated to the corresponding example without losing float32 and x64 coverage.
8. No engine, message-rule, numerical, serialization, or checkpoint behavior changes.
9. The ticket handoff lists which former README sections should become short links in the final README pass.

## Methodological Suggestions — HOW

These suggestions are optional. Another sound example layout is acceptable.

- Give every file a main() function and an if __name__ == "__main__" entry point.
- Favor one concept per file and deterministic seeds.
- Reuse small public model builders where they exist. Build a tiny schema-level model only when the builder itself is the capability being demonstrated.
- Test fast examples in-process when safe and use subprocesses when precision configuration or filesystem isolation matters.
- Applicable architecture change path: public example and executable-documentation path; no exact runtime row applies. Public builders and Engine are dependencies, not changed owners. Host/JIT and flooding/scheduled parity remain covered by existing engine tests; single/vmap and current/legacy checkpoint pairs are N/A unless an example explicitly claims them.

## Future work

- The downstream README ticket links every applicable capability-table row to these files and removes the superseded long walkthroughs.
- Full scientific benchmarks, notebooks, packaging, and hosted documentation are outside this ticket.
- No other future work is identified.

## Definitions

- Approximation boundary: The exact limit on what a reported objective or posterior is allowed to claim.
- Example fixture: A small set of model inputs chosen to demonstrate one workflow quickly and reproducibly.
- LGSSM: Linear Gaussian state-space model.
- MAP: Maximum a posteriori; the highest-scoring assignment under the model.
- Standalone program: A file that can run without names or data supplied by another document.
- Status gating: Refusing to use an inference result unless its status satisfies the capability contract.


## Notes

**2026-09-22T20:09:09Z**

Delegated extraction report

Question: Which model-specific README walkthroughs should become runnable files?
Answer: Move the newcomer agent workflow to examples/discrete_agent.py; generic finite-horizon discrete inference to discrete_rollout.py; LGSSM inference to linear_gaussian.py; conjugate exponential-family inference to exponential_family.py; hard constraints plus MAP to hard_constraint.py; nonlinear sigma-point inference to nonlinear_gaussian.py; declared learning and resumable bundles to learning_bundle.py; and hybrid/rSLDS inference to hybrid_rslds.py. The final README should replace those long walkthroughs with short explanations and direct links.
Source: Existing README workflow sections, the supported public builders and Engine readouts, and the eight executable files added by this ticket.
Confidence: High. All eight entry points reject unfinished inference; focused tests execute them, and the newcomer agent runs under both float32 and x64.
