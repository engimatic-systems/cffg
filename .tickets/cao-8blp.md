---
id: cao-8blp
status: closed
deps: [cao-avog]
links: []
created: 2026-11-24T22:49:07Z
type: task
priority: 3
assignee: coopwilliams
tags: [performance, numerics, documentation]
---
# Benchmark rounded float32 and float64 learning and document engine speed tuning

## Background in Plain English — WHY

CFFG's online mixed learner normally keeps the small remainder lost when a model parameter is rounded. Calling `bank.as_rounded()` discards those remainders and can make learning faster. The cost and the accuracy loss depend on the workload and on whether parameters use float32 or float64. Users should be able to discover this choice from the class help and find measured guidance in the user guide.

## Outcomes / Acceptance Criteria — WHAT

1. A reproducible benchmark compares the default and rounded banks in both float32 and float64 on the same data. It reports absolute warm times and relative changes for updating with fixed assignments and for scoring plus updating. It identifies hardware, software, model size, records per update, repetitions, timing variability and which work is included.
2. For each precision, results quantify error against a known answer in both an ordinary stream and a stress case where small updates can disappear. Show the effect on a learned quantity, such as a count or variance, after repeated updates. Timing and accuracy examples give readers enough context to judge their own workload.
3. The public `MixedBank` and `RoundedBank` class documentation makes the tuning choice discoverable through Python help. The `as_rounded()` documentation explains when the switch can help, which corrections it discards and how to return to compensated learning. The class help directs readers to the performance guide.
4. The existing README user guide has a discoverable section on tuning CFFG for speed. It gives a runnable opt-in example, presents the four precision/mode results, and explains how to time a representative workload and check accuracy before choosing a setting. It scopes claims to the measured APIs and distinguishes compilation time from recurring work. Link existing engine tuning guidance where applicable.

## Methodological Suggestions — HOW

Reuse the public score and update workloads and precision probes from cao-avog and cff-xfnx. Existing measurements and scripts are in `~/agent-share/axiom-online-precision/rounded-option/`, including `speed.json`, `speed-float64.json` and `update-speed-float64.json`. Check their workloads and timing boundaries before combining results. A versioned reproduction command and small results table can make the guide usable without access to that directory.

A useful accuracy comparison holds input values and assignment weights fixed across modes. Run with float64 explicitly enabled for float64 measurements. Warming up compiled functions, alternating modes and waiting for the device to finish each measured block are ways to reduce timing bias.

The relevant architecture change path is the runtime numerical or status invariant. The observed public surfaces are mixed scoring and updating in default and rounded mode. Inference schedules and message rules are outside this documentation and benchmark task.

## Future work

No downstream implementation is required. Changing defaults or expanding rounded mode to other engine APIs would be separate decisions.

## Definitions

Float32 and float64 store each floating-point number in 32 or 64 bits. A bank stores the parameters for a set of mixture components. An assignment weight is the fraction of an observation counted toward a component. Compensated updates carry rounding remainders forward. Rounded updates store the nearest representable result at each step. Warm time measures recurring work after compilation.

## Notes

**2026-11-28T04:19:30Z**

Change path: runtime numerical or status invariant (public mixed-learning option and measurement only). Both PyTree representations use public score/update under a compiled scan. Flood versus scheduled inference, parameter dispatch and checkpoint changes are inapplicable. Reuse the retained seeded data and measure matched fixed assignments and score plus update. Check ordinary and high-count streams against exact sums of the represented weights.

**2026-11-28T04:34:03Z**

Ran tools/axiom/mixed_learning_speed.py float32 and float64 in separate CPU processes on the M2 Max. JAX/jaxlib 0.10.2, NumPy 2.5.1, one record/32 slots, d=6, 12 paired blocks. Rounded/compensated paired median time ratios for fixed updates and score+update were .589/.618 in f32 and .556/.556 in f64. Four quarter-count increments recovered +1 from 16 in every mode. At 2^22 (f32) or 2^51 (f64), compensated gained one and rounded gained zero. Raw timings and checks are in ~/agent-share/mixed-learning-speed-HQQA0I/{float32,float64}.json. These are workload-specific measurements. Updated public docstrings and README. Isolation: the benchmark lives under tools, the only package changes are class/method docstrings, and the executable AST of mixed_state.py matches a774a54 exactly. Public help and both benchmark invocations passed.
