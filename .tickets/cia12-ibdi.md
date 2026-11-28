---
id: cia12-ibdi
status: closed
deps: []
links: []
created: 2026-11-28T03:11:02Z
type: task
priority: 1
assignee: coopwilliams
external-ref: supersedes-pr-140
tags: [testing, performance]
---
# Reduce the required short test suite to under ten minutes

## Background in Plain English — WHY
Every library change waits for a required test run before it can land. Cooper wants that run below ten minutes and wants further savings wherever the risk is small. The older PR #140 conflicts with current main. Its shared checkpoint and Kalman fixtures produced real savings on an older test collection, but its final required gate still took 16 minutes 41 seconds. A faster developer-only selection would leave the wait to merge in place.

## Outcomes / Acceptance Criteria — WHAT
1. Measure the end-to-end wall time of the required dashboard command on the project machine with the usual CPU and cache settings. Record the revision, selection, test counts, process configuration, peak memory if using parallel workers, and any competing jobs. Use comparable before and after measurements from required runs when available. Count collection, compilation and teardown. Report the actual result against 600 seconds, separately from estimates or focused-test time.
2. Aim below 600 seconds with headroom for normal variation. Prioritize low-risk reductions even after the first speedup. If the remaining savings require a serious coverage or resource tradeoff, record the measured safe result and ask Cooper about the specific tradeoff before removing protection.
3. Show which failure mechanisms the short gate catches before and after. Preserve independent numerical oracles, false-convergence and infeasibility detection, sparse support, float32, scale, mutation isolation, and distinct execution or persistence paths where relevant. Any moved case retains an executable home and an explanation of what still catches its failure. Verify that focused, partial or failed runs cannot masquerade as a complete pass.
4. Make the quicker workflow the one agents actually use. Align the project instructions, test configuration, dashboard and measured wait guidance. Reuse a passing run at unchanged code. Document the resulting required and opt-in coverage.
5. Shared validation machinery changes require the real short-suite gate. Preserve reproducible commands and evidence. Keep the ticket open if the time target is still outstanding unless Cooper explicitly accepts another outcome.

## Methodological Suggestions — HOW
Start with timings from the current integrated tree. Separate test setup and repeated JAX compilation from actual assertions and runner overhead. Check the old fixture reuse independently against current code. Prefer isolated changes that keep the same assertions. Consider bounded parallelism only if its memory use, determinism and CPU contention are measured. A shorter label or a fixed sleep estimate is no substitute for elapsed time.
Architecture handoff: test selection, execution and reporting are the affected surfaces. Audit pytest discovery, the dashboard, JUnit ingestion, and agent instructions together. Production Engine and Colony behavior, graph schemas and model serialization are outside scope unless a new decision expands it.

## Future work
A specific high-risk coverage reduction or hardware dependency belongs in a separate decision if the low-risk work cannot reach the target.

## Definitions
Wall time means elapsed time from starting the test command to its completion. A gate is the required validation for a change. A fixture is reusable test setup. An oracle supplies an independent expected answer. JUnit is the machine-readable test result.

## Notes

**2026-11-28T04:00:32Z**

Baseline on Apple M2 Max with 64 GiB: full dashboard command 1,250.18s at 04bc4c5, 2,837 passing; subsequent opt-in adjustments retained all 3,316 IDs. A representative sealed 116-case trial at the same source used 79.336s/33.744s/28.544s for 1/3/4 workers, with 5.12/6.02/5.98 GiB peak aggregate RSS and zero swapouts. Four workers merit a full qualification, not a suite-wide performance claim. Duration data are in ~/agent-share/axiom-integration-4RQ0bP/ and /tmp/cia12-ibdi-probe/measured/summary.json. The restored Kalman assertions and read-only hybrid fixture pass 24 focused tests in reverse order; no short case was removed in this ticket.

**2026-11-28T04:14:15Z**

Qualified e7cb35c on the same Apple M2 Max/64 GiB host with the actual required command. Outer elapsed time 431.397s (7m11s) vs 1,250.18s (20m50s), a 65.5% reduction. Inner pytest subprocess 431.246s, all 2,838 selected cases passed. Sampled aggregate peak process-tree RSS was 32.08 GiB, all pressure samples normal and zero swap use before/after. The serial reference and fallback remain available. The debugger can read the result on process exit. Forty-three focused dashboard/inventory checks passed. Integrity compares every selected node ID with its recorded JUnit outcome and fails closed for interrupted or narrowed runs. A separate comparison to the actual earlier collection verified 2,806 identical IDs plus two renamed/expanded passing dashboard cases, covering all 2,808 required prior/restored cases. The older HTML summary had flattened 23 class-qualified names. Raw, unrevised receipts are at ~/agent-share/short-suite-ten/gate-e7cb35cd13d4/ and ~/agent-share/axiom-integration-4RQ0bP/. One measured gate on this host; elapsed time on other machines remains unmeasured.
