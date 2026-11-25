---
id: chr-09in
status: closed
deps: []
links: []
created: 2026-11-25T07:23:58Z
type: bug
priority: 2
assignee: coopwilliams
parent: chr-vyg1
---
# Expose the Gaussian observation-role record through the compiler compatibility module

The integrated suite found a 21st compiled record, GaussianTapRole, introduced by exact latent conditioning. Existing compiler users import these frozen records through cffg.compiler, which must refer to the same class object owned by cffg.compiled. The new role was missing from that compatibility module.

## Design

Architecture change path: compiler facade and published values. Keep a single definition in cffg.compiled. The derived record and its arrays already exist. Numerical rules, host/JIT execution, Colony and persistence are unaffected by the alias.

## Acceptance Criteria

Export the existing record from the compiler facade and check its identity, immutability and public inventory alongside the other 20 records. Run the reproduced ownership failure, compiler and import regressions, and the integrated release suite.


## Notes

**2026-11-25T07:24:23Z**

At 8ec7ae2 the default suite passed 2,843 cases and failed only test_compatibility_exports_have_one_owner. Reproduced in isolation. The compiler now re-exports GaussianTapRole, and the ownership test checks that all 21 frozen dataclasses are also in the public export list. The focused compiled-value, compiler-contract and import suites passed all 88 tests in 3.57 s. Integrated rerun follows the test-budget review.

**2026-11-25T07:58:26Z**

Integrated gate f5b8ede passed all 2,751 default tests in 1,143.367 wall seconds. All 21 compiled records have a single immutable owner and have a public compiler alias.
