---
id: cff-epdp
status: closed
deps: []
links: []
created: 2026-08-13T04:26:16Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-uy48
tags: [paper-fidelity, audit]
---
# Inventory paper constraints and Part II Figure 5 messages

## Background in Plain English

The repository faithfully implements several central equations from the CFFG
papers, but its public language can make the supported subset sound broader
than it is. The papers define a general constrained variational language. The
runtime implements a finite set of factor and constraint combinations. For
example, delta, naive mean-field, and selected P-substitution paths execute,
while moment matching is currently reserved in the intermediate
representation without a registered rule.

Part II Figure 5 also defines several distinct messages. Some are ordinary
conjugate messages, one state message needs a nonlinear stationary solve, and
`mu3(A)` is a nonstandard functional message. A behavioral T-maze result does
not by itself show that every local update and ledger term is wired correctly.

This ticket creates the review ledger for the release gate. It records what the
papers define, what the repository represents, what actually executes, and
which independent evidence supports each claim. It does not turn the papers'
general framework into a universal solver.

## Outcomes / Acceptance Criteria

1. `docs/ARCHITECTURE.md` contains one compact inventory of every constraint
   construction defined by Parts I and II.
   1.1. Each row cites the paper section, equation, figure, or appendix that
   defines the construction.
   1.2. Each row states how the construction appears in the repository's
   template, compiler plan, or implicit factorization contract.
   1.3. Each row classifies execution as exact, a named approximation, syntax
   only, or unsupported.
   1.4. Each executable row names its production rule, objective-ledger owner,
   and direct test authority.
   1.5. Each non-executable row names the compile-time rejection, lint result,
   or documented limitation that prevents accidental use.
2. The inventory distinguishes edge form constraints, node factorization
   constraints, marginalization and normalization requirements, moment
   matching, and P-substitution without collapsing them into one generic bead.
3. `docs/ARCHITECTURE.md` contains one source-ordered inventory of every
   message in Part II Figure 5.
   3.1. Each row names the source and destination variables and the paper's
   message label.
   3.2. Each row states whether the message is closed form, solved through a
   stationary coordinate, or represented as a functional density.
   3.3. Each row names the production implementation and its factor-local
   energy or shared ledger contribution.
   3.4. Each row names an independent analytic, enumeration, quadrature, or
   literature oracle.
4. Direct tests cover every Figure 5 message that the public documentation
   describes as executable.
   4.1. Tests compare the local outgoing message before end-to-end
   normalization can hide a compensating error.
   4.2. Tests compare the corresponding local energy or normalizer with an
   independent reference.
   4.3. Tests include literal zero support, a vacuous incoming message,
   float32, and a domain at least ten times the smallest analytic fixture when
   those cases are mathematically defined.
5. The inventory distinguishes the paper's general facing-node derivation from
   the finite set of registered executable combinations.
6. The inventory distinguishes the paper's literal changing-constraint trial
   model from the repository's two compiled-window realization.
7. The inventory distinguishes the analytic learned-A approximation used by
   the existing learning experiment from a trial loop that actually carries
   functional `mu3(A)` messages.
8. Any advertised paper-core row that lacks required production behavior or a
   direct oracle receives a named implementation ticket and becomes a
   dependency of `cff-uy48` before this ticket closes.
9. `README.md` and `docs/LIMITATIONS.md` use the inventory's bounded claim
   language for exact, approximate, syntax-only, and unsupported combinations.
10. No generated audit, complete paper extraction, or duplicate survey is
    added to the repository. The canonical papers remain in `notes/` and the
    inventory remains compact.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Start with Part I's constrained Bethe Lagrangian and Part II's Figure 5 and
  appendices. Use paper equation identifiers rather than relying on current
  function names to define the rows.
- Trace each executable row from `ConstraintSpec` or the implicit
  factorization through compiler plan, message rule, and `_make_bfe` owner.
- Reuse existing enumeration, simplex minimization, quadrature, and Julia
  reference fixtures. Add only missing local comparisons.
- Use the **New factor or rule family** change path only if the inventory finds
  missing executable evidence that can be added without changing semantics.
  Otherwise the runtime execution pairs are not changed by the inventory.

## Future work

`cff-in0q` consumes the Figure 5 inventory when it wires `mu3(A)` into the
agent lifecycle. `cff-iqqk` consumes the constraint inventory when it compares
the literal and two-window trial models. Any other missing advertised behavior
gets a separate dependency of `cff-uy48`; it is not silently absorbed here.

## Definitions

- **Bead:** A constraint annotation attached to a variable edge or factor node
  in CFFG notation.
- **Direct oracle:** A calculation independent of the production path that
  checks one local message, energy, or normalizer.
- **Functional message:** A message represented as a function of a parameter,
  rather than a member of a fixed conjugate family.
- **Syntax only:** A construction can be serialized but has no executable rule
  for the named factor and constraint combination.
- **Vacuous message:** A neutral incoming message that contributes no
  information.

## Notes

**2026-08-18T20:22:12Z**

Implementation handoff, 2026-08-18. Applicable change path: documentation/evidence inventory; the New factor or rule family row is N/A unless the inventory exposes missing advertised behavior. Checked duplicated execution pairs: host/JIT N/A, flooding/scheduled N/A, single-instance/vmap N/A, current/legacy checkpoint N/A. This ticket will trace existing IR, compiler, rule, ledger, and direct-test authorities without changing runtime semantics. Any silent or missing non-executable guard becomes a separate implementation ticket and a dependency of cff-uy48.

**2026-08-18T20:33:32Z**

Research distillation, 2026-08-18. Question: Which paper constraint constructions belong in the release ledger? Answer: default normalization/marginalization, joint/naive/structured factorization, edge/node form constraints, known-data and optimized delta forms, moment matching, P-substitution, and the trial-time planning-to-data change are distinct rows. Source: notes/pt1.txt sections 2.1 and 5.1-5.5, equations 9 and 26-32, figures 4-13; notes/pt2.txt sections 2.3-4.1 and 6.3, figures 3, 4, and 7. Confidence: high. Question: What are all Figure 5 updates? Answer: conjugate mu1(c), direct categorical mu2(z), indirect stationary/root-solved mu2(z), and functional mu3(A), with Ux=-zbar*rho and the parameter contribution -log Z. Source: notes/pt2.txt Figure 5 and Appendix C.2-C.6; docs/DERIVATIONS.md D6b-D6c. Confidence: high. Question: Which repository owners and gaps correspond? Answer: compiler plans and rules/engine ledger owners are now named in ARCHITECTURE; raw mu1/mu2 assertions were missing, moment matching was silently ignored, and incoherent mu3 support was floored. Source: targeted traces in src/cffg/{core,compiler,rules,engine}.py and tests/test_{gfe,goal_hyperprior,mu3,mu3_derivation}.py. Confidence: high. The two runtime gaps were repaired and closed under c1-a2cb and c1-2h8y, both dependencies of cff-uy48.

**2026-08-18T20:45:56Z**

Completed 2026-08-18. ARCHITECTURE now contains the source-ordered constraint and Figure 5 ledgers with exact/named-approximation/syntax-only/unsupported boundaries, production/ledger owners, direct authorities, and the literal-versus-two-window and analytic-versus-functional distinctions. README and LIMITATIONS use the same bounded claims. Direct raw mu1/mu2 and isolated energy assertions were added. Discovered silent moment-matching execution and incoherent mu3 support were repaired in closed dependencies c1-a2cb and c1-2h8y. Validation: focused Figure 5 selection 31 passed, 3 slow deselected; final default short suite 1369 passed, 39 deselected, 0 failed in 709.29 seconds; dashboard refreshed at /Users/ravd/agent-share/cffg-tests.html. No slow behavioral run was required because production behavior beyond the focused mu3 support regression did not change.
