---
id: cff-domz
status: closed
deps: []
links: []
created: 2026-08-01T02:19:45Z
type: chore
priority: 2
assignee: coopwilliams
tags: [workflow, github]
---
# Document private-repository fallback for the GitHub skill

## Plain English

Agents will know that the official GitHub skill connector cannot currently access this private repository. The repository instructions will direct them immediately to the authenticated gh fallback after the connector returns 404, avoiding misleading retries while retaining the existing credential-store procedure.

## Scope

Update the existing GitHub access rule in `AGENTS.md` in place. This is a
documentation-only workflow correction; no GitHub operation or runtime code
changes.

## Acceptance Criteria

1. The rule names `engimatic-systems/cffg`, the connector-specific 404 failure,
   and the fact that both PR reads and creation are affected.
2. Agents still honor connector-first skill instructions, but after this known
   repository-specific 404 they immediately use authenticated `gh` rather than
   retrying or treating the repository as absent.
3. The existing credential-store command remains the single documented token
   path. `tk dep cycle` and `git diff --check` pass.

## Notes

**2026-08-01T02:21:50Z**

Updated AGENTS.md rule 5 beside the existing credential-store instructions. It now records that the official GitHub skill connector returns 404 for engimatic-systems/cffg PR reads and creation, requires only the connector-first attempt mandated by the skill, and directs immediate authenticated gh fallback without treating the repository as absent. This is documentation-only; tk dep cycle and git diff --check pass, so the short suite is not required by the repository protocol.
