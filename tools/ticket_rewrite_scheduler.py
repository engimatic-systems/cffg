#!/usr/bin/env python3
"""Select dependency-safe tickets for the intelligible-ticket rewrite campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
ACTIVE_STATUSES = {"open", "in_progress"}
REQUIRED_HEADINGS = (
    "## Background in Plain English",
    "## Outcomes / Acceptance Criteria",
    "## Methodological Suggestions",
    "## Future work",
    "## Definitions",
)
REVIEW_MARKER = "Intelligible-ticket review: complete."


class SchedulerError(ValueError):
    """The ticket graph or assignment request is inconsistent."""


def run_lines(command: list[str], *, repo: Path = REPO) -> list[str]:
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SchedulerError(f"{' '.join(command)} failed: {detail}")
    return result.stdout.splitlines()


def load_tickets(*, repo: Path = REPO) -> dict[str, dict[str, Any]]:
    tickets: dict[str, dict[str, Any]] = {}
    for line in run_lines(["tk", "query"], repo=repo):
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("status") not in ACTIVE_STATUSES:
            continue
        ticket_id = value.get("id")
        if not isinstance(ticket_id, str):
            raise SchedulerError("tk query returned a ticket without a string id")
        tickets[ticket_id] = value
    return tickets


def committed_ticket_text(ticket_id: str, *, repo: Path = REPO) -> str:
    path = f".tickets/{ticket_id}.md"
    return "\n".join(run_lines(["git", "show", f"HEAD:{path}"], repo=repo))


def has_review(text: str) -> bool:
    positions = [text.find(heading) for heading in REQUIRED_HEADINGS]
    return all(position >= 0 for position in positions) and positions == sorted(positions) and (
        REVIEW_MARKER in text
    )


def reviewed_ticket_ids(
    tickets: dict[str, dict[str, Any]], *, repo: Path = REPO
) -> set[str]:
    reviewed = set()
    for ticket_id in tickets:
        try:
            text = committed_ticket_text(ticket_id, repo=repo)
        except SchedulerError:
            continue
        if has_review(text):
            reviewed.add(ticket_id)
    return reviewed


def direct_dependents(tickets: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    result = {ticket_id: set() for ticket_id in tickets}
    for dependent_id, ticket in tickets.items():
        for prerequisite_id in ticket.get("deps", []):
            if prerequisite_id in result:
                result[prerequisite_id].add(dependent_id)
    return result


def schedule(
    tickets: dict[str, dict[str, Any]],
    reviewed: set[str],
    *,
    controller: str,
    assigned: set[str] | None = None,
    last_ticket: str | None = None,
) -> dict[str, Any]:
    assigned = assigned or set()
    active = set(tickets)
    if controller not in active:
        raise SchedulerError(f"controller is not active: {controller}")
    unknown_reviewed = reviewed - active
    if unknown_reviewed:
        raise SchedulerError(f"reviewed tickets are not active: {sorted(unknown_reviewed)}")

    remaining = active - reviewed - {controller}
    invalid_assigned = assigned - remaining
    if invalid_assigned:
        raise SchedulerError(
            f"assigned tickets are not unreviewed campaign tickets: {sorted(invalid_assigned)}"
        )
    if last_ticket is not None and last_ticket not in active:
        raise SchedulerError(f"last ticket is not active: {last_ticket}")

    dependents = direct_dependents(tickets)
    eligible_ids = {
        ticket_id
        for ticket_id in remaining - assigned
        if not (dependents[ticket_id] & remaining)
    }
    continuation_ids = set()
    if last_ticket is not None:
        continuation_ids = set(tickets[last_ticket].get("deps", [])) & eligible_ids

    def sort_key(ticket_id: str) -> tuple[int, int, str]:
        priority = int(tickets[ticket_id].get("priority", 4))
        return (ticket_id not in continuation_ids, priority, ticket_id)

    eligible = []
    for ticket_id in sorted(eligible_ids, key=sort_key):
        eligible.append(
            {
                "id": ticket_id,
                "priority": int(tickets[ticket_id].get("priority", 4)),
                "continues_from_last_ticket": ticket_id in continuation_ids,
                "active_dependencies": sorted(set(tickets[ticket_id].get("deps", [])) & active),
                "active_dependents": sorted(dependents[ticket_id]),
            }
        )

    blocked = []
    for ticket_id in sorted(remaining - assigned - eligible_ids):
        blocked.append(
            {
                "id": ticket_id,
                "waiting_on_dependents": sorted(dependents[ticket_id] & remaining),
            }
        )

    return {
        "schema_version": 1,
        "controller": controller,
        "last_ticket": last_ticket,
        "summary": {
            "active": len(active),
            "reviewed": len(reviewed - {controller}),
            "remaining": len(remaining),
            "assigned": len(assigned),
            "eligible": len(eligible),
        },
        "assigned": sorted(assigned),
        "eligible": eligible,
        "blocked": blocked,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--controller", default="cff-twzc")
    result.add_argument("--assigned", action="append", default=[])
    result.add_argument("--last-ticket")
    result.add_argument("--format", choices=("yaml", "json"), default="yaml")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        tickets = load_tickets()
        output = schedule(
            tickets,
            reviewed_ticket_ids(tickets),
            controller=args.controller,
            assigned=set(args.assigned),
            last_ticket=args.last_ticket,
        )
    except (json.JSONDecodeError, OSError, SchedulerError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        yaml.safe_dump(output, sys.stdout, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
