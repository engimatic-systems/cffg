"""Checks for dependency-safe ticket rewrite scheduling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCHEDULER_PATH = Path(__file__).parents[1] / "tools" / "ticket_rewrite_scheduler.py"


def load_scheduler_module():
    spec = importlib.util.spec_from_file_location("cffg_ticket_rewrite_scheduler", SCHEDULER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ticket(ticket_id: str, *, deps: list[str] | None = None, priority: int = 2) -> dict:
    return {
        "id": ticket_id,
        "status": "open",
        "deps": deps or [],
        "priority": str(priority),
    }


def graph() -> dict[str, dict]:
    return {
        "controller": ticket("controller", priority=1),
        "foundation": ticket("foundation"),
        "shared": ticket("shared"),
        "left": ticket("left", deps=["shared"]),
        "right": ticket("right", deps=["shared"]),
        "milestone": ticket("milestone", deps=["foundation"]),
    }


def test_has_review_requires_ordered_sections_and_completion_marker():
    scheduler = load_scheduler_module()
    text = "\n\n".join((*scheduler.REQUIRED_HEADINGS, scheduler.REVIEW_MARKER))

    assert scheduler.has_review(text)
    assert not scheduler.has_review(text.replace(scheduler.REVIEW_MARKER, "review pending"))
    embedded = "\n\n".join(scheduler.REQUIRED_HEADINGS)
    assert not scheduler.has_review(f"{embedded} embedded {scheduler.REVIEW_MARKER}")
    assert not scheduler.has_review(
        "\n\n".join(
            (
                scheduler.REQUIRED_HEADINGS[1],
                scheduler.REQUIRED_HEADINGS[0],
                *scheduler.REQUIRED_HEADINGS[2:],
                scheduler.REVIEW_MARKER,
            )
        )
    )


def test_initial_frontier_contains_only_terminal_tickets():
    scheduler = load_scheduler_module()

    result = scheduler.schedule(graph(), set(), controller="controller")

    assert [item["id"] for item in result["eligible"]] == ["left", "milestone", "right"]
    blocked = {item["id"]: item["waiting_on_dependents"] for item in result["blocked"]}
    assert blocked == {
        "foundation": ["milestone"],
        "shared": ["left", "right"],
    }


def test_fan_in_waits_for_every_direct_dependent():
    scheduler = load_scheduler_module()

    one_branch = scheduler.schedule(graph(), {"left"}, controller="controller")
    both_branches = scheduler.schedule(graph(), {"left", "right"}, controller="controller")

    assert "shared" not in {item["id"] for item in one_branch["eligible"]}
    assert "shared" in {item["id"] for item in both_branches["eligible"]}


def test_continuation_is_preferred_and_assignments_are_excluded():
    scheduler = load_scheduler_module()

    result = scheduler.schedule(
        graph(),
        {"milestone"},
        controller="controller",
        assigned={"left"},
        last_ticket="milestone",
    )

    assert result["eligible"][0]["id"] == "foundation"
    assert result["eligible"][0]["continues_from_last_ticket"]
    assert "left" not in {item["id"] for item in result["eligible"]}


def test_rejects_assignment_of_reviewed_ticket():
    scheduler = load_scheduler_module()

    with pytest.raises(scheduler.SchedulerError, match="not unreviewed"):
        scheduler.schedule(
            graph(),
            {"left"},
            controller="controller",
            assigned={"left"},
        )


def test_closed_controller_allows_a_completed_campaign_summary():
    scheduler = load_scheduler_module()
    tickets = graph()
    tickets.pop("controller")

    result = scheduler.schedule(tickets, set(tickets), controller="controller")

    assert result["summary"] == {
        "active": 5,
        "reviewed": 5,
        "remaining": 0,
        "assigned": 0,
        "eligible": 0,
    }
    assert result["eligible"] == []
    assert result["blocked"] == []
