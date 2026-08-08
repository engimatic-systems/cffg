"""Regression checks for the live ticket dashboard in ~/agent-share."""

from __future__ import annotations

import importlib.util
from pathlib import Path


DASHBOARD_PATH = Path.home() / "agent-share" / "ticket_dashboard.py"


def load_dashboard_module():
    spec = importlib.util.spec_from_file_location("cffg_ticket_dashboard", DASHBOARD_PATH)
    assert spec and spec.loader, f"dashboard script missing: {DASHBOARD_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ticket(
    dashboard,
    ticket_id: str,
    title: str,
    priority: int,
    *,
    tags: list[str],
    parent: str | None = None,
) -> dict:
    return {
        "id": ticket_id,
        "status": "open",
        "deps": [],
        "links": [],
        "created": "2026-07-12T00:00:00Z",
        "type": "task",
        "priority": priority,
        "assignee": "",
        "parent": parent or "",
        "tags": tags,
        **dashboard.parse_body(f"# {title}\n"),
    }


def test_render_exposes_priority_sort_and_tag_controls():
    dashboard = load_dashboard_module()
    tickets = {
        "cff-parent": ticket(dashboard, "cff-parent", "Parent", 2, tags=[]),
        "cff-child": ticket(
            dashboard,
            "cff-child",
            "Tagged child",
            0,
            tags=["Frontend"],
            parent="cff-parent",
        ),
        "cff-other": ticket(
            dashboard, "cff-other", "Other", 1, tags=["backend"]
        ),
    }
    dashboard.compute_relations(tickets)

    page = dashboard.render(tickets, "abc123")

    assert page.count('class="priority-pill ') == 3
    assert 'priority-p0" title="Priority 0" aria-label="Priority 0">P0' in page
    assert 'value="priority-high">priority: high' in page
    assert 'value="priority-low">priority: low' in page
    assert 'id="tagFilt"' in page
    assert 'data-v="frontend">Frontend</button>' in page
    assert 'data-v="backend">backend</button>' in page
    assert 'data-v="__untagged__">untagged</button>' in page

    # Tag data is machine-readable by the page's filterCards() implementation.
    assert 'data-tags="[&quot;frontend&quot;]"' in page
    assert "tags.includes(curTag)" in page
    assert "group.querySelectorAll('details.card')" in page
