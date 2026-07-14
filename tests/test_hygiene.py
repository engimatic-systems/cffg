"""Repo hygiene: the canonical-documents rule from AGENTS.md, enforced by machine.

Project state lives in tickets, durable knowledge lives in docstrings, and
executable claims live in tests. Markdown files multiplying in the repo is the
first symptom of context leaking out of that system — so this test fails the
build the moment an unsanctioned document appears.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CANONICAL_DOCS = {
    "AGENTS.md",
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/LIMITATIONS.md",
    "docs/DERIVATIONS.md",  # formal derivations + plain-English companions (cff-fk2u)
    "docs/READING.md",  # curated external reading list (Cooper, 2026-07-12)
}

# Directories whose markdown is none of our business.
EXEMPT_DIRS = {
    ".git",
    ".venv",
    ".tickets",  # tk stores tickets as markdown; tickets are the sanctioned pile
    "todo-notes",  # docs/todo-notes: one note per remaining thread, deleted on landing
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}


def test_no_contraband_markdown():
    found = {
        str(p.relative_to(REPO))
        for p in REPO.rglob("*.md")
        if not (set(p.relative_to(REPO).parts[:-1]) & EXEMPT_DIRS)
    }
    contraband = sorted(found - CANONICAL_DOCS)
    assert not contraband, (
        f"Contraband markdown detected: {contraband}. "
        "Per AGENTS.md document policy there is a fixed allowlist of canonical docs, "
        "updated in place. Session state belongs in tickets (tk add-note), "
        "research findings in the ticket that needed them, durable knowledge "
        "in docstrings, claims in tests. Delete the file and relocate its "
        "contents — or, if this is a genuine new pillar of the project, "
        "amend AGENTS.md and this allowlist in the same commit and be ready "
        "to defend that in review."
    )


def test_canonical_docs_exist():
    """The rule cuts both ways: the canonical documents must actually be there."""
    missing = sorted(d for d in CANONICAL_DOCS if not (REPO / d).is_file())
    assert not missing, f"Canonical documents missing: {missing}"


def test_todo_notes_name_their_ticket():
    """A todo-note without a ticket is a wish, not a plan: every note in
    docs/todo-notes must reference a tk ticket id (or say why not)."""
    import re

    for note in (REPO / "docs" / "todo-notes").glob("*.md"):
        text = note.read_text()
        assert re.search(r"cff-\w+|Ticket:\*?\*? *none", text), (
            f"{note.name}: no ticket reference — file one or state 'Ticket: none yet'"
        )
