"""Repo hygiene: the four-canonical-documents rule from AGENTS.md, enforced by machine.

Project state lives in tickets, durable knowledge lives in docstrings, and
executable claims live in tests. Markdown files multiplying in the repo is the
first symptom of context leaking out of that system — so this test fails the
build the moment a fifth document appears.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CANONICAL_DOCS = {
    "AGENTS.md",
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/LIMITATIONS.md",
}

# Directories whose markdown is none of our business.
EXEMPT_DIRS = {
    ".git",
    ".venv",
    ".tickets",  # tk stores tickets as markdown; tickets are the ONE sanctioned pile
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
        "Per AGENTS.md document policy there are exactly four canonical docs, "
        "updated in place. Session state belongs in tickets (tk add-note), "
        "research findings in the ticket that needed them, durable knowledge "
        "in docstrings, claims in tests. Delete the file and relocate its "
        "contents — or, if this is a genuine fifth pillar of the project, "
        "amend AGENTS.md and this allowlist in the same commit and be ready "
        "to defend that in review."
    )


def test_canonical_docs_exist():
    """The rule cuts both ways: the four documents must actually be there."""
    missing = sorted(d for d in CANONICAL_DOCS if not (REPO / d).is_file())
    assert not missing, f"Canonical documents missing: {missing}"
