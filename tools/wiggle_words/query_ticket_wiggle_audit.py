#!/usr/bin/env python3
"""Project bounded views of the ticket wiggle-word audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml


WIGGLE_WORDS_DIR = Path(__file__).resolve().parent
REPO = WIGGLE_WORDS_DIR.parent.parent
DEFAULT_AUDIT = WIGGLE_WORDS_DIR / "ticket-wiggle-word-audit-v2.yaml"
PROBLEM_MEANINGS = {"ambiguous", "unlisted_meaning", "pattern_false_positive"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def resolve_from_repo(path_value: str) -> Path:
    return REPO / path_value


def load_sources(audit_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    audit = load_yaml(audit_path)
    lexicon_path = resolve_from_repo(audit["provenance"]["lexicon"]["path"])
    return audit, load_yaml(lexicon_path), lexicon_path


def normalize_ticket(value: str) -> str:
    ticket = Path(value).name
    if ticket.endswith(".md"):
        ticket = ticket[:-3]
    if not ticket:
        raise ValueError("ticket id cannot be empty")
    return f".tickets/{ticket}.md"


def lexicon_by_id(lexicon: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["id"]: candidate for candidate in lexicon["candidates"]}


def source_block(audit_path: Path, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit": str(audit_path.relative_to(REPO)),
        "audit_schema_version": audit["schema_version"],
        "ticket_corpus_git_commit": audit["provenance"]["ticket_corpus"]["git_commit"],
        "lexicon": audit["provenance"]["lexicon"],
    }


def ticket_slice(
    audit_path: Path,
    ticket_value: str,
    candidate_id: str | None,
    problems_only: bool,
) -> dict[str, Any]:
    audit, lexicon, _ = load_sources(audit_path)
    meanings = lexicon_by_id(lexicon)
    ticket = normalize_ticket(ticket_value)
    selected: list[dict[str, Any]] = []

    for candidate in audit["candidates"]:
        if candidate_id is not None and candidate["id"] != candidate_id:
            continue
        occurrences = [item for item in candidate["occurrences"] if item["ticket"] == ticket]
        if problems_only:
            occurrences = [item for item in occurrences if item["meaning"] in PROBLEM_MEANINGS]
        if not occurrences:
            continue
        selected.append(
            {
                "id": candidate["id"],
                "category": candidate["category"],
                "pattern": candidate["pattern"],
                "listed_meanings": meanings[candidate["id"]]["meanings"],
                "ticket_usage_counts": dict(Counter(item["meaning"] for item in occurrences)),
                "occurrences": occurrences,
            }
        )

    occurrence_count = sum(len(candidate["occurrences"]) for candidate in selected)
    problem_count = sum(
        item["meaning"] in PROBLEM_MEANINGS
        for candidate in selected
        for item in candidate["occurrences"]
    )
    return {
        "schema_version": 1,
        "source": source_block(audit_path, audit),
        "query": {
            "mode": "slice",
            "ticket": ticket,
            "candidate": candidate_id,
            "problems_only": problems_only,
        },
        "summary": {
            "candidates": len(selected),
            "occurrences": occurrence_count,
            "problem_occurrences": problem_count,
        },
        "candidates": selected,
    }


def headings_by_line(lines: list[str]) -> list[str]:
    section = "Preamble"
    headings: list[str] = []
    for line in lines:
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            section = match.group(1)
        headings.append(section)
    return headings


def scan_ticket(
    audit_path: Path,
    ticket_value: str,
    candidate_id: str | None,
) -> dict[str, Any]:
    audit, lexicon, _ = load_sources(audit_path)
    ticket = normalize_ticket(ticket_value)
    ticket_path = REPO / ticket
    if not ticket_path.is_file():
        raise ValueError(f"ticket does not exist: {ticket}")

    lines = ticket_path.read_text(encoding="utf-8").splitlines()
    headings = headings_by_line(lines)
    selected: list[dict[str, Any]] = []
    for candidate in lexicon["candidates"]:
        if candidate_id is not None and candidate["id"] != candidate_id:
            continue
        pattern = re.compile(candidate["pattern"], re.IGNORECASE)
        occurrences = []
        for line_number, line in enumerate(lines, start=1):
            for match in pattern.finditer(line):
                occurrences.append(
                    {
                        "line": line_number,
                        "column": match.start() + 1,
                        "matched_text": match.group(0),
                        "section": headings[line_number - 1],
                        "line_text": line.strip(),
                    }
                )
        if occurrences:
            selected.append(
                {
                    "id": candidate["id"],
                    "category": candidate["category"],
                    "pattern": candidate["pattern"],
                    "listed_meanings": candidate["meanings"],
                    "occurrences": occurrences,
                }
            )

    return {
        "schema_version": 1,
        "source": source_block(audit_path, audit),
        "query": {"mode": "scan", "ticket": ticket, "candidate": candidate_id},
        "notice": [
            "This is a mechanical scan of the current ticket, not a contextual classification.",
            "Rewrite a match only when context does not select one meaning; "
            "intentional clear uses may remain.",
            "Check capability ownership separately because omitted ownership is lexically invisible.",
        ],
        "summary": {
            "candidates": len(selected),
            "occurrences": sum(len(candidate["occurrences"]) for candidate in selected),
        },
        "candidates": selected,
    }


def validate_audit(audit_path: Path) -> dict[str, Any]:
    audit, lexicon, lexicon_path = load_sources(audit_path)
    errors: list[str] = []
    audit_candidates = audit.get("candidates", [])
    lexicon_candidates = lexicon.get("candidates", [])
    audit_ids = [candidate.get("id") for candidate in audit_candidates]
    lexicon_ids = [candidate.get("id") for candidate in lexicon_candidates]
    lexicon_index = lexicon_by_id(lexicon)

    if audit.get("schema_version") != 2:
        errors.append("audit schema_version is not 2")
    if audit_ids != lexicon_ids:
        errors.append("audit candidate order or ids differ from the v2 lexicon")
    if len(audit_ids) != len(set(audit_ids)):
        errors.append("audit candidate ids are not unique")
    expected_hash = audit["provenance"]["lexicon"]["sha256"]
    actual_hash = hashlib.sha256(lexicon_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        errors.append("lexicon sha256 differs from audit provenance")

    total_occurrences = 0
    total_pairs = 0
    total_meanings: Counter[str] = Counter()
    for candidate in audit_candidates:
        candidate_id = candidate["id"]
        occurrences = candidate["occurrences"]
        raw_count = candidate["raw_count"]
        total_occurrences += len(occurrences)
        total_pairs += len(raw_count["by_ticket"])
        if candidate["pattern"] != lexicon_index[candidate_id]["pattern"]:
            errors.append(f"{candidate_id}: pattern differs from the v2 lexicon")
        if candidate["category"] != lexicon_index[candidate_id]["category"]:
            errors.append(f"{candidate_id}: category differs from the v2 lexicon")
        try:
            re.compile(candidate["pattern"], re.IGNORECASE)
        except re.error as exc:
            errors.append(f"{candidate_id}: pattern does not compile in the query tool: {exc}")
        if raw_count["occurrences"] != len(occurrences):
            errors.append(f"{candidate_id}: raw occurrence count does not reconcile")
        if raw_count["tickets"] != len(raw_count["by_ticket"]):
            errors.append(f"{candidate_id}: raw ticket count does not reconcile")
        ticket_counts = Counter(item["ticket"] for item in occurrences)
        if dict(ticket_counts) != raw_count["by_ticket"]:
            errors.append(f"{candidate_id}: by_ticket counts do not reconcile")
        usage_counts = Counter(item["meaning"] for item in occurrences)
        recorded_usage_counts = candidate["classification"]["usage_counts"]
        if any(usage_counts[meaning] != count for meaning, count in recorded_usage_counts.items()) or (
            set(usage_counts) - set(recorded_usage_counts)
        ):
            errors.append(f"{candidate_id}: usage counts do not reconcile")
        if candidate["classification"]["ambiguous_occurrences"] != usage_counts["ambiguous"]:
            errors.append(f"{candidate_id}: ambiguous occurrence count does not reconcile")
        ambiguous_tickets = sorted(
            {item["ticket"] for item in occurrences if item["meaning"] == "ambiguous"}
        )
        if candidate["classification"]["ambiguous_tickets"] != ambiguous_tickets:
            errors.append(f"{candidate_id}: ambiguous ticket list does not reconcile")
        listed_meanings = set(lexicon_index[candidate_id]["meanings"])
        for occurrence in occurrences:
            meaning = occurrence["meaning"]
            total_meanings[meaning] += 1
            if meaning not in listed_meanings | PROBLEM_MEANINGS:
                errors.append(f"{candidate_id}: unknown meaning {meaning!r}")
            has_observed = "observed_meaning" in occurrence
            expects_observed = meaning in {"unlisted_meaning", "pattern_false_positive"}
            if has_observed != expects_observed:
                errors.append(
                    f"{candidate_id}:{occurrence['ticket']}:{occurrence['line']}: "
                    "observed_meaning sentinel invariant failed"
                )

    summary = audit["summary"]
    expected_summary = {
        "candidates": len(audit_candidates),
        "raw_candidate_occurrences": total_occurrences,
        "candidate_ticket_pairs": total_pairs,
        "resolved_to_prelisted_meaning": total_occurrences
        - sum(total_meanings[meaning] for meaning in PROBLEM_MEANINGS),
        "ambiguous": total_meanings["ambiguous"],
        "unlisted_meaning": total_meanings["unlisted_meaning"],
        "pattern_false_positive": total_meanings["pattern_false_positive"],
    }
    for key, expected in expected_summary.items():
        if summary[key] != expected:
            errors.append(f"summary.{key} is {summary[key]!r}; expected {expected!r}")

    return {
        "schema_version": 1,
        "source": source_block(audit_path, audit),
        "valid": not errors,
        "checks": {
            "candidates": len(audit_candidates),
            "occurrences": total_occurrences,
            "candidate_ticket_pairs": total_pairs,
        },
        "errors": errors,
    }


def audit_summary(audit_path: Path) -> dict[str, Any]:
    audit = load_yaml(audit_path)
    return {
        "schema_version": 1,
        "source": source_block(audit_path, audit),
        "purpose": audit["purpose"],
        "method": audit["method"],
        "summary": audit["summary"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    result.add_argument("--format", choices=("yaml", "json"), default="yaml")
    commands = result.add_subparsers(dest="command", required=True)

    slice_parser = commands.add_parser("slice", help="project classified audit records for one ticket")
    slice_parser.add_argument("ticket")
    slice_parser.add_argument("--candidate")
    slice_parser.add_argument("--problems-only", action="store_true")

    scan_parser = commands.add_parser(
        "scan", help="mechanically scan one current ticket using the v2 lexicon"
    )
    scan_parser.add_argument("ticket")
    scan_parser.add_argument("--candidate")

    commands.add_parser("summary", help="print report-level metadata and counts")
    commands.add_parser("validate", help="check cross-file and count invariants")
    return result


def main() -> int:
    args = parser().parse_args()
    audit_path = args.audit.resolve()
    try:
        if args.command == "slice":
            output = ticket_slice(audit_path, args.ticket, args.candidate, args.problems_only)
        elif args.command == "scan":
            output = scan_ticket(audit_path, args.ticket, args.candidate)
        elif args.command == "summary":
            output = audit_summary(audit_path)
        else:
            output = validate_audit(audit_path)
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        yaml.safe_dump(output, sys.stdout, sort_keys=False, allow_unicode=True, width=110)
    return 0 if output.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
