"""Derived categorical-component classification for recurrence factors."""

from __future__ import annotations

from typing import NamedTuple

from cffg.core.ir import Template


class RecurrenceTopology(NamedTuple):
    """One ordinary categorical component containing recurrence factors."""

    members: tuple[str, ...]
    categorical_factors: tuple[str, ...]
    acyclic: bool


def classify_recurrence_components(
    template: Template,
    endpoints: dict[str, list[tuple[str, str]]],
    ordinary_bp: set[str],
    recurrence_members: set[str],
    parameter_edges: set[str],
) -> tuple[RecurrenceTopology, ...]:
    """Classify ordinary categorical topology around recurrence factors.

    Mean-field, GFE, parameter, Gaussian, and EF structures are boundaries.
    Their incoming messages are held unary cavities for this classification.
    """
    eligible = ordinary_bp | recurrence_members
    links: list[tuple[str, str]] = []
    adjacency: dict[str, list[str]] = {name: [] for name in eligible}
    for variable_name, ends in sorted(endpoints.items()):
        variable = template.variables[variable_name]
        if (
            variable.dim
            or variable.ef
            or variable_name in parameter_edges
            or len(ends) != 2
        ):
            continue
        left, right = ends[0][0], ends[1][0]
        if left not in eligible or right not in eligible:
            continue
        links.append((left, right))
        adjacency[left].append(right)
        adjacency[right].append(left)

    visited: set[str] = set()
    raw: list[RecurrenceTopology] = []
    for start in sorted(eligible):
        if start in visited:
            continue
        stack = [start]
        nodes: set[str] = set()
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            stack.extend(adjacency[node])
        visited.update(nodes)
        members = tuple(sorted(nodes & recurrence_members))
        if not members:
            continue
        edge_count = sum(len(adjacency[node]) for node in nodes) // 2
        raw.append(
            RecurrenceTopology(
                members=members,
                categorical_factors=tuple(sorted(nodes)),
                acyclic=edge_count == len(nodes) - 1,
            )
        )
    return tuple(sorted(raw, key=lambda component: component.members))


__all__ = ["RecurrenceTopology", "classify_recurrence_components"]
