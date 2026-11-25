"""Validate requested edge names before touching numerical state."""

from collections.abc import Callable, Iterable

from cffg.compiled import EdgeInfo
from cffg.readouts.views import EdgesView

EdgeSelection = Iterable[str] | None


def selected_edges(
    view: EdgesView, names: EdgeSelection, domain: str, width: Callable[[EdgeInfo], int],
) -> list[EdgeInfo]:
    if names is None:
        return [edge for edge in view.edges.values() if width(edge)]
    if isinstance(names, str):
        names = (names,)
    result = []
    for name in dict.fromkeys(names):
        if name not in view.edges:
            raise KeyError(f"unknown {domain} readout edge {name!r}")
        edge = view.edges[name]
        if not width(edge):
            raise ValueError(f"edge {name!r} is not {domain}")
        result.append(edge)
    return result


def grouped_edges(edges, width):
    groups = {}
    for edge in edges:
        groups.setdefault(width(edge), []).append(edge)
    return groups.items()
