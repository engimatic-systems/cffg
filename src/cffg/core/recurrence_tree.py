"""Ordered logistic gates for the conditional destination in a recurrence factor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    """An accepted or rejected branch names another gate or an integer destination."""

    name: str
    accept: str | int
    reject: str | int


def _check_size(k: int) -> None:
    if type(k) is not int or k < 2:
        raise ValueError(
            "a recurrence tree requires an integer cardinality of at least two"
        )


@dataclass(frozen=True)
class RecurrenceTree:
    """A full binary tree, with coefficient rows in root/accept/reject preorder."""

    gates: tuple[Gate, ...]

    def __post_init__(self) -> None:
        nodes: dict[str, Gate] = {}
        parents: set[str] = set()
        leaves: set[int] = set()
        for gate in self.gates:
            if (
                not isinstance(gate, Gate)
                or not isinstance(gate.name, str)
                or not gate.name
            ):
                raise ValueError("each recurrence gate requires a nonempty string name")
            if gate.name in nodes:
                raise ValueError(f"duplicate recurrence gate {gate.name!r}")
            nodes[gate.name] = gate
            for child in (gate.accept, gate.reject):
                if type(child) is int:
                    if child < 0 or child in leaves:
                        raise ValueError(
                            f"invalid or duplicate recurrence leaf {child}"
                        )
                    leaves.add(child)
                elif isinstance(child, str) and child:
                    if child in parents:
                        raise ValueError(
                            f"recurrence gate {child!r} has multiple incoming branches"
                        )
                    parents.add(child)
                else:
                    raise ValueError(
                        "a recurrence child must name a gate or an integer leaf"
                    )
        unknown = parents - nodes.keys()
        if unknown:
            raise ValueError(f"unknown recurrence gate {min(unknown)!r}")
        roots = nodes.keys() - parents
        if len(roots) != 1:
            raise ValueError("a recurrence tree requires one root and no cycles")
        if leaves != set(range(len(nodes) + 1)):
            raise ValueError(
                "recurrence leaves must be the consecutive labels 0 through K-1"
            )

        # Iterative preorder also supports long decision lists.
        stack = list(roots)
        ordered = []
        while stack:
            gate = nodes[stack.pop()]
            ordered.append(gate)
            for child in (gate.reject, gate.accept):
                if isinstance(child, str):
                    stack.append(child)
        if len(ordered) != len(nodes):
            raise ValueError("disconnected cycle in recurrence tree")
        object.__setattr__(self, "gates", tuple(ordered))

    @property
    def cardinality(self) -> int:
        return len(self.gates) + 1

    @classmethod
    def decision_list(cls, k: int) -> RecurrenceTree:
        """Test destination 0 first, continuing until the final rejected leaf."""
        _check_size(k)
        return cls(
            tuple(
                Gate(f"g{i}", i, f"g{i + 1}" if i + 2 < k else k - 1)
                for i in range(k - 1)
            )
        )

    @classmethod
    def balanced(cls, k: int) -> RecurrenceTree:
        """Split each consecutive label interval in half, larger half first."""
        _check_size(k)
        gates = []

        def branch(start: int, stop: int) -> str | int:
            if stop - start == 1:
                return start
            name = f"g{start}_{stop}"
            middle = (start + stop + 1) // 2
            gates.append(Gate(name, branch(start, middle), branch(middle, stop)))
            return name

        branch(0, k)
        return cls(tuple(gates))

    def is_decision_list(self) -> bool:
        """Whether the legacy ordered coefficient rows have their original meaning."""
        for i, gate in enumerate(self.gates):
            reject = (
                self.gates[i + 1].name
                if i + 1 < len(self.gates)
                else self.cardinality - 1
            )
            if gate.accept != i or gate.reject != reject:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "gates": [
                {"name": g.name, "accept": g.accept, "reject": g.reject}
                for g in self.gates
            ]
        }

    @classmethod
    def from_dict(cls, record: dict) -> RecurrenceTree:
        if (
            not isinstance(record, dict)
            or set(record) != {"gates"}
            or not isinstance(record["gates"], list)
        ):
            raise ValueError("a recurrence tree requires an array of gates")
        gates = []
        for row in record["gates"]:
            if not isinstance(row, dict) or set(row) != {"name", "accept", "reject"}:
                raise ValueError(
                    "each recurrence gate requires name, accept, and reject"
                )
            gates.append(Gate(**row))
        return cls(tuple(gates))
