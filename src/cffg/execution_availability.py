"""One host boundary between inspectable compiled identity and runtime support."""


class ExecutionUnavailableError(RuntimeError):
    """A compiled approximation has no implemented runtime execution path."""


def require_execution_available(arena, *, operation: str) -> None:
    """Reject D12 before callbacks, state initialization or JIT wrapping.

    Runtime support is an implementation decision, never a caller-supplied plan
    flag. All declared components remain unavailable until their state, ledger
    and execution consumers are implemented together.
    """
    if arena.hybrid_components:
        identities = ", ".join(
            f"{plan.name!r} ({plan.approximation})" for plan in arena.hybrid_components
        )
        raise ExecutionUnavailableError(
            f"{operation}: hybrid component {identities}: execution unavailable; "
            "immutable plan inspection does not enable inference"
        )
