"""Pure device support checks over the current state of an ordinary iteration."""

from cffg.compiled import Arena
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.schedule_program import make_support_check


def make_iteration_support(arena: Arena, executors=SCHEDULED_EXECUTORS):
    """Build the shared exact-support predicate with the ordinary State interface."""
    check = make_support_check(arena, executors)

    def bad_support(state):
        return check(state.messages, state.params)

    return bad_support
