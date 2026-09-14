"""Inspectable logical arrays and dynamic compilation interfaces for D12.

Logical leaves are not peak memory. Returning derivative captures extends their
lifetimes, so measure the enclosing iteration separately. Compiler/process RSS
is also separate from XLA's argument/output/temporary/alias byte counts.
"""

import jax
import jax.numpy as jnp
import numpy as np

from cffg import hybrid_joint as joint
from cffg import hybrid_joint_solver as solver


def arrays(value):
    """JSON-ready leaf paths, shapes, dtypes, element counts and logical bytes."""
    records = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        if not hasattr(leaf, 'shape') or not hasattr(leaf, 'dtype'):
            continue
        elements = int(np.prod(leaf.shape))
        records.append(dict(path=jax.tree_util.keystr(path), shape=list(leaf.shape),
                            dtype=str(leaf.dtype), elements=elements,
                            bytes=elements * np.dtype(leaf.dtype).itemsize))
    return records


def buffers(coordinate, parameters, sites, vector):
    """Held-point logical inventory, including the accepted JVP's captures.

    This evaluates and linearizes one coordinate; it performs no optimizer step.
    Inspect correction/origin leaves separately from packed unknowns. Repeated
    leaves may share buffers and must not be summed into a claimed physical peak.
    """
    value = joint.evaluate(coordinate, parameters, sites, vector)
    residual, forward, _ = joint.linearize(coordinate, parameters, sites, vector,
                                         value.scales, 1 + abs(vector))
    zero_n, zero_m = jnp.zeros_like(vector), jnp.zeros_like(residual)
    linear = solver.LinearState(jnp.int32(0), zero_n, zero_m, zero_n, zero_n,
                                jnp.asarray(False), jnp.asarray(True))
    trial = solver.TrialState(jnp.int32(0), vector, value, jnp.asarray(False), jnp.int32(0),
                              jnp.zeros(len(solver.Rejection), jnp.int32), jnp.int32(0))
    outer = solver.State(vector, value, *(jnp.int32(0) for _ in range(5)),
                         jnp.asarray(jnp.inf, vector.dtype), trial.rejection_counts,
                         jnp.int32(0), jnp.int32(solver.Outcome.RUNNING),
                         jnp.int32(solver.Reason.OUTER_BUDGET))
    groups = dict(metadata=coordinate.metadata(), context=(parameters, sites), coordinate=vector,
                  accepted_evaluation=value, coordinate_scales=1 + abs(vector),
                  residual_scales=value.scales, derivative_captures=forward,
                  linear_state=linear, linear_products=(zero_m, zero_n),
                  trial_state=trial, outer_state=outer)
    return dict(N=coordinate.size, M=coordinate.residual_size,
                groups={name: arrays(group) for name, group in groups.items()})


def programs(coordinate, budgets=solver.Budgets()):
    """Actual enclosing programs with dynamic context/coordinate arguments.

    ``residual`` takes (parameters, sites, vector).
    ``normal_product`` takes (parameters, sites, vector, scales, direction).
    ``linear_step`` takes (parameters, sites, vector, evaluation).
    ``iteration`` and ``certification`` take (parameters, sites, state).
    ``solve`` takes (parameters, sites, initial_xi); its default start is unperturbed.
    Use at multiple horizons and independently varied outer/linear caps. Report
    fixed captured metadata in addition to each compiled argument inventory.
    """
    def normal_product(parameters, sites, vector, scales, direction):
        _, forward, transpose = joint.linearize(coordinate, parameters, sites, vector, scales, 1 + abs(vector))
        return transpose(forward(direction))

    return dict(residual=lambda p, s, x: joint.evaluate(coordinate, p, s, x).residual,
                normal_product=normal_product,
                linear_step=lambda p, s, x, e: solver.gauss_newton(coordinate, p, s, x, e, budgets),
                iteration=lambda p, s, state: solver.iteration(coordinate, p, s, state, budgets),
                certification=lambda p, s, state: solver.certify(coordinate, p, s, state),
                solve=lambda p, s, xi: solver.solve(coordinate, p, s, xi, budgets=budgets))


def compiled_memory(function, *arguments):
    """Lower dynamic descriptors and return XLA memory data when available.

    No enclosing solve is executed. Compilation itself may consume substantial
    memory; the caller owns horizon/cap selection and process/RSS measurement.
    """
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), arguments)
    executable = jax.jit(function).lower(*abstract).compile()
    analysis = executable.memory_analysis()
    if analysis is None:
        return dict(available=False, substitute='logical leaf inventory and process peak RSS')
    fields = ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')
    return dict(available=True, **{name: int(getattr(analysis, name)) for name in fields})
