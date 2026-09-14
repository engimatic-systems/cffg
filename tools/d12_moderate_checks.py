"""Focused native-float32 checks. The default mode never compiles a D12 program.

``--status`` additionally exercises T2 initialization and certification. Complete
solver wrappers are host zero-budget calls and abstract traces; compiling their
full iteration bodies belongs to post-review evidence.
``--arithmetic`` compares only local projection/learning kernels with report
reconstruction. ``--projection-probe weighted|rare`` preserves unresolved local
numerical counterexamples; a failing probe is not qualification evidence.
"""

import argparse
from copy import deepcopy
import json

from d12_compile_scaling import native_jax, checked_arrays, compile_program


def tiny_context():
    """Small informative slice for API mechanisms, not a qualification fixture."""
    import jax
    from cffg.hybrid_joint import Coordinate
    from cffg.hybrid_natural import Natural
    from cffg.hybrid_separator import TransitionParameters
    from d12_compile_fixture import fixture_arrays

    p, s = fixture_arrays(32)
    sites = Natural(
        *(
            jax.numpy.asarray(a)
            for a in (
                s["constant"][:2, :2],
                s["linear"][:2, :2, :1],
                s["precision"][:2, :2, :1, :1],
                s["status"][:2],
                s["center"][:2, :2, :1],
            )
        )
    )
    parameters = TransitionParameters(
        *(
            jax.numpy.asarray(a)
            for a in (
                p["weights"][:1, :2, :1, :1],
                p["offsets"][:1, :2, :1],
                p["xi"][:1, :2, :1],
                p["matrix"][:1, :2, :1, :1],
                p["bias"][:1, :2, :1],
                p["noise"][:1, :2, :1, :1],
            )
        )
    )
    return Coordinate.from_sites(sites), parameters, sites


def check_native(jax):
    import numpy as np
    from cffg import hybrid_joint as joint
    from d12_moderate_plan import load_spec
    from d12_compile_fixture import context
    from d12_moderate_report import validate_compile, validate_logical, program_contract
    from d12_moderate_workload import wide_context, logical_inventory, compile_arguments
    from d12_moderate_runner import timed_call

    assert not jax.config.x64_enabled
    assert jax.config.jax_explicit_x64_dtypes.name == "ERROR"
    try:
        jax.numpy.asarray(1, dtype="float64")
    except ValueError:
        pass
    else:
        raise AssertionError("explicit float64 was accepted")
    for shape in load_spec()["storage"]["logical_shapes"]:
        t, k, d = (shape[n] for n in ("T", "K", "d"))
        ctx = context(t) if k == 4 else wide_context(t, k, d)
        joint.validate_context(*ctx)
        inv = logical_inventory(*ctx)
        validate_logical(inv, shape)
        bad = deepcopy(inv)
        leaf = next(
            r for r in bad["groups"]["outer_state"] if r["path"] == ".coordinate"
        )
        leaf["shape"][0] += 1
        leaf["elements"] += 1
        leaf["bytes"] += 4
        try:
            validate_logical(bad, shape)
        except ValueError:
            pass
        else:
            raise AssertionError("incorrect outer-state storage accepted")
        if k == 4:
            for program in ("iteration", "solve_from_coordinate"):
                arguments, outputs = program_contract(shape, program)
                actual = checked_arrays(compile_arguments(*ctx, program))
                assert {
                    r["path"]: (tuple(r["shape"]), r["dtype"]) for r in actual
                } == arguments
                actual = inv["groups"]["outer_state"]
                assert {
                    r["path"]: (tuple(r["shape"]), r["dtype"]) for r in actual
                } == outputs
    report, stages = {}, []
    x = jax.numpy.asarray([1, 2], jax.numpy.float32)
    retained = compile_program(lambda a: a + 1, (x,), report, stages.append)
    validate_compile(report, 0)
    result, seconds = timed_call(
        retained[1], (x,), report, stages.append, "scalar_call"
    )
    np.testing.assert_array_equal(result, [2, 3])
    validate_compile(report, 1)
    assert seconds >= 0 and stages == [
        "lower",
        "lower_inspection",
        "compile",
        "compiled_inspection",
        "scalar_call",
    ]
    return dict(
        native=True, logical_shapes=3, scalar_calls=1, moderate_optimizer_calls=0
    )


def check_status(jax):
    import numpy as np
    from cffg import hybrid_joint_solver as solver
    from d12_moderate_workload import raw_coordinate, output_values
    from d12_moderate_payload import flatten, validate_outputs

    c, p, s = tiny_context()
    raw = raw_coordinate(c, jax.numpy.float32(0))
    with jax.disable_jit():
        raw_state = solver.initial_state_from_coordinate(c, p, s, raw)
        feasible = solver.initial_state(c, p, s, 0.0)
        explicit = solver.solve_from_coordinate(
            c, p, s, feasible.coordinate, budgets=solver.Budgets(outer=0)
        )
        legacy = solver.solve(c, p, s, 0.0, budgets=solver.Budgets(outer=0))
    np.testing.assert_array_equal(raw_state.coordinate, raw)
    assert np.any(np.asarray(feasible.coordinate[: c.message_size]) != 0)
    for a, b in zip(jax.tree.leaves(explicit), jax.tree.leaves(legacy), strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)
    complete = lambda p, s, x: solver.solve_from_coordinate(
        c, p, s, x, budgets=solver.Budgets(outer=0)
    )
    descriptor = jax.eval_shape(complete, p, s, raw)
    assert (
        descriptor.coordinate.shape == raw.shape
        and descriptor.status.dtype == jax.numpy.int32
    )

    # The zero-budget boundary consists exactly of these shared entry/terminal
    # operations. Do not compile the unused iteration body for a focused check.
    def run(p, s, x):
        return solver.certify(c, p, s, solver.initial_state_from_coordinate(c, p, s, x))

    compiled = jax.jit(run)
    result = jax.block_until_ready(compiled(p, s, feasible.coordinate))
    assert int(result.status) == int(explicit.status) == solver.Outcome.EXHAUSTED
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(explicit), strict=True):
        np.testing.assert_allclose(
            a,
            b,
            rtol=64 * np.finfo(np.float32).eps,
            atol=64 * np.finfo(np.float32).eps,
            equal_nan=True,
        )
    negative = raw.at[c.message_size :].set(-1)
    nan = raw.at[0].set(jax.numpy.nan)
    changed = s._replace(constant=s.constant.at[0, 0].set(-jax.numpy.inf))
    vectors = jax.numpy.stack((raw, negative, nan, raw))
    sites = jax.tree.map(lambda a, b: jax.numpy.stack((a, a, a, b)), s, changed)
    batch = jax.block_until_ready(
        jax.jit(jax.vmap(run, in_axes=(None, 0, 0)))(p, sites, vectors)
    )
    for index in range(4):
        one = jax.block_until_ready(
            compiled(p, jax.tree.map(lambda a: a[index], sites), vectors[index])
        )
        assert int(one.status) == (
            solver.Outcome.EXHAUSTED if index == 0 else solver.Outcome.DIVERGED
        )
        if index:
            assert int(one.reason) == solver.Reason.INVALID_INITIALIZATION
        for a, b in zip(jax.tree.leaves(one), jax.tree.leaves(batch), strict=True):
            np.testing.assert_allclose(
                a,
                b[index],
                rtol=64 * np.finfo(np.float32).eps,
                atol=64 * np.finfo(np.float32).eps,
                equal_nan=True,
            )
        checked_arrays(one)
    for vector in (raw[:-1], raw.astype(jax.numpy.int32)):
        try:
            solver.initial_state_from_coordinate(c, p, s, vector)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed coordinate accepted")
    case = dict(T=2, K=2, d=1, xi=0, messages="raw-zero")
    validate_outputs(
        flatten(output_values(c, raw_state)),
        case,
        initial=True,
        parameters={name: np.asarray(value) for name, value in p._asdict().items()},
    )
    return dict(
        native=True,
        host_zero_budget_T2_calls=2,
        terminal_T2_calls=9,
        moderate_optimizer_calls=0,
        status_variants=4,
        complete_wrappers="host and abstract only",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--arithmetic", action="store_true")
    parser.add_argument("--projection-probe", choices=("weighted", "rare"))
    args = parser.parse_args()
    jax = native_jax()
    if args.arithmetic or args.projection_probe:
        from d12_moderate_arithmetic import check_arithmetic

        result = check_arithmetic(jax, probe=args.projection_probe)
    else:
        result = check_status(jax) if args.status else check_native(jax)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
