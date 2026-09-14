"""Validate complete moderate evidence and disclose sensitivity across starts."""

from itertools import combinations
import json
from pathlib import Path

import numpy as np

from d12_compile_fixture import fixture_arrays
from d12_compile_scaling import digest, rss_bytes
from d12_moderate_plan import BUDGETS, SCHEMA, identities, load_spec
from d12_moderate_payload import (
    output_contract,
    read_snapshot,
    require,
    validate_outputs,
)


def validate_compile(record, calls):
    require(
        record["calls"] == dict(lower=1, compile=1, execute=calls),
        "unexpected compilation/execution count",
    )
    for key in ("lower_seconds", "compile_seconds"):
        require(
            type(record[key]) in (int, float)
            and np.isfinite(record[key])
            and record[key] >= 0,
            "invalid compiler timing",
        )
    for key in ("arguments", "lowered_outputs", "compiled_outputs"):
        validate_leaves(record[key])
    require(
        set(record["stablehlo_dtypes"]["value_type_occurrences"]) == {"f32"},
        "non-native StableHLO",
    )
    memory = record["xla_memory"]
    if memory["available"]:
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
        ):
            value = memory["estimates"][name]
            require(
                type(value) is int and value >= 0, "missing/invalid XLA byte estimate"
            )
    else:
        require(
            bool(memory["substitute"]),
            "unavailable XLA memory needs a named substitute",
        )


def validate_leaves(leaves):
    require(
        bool(leaves) and len({x["path"] for x in leaves}) == len(leaves),
        "missing or duplicate logical leaves",
    )
    for leaf in leaves:
        dtype = np.dtype(leaf["dtype"])
        require(
            dtype == np.dtype("float32") or dtype.kind in "iub",
            "non-native logical leaf dtype",
        )
        require(
            all(type(n) is int and n >= 0 for n in leaf["shape"]), "invalid leaf shape"
        )
        count = int(np.prod(leaf["shape"]))
        require(
            leaf["elements"] == count and leaf["bytes"] == count * dtype.itemsize,
            "logical size mismatch",
        )


def state_contract(t, k, d):
    return {
        key.removeprefix("state")
        .replace(".targets.0.", ".targets[0].")
        .replace(".targets.1.", ".targets[1]."): value
        for key, value in output_contract(t, k, d).items()
        if key.startswith("state.")
    }


def program_contract(case, program):
    """Dynamic input and complete State output descriptors, including every field."""
    t, k, d = (case[name] for name in ("T", "K", "d"))
    outputs = state_contract(t, k, d)
    p, s = fixture_arrays(t)
    arguments = {
        f"[{index}].{name}": (value.shape, str(value.dtype))
        for index, group in enumerate((p, s))
        for name, value in group.items()
    }
    if program == "initialize":
        arguments.update({"[2]": ((), "float32"), "[3]": ((), "int32")})
    elif program == "solve_from_coordinate":
        arguments["[2]"] = outputs[".coordinate"]
    elif program == "iteration":
        arguments.update({f"[2]{key}": value for key, value in outputs.items()})
    else:
        raise ValueError("unknown enclosing program")
    return arguments, outputs


def validate_program(record, case, program, calls):
    validate_compile(record, calls)
    arguments, outputs = program_contract(case, program)
    for name, expected in (
        ("arguments", arguments),
        ("lowered_outputs", outputs),
        ("compiled_outputs", outputs),
    ):
        actual = {r["path"]: (tuple(r["shape"]), r["dtype"]) for r in record[name]}
        require(
            actual == expected,
            f"{program} {name} do not describe the complete dynamic program",
        )


def validate_runtime(worker, process):
    runtime = worker["runtime"]
    config = runtime["config"]
    require(
        runtime["backend"] == "cpu"
        and config["jax_enable_x64"] is False
        and config["jax_explicit_x64_dtypes"] == "ERROR",
        "native float32 setup missing",
    )
    require(
        config["jax_enable_compilation_cache"] is False
        and config["jax_persistent_cache_enable_xla_caches"] == "",
        "unexpected persistent cache policy",
    )
    require(
        all(
            runtime[key]
            for key in (
                "python",
                "executable",
                "platform",
                "machine",
                "jax",
                "jaxlib",
                "numpy",
                "devices",
            )
        ),
        "runtime provenance missing",
    )
    require(
        runtime["clock"]["monotonic"] and not runtime["clock"]["adjustable"],
        "unsuitable timer",
    )
    require(
        process["exit_code"] == 0 and process["stop"] is None,
        "worker failed or hit a resource stop",
    )
    peak = process["process_high_water"]
    system = "Darwin" if peak["raw_unit"] == "bytes" else "Linux"
    require(
        peak["raw_unit"] in ("bytes", "KiB")
        and peak["bytes"] > 0
        and peak["bytes"] == rss_bytes(peak["raw_ru_maxrss"], system),
        "process high-water unit mismatch",
    )
    require(
        bool(process["cache_evidence"]["log_sha256"]),
        "missing compiler/cache log provenance",
    )


def validate_fixture(arrays, horizon):
    p, s = fixture_arrays(horizon)
    for group, expected in (("parameters", p), ("sites", s)):
        for name, value in expected.items():
            require(
                np.array_equal(arrays[f"{group}.{name}"], value),
                f"frozen fixture mismatch: {group}.{name}",
            )
    require(
        np.all(arrays["metadata.origin"] == 0) and np.all(arrays["metadata.support"]),
        "fixed origin or support changed",
    )


def validate_work(arrays):
    n = {
        name: int(arrays[f"state.{name}"])
        for name in (
            "iterations",
            "evaluations",
            "rejected_trials",
            "linear_iterations",
            "gauss_newton_steps",
        )
    }
    i, e, r, linear, fallback = n.values()
    status = int(arrays["state.status"])
    require(
        i <= BUDGETS["outer"] and e == 2 + i + r, "outer/evaluation accounting mismatch"
    )
    require(
        fallback <= i + 1 and linear <= fallback * BUDGETS["linear"],
        "linear/fallback budget mismatch",
    )
    require(
        fallback * BUDGETS["trials"] <= r <= (i + 1 + fallback) * BUDGETS["trials"] - i,
        "trial budget mismatch",
    )
    if status == 0:
        require(
            fallback <= i and fallback <= linear, "accepted fallback lacks linear work"
        )
        require(
            r <= (i + fallback) * BUDGETS["trials"] - i,
            "successful trial budget mismatch",
        )


def comparison_group(key):
    if key.startswith("messages."):
        return "messages"
    if key == "xi":
        return "xi"
    if key in (
        "state.evaluation.local.free_energy",
        "state.evaluation.local.dual_energy",
    ):
        return "energy"
    for name in ("regions", "separators"):
        if key.startswith((f"{name}.", f"state.evaluation.local.{name}.")):
            return name
    if key.startswith(("learning.", "state.evaluation.auxiliary.")):
        return "learning"
    return None


def compare_starts(terminals):
    """All pairs, all returned physical/centered groups; no uniqueness assertion."""
    rows = []
    tau = 64 * np.finfo(np.float32).eps ** 0.75
    for (a, left), (b, right) in combinations(terminals.items(), 2):
        if a.split("-")[0] != b.split("-")[0]:
            continue
        groups = dict.fromkeys(
            ("messages", "xi", "energy", "regions", "separators", "learning"), 0.0
        )
        for key, x in left.items():
            group = comparison_group(key)
            if group is None or x.dtype.kind != "f":
                continue
            y = right[key]
            # Equal log-zero values have zero difference; unequal finite masks are material.
            equal_nonfinite = ~np.isfinite(x) & (x == y)
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.all(finite | equal_nonfinite):
                raise ValueError(
                    "stationary output comparison has inconsistent finite masks"
                )
            x, y = x[finite].astype(np.float64), y[finite].astype(np.float64)
            delta = np.max(abs(x - y) / (1 + np.maximum(abs(x), abs(y))), initial=0.0)
            groups[group] = max(groups[group], float(delta))
        rows.append(
            dict(
                left=a,
                right=b,
                scaled_maximum=groups,
                material=bool(max(groups.values()) > tau),
            )
        )
    return rows


def validate_logical(inventory, shape):
    t, k, d = (shape[name] for name in ("T", "K", "d"))
    n = 2 * (t - 1) * (k * (1 + d + d * (d + 1) // 2) - 1) + (t - 1) * k * (k - 1)
    m = n + 2 * (t - 1) * k * (1 + d + d * (d + 1) // 2)
    require(
        (inventory["N"], inventory["M"]) == (n, m),
        "coordinate/residual inventory mismatch",
    )
    expected = {
        "metadata",
        "context",
        "coordinate",
        "accepted_evaluation",
        "coordinate_scales",
        "residual_scales",
        "derivative_captures",
        "linear_state",
        "linear_products",
        "trial_state",
        "outer_state",
        "terminal_state",
    }
    require(
        set(inventory["groups"]) == expected, "incomplete enclosing logical inventory"
    )
    for leaves in inventory["groups"].values():
        validate_leaves(leaves)
    state = state_contract(t, k, d)
    evaluation = {
        key.removeprefix(".evaluation"): value
        for key, value in state.items()
        if key.startswith(".evaluation.")
    }
    contracts = {
        "coordinate": {"": ((n,), "float32")},
        "coordinate_scales": {"": ((n,), "float32")},
        "residual_scales": {"": ((m,), "float32")},
        "accepted_evaluation": evaluation,
        "outer_state": state,
        "terminal_state": state,
        "linear_state": {
            ".iteration": ((), "int32"),
            ".step": ((n,), "float32"),
            ".residual": ((m,), "float32"),
            ".normal_residual": ((n,), "float32"),
            ".direction": ((n,), "float32"),
            ".done": ((), "bool"),
            ".finite": ((), "bool"),
        },
        "linear_products": {"[0]": ((m,), "float32"), "[1]": ((n,), "float32")},
    }
    for group, expected in contracts.items():
        actual = {
            r["path"]: (tuple(r["shape"]), r["dtype"])
            for r in inventory["groups"][group]
        }
        require(actual == expected, f"incomplete or malformed {group} storage")
    for name in ("accepted_evaluation", "derivative_captures", "terminal_state"):
        leaves = inventory["groups"][name]
        require(
            any(x["dtype"] == "float32" for x in leaves), "missing floating buffers"
        )
    require(
        any(
            "correction" in x["path"]
            for x in inventory["groups"]["accepted_evaluation"]
        ),
        "retained covariance words missing",
    )
    require(
        any(
            np.dtype(x["dtype"]).kind in "iu"
            for x in inventory["groups"]["derivative_captures"]
        ),
        "derivative integer captures missing",
    )


def validate_report(directory, *, current_sources=True):
    """Reject incomplete runs. Return numerical gates and sensitivity, never a scope decision."""
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    spec = load_spec()
    require(
        report["schema"] == SCHEMA and report["status"] == "complete",
        "incomplete moderate report",
    )
    identity = report["identity"]
    require(not identity["git_status"], "qualification ran from a dirty tree")
    current = identities()
    require(identity["spec_sha256"] == current["spec_sha256"], "specification mismatch")
    if current_sources:
        require(
            identity["sources"] == current["sources"],
            "source-matched evidence required",
        )
    phase = report["phase"]
    require(phase in ("solves", "storage"), "unknown qualification phase")
    expected = (
        [f"t{t}" for t in (32, 100)]
        if phase == "solves"
        else ["inventory"] + [c["id"] for c in spec["storage"]["cases"]]
    )
    require(
        [j["id"] for j in report["jobs"]] == expected,
        "incomplete/duplicate/out-of-order job inventory",
    )
    require(
        json.loads((directory / "spec.json").read_text()) == spec,
        "saved specification differs from its identity",
    )
    terminals = {}
    for job in report["jobs"]:
        folder = directory / job["id"]
        require(
            digest(folder / "worker.json") == job["worker_sha256"]
            and digest(folder / "process.json") == job["process_sha256"],
            "worker/process report hash mismatch",
        )
        worker = json.loads((folder / "worker.json").read_text())
        process = json.loads((folder / "process.json").read_text())
        require(
            worker["identity"] == identity and worker["status"] == "complete",
            "incomplete worker/provenance mismatch",
        )
        require(
            digest(folder / "compiler.log") == process["cache_evidence"]["log_sha256"],
            "compiler log hash mismatch",
        )
        validate_runtime(worker, process)
        if phase == "storage":
            if job["id"] == "inventory":
                require(
                    worker["complete_program_calls"]
                    == dict(lower=0, compile=0, execute=0),
                    "shape-only inventory executed a complete program",
                )
                for shape, inventory in zip(
                    spec["storage"]["logical_shapes"],
                    worker["shape_inventory"],
                    strict=True,
                ):
                    require(inventory["shape"] == shape, "logical shape mismatch")
                    validate_logical(inventory["inventory"], shape)
                continue
            case = next(c for c in spec["storage"]["cases"] if c["id"] == job["id"])
            require(worker["case"] == case, "storage variant mismatch")
            validate_program(worker["compilation"], case, case["program"], 0)
            validate_logical(worker["logical"], case)
            continue
        cases = [c for c in spec["cases"] if f"t{c['T']}" == job["id"]]
        require(
            [r["case"] for r in worker["cases"]] == cases,
            "exact six-start inventory missing",
        )
        validate_fixture(read_snapshot(folder, worker["inputs"]), cases[0]["T"])
        validate_program(
            worker["initialization_compilation"], cases[0], "initialize", 6
        )
        validate_program(
            worker["solve_compilation"], cases[0], "solve_from_coordinate", 6
        )
        initials = {}
        for record in worker["cases"]:
            case = record["case"]
            initial = read_snapshot(folder, record["initial"])
            validate_outputs(initial, case, initial=True)
            require(int(initial["state.status"]) == 3, "initialization failed")
            terminal = read_snapshot(folder, record["terminal"])
            validate_outputs(terminal, case)
            validate_work(terminal)
            require(
                int(terminal["state.status"]) == 0, "moderate solve did not converge"
            )
            for key in ("initialization_seconds", "solve_seconds", "export_seconds"):
                require(
                    type(record[key]) in (int, float)
                    and np.isfinite(record[key])
                    and record[key] >= 0,
                    "missing synchronized timing",
                )
            initials[case["id"]] = initial
            terminals[case["id"]] = terminal
        for xi in (0, 1):
            name = f"{job['id']}-xi{xi}"
            a = initials[f"{name}-feasible"]["state.coordinate"]
            b = initials[f"{name}-perturbed"]["state.coordinate"]
            size = a.size - initials[f"{name}-feasible"]["xi"].size
            expected = a[:size] + np.float32(0.01) * np.sin(
                np.arange(size, dtype=np.float32) + np.float32(1)
            )
            require(
                np.allclose(
                    b[:size],
                    expected,
                    rtol=4 * np.finfo(np.float32).eps,
                    atol=4 * np.finfo(np.float32).eps,
                ),
                "declared non-oracle perturbation mismatch",
            )
    comparisons = compare_starts(terminals)
    require(phase != "solves" or len(comparisons) == 30, "start comparisons incomplete")
    return dict(
        phase=phase,
        complete=True,
        comparisons=comparisons,
        material_differences=sum(r["material"] for r in comparisons),
        qualification="A3WZ requires both phases and diagnosis of material differences; no recovery evidence",
    )
