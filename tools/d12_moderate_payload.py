"""Lossless immutable array snapshots and an explicit moderate output contract.

No JAX imports: reading evidence cannot change the consumer's precision policy.
Nonfinite failure values stay IEEE values in NPZ; JSON contains only manifests.
"""

import hashlib
from pathlib import Path

import numpy as np

from d12_compile_scaling import digest

CERTIFICATES = (
    "natural",
    "label_mass",
    "conditional_mean",
    "conditional_covariance",
    "auxiliary",
    "primal_dual",
)
COUNTERS = (
    "iterations",
    "evaluations",
    "rejected_trials",
    "linear_iterations",
    "gauss_newton_steps",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def flatten(value, prefix=""):
    if hasattr(value, "_asdict"):
        value = value._asdict()
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (tuple, list)):
        items = enumerate(value)
    elif value is None:
        return {}
    else:
        return {prefix: np.asarray(value)}
    return {
        key: array
        for name, item in items
        for key, array in flatten(
            item, f"{prefix}.{name}" if prefix else str(name)
        ).items()
    }


def array_manifest(arrays):
    return {
        name: dict(
            shape=list(a.shape),
            dtype=str(a.dtype),
            elements=int(a.size),
            bytes=int(a.nbytes),
            sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        )
        for name, a in sorted(arrays.items())
    }


def write_snapshot(directory, name, values):
    arrays = flatten(values)
    require(
        all(
            a.dtype in (np.dtype("float32"), np.dtype("int32"), np.dtype("bool"))
            for a in arrays.values()
        ),
        "snapshot contains a non-native dtype",
    )
    path = directory / f"{name}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return dict(file=path.name, sha256=digest(path), leaves=array_manifest(arrays))


def read_snapshot(directory, record):
    name = record["file"]
    require(
        Path(name).name == name and name.endswith(".npz"),
        "snapshot must be a local NPZ filename",
    )
    path = directory / name
    require(digest(path) == record["sha256"], "snapshot file hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), "duplicate snapshot keys")
        arrays = {name: saved[name] for name in saved.files}
    require(
        array_manifest(arrays) == record["leaves"], "snapshot leaf manifest mismatch"
    )
    return arrays


def output_contract(t, k=4, d=2):
    """Shapes follow local/regional axes, independently of the saved manifest."""
    l = t - 1
    triangle = d * (d + 1) // 2
    n = 2 * l * (k * (1 + d + triangle) - 1) + l * k * (k - 1)
    m = n + 2 * l * k * (1 + d + triangle)
    result = {}

    def add(prefix, shapes, dtype="float32"):
        result.update(
            {
                f"{prefix}.{name}": (tuple(shape), dtype)
                for name, shape in shapes.items()
            }
        )

    def natural(prefix):
        add(
            prefix,
            dict(
                constant=(l, k),
                linear=(l, k, d),
                precision=(l, k, d, d),
                center=(l, k, d),
            ),
        )
        add(prefix, dict(status=(l,)), "int32")

    for name, labels, dim, times in (
        ("regions", (l, k, k), 2 * d, l),
        ("separators", (t, k), d, t),
    ):
        add(
            f"state.evaluation.local.{name}",
            dict(
                probability=labels,
                log_probability=labels,
                displacement=(*labels, dim),
                center=(*labels, dim),
                covariance=(*labels, dim, dim),
                covariance_correction=(*labels, dim, dim),
                log_normalizer=(times,),
            ),
        )
        add(f"state.evaluation.local.{name}", dict(status=(times,)), "int32")
        add(
            name,
            dict(
                mean=(*labels, dim),
                second=(*labels, dim, dim),
                full_covariance=(*labels, dim, dim),
            ),
        )
    for prefix in (
        "state.evaluation.local.targets.0",
        "state.evaluation.local.targets.1",
        "messages.left",
        "messages.right",
    ):
        natural(prefix)
    add(
        "state.evaluation.local",
        {
            name: ()
            for name in (
                "free_energy",
                "dual_energy",
                "moment_residual",
                "quotient_residual",
            )
        },
    )
    add("state.evaluation.local", dict(status=()), "int32")
    add(
        "state.evaluation.auxiliary",
        dict(
            target=(l, k, k - 1),
            residual=(l, k, k - 1),
            exposure=(l, k, k - 1),
            linear=(l, k, k - 1, d + 1),
            precision=(l, k, k - 1, d + 1, d + 1),
            center=(l, k, d),
        ),
    )
    add("state.evaluation.auxiliary", dict(status=(l,)), "int32")
    add(
        "learning",
        dict(
            physical_linear=(l, k, k - 1, d + 1),
            physical_precision=(l, k, k - 1, d + 1, d + 1),
        ),
    )
    add("state.evaluation", dict(residual=(m,), scales=(m,)))
    add("state.evaluation.certificates", {name: () for name in CERTIFICATES})
    add("state.evaluation", dict(status=()), "int32")
    add("state", dict(coordinate=(n,), relative_energy_change=()))
    add(
        "state",
        {name: () for name in (*COUNTERS, "status", "reason", "last_rejection")},
        "int32",
    )
    add("state", dict(rejection_counts=(4,)), "int32")
    result["xi"] = ((l, k, k - 1), "float32")
    return result


def validate_outputs(arrays, case, *, initial=False, parameters=None):
    contract = output_contract(case["T"], case["K"], case["d"])
    require(set(arrays) == set(contract), "incomplete or unexpected output groups")
    for key, (shape, dtype) in contract.items():
        require(
            arrays[key].shape == shape and arrays[key].dtype == np.dtype(dtype),
            f"output shape/dtype: {key}",
        )
    x = arrays["state.coordinate"]
    xi = arrays["xi"]
    d = case["d"]
    constants, linear, precision = [], [], []
    for side in ("left", "right"):
        bank = {
            name: arrays[f"messages.{side}.{name}"]
            for name in ("constant", "linear", "precision", "center", "status")
        }
        require(np.all(bank["constant"][:, 0] == 0), "message scalar gauge changed")
        require(
            np.all(bank["center"] == 0) and np.all(bank["status"] == 0),
            "message metadata changed",
        )
        require(
            np.array_equal(
                bank["precision"], bank["precision"].swapaxes(-1, -2), equal_nan=True
            ),
            "message precision lost symmetry",
        )
        constants.append(bank["constant"][:, 1:].ravel())
        linear.append(bank["linear"].ravel())
        row, col = np.triu_indices(d)
        precision.append(bank["precision"][..., row, col].ravel())
    packed = np.concatenate((*constants, *linear, *precision, xi.ravel()))
    require(
        np.array_equal(x, packed, equal_nan=True),
        "natural messages/xi disagree with returned coordinate",
    )
    status, reason = int(arrays["state.status"]), int(arrays["state.reason"])
    require(status in ((2, 3) if initial else (0, 1, 2)), "invalid terminal outcome")
    allowed = {0: (0,), 1: (1, 2, 6), 2: (3, 4, 5), 3: (1,)}
    require(reason in allowed[status], "outcome/reason mismatch")
    counts = {name: int(arrays[f"state.{name}"]) for name in COUNTERS}
    require(all(v >= 0 for v in counts.values()), "negative work counter")
    histogram = arrays["state.rejection_counts"]
    require(
        np.all(histogram >= 0)
        and histogram[0] == 0
        and histogram.sum() == counts["rejected_trials"],
        "rejection histogram mismatch",
    )
    last = int(arrays["state.last_rejection"])
    require(
        0 <= last < 4 and (last == 0 or histogram[last] > 0), "last rejection mismatch"
    )
    if initial:
        require(
            counts
            == dict(
                iterations=0,
                evaluations=1,
                rejected_trials=0,
                linear_iterations=0,
                gauss_newton_steps=0,
            ),
            "initializer claims optimizer work",
        )
        require(
            np.isposinf(arrays["state.relative_energy_change"]),
            "initializer claims energy stability",
        )
        require(
            np.all(xi == case["xi"])
            or (status == 2 and case["messages"] != "raw-zero" and np.all(np.isnan(x))),
            "initializer changed named xi",
        )
        if case["messages"] == "raw-zero":
            require(
                np.all(x[: -xi.size] == 0),
                "raw zero contains feasible or perturbed messages",
            )
    numeric = int(arrays["state.evaluation.status"])
    if status != 2:
        require(
            numeric == 0 and np.all(xi >= 0),
            "nondiverged state has invalid evaluation/xi",
        )
        for key, value in arrays.items():
            if key == "state.relative_energy_change" or key.endswith(
                ".log_probability"
            ):
                continue
            require(np.all(np.isfinite(value)), f"nonfinite active output: {key}")
        for name in ("regions", "separators"):
            require(
                np.all(arrays[f"state.evaluation.local.{name}.status"] == 0),
                "improper active belief",
            )
            validate_belief(arrays, name)
        require(
            int(arrays["state.evaluation.local.status"]) == 0
            and np.all(arrays["state.evaluation.auxiliary.status"] == 0),
            "local/auxiliary failure",
        )
        for side in (0, 1):
            prefix = f"state.evaluation.local.targets.{side}."
            precision = arrays[prefix + "precision"]
            require(
                np.all(arrays[prefix + "center"] == 0)
                and np.all(arrays[prefix + "status"] == 0)
                and np.array_equal(precision, precision.swapaxes(-1, -2)),
                "target metadata or symmetric coordinate invalid",
            )
        from d12_moderate_reconstruct import scaled_blocks, physical_learning

        r, scales, expected, learning = scaled_blocks(arrays, case, parameters)
        close(arrays["state.evaluation.residual"], r, "reconstructed residual")
        close(arrays["state.evaluation.scales"], scales, "prescribed residual scales")
        for field in (
            "target",
            "residual",
            "exposure",
            "linear",
            "precision",
            "center",
        ):
            close(
                arrays[f"state.evaluation.auxiliary.{field}"],
                learning[field],
                f"source-conditioned learning {field}",
            )
        physical = physical_learning(
            *(
                arrays[f"state.evaluation.auxiliary.{field}"]
                for field in ("linear", "precision", "center")
            )
        )
        for field, value in physical.items():
            close(
                arrays[f"learning.{field}"],
                value,
                f"centered/physical learning {field}",
            )
            close(
                arrays[f"learning.{field}"],
                learning[field],
                f"source-conditioned learning {field}",
            )
        actual = np.array(
            [arrays[f"state.evaluation.certificates.{name}"] for name in CERTIFICATES]
        )
        close(actual, expected, "reconstructed certificates")
        # Preserve the original summary-to-saved-residual arithmetic check as
        # well as the independent reconstruction above.
        l, k = case["T"] - 1, case["K"]
        widths = (
            x.size - xi.size,
            2 * l * k,
            2 * l * k * d,
            l * k * d * (d + 1),
            xi.size,
        )
        stored_parts = np.split(
            arrays["state.evaluation.residual"] / arrays["state.evaluation.scales"],
            np.cumsum(widths)[:-1],
        )
        stored = [np.max(abs(part), initial=0) for part in stored_parts]
        stored.append(expected[-1])
        require(
            np.allclose(actual, stored, rtol=8 * np.finfo(np.float32).eps, atol=0),
            "certificate/saved residual summary mismatch",
        )
        require(np.all(actual >= 0), "negative residual certificate")
        if status == 0:
            tau = 64 * np.finfo(np.float32).eps ** 0.75
            change = float(arrays["state.relative_energy_change"])
            require(
                np.isfinite(change)
                and 0 <= change <= tau
                and np.max(actual) <= tau
                and np.max(expected) <= tau,
                "convergence without all terminal certificates",
            )
    elif reason == 3:
        require(numeric != 0, "invalid initialization missing numerical failure flags")
    return counts


def close(actual, expected, name):
    allowance = 512 * 4 * np.finfo(np.float32).eps
    require(
        np.all(np.isfinite(expected))
        and np.allclose(actual, expected, atol=allowance, rtol=allowance),
        name + " mismatch",
    )


def validate_belief(arrays, name):
    """Check saved properness and readout consistency at the frozen local allowance."""
    prefix = f"state.evaluation.local.{name}"
    probability = arrays[f"{prefix}.probability"]
    logs = arrays[f"{prefix}.log_probability"]
    high = arrays[f"{prefix}.covariance"]
    low = arrays[f"{prefix}.covariance_correction"]
    require(
        high.ndim == probability.ndim + 2
        and high.shape[:-2] == probability.shape
        and high.shape[-1] == high.shape[-2]
        and low.shape == high.shape
        and logs.shape == probability.shape
        and high.dtype
        == low.dtype
        == probability.dtype
        == logs.dtype
        == np.dtype("float32"),
        "belief word shape/dtype mismatch",
    )
    for word in (high, low):
        require(
            np.all(np.isfinite(word)) and np.array_equal(word, word.swapaxes(-1, -2)),
            "belief covariance words must be finite and symmetric",
        )
    info = np.finfo(high.dtype)
    require(
        np.all(abs(low) <= info.eps * abs(high))
        and np.all((low == 0) | (abs(low) >= info.tiny)),
        "invalid covariance low word",
    )
    active = np.isfinite(logs)
    require(
        np.all(active | np.isneginf(logs)) and np.array_equal(probability > 0, active),
        "belief support/finite-log underflow mismatch",
    )
    with np.errstate(under="ignore"):
        normalized = np.exp(logs)
    require(
        np.all(probability[active] >= info.tiny)
        and np.all(normalized[active] >= info.tiny),
        "native probability underflow",
    )
    for value in (high, low, arrays[f"{prefix}.displacement"]):
        require(np.all(value[~active] == 0), "nonzero inactive belief coordinates")
    allowance = 512 * 4 * np.finfo(np.float32).eps
    label_axes = tuple(range(1, probability.ndim))
    require(
        np.all(probability >= 0)
        and np.allclose(probability.sum(axis=label_axes), 1, atol=allowance, rtol=0),
        "belief probabilities do not normalize",
    )
    require(
        np.allclose(normalized, probability, atol=allowance, rtol=allowance),
        "belief log masses disagree",
    )
    covariance = high + low
    require(
        np.allclose(
            covariance, covariance.swapaxes(-1, -2), atol=allowance, rtol=allowance
        ),
        "belief covariance lost symmetry",
    )
    try:
        np.linalg.cholesky(covariance[probability > 0])
    except np.linalg.LinAlgError as error:
        raise ValueError("saved active covariance is not proper") from error
    mean = np.where(
        (probability > 0)[..., None],
        arrays[f"{prefix}.displacement"] + arrays[f"{prefix}.center"],
        0,
    )
    second = covariance + mean[..., None] * mean[..., None, :]
    for field, expected in (
        ("mean", mean),
        ("second", second),
        ("full_covariance", covariance),
    ):
        require(
            np.allclose(
                arrays[f"{name}.{field}"], expected, atol=allowance, rtol=allowance
            ),
            f"belief physical readout mismatch: {name}.{field}",
        )
