"""Saved PCGym controller constants, without executable pickle dependencies.

Sources: common/agent.py and common/graphs.py in the retained experiment.
Arrays were recovered by static inspection; the exact reported residual has
no attributed seed/version. Long constant-goal horizons are scale variants,
not a replay of the later changing preview schedule.
"""

import numpy as np

from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec


CANDIDATES = {
    "v3a0": (.7144090114227575, .2857077111861867, -3.6929274279797386e-5, .00011220092771246192),
    "v3a1": (.7144800784754491, .2855783037344391, -4.285648570903188e-5, .00011367953760745913),
    "v3a2": (.7143470897195969, .28567782106675776, -7.000279723140456e-5, .00011375372197826516),
    "v10": (.7142927729189439, .2858270440540955, -3.8439545854748584e-5, .00011074976499567734),
    "v11": (.7143146527529188, .28578301204567746, -3.847631376898361e-5, .00011166588308022319),
    "v12": (.7146330743666196, .28535471993985984, -6.928244201627833e-5, .00011210870474908996),
}


def first_order_model(
    candidate="v3a0", *, dtype=np.float64, source_dtype=np.float64,
    past=4, horizon=15,
):
    """Preserve source ordering and float64 covariance packing by default.

    ``dtype`` selects observation storage; callers select JAX precision before
    compilation. ``source_dtype=float32`` is a separate input-rounding variant.
    """
    a, gain, offset, process_variance = CANDIDATES[candidate]
    arrays = {
        "F": [[a, 1., gain], [0., 1., 0.], [0., 0., 1.]],
        "c": [offset, 0., 0.], "Q": np.diag([process_variance, .000025, 4.]),
        "m0": np.zeros(3), "P0": np.diag([1., .000625, 1.]),
        "G_past": np.eye(3)[[0, 2]], "R_past": np.diag([.0001, .000001]),
        "G_now": np.eye(3)[[0]], "R_now": [[.0001]],
        "G_goal": np.eye(3)[[0, 2]], "R_goal": np.diag([.0004, 1.]),
    }
    arrays.update(Fp=arrays["F"], cp=arrays["c"], Qp=arrays["Q"])
    arrays.update({f"z_{kind}": np.zeros(len(arrays[f"G_{kind}"])) for kind in ("past", "now", "goal")})
    arrays = {name: np.array(value, dtype=source_dtype) for name, value in arrays.items()}
    template = Template(name=f"retained_first_order_{candidate}_p{past}_h{horizon}")
    template.blobs = {name: BlobRef(name, f"{name}.npy") for name in arrays}

    def variable(name, dimension=3):
        template.variables[name] = VariableSpec(name, dim=dimension)
        return name

    template.factors["prior"] = FactorSpec(
        "prior", "gaussian", {"out": variable("s0")}, param_blobs={"mean": "m0", "cov": "P0"},
    )
    data = {}
    for index in range(past + horizon + 1):
        kind = "past" if index < past else "now" if index == past else "goal"
        emission = source = f"s{index}"
        if index < past + horizon:
            source, emission = variable(f"s{index}c"), variable(f"s{index}o")
            name = f"eq{index}"
            template.factors[name] = FactorSpec(name, "equality", {"a": f"s{index}", "b": source, "c": emission})
        port = variable(f"e{index}", len(arrays[f"G_{kind}"]))
        name = f"em{index}"
        template.factors[name] = FactorSpec(
            name, "gaussian_linear", {"in": emission, "out": port},
            param_blobs={"matrix": f"G_{kind}", "offset": f"z_{kind}", "cov": f"R_{kind}"},
        )
        template.ports[port] = PortSpec(port, port, PortRole.DATA)
        data[port] = None if kind == "past" else np.array([-1.] if kind == "now" else [-.6, 0.], dtype=dtype)
        if index < past + horizon:
            name = f"tr{index + 1}"
            suffix = "" if index < past else "p"
            template.factors[name] = FactorSpec(
                name, "gaussian_linear", {"in": source, "out": variable(f"s{index + 1}")},
                param_blobs={"matrix": "F" + suffix, "offset": "c" + suffix, "cov": "Q" + suffix},
            )
    return template, arrays, data


def dense_reference(arrays, data, *, past=4, runtime_dtype=None):
    """Independent float64 conditioning, optionally on runtime-rounded inputs.

    Packing inverts each source covariance before casting its precision. The
    rounded reference reproduces that conversion, not a float32 covariance
    inversion. It uses no CFFG messages, energies or readouts.
    """
    converted = {}
    for name, value in arrays.items():
        if runtime_dtype is None:
            converted[name] = np.asarray(value, dtype=np.float64)
        elif name in {"Q", "Qp", "P0", "R_past", "R_now", "R_goal"}:
            precision = np.linalg.inv(value).astype(runtime_dtype).astype(np.float64)
            converted[name] = np.linalg.inv(precision)
        else:
            converted[name] = np.asarray(value, dtype=runtime_dtype).astype(np.float64)
    arrays = converted
    dimension, length = 3, len(data)
    mean = np.zeros(length * dimension)
    covariance = np.zeros((len(mean), len(mean)))
    mean[:dimension], covariance[:dimension, :dimension] = arrays["m0"], arrays["P0"]
    for index in range(1, length):
        suffix = "" if index <= past else "p"
        matrix, offset, noise = (arrays[name + suffix] for name in ("F", "c", "Q"))
        start, end = index * dimension, (index + 1) * dimension
        mean[start:end] = matrix @ mean[start-dimension:start] + offset
        covariance[start:end, :start] = matrix @ covariance[start-dimension:start, :start]
        covariance[:start, start:end] = covariance[start:end, :start].T
        covariance[start:end, start:end] = noise + matrix @ covariance[start-dimension:start, start-dimension:start] @ matrix.T
    rows, noise, observations = [], [], []
    for index in range(length):
        if data[f"e{index}"] is None:
            continue
        kind = "past" if index < past else "now" if index == past else "goal"
        projection = arrays[f"G_{kind}"]
        row = np.zeros((len(projection), len(mean)))
        row[:, index*dimension:(index+1)*dimension] = projection
        rows.append(row)
        noise.extend(np.diag(arrays[f"R_{kind}"]))
        observed = np.asarray(data[f"e{index}"], dtype=runtime_dtype).astype(np.float64)
        observations.extend(observed - arrays[f"z_{kind}"])
    if not rows:
        return mean, covariance, 0.
    projection = np.concatenate(rows)
    innovation = np.diag(noise) + projection @ covariance @ projection.T
    residual = np.asarray(observations) - projection @ mean
    gain = np.linalg.solve(innovation, projection @ covariance).T
    posterior_mean = mean + gain @ residual
    posterior_covariance = covariance - gain @ projection @ covariance
    energy = .5 * (len(residual) * np.log(2 * np.pi) + np.linalg.slogdet(innovation)[1]
                    + residual @ np.linalg.solve(innovation, residual))
    return posterior_mean, posterior_covariance, energy
