"""Check whether public mixed-Gaussian statistics retain within-component spread."""
import argparse
import json
import subprocess

import jax
import numpy as np

from cffg.compiler import compile_template
from cffg.core.ir import BlobRef, FactorSpec, Template, VariableSpec
from cffg.engine import Engine


def check(origin, component_variance, dtype, *, displaced=False):
    """Observe two Gaussian components with a common Gaussian likelihood."""
    component_offset = np.array([0., 0.125]) if displaced else np.zeros(2)
    observation_offset = 1. if displaced else 0.
    observed_variance = 0.25 if displaced else component_variance
    variances = np.array([0.5, 1.]) if displaced else np.full(2, component_variance)
    arrays = {k: np.array(v, dtype=dtype) for k, v in {
        "m": (origin + component_offset)[:, None],
        "V": variances[:, None, None],
        "y": [origin + observation_offset], "R": [[observed_variance]], "p": [0.5, 0.5],
    }.items()}
    model = Template(
        name="mixed_feature_checkpoint",
        variables={"x": VariableSpec("x", dim=1), "s": VariableSpec("s", cardinality=2)},
        factors={
            "class": FactorSpec("class", "categorical", {"out": "s"}, param_blobs={"statistic": "p"}),
            "sensor": FactorSpec("sensor", "gaussian", {"out": "x"}, param_blobs={"mean": "y", "cov": "R"}),
            "object": FactorSpec("object", "gaussian_mixture", {"out": "x", "selector": "s"},
                                 param_blobs={"mean": "m", "cov": "V"}),
        },
        blobs={k: BlobRef(k, k + ".npy") for k in arrays},
    )
    engine = Engine(compile_template(model, arrays=arrays))
    # Each incoming message at object comes directly from a normalized leaf.
    state = engine.step(engine.init_state())
    stats = engine.gaussian_node_stats(state)["object"]
    assert bool(stats["proper"])
    reported_dtype = stats["weighted_second"].dtype
    assert reported_dtype == np.dtype(dtype), (reported_dtype, dtype)
    w = stats["responsibility"].astype(float)
    first = stats["weighted_mean"][:, 0].astype(float)
    second = stats["weighted_second"][:, 0, 0].astype(float)
    mean = first / w
    recovered = second / w - mean**2
    exact = 1 / (1 / variances + 1 / observed_variance)
    likelihood = np.exp(-0.5 * (component_offset-observation_offset)**2 /
                        (variances+observed_variance)) / np.sqrt(variances+observed_variance)
    exact_weight = likelihood / sum(likelihood)
    return dict(dtype=dtype.__name__, origin=origin, displaced=displaced,
                prior_variances=variances.tolist(), proper=True, weights=w.tolist(),
                exact_weights=exact_weight.tolist(), weighted_mean=first.tolist(),
                weighted_second=second.tolist(), recovered_variance=recovered.tolist(),
                exact_variance=exact.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dtype", choices=("float32", "float64"))
    dtype = np.dtype(parser.parse_args().dtype).type
    jax.config.update("jax_enable_x64", dtype is np.float64)
    rows = [check(origin, v, dtype) for origin in (0., 1., 128., 10000.)
            for v in (.25, 1.)]
    rows.extend(check(origin, .5, dtype, displaced=True) for origin in (0., 10000.))
    print(json.dumps({"repo_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "jax": jax.__version__,
        "device": str(jax.devices()[0]), "rows": rows}, indent=2, allow_nan=False))
