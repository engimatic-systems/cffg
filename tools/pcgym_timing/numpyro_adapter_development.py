"""Development-only comparison of NumPyro Laplace and full-covariance SVI fits.

Run in the single-threaded x64 harness environment with tools on PYTHONPATH.
This reads optional saved seed-0 windows. It never steps a plant.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import Adam
from scipy.linalg import solve_triangular

import cstr_runtime_fixture as fixture
from numpyro_adapter import Adapter, _pack, _SHAPE
from qualification import qualify

STEPS, PARTICLES = 256, 8


class SVICandidate:
    """Reinitialize from this window's Laplace fit, then fit all 3,240 Cholesky entries."""
    def __init__(self, arrays, config, *, affine=None):
        self.laplace = Adapter(arrays, config, affine=affine)
        dim = int(np.prod(_SHAPE))

        def whitened_model(observations, masks, center, basis):
            white = numpyro.sample("white", dist.ImproperUniform(dist.constraints.real, (), (dim,)))
            states = (center + basis @ white).reshape(_SHAPE)
            numpyro.factor("jacobian", jnp.log(jnp.diag(basis)).sum())
            conditioned = numpyro.handlers.condition(self.laplace.model, data={"states": states})
            conditioned(observations, masks)

        guide = AutoMultivariateNormal(whitened_model, init_scale=1.,
                                       init_loc_fn=init_to_value(values={"white": jnp.zeros(dim)}))
        optimizer = Adam(lambda i: .01 * (.05 ** (i / (STEPS - 1))))
        svi = SVI(whitened_model, guide, optimizer, Trace_ELBO(num_particles=PARTICLES))
        empty = _pack({f"e{i}": None for i in range(_SHAPE[0])})
        initial = svi.init(jax.random.key(41), *empty, jnp.zeros(dim), jnp.eye(dim))

        @jax.jit
        def fit(observations, masks, center, basis):
            def step(state, _):
                return svi.update(state, observations, masks, center, basis)
            final, losses = jax.lax.scan(step, initial, None, length=STEPS)
            q = guide.get_posterior(svi.get_params(final))
            return center + basis @ q.loc, basis @ q.scale_tril, losses

        self.refine = fit

    def solve(self, data, *, details=True):
        inputs = _pack(data)
        center, factor, info = self.laplace._fit(*inputs)
        if factor is None:
            return dict(status="laplace_initialization_failed", action=None, **info)
        basis = solve_triangular(factor[0], np.eye(center.size), lower=True, trans="T")
        center, scale, losses = map(np.asarray, self.refine(*inputs, center, basis))
        if any(not np.isfinite(v).all() for v in (center, scale, losses)):
            return dict(status="nonfinite_svi", action=None)
        block = scale[16:20]
        return dict(status="ok", action=float(center[19]), mean=center[16:20], cov=block @ block.T,
                    next_action_cov=float(block[3] @ scale[23]), steps=STEPS, particles=PARTICLES,
                    initial_mean_loss=float(losses[:16].mean()), final_mean_loss=float(losses[-16:].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Saved development seed-0 decision directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, arrays = fixture.model()
    results = {}
    windows = [("fresh", fixture.observations()),
               ("changed_goal", fixture.observations(2, history=True, goal=.3)),
               ("filter", fixture.observations(1, history=True))]
    windows[-1][1].update({f"e{i}": None for i in range(5, 20)})
    if args.archive:
        for tick in (0, 1, 4, 20, 40):
            data = json.loads((args.archive / f"decision-{tick:02d}.json").read_text())["data"]
            windows.append((f"archived_seed_0_tick_{tick}", data))
    for name, constructor in (("laplace", Adapter), ("svi", SVICandidate)):
        start = perf_counter()
        cases = qualify(constructor, arrays)
        result = dict(affine=cases, affine_total_s=perf_counter() - start)
        start = perf_counter()
        adapter = constructor(arrays, fixture.CONFIGURATION)
        result.update(setup_s=perf_counter() - start, nonlinear={})
        for window, data in windows + windows[:1]:
            label = window if window not in result["nonlinear"] else window + "_repeat"
            start = perf_counter()
            decision = adapter.solve(data, details=True)
            result["nonlinear"][label] = {**decision, "solve_s": perf_counter() - start}
            print(name, label, decision["status"], result["nonlinear"][label]["solve_s"], flush=True)
        results[name] = result
        args.output.write_text(json.dumps(results, allow_nan=False, indent=2,
                                         default=lambda x: x.tolist()) + "\n")


if __name__ == "__main__":
    main()
