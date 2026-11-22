"""Isolated protocol checks with a synthetic plant and adapter, and the real RK4 propagator."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import jax
import numpy as np

from pcgym_timing import forecast as batch


class SyntheticPlant:
    def __init__(self, params):
        self.steps = params["N"] - 1
        self.t = None

    def _observation(self):
        return np.array([.2 * np.sin(self.t / 42), .1 * np.random.normal(), -123.])

    def reset(self, seed):
        assert self.t is None and seed == batch.TRUTH_SEED
        self.t = 0
        self.state = np.array([.83, 324.])
        return self._observation(), {}

    def step(self, action):
        assert action.shape == (1,) and abs(action[0]) < 1
        self.t += 1
        self.state[:] = [.84 + .01 * np.cos(self.t / 16), 322 + action[0]]
        return self._observation(), 0., self.t == self.steps, False, {}

    def close(self):
        assert self.t == self.steps


class SyntheticFilter:
    def __init__(self, dataset, defects=False):
        self.dataset = dataset
        self.index = 0
        self.defects = defects

    def solve(self, data, *, details):
        assert details
        index = self.index
        self.index += 1
        tick = self.dataset["origin_ticks"][index]
        obs = self.dataset["observations_normalized"]
        controls = self.dataset["controls"]
        assert set(data) == {f"e{i}" for i in range(20)}
        assert all(data[f"e{i}"] is None for i in range(5, 20))
        for i in range(4):
            j = tick - 4 + i
            np.testing.assert_array_equal(data[f"e{i}"], [*obs[j], controls[j]])
        np.testing.assert_array_equal(data["e4"], obs[tick])
        mean = np.r_[data["e4"], .002, .99]
        cov = np.diag([.002, .003, .00001, .02])
        if self.defects:
            if index == 0:
                return dict(status="unconverged", error="synthetic native failure")
            if index == 1:
                mean[0] = np.nan
            if index == 2:
                cov[3, 3] = 0
            if index == 3:
                cov[0, 1] = .02
            if index == 4:
                raise RuntimeError("synthetic exception")
        return dict(status="ok", mean=mean, cov=cov)


def check_scalar_forecast(dataset, arrays, result, particles):
    """Reconstruct the first origin with separate per-particle scalar fixture calls."""
    row = result["origins"][0]
    tick = row["tick"]
    q = np.linalg.cholesky(arrays["Qp"][:3, :3])
    cov = np.linalg.cholesky(np.array(row["s4_cov"])[:3, :3])
    z = batch._draws(dataset["draw_seed"], tick, particles)
    states = np.array(row["s4_mean"])[:3] + z[0] @ cov.T
    cfg = batch.fixture.CONFIGURATION
    low, high = (np.asarray(cfg[k]) for k in ("x_lo", "x_hi"))
    advance = jax.jit(lambda s: batch.fixture.cstr_rk4(s, arrays["theta"], cfg))
    for h in range(1, 61):
        control = dataset["controls"][tick + h - 1]
        states = np.array([np.asarray(advance(np.r_[s, control]))[:3] for s in states]) + z[h] @ q.T
        if h in batch.HORIZONS:
            values = (states[:, :2] + 1) * (high - low) / 2 + low
            for i, name in enumerate(batch.VARIABLES):
                score = row["scores"][str(h)][name]
                expected = dataset["truth_physical"][tick + h][i]
                interval = np.quantile(values[:, i], [.05, .95])
                np.testing.assert_allclose([score["mean"], score["error"]], [values[:, i].mean(),
                                           values[:, i].mean() - expected], atol=1e-11, rtol=1e-11)
                np.testing.assert_allclose([score["lower90"], score["upper90"]], interval, atol=1e-11)
                assert score["truth"] == expected
                assert score["covered90"] == (interval[0] <= expected <= interval[1])


def main():
    _, arrays = batch.fixture.model()
    cfg = batch.fixture.CONFIGURATION
    params = dict(N=600, tsim=260., model="cstr", integration_method="casadi", normalise_a=True,
                  normalise_o=True, noise=True, noise_percentage=.001, x0=[.83, 324., .85],
                  SP={"Ca": [.85] * 600}, o_space={"low": cfg["x_lo"], "high": cfg["x_hi"]},
                  a_space={"low": [cfg["u_lo"]], "high": [cfg["u_hi"]]})
    particles = 8
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "truth.json"
        rng = np.random.get_state()
        manifest = batch.prepare(SyntheticPlant, params, arrays, path)
        restored = np.random.get_state()
        np.testing.assert_equal(rng, restored)
        assert json.loads(path.read_text()) == manifest == batch.prepare(SyntheticPlant, params, arrays, path)
        assert manifest["steps"] == 599 and len(set(manifest["origin_ticks"])) == 12
        assert manifest["origin_ticks"][-1] + 60 == manifest["steps"]
        result = batch.run(SyntheticFilter(manifest), arrays, path, particles=particles)
        assert result["complete"] and result["n_scored"] == 12
        assert all(h["n"] == 12 for h in result["summary"].values())
        check_scalar_forecast(manifest, arrays, result, particles)
        damaged = batch.run(SyntheticFilter(manifest, defects=True), arrays, manifest, particles=particles)
        assert not damaged["complete"] and damaged["n_failed"] == 5
        assert [r["filter_proper"] for r in damaged["origins"][:5]] == [None, False, False, False, None]
        assert all(h["n"] == 7 for h in damaged["summary"].values())
        for row, good in zip(damaged["origins"][5:], result["origins"][5:]):
            assert row["draws_sha256"] == good["draws_sha256"] and row["scores"] == good["scores"]
        with patch.object(batch, "_compile", return_value=lambda *args: np.full((60, particles, 3), np.inf)):
            nonfinite = batch.run(SyntheticFilter(manifest), arrays, manifest, particles=particles)
        assert not nonfinite["complete"] and nonfinite["n_failed"] == 12
        assert all(row["nonfinite_particle_steps"] == 60 * particles for row in nonfinite["origins"])
        assert nonfinite["summary"]["60"]["Ca"]["coverage90"] is None
        for output in (result, damaged, nonfinite):
            json.dumps(output, allow_nan=False)
    print("Matched inputs, deterministic truth, scalar RK4, interval scoring and failure reports passed.")


if __name__ == "__main__":
    main()
