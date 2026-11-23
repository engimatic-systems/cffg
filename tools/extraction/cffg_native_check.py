"""Run synthetic CFFG qualification with JAX_ENABLE_X64=1 and one BLAS thread."""

from dataclasses import replace
import json
from time import perf_counter

from cffg import Engine, compile_template
from cffg.core import BlobRef
from cffg.models import lgssm
import numpy as np
from scipy.linalg import block_diag, solve_triangular

from cffg_native import Backend


def fixture():
    rng = np.random.default_rng(2026)
    n, h = 11, 6
    m = rng.normal(size=n)
    full = rng.normal(size=(n, n))
    P = .001 * (np.eye(n) + full @ full.T / n)
    As = .85 * np.eye(n) + rng.normal(0, .035, (h, n, n))
    Bs = rng.normal(0, .002, (h, n, 2))
    bs = rng.normal(0, .1, (h, n))
    Qs = np.stack([.003 * P + np.eye(n) * 1e-6 for _ in range(h)])
    H = np.eye(n)[[8, 1]]
    R = np.array([[4., .8], [.8, 9.]]) * 1e-6
    y = H @ (As[0] @ m + bs[0]) + [.005, -.003]
    U0 = np.tile([75., 30.], (h, 1)) + rng.normal(size=(h, 2))
    Ucov = np.array([[225., 15.], [15., 64.]])
    targets = rng.normal(.3, .02, (h, 1))
    return (m, P, As[0], bs[0], Qs[0], H, R, y), (m, P, As, bs, Bs, Qs, H[:1], targets, .012**2, U0, Ucov)


def dense_update(m, P, A, b, Q, H, R, y):
    prediction = A @ m + b
    cov = A @ P @ A.T + Q
    if y is None:
        return prediction, cov, 0.
    residual = y - H @ prediction
    S = H @ cov @ H.T + R
    gain = np.linalg.solve(S, H @ cov).T
    nll = .5 * (len(y)*np.log(2*np.pi) + np.linalg.slogdet(S)[1] + residual @ np.linalg.solve(S, residual))
    return prediction + gain @ residual, cov - gain @ H @ cov, nll


def dense_plan(m, P, As, bs, Bs, Qs, C, targets, Rgoal, U0, Ucov):
    """Independent full trajectory fit of all state and control residuals by QR."""
    n, h, nu = len(m), len(As), 2
    controls = (h + 1) * n
    cols = controls + h * nu
    rows, rhs = [], []

    def add(parts, target, noise):
        design = np.zeros((len(target), cols))
        for start, values in parts:
            design[:, start:start+values.shape[1]] = values
        chol = np.linalg.cholesky(noise)
        rows.append(solve_triangular(chol, design, lower=True))
        rhs.append(solve_triangular(chol, target, lower=True))

    add([(0, np.eye(n))], m, P)
    for j in range(h):
        x, xp, u = n*j, n*(j+1), controls+nu*j
        add([(x, -As[j]), (u, -Bs[j]), (xp, np.eye(n))], bs[j], Qs[j])
        add([(u, np.eye(nu))], U0[j], Ucov)
        add([(xp, C)], targets[j], np.atleast_2d(Rgoal))
    q, r = np.linalg.qr(np.vstack(rows), mode="reduced")
    return solve_triangular(r, q.T @ np.concatenate(rhs))[controls:controls+nu]


def engine_update(args):
    m, P, A, b, Q, H, R, y = args
    template, arrays = lgssm(2, A, Q, H, R, m, P)
    arrays["zn"] = b
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), {"y0": None, "y1": y})
    state = engine.run(state, 60, tol=1e-10).require_converged()
    return (*engine.gaussian_marginal(state, "x1"), engine.bfe(state))


def engine_plan(args):
    """Public template with every state paired with its independent action."""
    m, P, As, bs, Bs, Qs, C, targets, Rgoal, U0, Ucov = args
    n, h = len(m), len(As)
    template, arrays = lgssm(h+1, np.eye(n+2), np.eye(n+2), np.c_[C, np.zeros((1,2))],
                             np.atleast_2d(Rgoal), np.r_[m, U0[0]], block_diag(P, Ucov))
    for j in range(h):
        factor = f"trans{j+1}"
        names = {}
        lower = np.zeros((2, n+2))
        for role, val in (("matrix", np.vstack((np.c_[As[j], Bs[j]], lower))),
                          ("cov", block_diag(Qs[j], Ucov)),
                          ("offset", np.r_[bs[j], U0[min(j+1, h-1)]])):
            name = f"{factor}_{role}"
            names[role] = name
            template.blobs[name] = BlobRef(name, f"blobs/{name}.npy")
            arrays[name] = val
        template.factors[factor] = replace(template.factors[factor], param_blobs=names)
    engine = Engine(compile_template(template, arrays=arrays))
    data = {"y0": None, **{f"y{j+1}": targets[j] for j in range(h)}}
    state = engine.bind_data(engine.init_state(), data)
    state = engine.run(state, 60, tol=1e-9).require_converged()
    return engine.gaussian_marginal(state, "x0")[0][-2:]


def run():
    update, plan = fixture()
    backend = Backend()
    expected = dense_update(*update)
    actual = (*backend.update(*update), backend.last_innovation_nll)
    public = engine_update(update)
    for ref in (expected, public):
        for value, target in zip(actual, ref):
            np.testing.assert_allclose(value, target, atol=1e-8, rtol=1e-9)
    missing = (*update[:-1], None)
    for value, target in zip(backend.update(*missing), dense_update(*missing)[:2]):
        np.testing.assert_allclose(value, target, atol=1e-12)
    assert backend.last_innovation_nll == 0.

    deterministic = list(update)
    deterministic[1] = np.diag([.001]*10 + [0.])
    deterministic[2] = np.eye(11)
    deterministic[4] = np.diag([1e-6]*10 + [0.])
    for value, target in zip(backend.update(*deterministic), dense_update(*deterministic)[:2]):
        np.testing.assert_allclose(value, target, atol=1e-12)
    invalid = list(update)
    invalid[4] = -invalid[4]
    with np.testing.assert_raises(ValueError):
        backend.update(*invalid)
    assert backend.last_innovation_nll is None
    smaller = [np.asarray(v, dtype=np.float32) for v in update]
    for value, target in zip(backend.update(*smaller), expected[:2]):
        assert value.dtype == np.float64
        np.testing.assert_allclose(value, target, atol=1e-6)

    raw = backend.plan(*plan)
    oracle = dense_plan(*plan)
    compiled = engine_plan(plan)
    for ref in (oracle, compiled):
        np.testing.assert_allclose(raw, ref, atol=1e-7, rtol=1e-10)
    altered = list(plan)
    altered[7] = plan[7] + .1
    assert np.all(np.abs(backend.plan(*altered) - raw) > 1e-4)
    for value, target in zip(backend.update(*update), actual[:2]):
        np.testing.assert_array_equal(value, target)
    short = list(plan)
    for index in (2, 3, 4, 5, 7, 9):
        short[index] = short[index][:1]
    np.testing.assert_allclose(backend.plan(*short), dense_plan(*short), atol=1e-9)
    uninformative = list(plan)
    uninformative[6] = np.zeros_like(plan[6])
    np.testing.assert_allclose(backend.plan(*uninformative), plan[9][0], atol=1e-12)

    timings = {}
    for name, method, args in (("update", backend.update, update), ("plan", backend.plan, plan)):
        times = []
        for _ in range(100):
            begin = perf_counter()
            method(*args)
            times.append((perf_counter() - begin)*1e3)
        timings[name] = dict(median_ms=float(np.median(times)), p95_ms=float(np.quantile(times, .95)))
    return dict(innovation_nll=actual[2], innovation_nll_error=abs(actual[2]-expected[2]),
                engine_mean_error=float(max(abs(public[0]-actual[0]))),
                engine_cov_error=float(np.max(abs(public[1]-actual[1]))),
                dense_action_error=float(max(abs(raw-oracle))), engine_action_error=float(max(abs(raw-compiled))),
                synthetic_first_action=raw.tolist(), synthetic_timings=timings)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
