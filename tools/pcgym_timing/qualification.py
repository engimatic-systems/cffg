"""Compare each native backend against independent dense, whitened least squares."""
import numpy as np
from scipy.linalg import solve_triangular

import cstr_runtime_fixture as fixture


def reference(arrays, data, matrix, offset):
    width = 4 * len(data)
    rows, targets = [], []

    def add(blocks, response, noise):
        design = np.zeros((len(response), width))
        for site, block in blocks:
            design[:, 4 * site:4 * site + 4] = block
        chol = np.linalg.cholesky(noise)
        rows.append(solve_triangular(chol, design, lower=True))
        targets.append(solve_triangular(chol, response, lower=True))

    identity = np.eye(4)
    add([(0, identity)], arrays["m0"], arrays["P0"])
    for i in range(len(data)):
        if i:
            noise = arrays["Q" if i <= fixture.PAST else "Qp"]
            add([(i-1, -matrix), (i, identity)], offset, noise)
        kind = "past" if i < fixture.PAST else "now" if i == fixture.PAST else "goal"
        y = data[f"e{i}"]
        if y is not None:
            add([(i, arrays["G_"+kind])], y - arrays["z_"+kind], arrays["R_"+kind])
    q, r = np.linalg.qr(np.vstack(rows), mode="reduced")
    center = solve_triangular(r, q.T @ np.concatenate(targets))
    inverse = solve_triangular(r, np.eye(width))
    covariance = inverse @ inverse.T
    return dict(action=center[19], mean=center[16:20], cov=covariance[16:20,16:20],
                next_action_cov=covariance[19,23])


def qualify(adapter_cls, arrays):
    matrix = np.array([[.91, .04, .15, .08], [.03,.89,.04,.12], [0,0,1,0], [0,0,0,1]])
    offset = np.array([.017,-.02,0.,0.])
    adapter = adapter_cls(arrays, fixture.CONFIGURATION, affine=(matrix, offset))
    results = []
    try:
        for name, data in (("fresh", fixture.observations()),
                           ("changed_goal", fixture.observations(2, history=True, goal=.3)),
                           ("filter", fixture.observations(1, history=True))):
            if name == "filter":
                data.update({f"e{i}": None for i in range(5,20)})
            expected = reference(arrays, data, matrix, offset)
            actual = adapter.solve(data, details=True)
            if actual["status"] != "ok":
                raise ValueError(f"{name} affine solve failed: {actual}")
            errors = {}
            for key, target in expected.items():
                value = np.asarray(actual[key])
                if value.shape != np.shape(target) or not np.isfinite(value).all():
                    raise ValueError(f"{name}: invalid {key} value")
                errors[key] = float(np.max(abs(value-target)))
            cov = expected["cov"]
            chol = np.linalg.cholesky(cov)
            d = np.linalg.solve(chol, np.asarray(actual["cov"]) - cov)
            mean_error = max(abs(np.linalg.solve(chol, np.asarray(actual["mean"]) - expected["mean"])))
            cov_error = np.max(abs(np.linalg.solve(chol, d.T)))
            # Error in current-posterior standard deviations (covariance in whitened units).
            passed = (mean_error <= 1e-3 and cov_error <= 1e-3 and errors["action"] <= 1e-3
                      and errors["next_action_cov"] <= 1e-4)
            results.append(dict(case=name, passed=bool(passed), absolute_errors=errors,
                                mean_whitened=float(mean_error), cov_whitened=float(cov_error)))
    finally:
        if hasattr(adapter, "close"):
            adapter.close()
    return results
