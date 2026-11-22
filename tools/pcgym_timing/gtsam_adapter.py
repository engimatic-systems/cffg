"""GTSAM sparse LM chain MAP with Gauss–Newton covariance at the returned mode.

Run setup_gtsam.sh with the harness Python and append the resulting site directory
to PYTHONPATH. The native transition uses forward-mode derivatives through RK4.
"""

import gtsam
import numpy as np

from cstr_runtime_fixture import HORIZON, PAST

SITES = PAST + HORIZON + 1


def whitener(covariance):
    return np.linalg.solve(np.linalg.cholesky(covariance), np.eye(len(covariance)))


class Adapter:
    def __init__(self, arrays, config, *, affine=None):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.origin = gtsam.Values()
        for i in range(SITES):
            self.initial.insert(i, arrays["m0"])
            self.origin.insert(i, np.zeros(4))

        self.graph.add(gtsam.PriorFactorVector(
            0, arrays["m0"], gtsam.noiseModel.Gaussian.Covariance(arrays["P0"])))
        if affine is None:
            from _gtsam_cstr import Reactor
            reactor = Reactor(arrays["theta"], config)
        for i in range(SITES - 1):
            cov = arrays["Q" if i < PAST else "Qp"]
            if affine is None:
                factor = reactor.factor(i, gtsam.noiseModel.Gaussian.Covariance(cov))
            else:
                matrix, offset = affine
                w = whitener(cov)
                linear = gtsam.JacobianFactor(i, -w @ matrix, i + 1, w, w @ offset,
                                              gtsam.noiseModel.Unit.Create(4))
                factor = gtsam.LinearContainerFactor(linear, self.origin)
            self.graph.add(factor)

        self.emissions = []
        for i in range(SITES):
            kind = "past" if i < PAST else "now" if i == PAST else "goal"
            w = whitener(arrays[f"R_{kind}"])
            self.emissions.append((w @ arrays[f"G_{kind}"], arrays[f"z_{kind}"], w,
                                   gtsam.noiseModel.Unit.Create(len(w))))

        self.params = gtsam.LevenbergMarquardtParams()
        self.params.setLinearSolverType("MULTIFRONTAL_CHOLESKY")
        self.params.setMaxIterations(100)
        self.params.setRelativeErrorTol(1e-9)
        self.params.setAbsoluteErrorTol(1e-10)
        self.method = ("affine Gaussian MAP and covariance" if affine is not None else
                       "local nonlinear MAP with Gauss-Newton covariance")

    def solve(self, data, *, details=False):
        graph = gtsam.NonlinearFactorGraph(self.graph)
        diagnostics = {"approximation": self.method}
        try:
            for i, (design, offset, w, unit) in enumerate(self.emissions):
                y = data[f"e{i}"]
                if y is not None:
                    y = np.asarray(y, dtype=np.float64)
                    if y.shape != offset.shape or not np.isfinite(y).all():
                        raise ValueError(f"Invalid observation at e{i}")
                    factor = gtsam.JacobianFactor(i, design, w @ (y - offset), unit)
                    graph.add(gtsam.LinearContainerFactor(factor, self.origin))

            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, self.initial, self.params)
            diagnostics["initial_error"] = optimizer.error()
            optimum = optimizer.optimize()
            final_error = optimizer.error()
            damping = optimizer.lambda_()
            iterations = optimizer.iterations()
            diagnostics.update(final_error=final_error, iterations=iterations, damping=damping)
            mean = optimum.atVector(PAST)
            if not np.isfinite(final_error) or not np.isfinite(mean).all():
                raise ValueError("Nonfinite LM result")
            if damping >= self.params.getlambdaUpperBound():
                raise ValueError("LM damping limit")
            if iterations >= self.params.getMaxIterations():
                raise ValueError("LM iteration budget exhausted")
            diagnostics["termination"] = "native LM cost tolerance"
            result = dict(action=float(mean[3]), status="ok", **diagnostics)
            if details:
                marginals = gtsam.Marginals(graph, optimum)
                joint = marginals.jointMarginalCovariance([PAST, PAST + 1])
                covariance = joint.at(PAST, PAST)
                cross = joint.at(PAST, PAST + 1)[3, 3]
                if not np.isfinite(covariance).all() or not np.isfinite(cross):
                    raise ValueError("Nonfinite marginal covariance")
                result.update(mean=mean, cov=covariance, next_action_cov=float(cross))
            return result
        except (RuntimeError, ValueError) as exc:
            return dict(action=None, status="failed", reason=str(exc), **diagnostics)
