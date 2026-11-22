"""Public directed CFFG controller; expose the frozen joint for the oracle audit."""
from dataclasses import replace

import numpy as np

import cstr_runtime_fixture as fixture
from cffg import Engine, FunctionSpec, compile_template
from cffg.bounded_chain import _frozen_joint


class Adapter:
    def __init__(self, arrays, config, *, affine=None):
        template, _ = fixture.model()
        if config != fixture.CONFIGURATION:
            raise ValueError("CFFG fixture and benchmark physics configuration differ")
        if affine is not None:
            matrix, offset = affine
            arrays = {**arrays, "theta": np.r_[matrix.ravel(), offset]}
            function = FunctionSpec("cffg.mean.affine", "1.0.0")
            for name, factor in template.factors.items():
                if factor.type == "gaussian_nonlinear":
                    template.factors[name] = replace(factor, function=function)
        self.engine = Engine(compile_template(template, arrays=arrays))
        self.binder = fixture.Binder(self.engine)
        self.base = self.engine.init_state()

    def solve(self, data, *, details=False):
        result = self.engine.run(self.binder.bind(self.base, data), fixture.BUDGET, tol=fixture.TOLERANCE)
        if result.termination_reason.value != "directed_chain_certified":
            return dict(status=result.termination_reason.value, action=None)
        state = result.require_converged()
        mean, cov = self.engine.gaussian_marginal(state, "s4")
        answer = dict(action=float(mean[3]), status="ok", residual=result.residual)
        if details:
            _, joint, _ = _frozen_joint(self.engine._directed, state)
            answer.update(mean=mean, cov=cov, next_action_cov=float(joint[19, 23]))
        return answer
