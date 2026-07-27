"""Executable factor registration (cff-ri8m): the FactorKernel contract.

The load-bearing claim: a *new* conjugate observation factor — exponential
observations of a gamma rate — registers from this test module with zero
edits to cffg compiler/engine/validation source, runs with an exact
independent oracle (hand conjugate posterior and −log evidence), and
round-trips through template JSON and .cffgz via its namespaced
(impl_id, version), failing loudly when the implementation is missing.
"""

from math import lgamma, log

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import kernels
from cffg.compiler import CompileError, compile_template
from cffg.core import (
    BlobRef,
    FACTOR_TYPES,
    FactorSpec,
    FactorType,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.engine import Engine
from cffg.kernels import FactorKernel, HostileInputs

CUSTOM = "exponential_obs"
IMPL, VERSION = "cffgtests.exponential_obs", "1.2.0"

_HOSTILE = HostileInputs(
    sparse_support="provably N/A: constant likelihood increment; no support pruning on EF edges",
    improper_messages="improper rate beliefs return +inf energy via the propriety guard",
    float32="increment and energy derive dtype from the incoming eta; no literals",
    scale="scalar observations; member count only batches the leading axis",
)


def _exp_increment(obs, mask):
    """x ~ Exponential(rate lam): likelihood lam*exp(-lam*x) has Gamma-natural
    T-coefficients (1, -x)."""
    return mask[..., None] * jnp.stack([jnp.ones_like(obs), -obs], axis=-1)


def _exp_message(inc, params, obs, mask):
    return _exp_increment(obs, mask)[:, None, :].astype(inc.dtype)


def _exp_energy(inc, params, obs, mask):
    from cffg import efrules

    eta_q = inc[:, 0] + _exp_increment(obs, mask).astype(inc.dtype)
    stats = efrules.JFAMILIES["gamma"].mean_stats(eta_q)
    e_logf = stats[..., 0] - obs * stats[..., 1]  # E[log lam] - x E[lam]
    return jnp.where(
        efrules.JFAMILIES["gamma"].is_proper(eta_q),
        -efrules.JFAMILIES["gamma"].entropy(eta_q) - mask * e_logf,
        jnp.inf,
    )


def _exp_readout(inc, params, obs, mask):
    eta_q = np.asarray(inc[:, 0]) + np.asarray(_exp_increment(jnp.asarray(obs), jnp.asarray(mask)))
    alpha, beta = eta_q[..., 0] + 1.0, -eta_q[..., 1]
    return {"rate_mean": alpha / beta}


def _validate_exponential(value) -> float:
    val = float(value)
    if not val >= 0:
        raise ValueError(f"Exponential observations are >= 0, got {value}")
    return val


def _make_kernel(**overrides) -> FactorKernel:
    spec = dict(
        factor_type=FactorType(CUSTOM, edge_roles=("out", "rate"), family="ef"),
        impl_id=IMPL,
        version=VERSION,
        arena="ef",
        family="gamma",
        message_roles=("rate",),
        edge_families={"out": "exponential", "rate": "gamma"},
        message=_exp_message,
        energy=_exp_energy,
        hostile=_HOSTILE,
        observed_out="exponential",
        validate_observation=_validate_exponential,
        readout=_exp_readout,
    )
    spec.update(overrides)
    return FactorKernel(**spec)


@pytest.fixture
def exponential_kernel():
    k = kernels.register_kernel(_make_kernel())
    try:
        yield k
    finally:
        kernels.unregister_kernel(CUSTOM)
    assert CUSTOM not in FACTOR_TYPES  # unregister removed what it added


def gamma_exponential(a0: float = 2.0, b0: float = 1.5) -> tuple[Template, dict]:
    t = Template(name="gamma_exponential")
    t.variables["lam"] = VariableSpec("lam", ef="gamma")
    t.variables["x0"] = VariableSpec("x0", ef="exponential")
    t.factors["prior"] = FactorSpec(
        "prior", "gamma", edges={"out": "lam"}, param_blobs={"shape": "a0", "rate": "b0"}
    )
    t.factors["obs"] = FactorSpec("obs", CUSTOM, edges={"out": "x0", "rate": "lam"})
    t.ports["draw"] = PortSpec("draw", "x0", PortRole.DATA)
    t.blobs["a0"] = BlobRef("a0", "blobs/a0.npy")
    t.blobs["b0"] = BlobRef("b0", "blobs/b0.npy")
    return t, {"a0": np.array(a0), "b0": np.array(b0)}


def _log_evidence(a0, b0, x):
    """p(x) = a0 * b0^a0 / (b0 + x)^(a0 + 1): the Lomax marginal."""
    return log(a0) + a0 * log(b0) - (a0 + 1.0) * log(b0 + x)


# --- the executable contract end to end ---------------------------------------


def test_custom_kernel_runs_with_independent_oracle(exponential_kernel):
    a0, b0, x = 2.0, 1.5, 0.7
    t, arrays = gamma_exponential(a0, b0)
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.bind_data(eng.init_state(), {"draw": x})
    st, deltas = eng.run(st, 10, tol=1e-12).require_converged_with_trace()
    assert deltas[-1] < 1e-12

    post = eng.ef_marginals(st)["lam"]
    assert post.family == "gamma" and post.proper
    np.testing.assert_allclose(post.params, (a0 + 1.0, b0 + x), rtol=1e-12)
    np.testing.assert_allclose(eng.bfe(st), -_log_evidence(a0, b0, x), rtol=1e-12)

    readout = eng.kernel_readouts(st)
    np.testing.assert_allclose(readout["obs"]["rate_mean"], (a0 + 1.0) / (b0 + x), rtol=1e-12)


def test_custom_kernel_requires_no_central_source_edits(exponential_kernel):
    """AC3 made executable: the dispatching modules never name the type."""
    import importlib
    from pathlib import Path

    for name in ("cffg.compiler", "cffg.engine", "cffg.core.validate"):
        module = importlib.import_module(name)
        assert CUSTOM not in Path(module.__file__).read_text(), name


def test_missing_observation_drops_the_node(exponential_kernel):
    a0, b0 = 2.0, 1.5
    t, arrays = gamma_exponential(a0, b0)
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.bind_data(eng.init_state(), {"draw": None})
    st = eng.run(st, 10, tol=1e-12).require_converged()
    post = eng.ef_marginals(st)["lam"]
    np.testing.assert_allclose(post.params, (a0, b0), rtol=1e-12)  # prior intact


def test_observation_validation_dispatches_to_the_kernel(exponential_kernel):
    t, arrays = gamma_exponential()
    eng = Engine(compile_template(t, arrays=arrays))
    with pytest.raises(ValueError, match="port 'draw'.*Exponential observations are >= 0"):
        eng.bind_data(eng.init_state(), {"draw": -0.5})


def test_float32_run_stays_finite(exponential_kernel):
    with jax.enable_x64(False):
        t, arrays = gamma_exponential()
        eng = Engine(compile_template(t, arrays=arrays))
        st = eng.bind_data(eng.init_state(), {"draw": 0.7})
        st = eng.run(st, 10, tol=1e-6).require_converged()
        assert np.isfinite(eng.bfe(st))
        assert eng.ef_marginals(st)["lam"].proper


# --- serialization: namespaced identity + loud failure -------------------------


def test_json_roundtrip_records_impl_identity(exponential_kernel):
    t, _ = gamma_exponential()
    text = to_json(t)
    assert IMPL in text and VERSION in text
    back = from_json(text)
    assert back.factors["obs"].type == CUSTOM
    assert structure_hash(back) == structure_hash(t)


def test_cffgz_roundtrip(exponential_kernel, tmp_path):
    from cffg.bundle import load_bundle, save_bundle

    t, arrays = gamma_exponential()
    path = tmp_path / "exp.cffgz"
    save_bundle(path, t, arrays)
    loaded_t, loaded_arrays, _ = load_bundle(path)
    assert structure_hash(loaded_t) == structure_hash(t)
    eng = Engine(compile_template(loaded_t, arrays=loaded_arrays))
    st = eng.bind_data(eng.init_state(), {"draw": 0.7})
    st = eng.run(st, 10, tol=1e-12).require_converged()
    np.testing.assert_allclose(eng.bfe(st), -_log_evidence(2.0, 1.5, 0.7), rtol=1e-12)


def test_missing_implementation_fails_loudly(exponential_kernel):
    text = to_json(gamma_exponential()[0])
    kernels.unregister_kernel(CUSTOM)
    try:
        with pytest.raises(ValueError, match=f"{CUSTOM}.*{IMPL}.*{VERSION}"):
            from_json(text)
    finally:  # restore for the fixture's own unregister
        kernels.register_kernel(_make_kernel())


def test_missing_implementation_fails_when_factor_type_was_pre_registered():
    """A structural FactorType is not evidence that its executable kernel is
    installed: the serialized implementation requirement must still fail at
    load time (cff-ri8m)."""
    from cffg.core import register_factor_type
    from cffg.core.ir import EF_EDGE_FAMILIES

    factor_type = "predeclared_exponential_obs"
    impl = "cffgtests.predeclared_exponential_obs"
    ft = register_factor_type(
        FactorType(factor_type, edge_roles=("out", "rate"), family="ef")
    )
    try:
        kernels.register_kernel(_make_kernel(factor_type=ft, impl_id=impl))
        t, _ = gamma_exponential()
        t.factors["obs"] = FactorSpec(
            "obs", factor_type, edges={"out": "x0", "rate": "lam"}
        )
        text = to_json(t)

        # unregister_kernel deliberately leaves a FactorType it did not add.
        # Loading must check the executable manifest, not just that structural
        # registry entry.
        kernels.unregister_kernel(factor_type)
        assert factor_type in FACTOR_TYPES
        with pytest.raises(ValueError, match=f"{factor_type}.*{impl}.*{VERSION}"):
            from_json(text)
    finally:
        kernels.unregister_kernel(factor_type)
        FACTOR_TYPES.pop(factor_type, None)
        EF_EDGE_FAMILIES.pop(factor_type, None)


def test_incompatible_major_version_fails_loudly(exponential_kernel):
    text = to_json(gamma_exponential()[0])
    kernels.unregister_kernel(CUSTOM)
    kernels.register_kernel(_make_kernel(version="2.0.0"))
    with pytest.raises(ValueError, match="incompatible"):
        from_json(text)


# --- registration is refused without the declarations --------------------------


def test_registration_requires_hostile_declarations():
    blank = HostileInputs(sparse_support="", improper_messages="x", float32="x", scale="x")
    with pytest.raises(ValueError, match="hostile-input declaration 'sparse_support'"):
        kernels.register_kernel(_make_kernel(hostile=blank))
    assert CUSTOM not in FACTOR_TYPES  # refused registration leaves no trace


def test_registration_guards_identity_and_structure():
    with pytest.raises(ValueError, match="namespaced"):
        kernels.register_kernel(_make_kernel(impl_id="nodots"))
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        kernels.register_kernel(_make_kernel(version="1.0"))
    with pytest.raises(ValueError, match="validate_observation"):
        kernels.register_kernel(_make_kernel(validate_observation=None))
    with pytest.raises(ValueError, match="arena"):
        kernels.register_kernel(_make_kernel(arena="discrete"))
    assert CUSTOM not in FACTOR_TYPES


def test_unregistered_ef_type_fails_at_compile():
    """A structural registration without a kernel cannot silently compile."""
    from cffg.core import register_factor_type

    mystery = "mystery_prior"
    register_factor_type(FactorType(mystery, edge_roles=("out",), family="ef"))
    try:
        t, arrays = gamma_exponential()
        del t.factors["obs"], t.variables["x0"], t.ports["draw"]
        t.factors["mystery"] = FactorSpec("mystery", mystery, edges={"out": "lam"})
        with pytest.raises(CompileError, match="no registered.*FactorKernel"):
            compile_template(t, arrays=arrays)
    finally:
        FACTOR_TYPES.pop(mystery, None)
