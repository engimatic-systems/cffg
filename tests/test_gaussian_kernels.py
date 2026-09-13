"""End-to-end Gaussian registration against covariance and KL oracles."""

from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gaussian_kernel_fixture import (
    CUSTOM,
    IMPL,
    ROLES,
    covariance_oracle,
    declaration,
    model,
    role_indices,
)
from scipy.linalg import block_diag

from cffg import inits, kernels
from cffg.compiler import CompileError, compile_template
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.engine import Engine, RunStatus, _replace
from cffg.estimate import estimate


@pytest.fixture
def registered():
    kernel = kernels.register_kernel(declaration())
    yield kernel
    kernels.unregister_kernel(CUSTOM)


@pytest.mark.parametrize("schedule", ["auto", "flood"])
@pytest.mark.parametrize(
    "conditioned,present", [(False, False), (True, False), (True, True)]
)
def test_external_joint_and_middle_role_conditioning(
    registered, schedule, conditioned, present
):
    dims = (3, 2, 1)
    t, arrays = model(dims, observed=conditioned)
    eng = Engine(compile_template(t, arrays=arrays), schedule=schedule)
    state = eng.init_state()
    observation = np.array([0.8, -0.3]) if present else None
    if conditioned:
        state = eng.bind_data(state, {"sensor": observation})
        assert eng.arena.gauss_groups[0].observed_role == 1
        assert state.g_obs.shape == (2,)
    assert not np.any(state.g_eta) and not np.any(state.g_lam)
    state = eng.run(state, 20, tol=1e-11).require_converged()
    mean, covariance, energy = covariance_oracle(arrays, dims, observation, conditioned)
    np.testing.assert_allclose(eng.bfe(state), energy, atol=1e-11)
    for values in (
        eng.kernel_readouts(state)["joint"],
        eng.gaussian_node_stats(state)["joint"],
    ):
        assert values["proper"]
        np.testing.assert_allclose(values["mean"], mean, atol=1e-11)
        np.testing.assert_allclose(values["covariance"], covariance, atol=1e-11)
    remaining = (0, 2) if conditioned else (0, 1, 2)
    offsets = np.cumsum((0, *(dims[r] for r in remaining)))
    marginals = eng.gaussian_marginals(state)
    for local, role in enumerate(remaining):
        section = slice(offsets[local], offsets[local + 1])
        np.testing.assert_allclose(marginals[ROLES[role]][0], mean[section], atol=1e-11)
        np.testing.assert_allclose(
            marginals[ROLES[role]][1], covariance[section, section], atol=1e-11
        )
    if conditioned:
        # DATA role still receives its cavity predictive, excluding its own value.
        m, v, _ = covariance_oracle(arrays, dims)
        indices = role_indices(dims)[1]
        np.testing.assert_allclose(marginals["sensor"][0], m[indices], atol=1e-11)
        np.testing.assert_allclose(
            marginals["sensor"][1], v[np.ix_(indices, indices)], atol=1e-11
        )


@pytest.mark.parametrize("conditioned", [False, True])
def test_nonvacuous_cavities_messages_and_kl_energy(registered, conditioned):
    dims = (3, 2, 1)
    t, arrays = model(dims, observed=conditioned)
    eng = Engine(compile_template(t, arrays=arrays), schedule="flood")
    state = eng.init_state()
    observation = np.array([0.5, -0.2]) if conditioned else None
    if conditioned:
        state = eng.bind_data(state, {"sensor": observation})
    group = eng.arena.gauss_groups[0]
    incoming_e = [np.linspace(0.2, 0.7, d) for d in dims]
    incoming_p = [np.eye(d) * (0.4 + r * 0.2) for r, d in enumerate(dims)]
    for r, d in enumerate(dims):
        if conditioned and r == 1:
            incoming_e[r][:] = 0
            incoming_p[r][:] = 0
        state = _replace(
            state,
            g_eta=state.g_eta.at[eng.arena.g_eta_indices(group.gather[:, r], d)].set(
                incoming_e[r][None]
            ),
            g_lam=state.g_lam.at[eng.arena.g_lam_indices(group.gather[:, r], d)].set(
                incoming_p[r][None]
            ),
        )
    updated = eng.step(state)
    for destination, dim in enumerate(dims):
        reduced = conditioned and destination != 1
        base_m, base_v, _ = covariance_oracle(arrays, dims, observation, reduced)
        roles = (0, 2) if reduced else (0, 1, 2)
        e = np.concatenate(
            [np.zeros(dims[r]) if r == destination else incoming_e[r] for r in roles]
        )
        p = block_diag(
            *[
                np.zeros_like(incoming_p[r]) if r == destination else incoming_p[r]
                for r in roles
            ]
        )
        base_w = np.linalg.inv(base_v)
        v = np.linalg.inv(base_w + p)
        m = v @ (base_w @ base_m + e)
        sections = role_indices(tuple(dims[r] for r in roles))
        idx = sections[roles.index(destination)]
        expected_p = np.linalg.inv(v[np.ix_(idx, idx)])
        expected_e = expected_p @ m[idx]
        actual_e = updated.g_eta[
            eng.arena.g_eta_indices(group.scatter[:, destination], dim)
        ][0]
        actual_p = updated.g_lam[
            eng.arena.g_lam_indices(group.scatter[:, destination], dim)
        ][0]
        np.testing.assert_allclose(actual_e, expected_e, atol=1e-11)
        np.testing.assert_allclose(actual_p, expected_p, atol=1e-11)
    base_m, base_v, evidence = covariance_oracle(arrays, dims, observation, conditioned)
    roles = (0, 2) if conditioned else (0, 1, 2)
    base_w = np.linalg.inv(base_v)
    p = block_diag(*(incoming_p[r] for r in roles))
    e = np.concatenate([incoming_e[r] for r in roles])
    v = np.linalg.inv(base_w + p)
    m = v @ (base_w @ base_m + e)
    delta = m - base_m
    kl = 0.5 * (
        np.trace(base_w @ v)
        + delta @ base_w @ delta
        - len(m)
        + np.linalg.slogdet(base_v)[1]
        - np.linalg.slogdet(v)[1]
    )
    np.testing.assert_allclose(eng.bfe(state), kl + evidence, atol=1e-11)


def test_observation_validation_and_unsupported_role(registered):
    t, arrays = model(observed=True)
    eng = Engine(compile_template(t, arrays=arrays))
    for value in ([1.0], [np.nan, 0.0], [[0.0, 1.0]]):
        with pytest.raises(ValueError, match="sensor requires a finite vector"):
            eng.bind_data(eng.init_state(), {"sensor": value})
    t.ports["field"] = PortSpec("field", "field", PortRole.DATA)
    with pytest.raises(CompileError, match="registered conditioning role"):
        compile_template(t, arrays=arrays)


def test_reconstruction_identity_files_initializers_and_override(registered, tmp_path):
    from cffg.bundle import load_bundle, save_bundle

    t, arrays = model(observed=True)
    text = to_json(t)
    assert IMPL in text and "1.3.0" in text
    assert structure_hash(from_json(text)) == structure_hash(t)
    bundle = tmp_path / "joint.cffgz"
    save_bundle(bundle, t, arrays)
    back, values, _ = load_bundle(bundle)
    for key, value in arrays.items():
        np.save(tmp_path / f"{key}.npy", value)
    sources = (
        compile_template(back, arrays=values),
        compile_template(t, root=tmp_path),
    )
    y = np.array([0.2, 0.8])
    for arena in sources:
        eng = Engine(arena)
        state = eng.bind_data(eng.init_state(), {"sensor": y})
        state = eng.run(state, 10).require_converged()
        np.testing.assert_allclose(
            eng.bfe(state), covariance_oracle(arrays, (3, 2, 1), y, True)[2], atol=1e-11
        )
    t.blobs["center"] = BlobRef("center", "center.npy", inits.spec("zeros"))
    t.blobs["precision_root"] = BlobRef(
        "precision_root", "precision_root.npy", inits.spec("eye", scale=2.0)
    )
    initialized = compile_template(t)
    group = initialized.gauss_groups[0]
    paths = dict(group.parameter_paths)
    np.testing.assert_array_equal(
        initialized.log_params[paths["joint_precision"]], 4 * np.eye(6)[None]
    )
    filed = compile_template(t, root=tmp_path)
    np.testing.assert_array_equal(
        filed.log_params[paths["joint_precision"]],
        sources[1].log_params[paths["joint_precision"]],
    )
    override = compile_template(t, arrays={"precision_root": np.eye(6)}, root=tmp_path)
    np.testing.assert_array_equal(
        override.log_params[paths["joint_precision"]], np.eye(6)[None]
    )
    kernels.unregister_kernel(CUSTOM)
    with pytest.raises(ValueError, match=f"{IMPL}.*1.3.0"):
        from_json(text)
    with pytest.raises(ValueError, match=IMPL):
        load_bundle(bundle)
    kernels.register_kernel(replace(registered, version="2.0.0"))
    with pytest.raises(ValueError, match="incompatible"):
        from_json(text)


@pytest.mark.parametrize(
    "dtype",
    [np.float32, pytest.param(np.float64, marks=pytest.mark.pr_head)],
)
def test_large_flat_storage_host_jit_and_distinct_batches(registered, dtype):
    with jax.enable_x64(dtype == np.float64):
        dims = (100, 2, 1)
        t, arrays = model(dims, observed=True)
        t.variables["tiny"] = VariableSpec("tiny", dim=1)
        t.ports["tiny"] = PortSpec("tiny", "tiny", PortRole.GLUE)
        t.factors["tiny"] = FactorSpec(
            "tiny",
            "gaussian",
            {"out": "tiny"},
            param_blobs={"mean": "tiny_mean", "cov": "tiny_cov"},
        )
        arrays.update(tiny_mean=np.zeros(1), tiny_cov=np.eye(1))
        t.blobs.update({k: BlobRef(k, f"{k}.npy") for k in ("tiny_mean", "tiny_cov")})
        eng = Engine(compile_template(t, arrays=arrays), schedule="flood")
        s0 = eng.bind_data(eng.init_state(), {"sensor": np.array([0.3, 0.7])})
        s1 = eng.bind_data(eng.init_state(), {"sensor": np.array([-0.8, 0.1])})

        def numerical(state):
            e = eng._step(
                state.messages,
                state.g_eta,
                state.g_lam,
                state.g_obs,
                state.g_mask,
                state.ef_eta,
                state.ef_obs,
                state.ef_mask,
                state.params,
            )
            return _replace(
                state, messages=e[0], g_eta=e[1], g_lam=e[2], ef_eta=e[3]
            ), e[5]

        with jax.disable_jit():
            host = eng.step(s0)
        direct = [eng.step(s) for s in (s0, s1)]
        batch = jax.tree_util.tree_map(lambda a, b: jnp.stack((a, b)), s0, s1)
        together, statuses = eqx.filter_jit(eqx.filter_vmap(numerical))(batch)
        np.testing.assert_array_equal(statuses, [0, 0])
        for name in ("g_eta", "g_lam"):
            np.testing.assert_allclose(
                getattr(host, name), getattr(direct[0], name), atol=2e-5
            )
            for i in range(2):
                np.testing.assert_allclose(
                    getattr(together, name)[i], getattr(direct[i], name), atol=2e-5
                )
        assert not np.array_equal(direct[0].g_eta, direct[1].g_eta)
        state = eng.run(s0, 10, tol=2e-5).require_converged()
        mean, covariance, energy = covariance_oracle(arrays, dims, [0.3, 0.7], True)
        values = eng.kernel_readouts(state)["joint"]
        np.testing.assert_allclose(values["mean"], mean, atol=3e-5)
        np.testing.assert_allclose(values["covariance"], covariance, atol=3e-5)
        np.testing.assert_allclose(eng.bfe(state), energy, atol=8e-5)
        assert state.g_eta.shape == (2 * (sum(dims) + 1),)
        assert state.g_lam.shape == (2 * (sum(d * d for d in dims) + 1),)
        est = estimate(eng, calibrate=False)
        assert est.state_bytes == sum(
            x.nbytes for x in jax.tree_util.tree_leaves(state)
        )
        assert est.param_bytes == sum(x.nbytes for x in state.params.values())


@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_improper_and_nonfinite_results_are_loud(registered, schedule):
    t, arrays = model()
    eng = Engine(compile_template(t, arrays=arrays), schedule=schedule)
    group = eng.arena.gauss_groups[0]
    key = dict(group.parameter_paths)["joint_precision"]
    for problem in ("negative", "asymmetric", "nonfinite"):
        state = eng.init_state()
        params = dict(state.params)
        if problem == "negative":
            params[key] = -params[key]
        elif problem == "asymmetric":
            params[key] = params[key].at[:, 0, 1].add(1.0)
        else:
            params[key] = params[key].at[:, 0, 0].set(jnp.nan)
        state = _replace(state, params=params)
        assert eng.run(state, 10).status is RunStatus.DIVERGED
        assert not eng.gaussian_node_stats(state)["joint"]["proper"]
    for field in ("message", "energy"):
        kernels.unregister_kernel(CUSTOM)
        if field == "message":

            def broken(e, p, params, dims, *args):
                return tuple(
                    (jnp.full((e.shape[0], d), jnp.nan), jnp.zeros((e.shape[0], d, d)))
                    for d in dims
                )
        else:

            def broken(e, *args):
                return jnp.full((e.shape[0],), jnp.inf)

        kernels.register_kernel(
            replace(registered, binding=replace(registered.binding, **{field: broken}))
        )
        eng = Engine(compile_template(t, arrays=arrays), schedule=schedule)
        assert eng.run(eng.init_state(), 10).status is RunStatus.DIVERGED


def test_registration_and_compilation_reject_incomplete_contracts(registered):
    kernels.unregister_kernel(CUSTOM)
    for field in ("sparse_support", "improper_messages", "float32", "scale"):
        for value in (None, "", " "):
            with pytest.raises(ValueError, match="hostile-input declaration"):
                kernels.register_kernel(
                    replace(
                        registered,
                        hostile=replace(registered.hostile, **{field: value}),
                    )
                )
    bad_bindings = (
        replace(registered.binding, message=None),
        replace(registered.binding, energy=None),
        replace(
            registered.binding,
            conditioning=replace(registered.binding.conditioning, role="missing"),
        ),
        replace(
            registered.binding,
            conditioning=replace(registered.binding.conditioning, validate=None),
        ),
    )
    for binding in bad_bindings:
        with pytest.raises(ValueError):
            kernels.register_kernel(replace(registered, binding=binding))
    bad_callbacks = (
        replace(registered.binding, message=lambda *args: ()),
        replace(registered.binding, pack_params=lambda *args: {}),
        replace(registered.binding, pack_params=lambda *args: {"extra": np.ones(1)}),
    )
    for binding in bad_callbacks:
        kernels.register_kernel(replace(registered, binding=binding))
        t, arrays = model()
        with pytest.raises(CompileError):
            compile_template(t, arrays=arrays)
        kernels.unregister_kernel(CUSTOM)


def test_custom_names_require_no_central_dispatch_edits(registered):
    import importlib

    for name in ("cffg.compiler", "cffg.engine", "cffg.core.validate"):
        text = Path(importlib.import_module(name).__file__).read_text()
        assert CUSTOM not in text
        assert "joint_precision" not in text


def test_registration_rejects_aliases_and_duplicate_parameter_addresses(registered):
    kernels.unregister_kernel(CUSTOM)
    with pytest.raises(ValueError, match="reserved for built-ins"):
        kernels.register_kernel(
            replace(
                registered,
                binding=replace(
                    registered.binding,
                    conditioning=replace(
                        registered.binding.conditioning, plan_type="shared_label"
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="reserved for a conditioned plan"):
        kernels.register_kernel(
            replace(
                registered,
                factor_type=replace(registered.factor_type, name="gaussian_linear_obs"),
            )
        )
    specs = {
        role: replace(spec, key=key, group_key=True)
        for (role, spec), key in zip(
            registered.binding.parameters.items(), registered.param_keys
        )
    }
    with pytest.raises(ValueError, match="only one named packed parameter"):
        kernels.register_kernel(
            replace(registered, binding=replace(registered.binding, parameters=specs))
        )
    assert kernels.kernel_for(CUSTOM) is None


def test_derived_group_name_collision_fails_before_parameter_overwrite(registered):
    alias = f"{CUSTOM}:observed:sensor"
    kernels.register_kernel(
        replace(
            registered,
            factor_type=replace(registered.factor_type, name=alias),
            binding=replace(registered.binding, conditioning=None),
        )
    )
    try:
        t, arrays = model(observed=True)
        for role, dim in zip(ROLES, (3, 2, 1)):
            name = f"second_{role}"
            t.variables[name] = VariableSpec(name, dim=dim)
            t.ports[name] = PortSpec(name, name, PortRole.GLUE)
        t.factors["second"] = FactorSpec(
            "second",
            alias,
            {role: f"second_{role}" for role in ROLES},
            param_blobs=dict(t.factors["joint"].param_blobs),
        )
        with pytest.raises(CompileError, match="Gaussian group name collision"):
            compile_template(t, arrays=arrays)
    finally:
        kernels.unregister_kernel(alias)


@pytest.mark.parametrize("source", ["arrays", "file"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_builtin_covariance_uses_source_dtype_before_packing(tmp_path, source, dtype):
    covariance = np.array([[1, 0.4], [0.4, 1]], dtype=dtype)
    if dtype != np.float16:
        covariance[1, 0] = np.nextafter(covariance[1, 0], dtype(1))
    t = Template(
        name="covariance_roundoff",
        variables={"x": VariableSpec("x", dim=2)},
        factors={
            "prior": FactorSpec(
                "prior",
                "gaussian",
                {"out": "x"},
                param_blobs={"mean": "mean", "cov": "cov"},
            )
        },
        ports={"x": PortSpec("x", "x", PortRole.GLUE)},
        blobs={key: BlobRef(key, f"{key}.npy") for key in ("mean", "cov")},
    )
    arrays = {"mean": np.zeros(2), "cov": covariance}
    if source == "file":
        for key, value in arrays.items():
            np.save(tmp_path / f"{key}.npy", value)
        arena = compile_template(t, root=tmp_path)
    else:
        arena = compile_template(t, arrays=arrays)
    engine = Engine(arena)
    address = arena.member_of["prior"].parameter("cov")
    np.testing.assert_array_equal(
        engine.init_state().params[address.key][address.index],
        np.linalg.inv(covariance.astype(np.float64)),
    )
    for bad in (
        np.array([[1, 0.4], [0.5, 1]], dtype=np.float32),
        np.diag([1, -1]),
        np.full((2, 2), np.nan),
    ):
        with pytest.raises(CompileError, match="covariance"):
            compile_template(t, arrays={**arrays, "cov": bad})
