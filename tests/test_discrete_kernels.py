"""External discrete registration, checked against scalar enumeration.

This factor has two unequal domains and four independently packed parameters.
Its mathematics deliberately does not call production categorical rules.
"""

from dataclasses import replace
from itertools import product
from math import exp, log
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

from cffg import kernels
from cffg.compiler import NEG, CompileError, compile_template
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    FactorType,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.engine import Engine, RunStatus, _disc_both, _disc_rows, _replace
from cffg.estimate import estimate
from cffg.kernel_bindings import DiscreteBinding, DiscreteMode, ParameterSpec
from cffg.kernel_runtime import discrete_energy, parameters
from cffg.kernels import FactorKernel, HostileInputs

CUSTOM = "bilinear_spin_color"
IMPL = "cffgtests.bilinear_spin_color"
ROLES = ("spin", "color")
KEYS = {
    "coupling": "strength",
    "spin_bias": "spin_tilt",
    "color_bias": "color_tilt",
    "support": "allowed",
}
HOSTILE = HostileInputs(
    "Forbidden states preserve NEG; empty intersection is infeasible.",
    "Vacuous cavities are neutral; nonfinite computations diverge.",
    "Log-sum-exp uses the input dtype; no fixed numerical floors.",
    "Test cardinality128 beside cardinality2 with local padding only.",
)


def potential(params, cards):
    a = jnp.linspace(-1.0, 1.0, cards[0], dtype=params["strength"].dtype)
    b = jnp.linspace(-1.0, 1.0, cards[1], dtype=params["strength"].dtype)
    value = (
        params["strength"][:, None, None] * a[None, :, None] * b[None, None, :]
        + params["spin_tilt"][:, :, None]
        + params["color_tilt"][:, None, :]
    )
    return jnp.where(params["allowed"] > 0, value, jnp.asarray(NEG, value.dtype))


def exact_log(value):
    return jnp.where(value <= NEG, -jnp.inf, value)


def normalized(value):
    total = logsumexp(value, axis=-1, keepdims=True)
    result = value - jnp.where(jnp.isneginf(total), 0.0, total)
    return jnp.where(jnp.isneginf(result), jnp.asarray(NEG, result.dtype), result)


def padded(values, cards, width):
    return jnp.stack(
        [
            jnp.pad(normalized(v), ((0, 0), (0, width - card)), constant_values=NEG)
            for v, card in zip(values, cards)
        ],
        axis=1,
    )


def bp_message(inc, params, cards, width, reduction):
    lf = exact_log(potential(params, cards))
    spin = reduction(lf + exact_log(inc[:, 1, None, : cards[1]]), axis=2)
    color = reduction(lf + exact_log(inc[:, 0, : cards[0], None]), axis=1)
    return padded((spin, color), cards, width)


def bp_joint(inc, params, cards):
    lf = exact_log(potential(params, cards))
    value = (
        lf
        + exact_log(inc[:, 0, : cards[0], None])
        + exact_log(inc[:, 1, None, : cards[1]])
    )
    lq = value - logsumexp(value, axis=(1, 2), keepdims=True)
    return jnp.exp(lq), lq


def bp_energy(inc, params, cards):
    q, lq = bp_joint(inc, params, cards)
    return jnp.sum(
        jnp.where(q > 0, q * (lq - exact_log(potential(params, cards))), 0.0),
        axis=(1, 2),
    )


def expected(q, values, axis):
    return jnp.sum(jnp.where(q > 0, q * values, 0.0), axis=axis)


def vmp_message(beliefs, params, cards, width):
    lf = exact_log(potential(params, cards))
    return padded(
        (
            expected(beliefs[1][0][:, None, :], lf, 2),
            expected(beliefs[0][0][:, :, None], lf, 1),
        ),
        cards,
        width,
    )


def vmp_energy(beliefs, params, cards):
    (qa, la), (qb, lb) = beliefs
    entropy_terms = expected(qa, la, 1) + expected(qb, lb, 1)
    return entropy_terms - expected(
        qa[:, :, None] * qb[:, None, :], exact_log(potential(params, cards)), (1, 2)
    )


def correlation(q, cards):
    a = jnp.linspace(-1.0, 1.0, cards[0], dtype=q.dtype)
    b = jnp.linspace(-1.0, 1.0, cards[1], dtype=q.dtype)
    return {
        "correlation": jnp.sum(q * a[None, :, None] * b[None, None, :], axis=(1, 2))
    }


def bp_readout(inc, params, cards):
    return correlation(bp_joint(inc, params, cards)[0], cards)


def vmp_readout(beliefs, params, cards):
    return correlation(beliefs[0][0][:, :, None] * beliefs[1][0][:, None, :], cards)


def pack(template, members, resolved):
    return {
        key: np.stack(
            [resolved[template.factors[m].param_blobs[role]] for m in members]
        )
        for role, key in KEYS.items()
    }


def declaration():
    shapes = {
        "coupling": lambda s: (),
        "spin_bias": lambda s: (s["spin"],),
        "color_bias": lambda s: (s["color"],),
        "support": lambda s: (s["spin"], s["color"]),
    }
    return FactorKernel(
        FactorType(CUSTOM, ROLES, tuple(KEYS)),
        IMPL,
        "1.2.0",
        hostile=HOSTILE,
        binding=DiscreteBinding(
            ROLES,
            {r: ParameterSpec(shapes[r], k) for r, k in KEYS.items()},
            tuple(KEYS.values()),
            pack,
            potential,
            DiscreteMode(bp_message, bp_energy, bp_readout),
            DiscreteMode(vmp_message, vmp_energy, vmp_readout),
        ),
    )


@pytest.fixture
def registered():
    kernel = kernels.register_kernel(declaration())
    yield kernel
    kernels.unregister_kernel(CUSTOM)


def model(cards=(2, 3), mean_field=False):
    arrays = {
        "coupling": np.asarray(0.8),
        "spin_bias": np.linspace(0.1, 0.3, cards[0]),
        "color_bias": np.linspace(0.2, -0.1, cards[1]),
        "support": np.ones(cards),
        "spin_prior": np.arange(1, cards[0] + 1, dtype=float),
        "color_prior": np.arange(cards[1], 0, -1, dtype=float),
    }
    for role in ROLES:
        arrays[f"{role}_prior"] /= arrays[f"{role}_prior"].sum()
    template = Template(
        name="registered_bilinear",
        variables={r: VariableSpec(r, k) for r, k in zip(ROLES, cards)},
        factors={
            "interaction": FactorSpec(
                "interaction",
                CUSTOM,
                dict(zip(ROLES, ROLES)),
                param_blobs={r: r for r in KEYS},
            )
        },
        blobs={key: BlobRef(key, f"{key}.npy") for key in arrays},
    )
    for role in ROLES:
        template.factors[role] = FactorSpec(
            role,
            "categorical",
            {"out": role},
            param_blobs={"statistic": f"{role}_prior"},
        )
    if mean_field:
        template.constraints.append(
            ConstraintSpec(ConstraintKind.MEAN_FIELD, "interaction")
        )
    return template, arrays


def scalar_potential(arrays):
    cards = arrays["support"].shape
    values = np.empty(cards)
    for i, j in product(*(range(k) for k in cards)):
        a, b = -1 + 2 * i / (cards[0] - 1), -1 + 2 * j / (cards[1] - 1)
        values[i, j] = (
            (
                float(arrays["coupling"]) * a * b
                + arrays["spin_bias"][i]
                + arrays["color_bias"][j]
            )
            if arrays["support"][i, j]
            else -np.inf
        )
    return values


def oracle(arrays):
    lf = scalar_potential(arrays)
    weights = {
        (i, j): exp(lf[i, j]) * arrays["spin_prior"][i] * arrays["color_prior"][j]
        for i, j in product(*(range(k) for k in lf.shape))
    }
    z = sum(weights.values())
    joint = np.array(
        [[weights[i, j] / z for j in range(lf.shape[1])] for i in range(lf.shape[0])]
    )
    return joint, -log(z), max(weights, key=weights.get), -log(max(weights.values()))


@pytest.mark.parametrize("schedule", ["auto", "flood"])
@pytest.mark.parametrize("semiring", ["sum", "max"])
def test_external_execution_and_objectives(registered, schedule, semiring):
    t, arrays = model()
    eng = Engine(
        compile_template(t, arrays=arrays, semiring=semiring), schedule=schedule
    )
    state = eng.run(eng.init_state(), 30, tol=1e-11).require_converged()
    joint, energy, assignment, score = oracle(arrays)
    group = next(g for g in eng.arena.groups if g.type == CUSTOM)
    assert group.name not in state.params
    if semiring == "sum":
        np.testing.assert_allclose(
            eng.marginals(state)["spin"], joint.sum(axis=1), atol=1e-12
        )
        np.testing.assert_allclose(
            eng.marginals(state)["color"], joint.sum(axis=0), atol=1e-12
        )
        np.testing.assert_allclose(
            eng.node_beliefs(state)["interaction"], joint, atol=1e-12
        )
        np.testing.assert_allclose(eng.bfe(state), energy, atol=1e-12)
        expected_corr = sum(
            joint[i, j] * (-1 + 2 * i) * (-1 + j)
            for i, j in product(range(2), range(3))
        )
        np.testing.assert_allclose(
            eng.kernel_readouts(state)["interaction"]["correlation"],
            expected_corr,
            atol=1e-12,
        )
    else:
        assert tuple(eng.decode(state)[r] for r in ROLES) == assignment
        np.testing.assert_allclose(eng.objective(state), score, atol=1e-12)
        with pytest.raises(RuntimeError, match="not a free energy"):
            eng.bfe(state)


def test_two_uncertain_neighbors_use_mode_specific_inputs(registered):
    outputs = []
    for mean_field in (False, True):
        t, arrays = model(mean_field=mean_field)
        eng = Engine(compile_template(t, arrays=arrays), schedule="flood")
        state = eng.init_state()
        group = next(g for g in eng.arena.groups if g.type == CUSTOM)
        cavities = (np.array([0.65, 0.35]), np.array([0.2, 0.5, 0.3]))
        own = (np.array([0.4, 0.6]), np.array([0.5, 0.3, 0.2]))
        messages = state.messages
        for r, card in enumerate(group.cards):
            messages = messages.at[
                eng.arena.message_indices(group.gather[:, r], card)
            ].set(np.log(cavities[r])[None])
            messages = messages.at[
                eng.arena.message_indices(group.scatter[:, r], card)
            ].set(np.log(own[r])[None])
        state = _replace(state, messages=messages)
        beliefs = tuple(c * o / np.dot(c, o) for c, o in zip(cavities, own))
        lf = scalar_potential(arrays)
        expected_messages = []
        for r, card in enumerate(group.cards):
            other = 1 - r
            vals = []
            for i in range(card):
                terms = [
                    lf[i, j] if r == 0 else lf[j, i] for j in range(group.cards[other])
                ]
                vals.append(
                    sum(beliefs[other][j] * term for j, term in enumerate(terms))
                    if mean_field
                    else log(
                        sum(
                            cavities[other][j] * exp(term)
                            for j, term in enumerate(terms)
                        )
                    )
                )
            probs = np.exp(vals) / np.exp(vals).sum()
            expected_messages.append(np.log(probs))
        updated = eng.step(state)
        for r, card in enumerate(group.cards):
            actual = np.asarray(
                updated.messages[eng.arena.message_indices(group.scatter[:, r], card)]
            )[0]
            np.testing.assert_allclose(actual, expected_messages[r], atol=1e-12)
        outputs.append(expected_messages[0])
        inc = (
            _disc_both(eng.arena, state.messages, group.gather_both, group.cards)
            if mean_field
            else _disc_rows(eng.arena, state.messages, group.gather, group.cards)
        )
        if mean_field:
            joint = np.outer(*beliefs)
        else:
            joint = np.exp(lf) * np.outer(*cavities)
            joint /= joint.sum()
        energy = sum(
            joint[i, j] * (log(joint[i, j]) - lf[i, j])
            for i, j in product(range(2), range(3))
        )
        np.testing.assert_allclose(
            discrete_energy(group, inc, parameters(group, state.params)),
            [energy],
            atol=1e-12,
        )
        corr = sum(
            joint[i, j] * (-1 + 2 * i) * (-1 + j)
            for i, j in product(range(2), range(3))
        )
        np.testing.assert_allclose(
            eng.kernel_readouts(state)["interaction"]["correlation"], corr, atol=1e-12
        )
    assert not np.allclose(*outputs)


def test_reconstruction_identity_sources_and_parameter_replacement(
    registered, tmp_path
):
    from cffg.bundle import load_bundle, save_bundle
    from cffg.params import set_params

    t, arrays = model()
    text = to_json(t)
    assert IMPL in text and "1.2.0" in text
    assert structure_hash(from_json(text)) == structure_hash(t)
    path = tmp_path / "bilinear.cffgz"
    save_bundle(path, t, arrays)
    loaded, values, _ = load_bundle(path)
    for key, arr in arrays.items():
        np.save(tmp_path / f"{key}.npy", arr)
    engines = [
        Engine(compile_template(loaded, arrays=values)),
        Engine(compile_template(t, root=tmp_path)),
    ]
    for eng in engines:
        st = eng.run(eng.init_state(), 10).require_converged()
        np.testing.assert_allclose(eng.bfe(st), oracle(arrays)[1], atol=1e-12)
    eng = engines[0]
    params = set_params(
        eng, eng.init_state().params, {"interaction": {"coupling": np.asarray(0.3)}}
    )
    state = _replace(eng.init_state(), params=params)
    state = eng.run(state, 10).require_converged()
    changed = {**arrays, "coupling": np.asarray(0.3)}
    np.testing.assert_allclose(eng.bfe(state), oracle(changed)[1], atol=1e-12)
    kernels.unregister_kernel(CUSTOM)
    with pytest.raises(ValueError, match=f"{IMPL}.*1.2.0"):
        from_json(text)
    kernels.register_kernel(replace(registered, version="2.0.0"))
    with pytest.raises(ValueError, match="incompatible"):
        from_json(text)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sparse_large_flat_storage_and_distinct_instance_batch(registered, dtype):
    with jax.enable_x64(dtype == np.float64):
        t, arrays = model((128, 3))
        # A disconnected small factor proves that the arena has no global padding.
        t.variables["small"] = VariableSpec("small", 2)
        t.ports["small"] = PortSpec("small", "small", PortRole.GLUE)
        t.factors["small"] = FactorSpec(
            "small", "categorical", {"out": "small"}, param_blobs={"statistic": "small"}
        )
        t.blobs["small"] = BlobRef("small", "small.npy")
        arrays["small"] = np.array([0.6, 0.4])
        arrays["support"][0] = 0
        eng = Engine(compile_template(t, arrays=arrays), schedule="flood")
        s0 = eng.init_state()
        params = dict(s0.params)
        group = next(g for g in eng.arena.groups if g.type == CUSTOM)
        key = dict(group.parameter_paths)["strength"]
        params[key] = params[key] + jnp.asarray(0.25, dtype)
        s1 = _replace(s0, params=params)
        batch = jax.tree_util.tree_map(lambda a, b: jnp.stack((a, b)), s0, s1)
        with jax.disable_jit():
            host = eng.step(s0)
        direct = [eng.step(s) for s in (s0, s1)]

        def numerical_step(state):
            messages, eta, lam, ef_eta, _, status, *_ = eng._step(
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
                state, messages=messages, g_eta=eta, g_lam=lam, ef_eta=ef_eta
            ), status

        together, statuses = eqx.filter_jit(eqx.filter_vmap(numerical_step))(batch)
        np.testing.assert_array_equal(statuses, [0, 0])
        np.testing.assert_allclose(host.messages, direct[0].messages, atol=4e-6)
        for i in range(2):
            np.testing.assert_allclose(
                together.messages[i], direct[i].messages, atol=4e-6
            )
        assert not np.array_equal(direct[0].messages, direct[1].messages)
        st = eng.run(
            s0, 20, tol=2e-5 if dtype == np.float32 else 1e-11
        ).require_converged()
        assert eng.marginals(st)["spin"][0] == 0
        index = eng.arena.message_indices(group.scatter[:, 0], 128)[0, 0]
        assert st.messages[index] == dtype(NEG)
        assert st.messages.shape == (2 * (128 + 3) + 4,)
        est = estimate(eng, calibrate=False)
        assert est.state_bytes == sum(x.nbytes for x in jax.tree_util.tree_leaves(st))
        assert est.param_bytes == sum(x.nbytes for x in st.params.values())


@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_empty_support_and_nonfinite_callbacks(registered, schedule):
    t, arrays = model()
    arrays["support"][:] = 0
    eng = Engine(compile_template(t, arrays=arrays), schedule=schedule)
    assert eng.run(eng.init_state(), 10).status is RunStatus.INFEASIBLE
    for field in ("message", "energy"):
        kernels.unregister_kernel(CUSTOM)
        bp = replace(
            registered.binding.bp,
            **{
                field: (
                    lambda inc, params, cards, *args: jnp.full(
                        (inc.shape[0], len(cards), max(cards))
                        if args
                        else (inc.shape[0],),
                        jnp.nan,
                    )
                )
            },
        )
        kernels.register_kernel(
            replace(registered, binding=replace(registered.binding, bp=bp))
        )
        t, arrays = model()
        eng = Engine(compile_template(t, arrays=arrays), schedule=schedule)
        assert eng.run(eng.init_state(), 10).status is RunStatus.DIVERGED


def test_declaration_and_packing_failures_leave_no_partial_registration(registered):
    kernels.unregister_kernel(CUSTOM)
    for binding in (
        replace(registered.binding, potential=None),
        replace(registered.binding, bp=DiscreteMode(None, bp_energy)),
        replace(registered.binding, parameters={}),
        replace(registered.binding, roles=("color", "spin")),
    ):
        with pytest.raises(ValueError):
            kernels.register_kernel(replace(registered, binding=binding))
        assert kernels.kernel_for(CUSTOM) is None
    for packed in ({}, {"unexpected": np.ones((1,))}):
        kernels.register_kernel(
            replace(
                registered,
                binding=replace(
                    registered.binding,
                    pack_params=lambda *args, _packed=packed: _packed,
                ),
            )
        )
        t, arrays = model()
        with pytest.raises(CompileError, match="declared param_keys"):
            compile_template(t, arrays=arrays)
        kernels.unregister_kernel(CUSTOM)
    bad = replace(registered.binding.bp, message=lambda inc, *args: inc[:, :1])
    kernels.register_kernel(
        replace(registered, binding=replace(registered.binding, bp=bad))
    )
    t, arrays = model()
    with pytest.raises(CompileError, match="outgoing message shape"):
        compile_template(t, arrays=arrays)


def test_custom_names_require_no_central_dispatch_edits(registered):
    import importlib

    for name in ("cffg.compiler", "cffg.engine", "cffg.core.validate"):
        text = Path(importlib.import_module(name).__file__).read_text()
        assert CUSTOM not in text
        assert "spin_tilt" not in text


def test_frozen_bindings_remain_copyable_and_malformed_pack_is_rejected(registered):
    from copy import deepcopy

    for kernel in (registered, kernels.kernel_for("gamma")):
        copied = deepcopy(kernel)
        assert copied == kernel
    kernels.unregister_kernel(CUSTOM)
    with pytest.raises(ValueError, match="pack_params must be callable"):
        kernels.register_kernel(
            replace(registered, binding=replace(registered.binding, pack_params=3))
        )
    specs = dict(registered.binding.parameters)
    specs["coupling"] = replace(
        specs["coupling"], transform="probability_to_log", probability_axis=3
    )
    kernels.register_kernel(
        replace(registered, binding=replace(registered.binding, parameters=specs))
    )
    t, arrays = model()
    with pytest.raises(CompileError, match="probability axis"):
        compile_template(t, arrays=arrays)


def test_vmp_ledger_and_zero_probability_expectations(registered):
    t, arrays = model(mean_field=True)
    eng = Engine(compile_template(t, arrays=arrays), schedule="flood")
    state = eng.run(eng.init_state(), 100, tol=1e-11, damp=0.3).require_converged()
    qs = eng.marginals(state)
    joint = np.outer(qs["spin"], qs["color"])
    lf = scalar_potential(arrays)
    expected_energy = sum(
        joint[i, j]
        * (
            log(joint[i, j])
            - lf[i, j]
            - log(arrays["spin_prior"][i])
            - log(arrays["color_prior"][j])
        )
        for i, j in product(range(2), range(3))
    )
    np.testing.assert_allclose(eng.bfe(state), expected_energy, atol=1e-10)
    group = next(g for g in eng.arena.groups if g.type == CUSTOM)
    gp = parameters(group, state.params)
    gp["allowed"] = gp["allowed"].at[:, 0, :].set(0)
    qa, qb = jnp.array([[0.0, 1.0]]), jnp.array([[0.2, 0.5, 0.3]])
    beliefs = ((qa, jnp.log(qa)), (qb, jnp.log(qb)))
    assert np.isfinite(vmp_energy(beliefs, gp, group.cards)).all()
    finite = vmp_message(beliefs, gp, group.cards, 3)
    assert finite[0, 0, 0] == NEG
    assert np.isfinite(finite).all()
    bad = ((jnp.array([[0.5, 0.5]]), jnp.log(jnp.array([[0.5, 0.5]]))), beliefs[1])
    assert np.isposinf(vmp_energy(bad, gp, group.cards)).all()


def test_initializer_shape_metadata_and_override_precedence(registered, tmp_path):
    from cffg import inits

    t, arrays = model()
    for key in KEYS:
        value = 0.8 if key == "coupling" else 1.0 if key == "support" else 0.0
        t.blobs[key] = BlobRef(key, f"{key}.npy", inits.spec("constant", value=value))
    supplied = {key: value for key, value in arrays.items() if key not in KEYS}
    initialized = compile_template(t, arrays=supplied)
    group = next(g for g in initialized.groups if g.type == CUSTOM)
    packed = parameters(group, initialized.log_params)
    assert packed["strength"].shape == (1,)
    assert packed["allowed"].shape == (1, 2, 3)
    np.testing.assert_array_equal(packed["spin_tilt"], np.zeros((1, 2)))
    np.save(tmp_path / "coupling.npy", np.asarray(0.2))
    filed = compile_template(t, arrays=supplied, root=tmp_path)
    assert parameters(group, filed.log_params)["strength"][0] == 0.2
    override = compile_template(
        t, arrays={**supplied, "coupling": np.asarray(0.6)}, root=tmp_path
    )
    assert parameters(group, override.log_params)["strength"][0] == 0.6
    with pytest.raises(CompileError, match="shape"):
        compile_template(t, arrays={**supplied, "spin_bias": np.zeros(4)})


def chain_model(n=32):
    _, arrays = model((2, 2))
    t = Template(
        name="registered_chain",
        variables={f"z{i}": VariableSpec(f"z{i}", 2) for i in range(n + 1)},
        factors={
            f"f{i:03}": FactorSpec(
                f"f{i:03}",
                CUSTOM,
                {"spin": f"z{i}", "color": f"z{i + 1}"},
                param_blobs={r: r for r in KEYS},
            )
            for i in range(n)
        },
        blobs={key: BlobRef(key, f"{key}.npy") for key in arrays},
    )
    for name, var, prior in (
        ("left", "z0", "spin_prior"),
        ("right", f"z{n}", "color_prior"),
    ):
        t.factors[name] = FactorSpec(
            name, "categorical", {"out": var}, param_blobs={"statistic": prior}
        )
    return t, arrays


def test_external_chain_scans_and_parameter_changes_reuse_trace(registered):
    from cffg.schedule import ScanStage
    from cffg.schedule_program import _flat_scan_fn

    traces = []

    def counted(*args):
        traces.append(None)
        return bp_message(*args)

    kernels.unregister_kernel(CUSTOM)
    bp = replace(registered.binding.bp, message=counted)
    kernels.register_kernel(
        replace(registered, binding=replace(registered.binding, bp=bp))
    )
    t, arrays = chain_model()
    arena = compile_template(t, arrays=arrays)
    eng = Engine(arena)
    assert eng.schedule.kind == "two_pass" and all(eng.scan_registerized)
    initial = eng.init_state()
    first = eng.step(initial)
    count = len(traces)
    gp = next(g for g in arena.groups if g.type == CUSTOM)
    changed = dict(initial.params)
    key = dict(gp.parameter_paths)["strength"]
    changed[key] = changed[key] + 0.1
    second = eng.step(_replace(initial, params=changed))
    assert len(traces) == count
    assert not np.array_equal(first.messages, second.messages)
    exact = eng.run(initial, 1).require_converged()
    flooded = (
        Engine(arena, schedule="flood").run(initial, 80, tol=1e-11).require_converged()
    )
    np.testing.assert_allclose(exact.messages, flooded.messages, atol=1e-11)
    # Exercise fallback transport with the same custom local callback.
    args = (
        initial.messages,
        initial.g_eta,
        initial.g_lam,
        initial.ef_eta,
        initial.g_obs,
        initial.g_mask,
        initial.ef_obs,
        initial.ef_mask,
        initial.params,
    )
    current = args[:4]
    from cffg.schedule_program import _batch_stage_fn

    for stage in eng.schedule.stages:
        fn = (
            _flat_scan_fn(arena, stage, logsumexp)
            if isinstance(stage, ScanStage)
            else _batch_stage_fn(arena, stage, logsumexp)
        )
        current = eqx.filter_jit(fn)(*current[:4], *args[4:])
    np.testing.assert_allclose(current[0], exact.messages, atol=1e-12)

    closed, _, _ = eqx.filter_make_jaxpr(eng._propagate)(*args)
    scans = []

    def visit(jaxpr):
        for equation in jaxpr.eqns:
            if equation.primitive.name == "scan":
                body = equation.params["jaxpr"].jaxpr
                start = equation.params["num_consts"]
                carry = body.invars[start : start + equation.params["num_carry"]]
                scans.extend(v.aval.shape for v in carry)
            for value in equation.params.values():
                if hasattr(value, "jaxpr"):
                    visit(value.jaxpr)
                elif hasattr(value, "eqns"):
                    visit(value)

    visit(closed.jaxpr)
    assert scans and all(shape in {(), (1, 2)} for shape in scans)
