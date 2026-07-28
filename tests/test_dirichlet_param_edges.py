"""In-graph Dirichlet parameter edges (cff-fr5c / DERIVATIONS D6a)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import gammaln

from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import CompileError, compile_template
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.engine import Engine, RunStatus, _disc_both
from cffg.learn import Dirichlet
from cffg.rules import dirichlet_vmp_energy_parts


FAMILY = Dirichlet()


def _parameter_role(ftype):
    return {
        "categorical": "statistic",
        "matrix_cat": "matrix",
        "transition_mixture": "matrices",
    }[ftype]


def _shape(ftype, cards):
    if ftype == "categorical":
        return (cards[0],)
    if ftype == "matrix_cat":
        return cards
    ko, ki, ku = cards
    return (ku, ko, ki)


def _make_template(ftype, cards, *, latent_parents=False, constrained=True):
    roles = {
        "categorical": ("out",),
        "matrix_cat": ("out", "in"),
        "transition_mixture": ("out", "in", "selector"),
    }[ftype]
    variables = {
        role: VariableSpec(role, card) for role, card in zip(roles, cards)
    }
    shape = _shape(ftype, cards)
    variables["theta"] = VariableSpec("theta", int(np.prod(shape)))

    factors = {
        "theta_prior": FactorSpec(
            "theta_prior",
            "dirichlet",
            edges={"out": "theta"},
            param_blobs={"counts": "alpha0"},
        ),
        "lik": FactorSpec(
            "lik",
            ftype,
            edges={role: role for role in roles},
            param_edges={_parameter_role(ftype): "theta"},
        ),
    }
    ports = {}
    blobs = {"alpha0": BlobRef("alpha0", "alpha0.npy")}
    if latent_parents:
        # The output remains observed. Every other role gets a fixed
        # categorical prior, making its mean-field marginal nontrivial.
        ports["out"] = PortSpec("out", "out", PortRole.DATA)
        for role, card in zip(roles[1:], cards[1:]):
            blob = f"d_{role}"
            factor = f"prior_{role}"
            factors[factor] = FactorSpec(
                factor,
                "categorical",
                edges={"out": role},
                param_blobs={"statistic": blob},
            )
            blobs[blob] = BlobRef(blob, f"{blob}.npy")
    else:
        ports = {
            role: PortSpec(role, role, PortRole.DATA) for role in roles
        }
    constraints = (
        [ConstraintSpec(ConstraintKind.MEAN_FIELD, "lik")]
        if constrained
        else []
    )
    return Template(
        f"dir-{ftype}",
        variables=variables,
        factors=factors,
        constraints=constraints,
        ports=ports,
        blobs=blobs,
    )


def _log_beta(alpha, axis):
    alpha = jnp.asarray(alpha)
    return jnp.sum(gammaln(alpha), axis=axis) - gammaln(
        jnp.sum(alpha, axis=axis)
    )


def test_categorical_parameter_edge_observation_is_exact_conjugacy():
    alpha0 = np.array([0.3, 1.2, 2.5])
    template = _make_template("categorical", (3,))
    engine = Engine(compile_template(template, arrays={"alpha0": alpha0}))
    state = engine.bind_data(engine.init_state(), {"out": 1})
    result = engine.run(state, 20, tol=1e-12)
    assert result.status is RunStatus.CONVERGED

    posterior = engine.dirichlet_marginals(result.state)["theta"]
    np.testing.assert_array_equal(posterior.alpha, alpha0 + [0.0, 1.0, 0.0])
    assert posterior.proper
    np.testing.assert_allclose(
        engine.bfe(result.state),
        _log_beta(alpha0, 0) - _log_beta(posterior.alpha, 0),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("ftype", "cards", "observed", "parent_priors"),
    [
        (
            "matrix_cat",
            (3, 2),
            2,
            {"d_in": np.array([0.65, 0.35])},
        ),
        (
            "transition_mixture",
            (3, 2, 2),
            1,
            {
                "d_in": np.array([0.7, 0.3]),
                "d_selector": np.array([0.25, 0.75]),
            },
        ),
    ],
)
def test_in_graph_posterior_matches_learnables_fold_and_direct_vfe(
    ftype, cards, observed, parent_priors
):
    shape = _shape(ftype, cards)
    alpha0 = np.linspace(0.25, 1.75, np.prod(shape)).reshape(shape)
    arrays = {"alpha0": alpha0, **parent_priors}
    template = _make_template(ftype, cards, latent_parents=True)
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), {"out": observed})
    result = engine.run(state, 200, tol=1e-11, damp=0.25)
    assert result.status is RunStatus.CONVERGED, (
        result.status,
        result.residual,
    )
    state = result.state

    node_belief = engine.node_beliefs(state)["lik"]
    stats = FAMILY.stats(ftype, node_belief)
    folded = FAMILY.fold({"alpha": alpha0}, stats)["alpha"]
    posterior = engine.dirichlet_marginals(state)["theta"]
    np.testing.assert_allclose(posterior.alpha, folded, atol=2e-10)

    structural = engine.marginals(state)
    direct = _log_beta(alpha0, 1 if ftype == "transition_mixture" else 0)
    direct = jnp.sum(
        direct
        - _log_beta(
            posterior.alpha, 1 if ftype == "transition_mixture" else 0
        )
    )
    for role, prior in parent_priors.items():
        variable = role.removeprefix("d_")
        q = jnp.asarray(structural[variable])
        direct = direct + jnp.sum(
            jnp.where(q > 0, q * (jnp.log(q) - jnp.log(prior)), 0.0)
        )
    np.testing.assert_allclose(engine.bfe(state), direct, atol=2e-9)

    # Literal node terms plus the parameter-edge correction equal the stable
    # combined ledger used by Engine.bfe.
    group = engine.arena.dirichlet_groups[0]
    inc2 = _disc_both(engine.arena, state.messages, group.gather_both, group.cards)
    parts = dirichlet_vmp_energy_parts(
        ftype,
        inc2,
        jnp.asarray(posterior.alpha)[None, ...],
        jnp.asarray(alpha0)[None, ...],
        cards,
    )
    local_from_parts = sum(float(jnp.sum(part)) for part in parts)
    local_direct = float(
        sum(
            jnp.sum(jnp.where(q > 0, q * jnp.log(q), 0.0))
            for role in ("out", *tuple(r.removeprefix("d_") for r in parent_priors))
            for q in [jnp.asarray(structural[role])]
        )
        + jnp.sum(
            _log_beta(alpha0, 1 if ftype == "transition_mixture" else 0)
            - _log_beta(
                posterior.alpha,
                1 if ftype == "transition_mixture" else 0,
            )
        )
    )
    assert local_from_parts == pytest.approx(local_direct, abs=2e-9)


def test_unconstrained_nonclosed_use_and_bad_priors_fail_loudly():
    alpha0 = np.ones((3, 2))
    unconstrained = _make_template("matrix_cat", (3, 2), constrained=False)
    with pytest.raises(CompileError, match="not Dirichlet-closed"):
        compile_template(unconstrained, arrays={"alpha0": alpha0})

    wrong_facing = _make_template("matrix_cat", (3, 2))
    wrong_facing.factors["theta_prior"] = FactorSpec(
        "theta_prior",
        "categorical",
        edges={"out": "theta"},
        param_blobs={"statistic": "alpha0"},
    )
    with pytest.raises(CompileError, match="must face a dirichlet prior"):
        compile_template(wrong_facing, arrays={"alpha0": np.ones(6) / 6})

    template = _make_template("matrix_cat", (3, 2))
    for bad in (
        np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        np.array([[np.nan, 1.0], [1.0, 1.0], [1.0, 1.0]]),
    ):
        with pytest.raises(CompileError, match="finite and > 0"):
            compile_template(template, arrays={"alpha0": bad})
    with pytest.raises(CompileError, match="concentration shape"):
        compile_template(template, arrays={"alpha0": np.ones(6)})


def test_parameter_edges_and_constraints_round_trip_json_and_bundle(tmp_path):
    template = _make_template("matrix_cat", (3, 2), latent_parents=True)
    arrays = {
        "alpha0": np.arange(1, 7, dtype=float).reshape(3, 2),
        "d_in": np.array([0.4, 0.6]),
    }
    restored = from_json(to_json(template))
    assert restored.factors["lik"].param_edges == {"matrix": "theta"}
    assert restored.constraints == template.constraints
    assert structure_hash(restored) == structure_hash(template)

    path = tmp_path / "dirichlet.cffgz"
    save_bundle(path, template, arrays)
    bundled, bundled_arrays, manifest = load_bundle(path)
    assert bundled.factors["lik"].param_edges == {"matrix": "theta"}
    assert bundled.constraints == template.constraints
    assert manifest["structure_hash"] == structure_hash(bundled)

    before = Engine(compile_template(template, arrays=arrays))
    after = Engine(compile_template(bundled, arrays=bundled_arrays))
    for engine in (before, after):
        state = engine.bind_data(engine.init_state(), {"out": 2})
        state = engine.run(state, 100, tol=1e-11).require_converged()
        np.testing.assert_allclose(
            engine.dirichlet_marginals(state)["theta"].alpha,
            before.dirichlet_marginals(
                before.run(
                    before.bind_data(before.init_state(), {"out": 2}),
                    100,
                    tol=1e-11,
                ).require_converged()
            )["theta"].alpha,
            atol=1e-10,
        )


def test_float32_sparse_zero_support_and_uncomfortable_scale():
    cards = (64, 48, 20)
    shape = _shape("transition_mixture", cards)
    template = _make_template("transition_mixture", cards)
    alpha0 = np.full(shape, 1e-3, dtype=np.float32)
    alpha0[:, 0, :] = 2.0

    with jax.enable_x64(False):
        engine = Engine(
            compile_template(template, arrays={"alpha0": alpha0}),
            schedule="flood",
        )
        state = engine.bind_data(
            engine.init_state(), {"out": 17, "in": 31, "selector": 7}
        )
        result = engine.run(state, 20, tol=1e-5)
        assert result.status is RunStatus.CONVERGED, (
            result.status,
            result.residual,
        )
        posterior = engine.dirichlet_marginals(result.state)["theta"]
        assert posterior.alpha.dtype == np.float32
        assert posterior.expected_log.dtype == np.float32
        assert np.all(np.isfinite(posterior.expected_log))
        expected = alpha0.copy()
        expected[7, 17, 31] += 1.0
        np.testing.assert_array_equal(posterior.alpha, expected)
        # Every impossible structural assignment contributes exact zero.
        np.testing.assert_array_equal(
            posterior.alpha - alpha0,
            np.eye(1, np.prod(shape), 7 * 64 * 48 + 17 * 48 + 31)
            .reshape(shape)
            .astype(np.float32),
        )
        assert np.isfinite(engine.bfe(result.state))
