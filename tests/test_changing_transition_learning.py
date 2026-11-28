"""Changing transition columns through the normal factored learning and checkpoint APIs."""

from functools import partial

import jax
import numpy as np
import pytest

from cffg.agent import RollingAgent
from cffg.bundle import load_bundle, save_bundle
from cffg.compiled import NEG
from cffg.ledger_terms import add_transition_gfe_terms
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.models.factored import ObservationSpec, StateBlockSpec, build_factored_rollout
from test_transition_change_derivation import update


OWNER = "valve_motion"


def model(hazard, *, sparse=False, initial=(1.0, 0.0), dtype=np.float64):
    b = np.array([[[0.7, 0.25], [0.3, 0.75]], [[0.4, 0.6], [0.6, 0.4]]], dtype=dtype)
    if sparse:
        b[:, :, 1] = [0, 1]
    prior = b * np.array([5, 9], dtype=dtype)[:, None, None]
    sensor = np.array([[0.8, 0.15], [0.2, 0.85]], dtype=dtype)
    return partial(
        build_factored_rollout,
        "valve",
        blocks=(
            StateBlockSpec(
                "phase", np.array(initial, dtype=dtype), b, transition_learnable=OWNER
            ),
        ),
        actions=np.array([0.5, 0.5], dtype=dtype),
        observation=ObservationSpec(
            "mark", "phase", sensor, np.array([0.5, 0.5], dtype=dtype)
        ),
        transition_counts={OWNER: prior},
        transition_hazards={} if hazard is None else {OWNER: hazard},
    )


def agent(builder):
    return RollingAgent(builder, 1, learn=(OWNER,), epistemic_b=True, select="map")


def project_table(prior, before, counts, hazard):
    out = np.empty_like(before)
    for action in range(out.shape[0]):
        for source in range(out.shape[2]):
            out[action, :, source] = update(
                prior[action, :, source],
                before[action, :, source],
                counts[action, :, source],
                hazard,
            )
    return out


def test_real_fractional_event_and_exact_restart(tmp_path):
    h = 0.2
    builder = model(h, sparse=True)
    learner = agent(builder)
    bank = learner.learnables
    prior = bank.initial_params[OWNER]["alpha"]
    for action, observation in [(0, 0), (1, 1), (0, 1)]:
        before = bank.params[OWNER]["alpha"].copy()
        predicted = project_table(prior, before, np.zeros_like(prior), h)
        np.testing.assert_allclose(
            bank.predicted_transition_counts(OWNER), predicted, rtol=2e-12
        )
        learner.act()
        virtual = learner.plan.run(
            learner._with_params(learner.plan), learner.n_sweeps, tol=1e-10
        )
        bank.replace(learner.plan, virtual)
        np.testing.assert_array_equal(bank.params[OWNER]["alpha"], before)
        learner.observe(action, observation)
        post = learner._last_filter_result.require_converged()
        src = learner.filter.arena.learn_plans[0].sources
        (source,) = src
        count = learner.filter.node_beliefs(post)[source].transpose(2, 0, 1)
        assert (count[(count > 0) & (count < 1)]).size
        assert np.count_nonzero(count[1 - action]) == 0
        expected = project_table(prior, before, count, h)
        np.testing.assert_allclose(bank.params[OWNER]["alpha"], expected, rtol=3e-12)
        assert np.all(expected[prior == 0] == 0)
        # The expectation target is a packed [dest,source,action] BP table.
        plan = learner.filter.arena.learn_plans[0]
        target = next(t for t in plan.targets if t.form == "expected")
        writes = bank.write(learner.filter, learner.filter.init_state().params)
        actual = np.asarray(writes[target.key][target.index])
        np.testing.assert_array_equal(actual[(prior == 0).transpose(1, 2, 0)], NEG)

    checkpoint = tmp_path / "agent.npz"
    learner.checkpoint(checkpoint)
    replay = agent(builder)
    replay.restore(checkpoint)
    assert learner.act() == replay.act()
    learner.observe(1, 0)
    replay.observe(1, 0)
    np.testing.assert_array_equal(
        bank.params[OWNER]["alpha"], replay.learnables.params[OWNER]["alpha"]
    )

    t, arrays = builder(steps=1, past=0, learn=(OWNER,), epistemic_b=True)
    assert structure_hash(t) == structure_hash(from_json(to_json(t)))
    saved = tmp_path / "changing.cffgz"
    save_bundle(saved, t, arrays, learnables=bank)
    restored = load_bundle(saved).learnables
    np.testing.assert_array_equal(
        restored.predicted_transition_counts(OWNER),
        bank.predicted_transition_counts(OWNER),
    )
    with pytest.raises(ValueError, match="(declaration|identity|ownership|shapes)"):
        agent(model(0.3, sparse=True)).learnables.load_state_dict(
            bank.state_dict(), exact=True
        )
    with pytest.raises(ValueError, match="legacy"):
        bank.load_state_dict({OWNER: bank.params[OWNER]["alpha"]}, exact=True)


@pytest.mark.parametrize(
    "dtype,initial", [(np.float64, (1.0, 0.0)), (np.float32, (0.3, 0.7))]
)
def test_stationary_limit_through_real_engine(dtype, initial):
    eps = np.finfo(dtype).eps
    with jax.enable_x64(dtype == np.float64):
        explicit, ordinary = [
            agent(model(h, initial=initial, dtype=dtype)) for h in (0.0, None)
        ]
        for action, observation in ((0, 0), (1, 1), (0, 1)):
            assert explicit.act() == ordinary.act()
            runs = [
                a.plan.run(a._with_params(a.plan), a.n_sweeps, tol=1e-10)
                for a in (explicit, ordinary)
            ]
            assert runs[0].status is runs[1].status
            for r in runs:
                r.require_converged()
            np.testing.assert_allclose(
                runs[0].state.messages,
                runs[1].state.messages,
                rtol=128 * eps,
                atol=128 * eps,
            )
            np.testing.assert_allclose(
                runs[0].free_energies,
                runs[1].free_energies,
                rtol=256 * eps,
                atol=256 * eps,
            )
            local = []
            ranks = []
            for current, r in zip((explicit, ordinary), runs, strict=True):
                assert (
                    sum(len(g.members) for g in current.plan.arena.gfe_tm_groups) == 1
                )
                local.append(
                    add_transition_gfe_terms(
                        0.0, current.plan.arena, r.state.messages, r.state.params
                    )
                )
                marginal = current.plan.marginals(r.state)[current._plan_spec.action]
                ranks.append(np.argsort(-np.asarray(marginal), kind="stable"))
            np.testing.assert_allclose(*local, rtol=256 * eps, atol=256 * eps)
            np.testing.assert_array_equal(*ranks)
            for a in (explicit, ordinary):
                a.observe(action, observation)
            np.testing.assert_allclose(
                explicit.learnables.params[OWNER]["alpha"],
                ordinary.learnables.params[OWNER]["alpha"],
                rtol=128 * eps,
                atol=128 * eps,
            )


def test_float32_sparse_next_event_and_invalid_restore_are_atomic():
    with jax.enable_x64(False):
        learner = agent(model(0.15, sparse=True, initial=(0.4, 0.6), dtype=np.float32))
        learner.observe(0, 1)
        params = learner.learnables
        before = params.params[OWNER]["alpha"].copy()
        result = learner.plan.run(
            learner._with_params(learner.plan), learner.n_sweeps, tol=1e-10
        )
        state = result.require_converged()
        assert state.messages.dtype == np.float32 and np.isfinite(state.messages).all()
        assert np.isfinite(result.free_energies[-1])
        data = params.state_dict()
        key = next(k for k in data if np.shape(data[k]) == before.shape)
        data[key] = data[key].copy()
        data[key][0, 0, 1] = 1
        with pytest.raises(ValueError, match="digest"):
            params.load_state_dict(data)
        np.testing.assert_array_equal(params.params[OWNER]["alpha"], before)


def test_tied_sensors_pool_one_clock_and_copy_declarations():
    from dataclasses import replace

    from cffg.compiler import compile_template
    from cffg.core.compose import scoped
    from cffg.core.ir import PlateSpec, Template
    from cffg.core.plates import expand_plates
    from cffg.engine import Engine
    from cffg.learn import Learnables
    from test_learnables import two_chain_template

    t, arrays = two_chain_template(start_a=0, start_b=1)
    t.learnables["B"] = replace(t.learnables["B"], change_hazard=0.2)
    assert scoped(t, "cell").learnables["cell/B"].change_hazard == 0.2
    plate = Template(
        name="plate", plates={"r": PlateSpec(t, (2,), per_copy_blobs=("Bs",))}
    )
    copied = expand_plates(plate)
    assert len(copied.learnables) == 2 and all(
        d.change_hazard == 0.2 for d in copied.learnables.values()
    )

    engine = Engine(compile_template(t, arrays=arrays))
    prior = np.ones_like(arrays["Bs"])
    bank = Learnables(engine, priors={"B": prior})
    result = engine.run(
        engine.bind_data(
            engine.init_state(), {"acta": 0, "seea": 1, "actb": 1, "seeb": 2}
        ),
        40,
        tol=1e-12,
    )
    joint = engine.node_beliefs(result.require_converged())
    counts = (joint["tma"] + joint["tmb"]).transpose(2, 0, 1)
    np.testing.assert_allclose(counts.sum(), 2, atol=1e-12)
    for _ in range(2):
        expected = project_table(prior, bank.params["B"]["alpha"], counts, 0.2)
        bank.accumulate(engine, result)
        np.testing.assert_allclose(bank.params["B"]["alpha"], expected, rtol=2e-12)


@pytest.mark.parametrize("rare", [1e-39, 1e-50])
def test_dormant_support_cannot_disappear_in_real_float32(rare):
    from cffg.compiler import compile_template
    from cffg.engine import Engine
    from cffg.learn import Learnables

    builder = model(0.2, sparse=True)
    t, arrays = builder(steps=1, past=0, learn=(OWNER,), epistemic_b=True)
    owner = t.learnables[OWNER]
    (blob,) = owner.blobs
    arrays[blob][0, 0, 0] = rare
    with jax.enable_x64(True):
        e = Engine(compile_template(t, arrays=arrays))
        learner = Learnables(e)
        assert np.all(np.isfinite(learner.predicted_transition_counts(OWNER)))
        assert e.init_state().messages.dtype == np.float64
    with jax.enable_x64(False), pytest.raises(ValueError, match="unrepresentable"):
        e.init_state()


def test_independent_transition_clocks_and_named_write_support():
    from dataclasses import replace

    from cffg.compiler import compile_template
    from cffg.core.ir import BlobRef, LearnableSpec
    from cffg.engine import Engine
    from cffg.learn import Learnables
    from cffg.params import set_params
    from test_learnables import two_chain_template

    template, arrays = two_chain_template()
    template.learnables["B"] = replace(template.learnables["B"], change_hazard=0.2)
    template.blobs["B2"] = BlobRef("B2", "B2.npy")
    second = arrays["Bs"].copy()
    second[:, :, 0] = [1, 0, 0]
    arrays["B2"] = second
    template.factors["tmb"] = replace(
        template.factors["tmb"], param_blobs={"matrices": "B2"}
    )
    template.learnables["second"] = LearnableSpec(
        "second", "dirichlet", ("B2",), change_hazard=0.7
    )
    engine = Engine(compile_template(template, arrays=arrays))
    priors = {"B": np.ones_like(arrays["Bs"]), "second": 4 * second}
    bank = Learnables(engine, priors=priors)
    params = engine.init_state().params
    with pytest.raises(ValueError, match="changes transition support"):
        set_params(engine, params, {"tmb": {"matrices": arrays["Bs"]}})
    set_params(engine, params, {"tmb": {"matrices": second}})
    with pytest.raises(ValueError, match="real"):
        Learnables(engine, priors={**priors, "second": second + 1j})
    result = engine.run(
        engine.bind_data(
            engine.init_state(), {"acta": 0, "seea": 1, "actb": 1, "seeb": 2}
        ),
        40,
        tol=1e-12,
    )
    bank.accumulate(engine, result)
    second_before = bank.params["second"]["alpha"].copy()
    first_before = bank.params["B"]["alpha"].copy()
    bank.accumulate(engine, result, only=("B",))
    np.testing.assert_array_equal(bank.params["second"]["alpha"], second_before)
    assert not np.array_equal(first_before, bank.params["B"]["alpha"])
    np.testing.assert_array_equal(bank.params["second"]["alpha"][:, 1:, 0], 0)


@pytest.mark.parametrize("forbidden", [1e-50, -1e-50])
def test_float32_packing_rejects_source_mass_outside_the_face(forbidden):
    builder = model(0.2, sparse=True, dtype=np.float32)
    counts = builder.keywords["transition_counts"][OWNER].astype(np.float64)
    counts[0, 0, 1] = forbidden
    with pytest.raises(ValueError, match="transition support"):
        builder(
            steps=1,
            past=0,
            learn=(OWNER,),
            epistemic_b=True,
            transition_counts={OWNER: counts},
        )
