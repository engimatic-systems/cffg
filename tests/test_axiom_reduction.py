"""Independent BMR score, sequencing, sampler and restart checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, gammaln

from cffg.component_bank import ComponentBank, SlotStatus
from cffg.experiments import axiom_reduction as bmr
from cffg.experiments.axiom_evidence import initialized_view
from cffg.mixed_learning import MixedBank


def fixture(dtype, k=2, d=1, widths=(2,2,2,2,2,2), live=2):
    def repeat(x):
        return np.broadcast_to(np.array(x, dtype), (k, *np.shape(x))).copy()
    p = MixedBank(repeat(np.zeros(d)), repeat(2), repeat(np.eye(d)), repeat(d+3),
                  repeat(1), tuple(repeat(np.ones(n)) for n in widths), np.arange(k)<live)
    original = ComponentBank(p)
    record = original.state_dict()
    # Each occupied row has one category-only reading, including reward/switch
    # zero. Its action is the row's parity, yielding two contextual winners.
    for i in range(live):
        record["posterior/alpha"][i] += 1
        for head in range(6):
            record[f"posterior/category_{head}"][i, i%2 if head == 3 else 0] += 1
    return ComponentBank.from_state_dict(record)


def tape(outcome, dtype=np.float64):
    categories = np.array([[0,0,0,0,outcome,outcome],[0,0,0,1,outcome,outcome]], np.int32)
    return bmr.Records(np.zeros((2,1), dtype), categories)


def independent_scales(raw, *, minimum=None):
    mu = np.asarray(raw.mu, float) + np.asarray(raw.carry.mu, float)
    kappa = np.asarray(raw.kappa, float) + np.asarray(raw.carry.kappa, float)
    scale = np.asarray(raw.psi, float) + np.asarray(raw.carry.psi, float)
    d = mu.shape[1]
    ids = np.arange(d)
    factor = np.sqrt(np.maximum(1, .5 * (scale[:, ids, ids] + kappa[:, None] * mu**2)))
    norm = factor[:, :, None] * factor[:, None, :]
    scaled = scale / norm
    scaled[np.abs(scaled) < .001] = 0
    scaled[:, ids, ids] = np.maximum(.001, scaled[:, ids, ids])
    if minimum is None:
        minimum = min(0., float(np.min(np.linalg.eigvalsh(scaled))))
    if minimum < 0:
        scaled[:, ids, ids] += 2 * (np.finfo(raw.mu.dtype).eps - minimum)
    return scaled * norm, minimum


def scalar_score(bank, records, virgin=None, *, minimum=None):
    # Score every row, including dormant ones, from the NIW and Dirichlet formulas.
    projected = initialized_view(bank.posterior, bank.generations, virgin)
    scales, _ = independent_scales(projected, minimum=minimum)
    mass = np.asarray(bank.posterior.alpha, float) + np.asarray(bank.posterior.carry.alpha, float)
    base = np.asarray(bank.reset_prior.alpha, float) + np.asarray(bank.reset_prior.carry.alpha, float)
    outputs, winners = [], []
    for x, labels in zip(*records, strict=True):
        contexts, held = [], []
        for j in range(len(mass)):
            mu = np.asarray(projected.mu[j],float) + np.asarray(projected.carry.mu[j],float)
            psi = scales[j]
            nu = float(projected.nu[j]) + float(projected.carry.nu[j])
            kappa = float(projected.kappa[j]) + float(projected.carry.kappa[j])
            d = len(mu)
            quad = (x - mu) @ np.linalg.solve(psi, x - mu)
            gauss = .5*(sum(digamma((nu-i)/2) for i in range(d)) - d*np.log(np.pi)
                        - np.linalg.slogdet(psi)[1] - d/kappa - nu*quad)
            elog = []
            for h, y in enumerate(labels):
                a = np.asarray(projected.categories[h][j],float) + np.asarray(projected.carry.categories[h][j],float)
                elog.append(digamma(a[y]) - digamma(sum(a)))
            contexts.append(gauss + sum(elog[:4]))
            held.append(sum(elog[4:]) if mass[j] > base[j] else 0.)
        winner = np.argmax(contexts)
        outputs.append(held[winner])
        winners.append(winner)
    kl = gammaln(sum(mass)) - gammaln(sum(base))
    for a, p in zip(mass, base, strict=True):
        kl += gammaln(p) - gammaln(a) + (a-p)*(digamma(a)-digamma(sum(mass)))
    return np.mean(outputs)-kl, winners


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("label","delta","status"), [(0,.0264805138932786,"accepted"),
                                                      (1,-.9735194861067214,"rejected")])
def test_frozen_reference_deltas(dtype, label, delta, status):
    bank = fixture(dtype)
    data = tape(label,dtype)
    out, before, after, log = bmr.reduce_on_tape(bank, data, np.array([[0,1],[1,0]]))
    first = log[0]
    independent, winners = scalar_score(bank,data)
    assert winners == [0,1]
    assert before == pytest.approx(independent, abs=2e-12)
    assert first.proposed - first.before == pytest.approx(delta,abs=2e-12)
    assert first.status == status
    assert after == pytest.approx(scalar_score(out,data)[0],abs=2e-12)
    if status == "accepted":
        assert log[1].status == "inactive" and out.reference(1) is None
        assert out.posterior.alpha[0] == 3 and out.posterior.alpha[1] == 1
    else:
        assert out is bank


def test_random_virgin_winner_has_zero_output_and_uses_raw_kl():
    bank = fixture(np.float64, k=3)
    data = tape(0)._replace(features=np.full((2,1),10.))
    raw = bank.posterior
    # This factory-only head lies below the raw reset concentration.
    seeded_heads = tuple(a.at[2].set(jnp.array([.1, .05])) for a in raw.categories)
    virgin = raw._replace(mu=raw.mu.at[2].set([10.]), categories=seeded_heads)
    expected, winners = scalar_score(bank,data,virgin)
    assert winners == [2,2]
    _, initial, _, log = bmr.reduce_on_tape(bank,data,np.array([[0,2]]),dormant_initial=virgin)
    assert initial == pytest.approx(expected,abs=1e-12)
    assert log[0].status == "inactive"
    no_overlay = bmr.reduce_on_tape(bank,data,np.empty((0,2),int))[1]
    assert abs(no_overlay-initial) > .1
    assert np.all(np.asarray(bank.posterior.categories[0][2]) == 1)


def test_incremental_rankings_and_sequential_decisions_match_full_rescans():
    bank = fixture(np.float64,k=9,live=7)
    rec = bank.state_dict()
    rec["posterior/mu"][:7,0] = [-1.,-2.,.5,1.25,1.5,3.,-1.25]
    bank = ComponentBank.from_state_dict(rec)
    data = bmr.Records(np.linspace(-3,3,19)[:,None], np.resize(tape(0).categories,(19,6)))
    pairs = np.array([[0,2],[1,4],[0,3],[2,6],[5,1],[4,0],[3,8],[1,1]])
    running = bank
    result, _, _, decisions = bmr.reduce_on_tape(bank,data,pairs)
    for (i,j), actual in zip(pairs,decisions,strict=True):
        before = scalar_score(running,data)[0]
        assert actual.before == pytest.approx(before,abs=2e-12)
        r, d = running.reference(int(i)), running.reference(int(j))
        if r is None or d is None:
            assert actual.status == "inactive"
            continue
        candidate = running.propose_merge(r,d)
        if candidate.status != SlotStatus.OK:
            assert actual.status == candidate.status.value
            continue
        score = scalar_score(candidate.bank,data)[0]
        assert actual.proposed == pytest.approx(score,abs=2e-12)
        if score >= before:
            assert actual.status == "accepted"
            running = candidate.bank
        else:
            assert actual.status == "rejected"
    assert result.logical_ids == running.logical_ids


def test_equal_score_is_accepted():
    p = fixture(np.float64).reset_prior
    empty_stats = ComponentBank(p)
    result, before, after, decisions = bmr.reduce_on_tape(empty_stats,tape(0),np.array([[0,1]]))
    assert before == after == 0 and decisions[0].status == "accepted"
    assert sum(result.posterior.active) == 1


def test_generated_gaussian_uses_projected_posterior_mean_covariance():
    bank = fixture(np.float64,k=2,live=1)
    state = bank.state_dict()
    state["posterior/mu"][0,0] = 100
    state["posterior/nu"][0] = 5
    bank = ComponentBank.from_state_dict(state)
    key = jax.random.key(119)
    data = bmr._draw(bank, bank.posterior, key, 3)
    # Scalar release clipping at mu=100, kappa=2, psi=1 raises the scale.
    # Outer split skips key 0; subkey 2 independently drives the Gaussian.
    scale = .001 * (.5 * (1 + 2*100**2))
    normals = [jax.random.normal(jax.random.split(k,9)[2],(1,),jnp.float64)[0]
               for k in jax.random.split(key,4)[1:]]
    np.testing.assert_allclose(data.features[:,0], 100 + np.sqrt(scale/(5-1-1))*np.asarray(normals),
                               atol=5e-14,rtol=0)


def test_generated_history_survives_npz_checkpoint_and_repeated_step(tmp_path):
    bank = fixture(np.float32,k=6,live=4)
    snapshot = bank.state_dict(exact=True)
    snapshot["posterior/carry_alpha"][0] = np.float32(2.**-28)
    bank = ComponentBank.from_state_dict(snapshot)
    state = bmr.ReductionState.start(bank,jax.random.key(911))
    assert bmr.reduce_interactions(bank,state,step=499).state is state
    first = bmr.reduce_interactions(bank,state,step=500,sample_count=12,pair_count=3)
    assert first.status == "ok" and len(first.state.records.features) == 12
    assert bmr.reduce_interactions(first.bank,first.state,step=500).status == "not_due"
    path = tmp_path/"bmr.npz"
    np.savez(path,**first.state.state_dict())
    with np.load(path,allow_pickle=False) as saved:
        restored = bmr.ReductionState.from_state_dict(saved,first.bank)
    restored_bank = ComponentBank.from_state_dict(first.bank.state_dict(exact=True))
    normal = bmr.reduce_interactions(first.bank,first.state,step=1000,sample_count=9,pair_count=3)
    resumed = bmr.reduce_interactions(restored_bank,restored,step=1000,sample_count=9,pair_count=3)
    assert normal.status == resumed.status == "ok"
    assert normal.decisions == resumed.decisions and normal.final_score == resumed.final_score
    assert len(normal.state.records.features) == 21
    for a,b,c in zip(first.state.records, normal.state.records, resumed.state.records, strict=True):
        np.testing.assert_array_equal(a,b[:12])
        np.testing.assert_array_equal(b,c)
    np.testing.assert_array_equal(jax.random.key_data(normal.state.key),jax.random.key_data(resumed.state.key))
    expected = normal.bank.state_dict(exact=True)
    actual = resumed.bank.state_dict(exact=True)
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(expected[name], actual[name], err_msg=name)


def test_empty_support_and_invalid_tape_are_visible():
    bank = ComponentBank(fixture(np.float64).reset_prior)
    state = bmr.ReductionState.start(bank,jax.random.key(1))
    result = bmr.reduce_interactions(bank,state,step=500)
    assert result.status == "empty_support" and result.bank is bank and result.state is state
    bad = tape(0)._replace(categories=np.full((2,6),-1))
    with pytest.raises(ValueError, match="indices"):
        bmr.reduce_on_tape(bank,bad,np.array([[0,1]]))


@pytest.mark.slow
def test_5000_rows_and_2000_accumulated_records():
    bank = fixture(np.float32,k=5000,d=7,widths=(32,33,2,4,3,500))
    cats = np.resize(tape(0).categories,(2000,6))
    data = bmr.Records(np.zeros((2000,7),np.float32),cats)
    # Only two rows are live. After acceptance, repeats are frozen stale pairs.
    pairs = np.tile([[0,1]],(2000,1))
    result, before, after, log = bmr.reduce_on_tape(bank,data,pairs)
    assert np.isfinite(before) and np.isfinite(after) and len(log) == 2000
    assert log[0].status == "accepted" and all(event.status == "inactive" for event in log[1:])
    assert np.sum(result.posterior.active) == 1


def shifting_bank(dtype, k=4, d=3):
    bank = fixture(dtype, k=k, d=d, live=4)
    state = bank.state_dict()
    ids = np.arange(d)
    for prefix in ("prior", "posterior"):
        psi = state[f"{prefix}/psi"]
        psi[0] = np.eye(d) * .04
        psi[1] = np.eye(d) * .0001
        psi[2] = np.eye(d) * .001
        psi[3] = np.eye(d) * .003
        state[f"{prefix}/mu"][2:4] = .25
    psi = state["posterior/psi"][1]
    psi[ids, ids] = .002
    psi[:3,:3] = [[.002,.0015,.0008],[.0015,.002,.0015],[.0008,.0015,.002]]
    state["posterior/category_3"][:4] = [2,1]
    for h in (4,5):
        state[f"posterior/category_{h}"][3] = [1,2]
    return ComponentBank.from_state_dict(state)


@pytest.mark.parametrize("dtype", [pytest.param(np.float64, marks=pytest.mark.slow), np.float32])
@pytest.mark.parametrize("outcome, verdict", [(0, "rejected"), (1, "accepted")])
def test_global_repair_changes_untouched_winner_and_merge_decision(dtype, outcome, verdict):
    bank = shifting_bank(dtype)
    data = bmr.Records(np.array([[.28,.25,.25]],dtype), np.array([[0,0,0,0,outcome,outcome]]))
    candidate = bank.propose_merge(bank.reference(0), bank.reference(1)).bank
    before, first = scalar_score(bank, data)
    proposed, second = scalar_score(candidate, data)
    minimum = independent_scales(bank.posterior)[1]
    assert minimum < 0 and independent_scales(candidate.posterior)[1] == 0
    assert first == [2] and second == [3]
    _, stale = scalar_score(candidate, data, minimum=minimum)
    assert stale == first
    result, initial, final, (decision,) = bmr.reduce_on_tape(bank,data,np.array([[0,1]]))
    assert initial == pytest.approx(before,abs=1e-12)
    assert decision.proposed == pytest.approx(proposed,abs=1e-12)
    assert decision.full_rescore and decision.minimum_before == pytest.approx(minimum,abs=1e-9)
    assert decision.minimum_proposed == 0 and decision.status == verdict
    assert final == pytest.approx(max(proposed,before), abs=1e-12)
    assert (result is bank) == (verdict == "rejected")


def test_global_repair_applies_to_dirty_rows_even_when_minimum_stays_fixed():
    bank = shifting_bank(np.float64)
    data = bmr.Records(np.array([[.28,.25,.25]]), np.array([[0,0,0,0,1,1]]))
    candidate = bank.propose_merge(bank.reference(2),bank.reference(3)).bank
    expected, _ = scalar_score(candidate,data)
    _, _, _, (decision,) = bmr.reduce_on_tape(bank,data,np.array([[2,3]]))
    assert decision.minimum_before == decision.minimum_proposed < 0
    assert not decision.full_rescore and decision.proposed == pytest.approx(expected,abs=1e-12)


def test_sampler_repairs_each_expected_covariance_independently():
    bank = shifting_bank(np.float64)
    key = jax.random.key(722)
    keys = jax.random.split(key,18)[1:]
    data = bmr._draw(bank,bank.posterior,key,len(keys))
    scales, _ = independent_scales(bank.posterior,minimum=0)
    expected, rows = [], []
    for sample_key in keys:
        subkeys = jax.random.split(sample_key,9)
        row = int(jax.random.choice(subkeys[1],4,p=jnp.full(4,.25)))
        rows.append(row)
        cov = scales[row] / (6-3-1)
        minimum = np.linalg.eigvalsh(cov)[0]
        if minimum < 0:
            cov += 2*(np.finfo(float).eps-minimum)*np.eye(3)
        noise = np.asarray(jax.random.normal(subkeys[2],(3,),jnp.float64))
        expected.append(np.asarray(bank.posterior.mu[row])+np.linalg.cholesky(cov)@noise)
    assert set(rows) == {0,1,2,3}
    np.testing.assert_allclose(data.features,expected,rtol=1e-12,atol=1e-14)


@pytest.mark.slow
def test_full_bank_rescore_5000_slots_and_2000_records():
    bank = shifting_bank(np.float32,k=5000)
    data = bmr.Records(np.tile(np.array([[.28,.25,.25]],np.float32),(2000,1)),
                       np.tile([[0,0,0,0,1,1]],(2000,1)))
    _, before, after, (event,) = bmr.reduce_on_tape(bank,data,np.array([[0,1]]))
    assert event.status == "accepted" and event.full_rescore
    assert event.minimum_before < 0 and event.minimum_proposed == 0
    assert after > before


def test_soft_neighbors_and_modal_pair_screen_match_scalar_queries():
    bank = fixture(np.float64, k=7, live=5)
    state = bank.state_dict()
    # Rows 0, 1 and 5 share location and modes. A one-hot modal query prefers
    # row 1, while the soft query from row 0 should select the dormant row 5.
    means = np.array([0,0,7,7,14,0,14.01])
    for prefix in ("prior","posterior"):
        state[f"{prefix}/mu"][:,0] = means
    counts = np.array([[3,2],[99,1],[10,1],[1,10],[5,4],[60,40],[50,40]])
    for h in range(6):
        state[f"posterior/category_{h}"][:] = counts
        state[f"prior/category_{h}"][5:] = counts[5:]
    bank = ComponentBank.from_state_dict(state)
    tables, _ = bmr._model_scores(bank.posterior)
    p = counts / counts.sum(axis=1,keepdims=True)
    expected_log = digamma(counts) - digamma(counts.sum(axis=1,keepdims=True))
    def nearest(i, query):
        # All seven unit scales, kappa, nu and log determinants are equal.
        scores = -2*(means - means[i])**2 + 6*(expected_log @ query)
        scores[i] = -np.inf
        return scores.argmax()
    assert nearest(0,p[0]) == 5 and nearest(0,[1,0]) == 1
    sources = np.arange(5)
    nearest_ids = np.array([nearest(i,p[i]) for i in sources])
    np.testing.assert_array_equal(bmr.soft_neighbors(tables,sources),nearest_ids)
    key = jax.random.key(71)
    drawn = np.asarray(jax.random.choice(key,5,shape=(4,),replace=False))
    modes = counts.argmax(axis=1)
    expected = [(i,nearest_ids[i]) for i in drawn if modes[i] == modes[nearest_ids[i]]]
    assert 0 < len(expected) < len(drawn)
    pairs = bmr._pairs(tables, np.asarray(bank.posterior.active), key, 4)
    np.testing.assert_array_equal(pairs,expected)


def test_undefined_draw_covariance_preserves_bank_history_and_key():
    bank = fixture(np.float32,k=2,live=1)
    state = bank.state_dict()
    state["prior/nu"][0] = state["posterior/nu"][0] = 2
    bank = ComponentBank.from_state_dict(state)
    history = bmr.ReductionState.start(bank,jax.random.key(514))
    run = bmr.reduce_interactions(bank,history,step=500,sample_count=2)
    assert run.status.startswith("invalid_model: posterior-mean covariance is undefined")
    assert run.bank is bank and run.state is history and not run.decisions
    for bad in (jax.random.split(history.key,2),jax.random.split(jax.random.PRNGKey(9),2)):
        with pytest.raises(ValueError, match="expected one"):
            bmr.ReductionState.start(bank,bad)
