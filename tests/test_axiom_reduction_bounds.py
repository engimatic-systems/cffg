"""Fixed-row likelihood bounds and exact tie behavior throughout a greedy pass."""

from unittest.mock import patch

import numpy as np

from cffg.experiments._axiom_reduction_scores import Competition, Records, Tables, _fixed_bounds


def tables(means, constants, dim=1):
    k = len(constants)
    mu = np.broadcast_to(np.array(means,dtype=float)[:,None], (k,dim)).copy()
    precision = np.broadcast_to(np.eye(dim), (k,dim,dim)).copy()
    head = np.zeros((k, 1))
    return Tables(mu, precision, np.asarray(constants, float), (head,)*6, (np.ones_like(head),)*6)


def test_strict_upper_bound_and_lower_index_ties_after_new_proposal():
    t = tables([10,0,10,0], [-9,0,0,-10])
    tape = Records(np.array([[0.],[10.]]), np.zeros((2,6),dtype=int))
    app = Competition(t,tape,[1,2],active=np.array([False,True,True,False]))
    np.testing.assert_array_equal(app.winners(), [1,2])
    assert (app.dormant.winner == -1).all()
    # Slot 0 ties slot 2 on record 1 only after the trial and wins by index.
    next_rows = tables([0,10],[-8,-9])
    np.testing.assert_array_equal(app.trial_winners([1,2],next_rows), [1,0])
    np.testing.assert_array_equal(app.dormant.winner, [-1,0])
    with patch.object(t, "context", side_effect=AssertionError("unnecessary fixed rescan")):
        np.testing.assert_array_equal(app.trial_winners([1,2],next_rows), [1,0])
    app.accept([1,2],next_rows)
    np.testing.assert_array_equal(app.winners(), [1,0])


def test_full_fallback_without_diagonal_precision_or_safe_product_range():
    t = tables([0,1,0], [-200,-200,-200],dim=2)
    t.precision[0,0,1] = t.precision[0,1,0] = .5
    t.log_heads = tuple(np.array([[-2.],[-4.],[-5.]]) for _ in range(6))
    records = Records(np.zeros((3,2)),np.zeros((3,6),int))
    eligible, upper = _fixed_bounds(t, records, np.arange(3))
    np.testing.assert_array_equal(eligible,[False,True,True])
    assert upper == -216.
    extreme = Records(np.full((1,2),1e80), np.zeros((1,6),int))
    eligible, _ = _fixed_bounds(t,extreme,np.arange(3))
    assert not eligible.any()
    # All rows are fixed and none are live. Resolve both dense and bounded rows.
    cache = Competition(t,records,[],active=np.zeros(3,bool))
    np.testing.assert_array_equal(cache.winners(), t.context(records,np.arange(3)).argmax(axis=1))
    t.precision[0] = np.eye(2)
    cache = Competition(t,records,[],active=np.zeros(3,bool))
    np.testing.assert_array_equal(cache.winners(), t.context(records,np.arange(3)).argmax(axis=1))
