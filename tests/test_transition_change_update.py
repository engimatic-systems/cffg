"""Native changing-column projection versus independent D1a quadrature and census."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import digamma

from cffg.transition_change import update_column
from test_transition_change_derivation import update


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_sequence_matches_reference_and_has_transactional_failures(dtype):
    eps = np.finfo(dtype).eps
    with jax.enable_x64(dtype == np.float64):
        prior = np.array([1., 0., 1.], dtype)
        carry = np.array([8., 0., 2.], dtype)
        reference = carry.astype(float)
        support = prior > 0
        staged = jax.jit(update_column)
        for data in ([0.,0.,1.],[.8,0.,.2],[.05,0.,.95],[0.,0.,1.],[0.,0.,0.]):
            counts = np.array(data,dtype)
            reference = update(prior.astype(float), reference, counts.astype(float), .25)
            before = carry.copy()
            carry, valid = staged(prior, carry, counts, support, .25)
            eager = update_column(prior, before, counts, support, .25)
            assert valid and eager.valid and carry.dtype == dtype and carry[1] == 0
            np.testing.assert_allclose(carry, reference, rtol=128*eps, atol=128*eps)
            np.testing.assert_allclose(carry, eager.concentration, rtol=128*eps, atol=128*eps)
        for values, mask, h in (([0,1,0],support,.3), ([-1,0,0],support,.3),
                                ([0,0,0],~support,.3), ([0,0,0],support,np.nan)):
            bad = staged(prior,carry,np.array(values,dtype),mask,h)
            assert not bad.valid
            np.testing.assert_array_equal(bad.concentration, carry)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_batches_large_counts_endpoints_and_singleton(dtype):
    with jax.enable_x64(dtype == np.float64):
        rng = np.random.default_rng(8)
        shape = (80, 6)
        prior = rng.uniform(.3, 2., shape).astype(dtype)
        carry = rng.uniform(1e5, 5e5, shape).astype(dtype)
        counts = rng.uniform(0, 1, shape).astype(dtype)
        counts /= counts.sum(axis=1,keepdims=True)
        support = np.ones(shape,bool)
        batch = jax.jit(jax.vmap(update_column,in_axes=(0,0,0,0,None)))
        result, valid = batch(prior,carry,counts,support,.15)
        assert valid.all() and result.shape == shape
        # Integrate the derivative of log Gamma independently; direct SciPy
        # beta-ratio differences at million-scale concentrations can lose digits.
        def log_evidence(alpha, c):
            increments = [quad(lambda t: digamma(x+t), 0., y)[0] for x,y in zip(alpha,c)]
            return sum(increments) - quad(lambda t: digamma(alpha.sum()+t), 0., c.sum())[0]
        for a,b,c,row in zip(prior.astype(float),carry.astype(float),counts.astype(float),result):
            log_odds = np.log(.15/.85) + log_evidence(a,c) - log_evidence(b,c)
            w = 1/(1+np.exp(-log_odds))
            means = np.stack([(b+c)/(b+c).sum(), (a+c)/(a+c).sum()])
            mu = (1-w)*means[0] + w*means[1]
            tr = (means*(1-means)).sum(axis=1)
            within = tr/(1+np.array([(b+c).sum(), (a+c).sum()]))
            v = (1-w)*within[0] + w*within[1] + w*(1-w)*np.sum((means[0]-means[1])**2)
            expected = mu*((1-mu@mu)/v-1)
            np.testing.assert_allclose(row,expected,rtol=128*np.finfo(dtype).eps,atol=128*np.finfo(dtype).eps)
        for h,base in ((0,carry),(1,prior)):
            r, valid = batch(prior,carry,counts,support,h)
            assert valid.all()
            np.testing.assert_array_equal(r,base+counts)
        r=update_column(jnp.array([0,2],dtype),jnp.array([0,9],dtype),
                        jnp.array([0,1],dtype),jnp.array([False,True]),.5)
        assert r.valid
        np.testing.assert_array_equal(r.concentration,[0,2])
