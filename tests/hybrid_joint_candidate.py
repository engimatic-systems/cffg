"""Test-only adapters between independent tiny references and the JAX candidate."""

import jax
import jax.numpy as jnp
import numpy as np

from cffg.hybrid_joint import Coordinate
from cffg.hybrid_natural import Natural
from cffg.hybrid_separator import TransitionParameters
from hybrid_joint_reference import Natural as ReferenceNatural, load_case, tilted, zero_messages


def context(name, dtype=jnp.float64):
    model = load_case(name)
    arrays = [jnp.asarray(np.stack([s.arrays()[rank] for s in model.sites]), dtype) for rank in range(3)]
    sites = Natural(*arrays, jnp.zeros((model.shape[0],), jnp.int32), jnp.asarray(model.origins, dtype))
    parameters = TransitionParameters(*(jnp.asarray(x, dtype) for x in
                                       (model.weights, model.offsets, np.zeros_like(model.offsets),
                                        model.matrix, model.bias, model.noise)))
    return Coordinate.from_sites(sites), parameters, sites, model


def from_reference(coordinate, messages, xi):
    banks = tuple(Natural(*(jnp.asarray(np.stack([time[side].arrays()[rank] for time in messages]), coordinate.origin.dtype)
                            for rank in range(3)),
                          jnp.zeros((coordinate.shape[0] - 1,), jnp.int32),
                          coordinate.origin[side:side + coordinate.shape[0] - 1]) for side in range(2))
    return coordinate.pack(banks, jnp.asarray(xi, coordinate.origin.dtype))


def held_coordinate(coordinate, model, start=0., perturb=True):
    xi = np.full_like(model.offsets, start)
    xi[~model.auxiliary_support] = 0
    _, _, _, messages = tilted(model, zero_messages(model), xi)
    vector = from_reference(coordinate, messages, xi)
    if perturb:
        vector = vector.at[:coordinate.message_size].add(
            .01 * jnp.sin(jnp.arange(coordinate.message_size, dtype=vector.dtype) + 1))
    return vector


def to_reference(coordinate, vector):
    banks, xi = coordinate.unpack(vector)
    messages = []
    for t in range(coordinate.shape[0] - 1):
        time = []
        for side, bank in enumerate(banks):
            active = np.asarray(coordinate.support[t + side])
            # Independent storage uses zero for absent constants. Support is in
            # unary sites; the native candidate retains -inf metadata instead.
            arrays = [np.where(active.reshape((*active.shape, *((1,) * rank))), np.asarray(a[t]), 0)
                      for rank, a in enumerate(bank.arrays())]
            time.append(ReferenceNatural(*arrays))
        messages.append(time)
    return messages, np.asarray(xi)


def output_arrays(coordinate, vector, evaluation):
    messages, xi = to_reference(coordinate, vector)
    groups = dict(messages=np.concatenate([x.ravel() for bank in messages for n in bank for x in n.arrays()]),
                  xi=xi.ravel(), energy=np.asarray([evaluation.local.free_energy, evaluation.local.dual_energy]))
    for name in ('regions', 'separators'):
        belief = getattr(evaluation.local, name)
        for field in ('probability', 'displacement', 'mean', 'covariance', 'second'):
            value = belief.full_covariance if field == 'covariance' else getattr(belief, field)
            groups[f'{name}_{field}'] = np.asarray(value).ravel()
    learning = evaluation.auxiliary
    for name, value in [('target', learning.target), ('exposure', learning.exposure),
                        ('centered_linear', learning.linear), ('centered_precision', learning.precision),
                        ('linear', jax.vmap(lambda x: x.physical_linear)(learning)),
                        ('precision', jax.vmap(lambda x: x.physical_precision)(learning))]:
        groups[f'learning_{name}'] = np.asarray(value).ravel()
    return groups
