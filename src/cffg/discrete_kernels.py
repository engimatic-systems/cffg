"""Registered adapters for the shipped ordinary categorical factors.

Numerical expressions remain in rules.py. This module owns their serialized
parameter layouts and connects those rules to the single-arena contract.
Imports of numerical JAX rules are deferred until execution.
"""

from __future__ import annotations

import numpy as np

from cffg.core.ir import FACTOR_TYPES
from cffg.kernel_bindings import DiscreteBinding, DiscreteMode, ParameterSpec


def _shape(order):
    return lambda sizes: tuple(sizes[role] for role in order)


def _pack(role, spec):
    def pack(template, members, resolved):
        from cffg.compiler import _expect, _log_of, _stochastic

        rows = []
        for name in members:
            factor = template.factors[name]
            sizes = {
                r: template.variables[v].cardinality for r, v in factor.edges.items()
            }
            value = resolved[factor.param_blobs[role]]
            _expect(name, value, spec.shape(sizes))
            columns = np.moveaxis(value, spec.probability_axis, 0).reshape(
                value.shape[spec.probability_axis], -1
            )
            _stochastic(name, columns)
            if spec.permutation:
                value = value.transpose(spec.permutation)
            rows.append(_log_of(value))
        return {"logf": np.stack(rows)}

    return pack


def _bp(name):
    def message(inc, params, cards, width, op):
        from cffg.rules import BP_UPDATES

        return BP_UPDATES[name](inc, params["logf"], cards, width, op)

    def energy(inc, params, cards):
        from cffg.rules import NODE_ENERGIES

        return NODE_ENERGIES[name](inc, params["logf"], cards)

    return DiscreteMode(message, energy)


def _vmp(name):
    def message(beliefs, params, cards, width):
        from cffg.rules import VMP_UPDATES

        return VMP_UPDATES[name](beliefs, params["logf"], cards, width)

    def energy(beliefs, params, cards):
        from cffg.rules import VMP_ENERGIES

        return VMP_ENERGIES[name](beliefs, params["logf"], cards)

    return DiscreteMode(message, energy)


def install(register, kernel, hostile):
    specifications = (
        ("categorical", "statistic", ("out",), 0, ()),
        ("matrix_cat", "matrix", ("out", "in"), 0, ()),
        ("transition_mixture", "matrices", ("selector", "out", "in"), 1, (1, 2, 0)),
    )
    for name, role, order, axis, permutation in specifications:
        spec = ParameterSpec(
            _shape(order), "logf", "probability_to_log", axis, permutation, True
        )
        register(
            kernel(
                FACTOR_TYPES[name],
                f"cffg.discrete.{name}",
                "1.0.0",
                hostile=hostile,
                binding=DiscreteBinding(
                    FACTOR_TYPES[name].edge_roles,
                    {role: spec},
                    ("logf",),
                    _pack(role, spec),
                    lambda params, cards: params["logf"],
                    _bp(name),
                    _vmp(name),
                ),
            )
        )
