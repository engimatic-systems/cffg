"""Identify shared variable dimensions through factor roles and tied array axes."""

from collections import defaultdict

from cffg.core.ir import FACTOR_TYPES

# Each role names the variable that sets an axis length. A recurrence has K-1 sticks.
PARAMETER_AXES = {
    "gaussian": {"mean": ("out",), "cov": ("out", "out")},
    "gaussian_linear": {
        "matrix": ("out", "in"),
        "offset": ("out",),
        "cov": ("out", "out"),
    },
    "gaussian_mixture_linear": {
        "matrix": ("selector", "out", "in"),
        "offset": ("selector", "out"),
        "cov": ("selector", "out", "out"),
    },
    "recurrence_link": {
        "weights": ("in", "out-1", "drive"),
        "offsets": ("in", "out-1"),
    },
    "categorical": {"statistic": ("out",)},
    "matrix_cat": {"matrix": ("out", "in")},
    "transition_mixture": {"matrices": ("selector", "out", "in")},
}


def dependencies(template, variable, recipe_axes):
    groups = defaultdict(set)
    blobs = {}
    for factor in template.factors.values():
        if FACTOR_TYPES[factor.type].symmetric_domain:
            groups["factor", factor.name].update(factor.edges.values())
        for role, blob in factor.param_blobs.items():
            spec = PARAMETER_AXES.get(factor.type, {}).get(role)
            if spec is None:
                continue
            dims = []
            for i, r in enumerate(spec):
                edge = factor.edges[r.removesuffix("-1")]
                groups["array", blob, i].add(edge)
                dims.append((edge, int(r.endswith("-1"))))
            blobs[blob] = dims
    for name, axis in recipe_axes.items():
        if name in template.variables:
            groups["recipe", axis].add(name)
    names = {variable}
    while True:
        count = len(names)
        for adjacent in groups.values():
            if names.intersection(adjacent):
                names.update(adjacent)
        if count == len(names):
            return names, blobs
