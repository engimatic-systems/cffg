"""Replayable local structure proposals for a hidden mixture selector or affine input."""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from cffg.compiler import compile_template
from cffg.core import FactorSpec
from cffg.core.edits import AddVariable, RemoveBlob, RemoveFactor, RemoveVariable, ReplaceFactor
from cffg.structure_proposals import edit_step, record_proposal, replay_proposal


@dataclass(frozen=True)
class SplitState:
    """Divide one latent state's mass, optionally supplying two new emission columns."""

    selector: str
    state: int
    fraction: float = 0.5
    emissions: tuple | None = None


@dataclass(frozen=True)
class MergeStates:
    """Combine two Dirichlet concentrations and pool their categorical emissions."""

    selector: str
    recipient: int
    donor: int


@dataclass(frozen=True)
class PruneState:
    """Remove a categorical state or Gaussian mixture component."""

    selector: str
    state: int


@dataclass(frozen=True)
class AddRegime:
    """Propose a Gaussian at a training observation using the AXIOM online threshold."""

    selector: str
    observation: np.ndarray
    covariance: np.ndarray
    concentration: float
    log_base_density: float
    capacity: int | None = None


@dataclass(frozen=True)
class PruneGaussianInput:
    """Remove an affine factor's dependence on its input by fixing the slope to zero."""

    factor: str


@dataclass(frozen=True)
class StructuralCandidate:
    record: dict
    inputs: dict[str, np.ndarray]


@dataclass(frozen=True)
class RejectedEdit:
    code: str
    detail: str


class _Invalid(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _index(index, size):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < size:
        raise _Invalid("invalid_state", f"expected a state index in [0, {size})")
    return int(index)


def _one(factors, description):
    if len(factors) != 1:
        raise _Invalid("unsupported_graph", f"expected exactly one {description}")
    return factors[0]


def _weights(template, categorical):
    """Locate the mixing probabilities or the actual in-graph Dirichlet concentration."""
    if "statistic" in categorical.param_blobs:
        return categorical.param_blobs["statistic"], None, categorical.name
    carrier = categorical.param_edges["statistic"]
    prior = _one([f for f in template.factors.values()
                  if f.type == "dirichlet" and f.edges.get("out") == carrier],
                 "Dirichlet prior on the mixing parameter")
    return prior.param_blobs["counts"], carrier, prior.name


def _preserve_support(values, source):
    dtype = source.dtype if source.dtype.kind == "f" else np.dtype(np.float64)
    result = np.asarray(values, dtype=dtype)
    if not np.isfinite(result).all() or ((np.asarray(values) > 0) & (result == 0)).any():
        raise _Invalid("numeric_range", "new positive mass is outside the parameter dtype")
    return result


def _emission_columns(columns, shape):
    columns = np.asarray(columns)
    if columns.shape != shape or columns.dtype.kind not in "fiu":
        raise _Invalid("invalid_emission", "expected two emission vectors in the observed alphabet")
    if not np.isfinite(columns).all() or (columns < 0).any():
        raise _Invalid("invalid_emission", "emission probabilities must be finite and nonnegative")
    return columns


def _grow(op, weights, means, covariances):
    if op.capacity is not None:
        if isinstance(op.capacity, bool) or not isinstance(op.capacity, Integral) or op.capacity < 1:
            raise _Invalid("invalid_capacity", "capacity must be a positive integer")
        if len(weights) >= op.capacity:
            raise _Invalid("capacity", "the requested regime capacity is full")
    point = np.asarray(op.observation)
    birth_cov = np.asarray(op.covariance)
    d = means.shape[1]
    if (point.shape != (d,) or birth_cov.shape != (d, d)
            or point.dtype.kind not in "fiu" or birth_cov.dtype.kind not in "fiu"):
        raise _Invalid("invalid_parameter", "new Gaussian parameters must match the observed dimension")
    if not (np.isfinite(point).all() and np.isfinite(birth_cov).all()
            and np.isfinite(op.concentration) and op.concentration > 0
            and np.isfinite(op.log_base_density)):
        raise _Invalid("invalid_parameter", "growth requires finite inputs and a positive concentration")
    # The existing emissions are fixed Gaussians, so their expected log likelihood is exact.
    active = weights > 0
    chol = np.linalg.cholesky(covariances[active].astype(np.float64))
    residual = np.linalg.solve(chol, (point - means[active].astype(np.float64))[..., None])[..., 0]
    logpdf = -0.5 * ((residual**2).sum(axis=1) + d * np.log(2 * np.pi))
    logpdf -= np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=1)
    if not np.isfinite(logpdf).all():
        raise _Invalid("numeric_range", "component likelihood is outside the scoring range")
    if logpdf.max() >= op.log_base_density + np.log(op.concentration):
        raise _Invalid("existing_regime", "an occupied component meets the online growth threshold")
    return (np.append(weights, op.concentration),
            np.concatenate((means, point[None]), axis=0),
            np.concatenate((covariances, birth_cov[None]), axis=0))


def _split_columns(op, weights, table):
    size = len(weights)
    i = _index(op.state, size)
    if not 0 < op.fraction < 1:
        raise _Invalid("invalid_fraction", "both split children need a positive share")
    new_weights = np.append(weights, weights[i] * (1 - op.fraction))
    new_weights[i] *= op.fraction
    if weights[i] > 0 and (new_weights[i] == 0 or new_weights[-1] == 0):
        raise _Invalid("numeric_range", "a positive split concentration rounded to zero")
    columns = (table[:, i], table[:, i]) if op.emissions is None else op.emissions
    left, right = _emission_columns(columns, (2, table.shape[0]))
    table = np.column_stack((table, right))
    table[:, i] = left
    return new_weights, table


def _merge_columns(op, weights, table):
    size = len(weights)
    i, j = _index(op.recipient, size), _index(op.donor, size)
    if i == j:
        raise _Invalid("invalid_state", "the donor and recipient must be distinct")
    mass = weights[i] + weights[j]
    table = table.astype(np.float64)
    if mass > 0:
        table[:, i] = (weights[i] / mass) * table[:, i] + (weights[j] / mass) * table[:, j]
    weights[i] = mass
    new_weights = np.delete(weights, j)
    table = np.delete(table, j, axis=1)
    return new_weights, table


def _state_edit(template, arrays, op):
    selector = op.selector
    size = template.variables[selector].cardinality
    touching = [f for f in template.factors.values() if selector in f.all_edges().values()]
    prior = _one([f for f in touching if f.type == "categorical" and f.edges.get("out") == selector],
                 "categorical mixing factor")
    emission = _one([f for f in touching if f is not prior], "emission factor")
    gaussian = emission.type == "gaussian_mixture" and emission.edges.get("selector") == selector
    categorical = emission.type == "matrix_cat" and emission.edges.get("in") == selector
    if not (gaussian or categorical) or emission.param_edges:
        raise _Invalid("unsupported_graph", "expected fixed categorical or Gaussian emissions")
    weight_key, carrier, weight_owner = _weights(template, prior)
    size_vars = [selector] + ([carrier] if carrier is not None else [])
    if any(p.variable in size_vars for p in template.ports.values()):
        raise _Invalid("observed_state", "edited state and mixing parameter must be internal")
    weights = arrays[weight_key].astype(np.float64)
    if weights.shape != (size,) or not np.isfinite(weights).all() or not np.any(weights > 0):
        raise _Invalid("invalid_prior", "mixing vector must have positive finite total mass")
    if (weights < 0).any() or (carrier is not None and (weights == 0).any()):
        raise _Invalid("invalid_prior", "invalid mixing mass or improper Dirichlet concentration")

    keys = [emission.param_blobs[r] for r in (("mean", "cov") if gaussian else ("matrix",))]
    values = [arrays[key] for key in keys]
    if len(set([weight_key, *keys])) != 1 + len(keys):
        raise _Invalid("shared_parameter", "edited parameter roles share a blob")
    if not gaussian and (values[0] < 0).any():
        raise _Invalid("invalid_emission", "emission probabilities must be nonnegative")
    axes = [0, 0] if gaussian else [1]
    if isinstance(op, AddRegime):
        if not gaussian:
            raise _Invalid("unsupported_graph", "growth requires a Gaussian mixture")
        new_weights, *values = _grow(op, weights, *values)
    elif isinstance(op, PruneState):
        i = _index(op.state, size)
        new_weights = np.delete(weights, i)
        if not np.any(new_weights > 0):
            raise _Invalid("empty_support", "prune would remove every occupied state")
        values = [np.delete(v, i, axis=a) for v, a in zip(values, axes, strict=True)]
    else:
        if not categorical:
            raise _Invalid("unsupported_graph", "this split or merge expects categorical emissions")
        table = values[0]
        if isinstance(op, SplitState):
            new_weights, table = _split_columns(op, weights, table)
        else:
            new_weights, table = _merge_columns(op, weights, table)
        values = [table]

    if carrier is None and isinstance(op, (PruneState, AddRegime)):
        total = new_weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise _Invalid("numeric_range", "remaining mixing mass cannot be normalized")
        normalized = new_weights / total
        if np.any((normalized == 0) & (new_weights > 0)):
            raise _Invalid("numeric_range", "normalization erased a positive state")
        new_weights = normalized
    updates = {weight_key: _preserve_support(new_weights, arrays[weight_key])}
    for key, value in zip(keys, values, strict=True):
        dtype = np.result_type(arrays[key].dtype, np.float32)
        updates[key] = (np.asarray(value, dtype=dtype) if gaussian
                        else _preserve_support(value, arrays[key]))
    readers = {weight_owner, emission.name}
    actions = []
    for var in size_vars:
        actions.extend((RemoveVariable(var), AddVariable(replace(
            template.variables[var], cardinality=len(new_weights)))))
    return actions, updates, readers


def _prune_input(template, op):
    factor = template.factors[op.factor]
    if factor.type != "gaussian_linear" or factor.param_edges:
        raise _Invalid("unsupported_graph", "expected an affine Gaussian with fixed parameters")
    x = factor.edges["in"]
    neighbors = [f for f in template.factors.values() if f is not factor and x in f.all_edges().values()]
    prior = _one(neighbors, "normalized unary prior on the removed input")
    if prior.type != "gaussian" or prior.edges != {"out": x} or prior.param_edges:
        raise _Invalid("unsupported_graph", "removed input needs a normalized fixed Gaussian prior")
    if any(port.variable == x for port in template.ports.values()):
        raise _Invalid("observed_state", "removing an observed input would change the data")
    remaining = FactorSpec(
        factor.name, "gaussian", {"out": factor.edges["out"]},
        param_blobs={"mean": factor.param_blobs["offset"], "cov": factor.param_blobs["cov"]},
    )
    owners = {factor.name, prior.name}
    old_keys = set(prior.param_blobs.values()) | set(factor.param_blobs.values())
    if any(old_keys.intersection(owner.blobs) for owner in template.learnables.values()):
        raise _Invalid("learnable_parameter", "freeze fitted Gaussian parameters before removing their edge")
    live_keys = set(remaining.param_blobs.values())
    for other in template.factors.values():
        if other.name not in owners:
            live_keys.update(other.param_blobs.values())
    actions = [RemoveFactor(prior.name), RemoveVariable(x), ReplaceFactor(remaining)]
    actions.extend(RemoveBlob(key) for key in sorted(old_keys - live_keys))
    return actions, {}, owners


def propose_operator(source_id, models, operation, *, parent_id=None):
    """Make a replayable graph edit from fitted source arrays, or return a rejection reason."""
    try:
        template, source = models[source_id]
        if template.plates:
            raise _Invalid("unsupported_graph", "expand the plate before editing one of its parameter owners")
        arrays = {name: np.asarray(value) for name, value in source.items()}
        compile_template(template, arrays=arrays)
        if isinstance(operation, PruneGaussianInput):
            actions, updates, readers = _prune_input(template, operation)
        elif isinstance(operation, (AddRegime, SplitState, PruneState, MergeStates)):
            actions, updates, readers = _state_edit(template, arrays, operation)
        else:
            raise _Invalid("unknown_operation", "unknown structure operator")
        for factor in template.factors.values():
            if factor.name not in readers and set(factor.param_blobs.values()) & updates.keys():
                raise _Invalid("shared_parameter", f"parameter also supplies {factor.name!r}")
        for owner in template.learnables.values():
            if set(owner.blobs) & updates.keys():
                raise _Invalid("learnable_parameter", f"edit requires migration of learner {owner.name!r}")
        record = record_proposal(
            source_id=source_id, parent_id=parent_id, proposal_source=type(operation).__name__,
            steps=[edit_step(actions, parameters={k: k for k in updates})],
            models=models, inputs=updates,
        )
        candidate, parameters = replay_proposal(record, models, inputs=updates)
        compile_template(candidate, arrays=parameters)
        return StructuralCandidate(record, updates)
    except _Invalid as error:
        return RejectedEdit(error.code, str(error))
    except (ValueError, KeyError, TypeError, IndexError, FloatingPointError, np.linalg.LinAlgError) as error:
        return RejectedEdit("invalid_model", str(error))
