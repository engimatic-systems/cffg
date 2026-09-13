"""Family-owned concentration, probability, covariance and mixed-factor packing."""

import numpy as np

from cffg.compiled import NEG
from cffg.compiler_errors import CompileError
from cffg.core.ir import Template
from cffg.registered.packing import pack_gaussian_parameters


def requirement_arrays(requirement, resolved):
    return {source.blob: resolved[source.blob] for source in requirement.inputs}


def declared_parameters(requirement, packed):
    values = {f"{requirement.owner}:{key}": value for key, value in packed.items()}
    expected = {destination.name for destination in requirement.destinations}
    if set(values) != expected:
        raise CompileError(
            f"plan {requirement.owner!r}: packed keys {sorted(values)} "
            f"do not match declared destinations {sorted(expected)}"
        )
    return values


def parameter_priors(template, drafts, resolved):
    """Keep failed concentration claims and the combined orphan diagnostic."""
    concentrations, goals, problems, consumed = {}, {}, [], set()
    for claim in drafts.parameters:
        shape, prior_name = claim.layout.shape, claim.prior
        if claim.problem:
            problems.append(claim.problem)
            continue
        prior = template.factors[prior_name]
        alpha0 = resolved[prior.param_blobs["counts"]]
        if alpha0.shape != shape:
            problems.append(
                f"Dirichlet prior {prior_name!r}: concentration shape {alpha0.shape} "
                f"!= likelihood table shape {shape}"
            )
            continue
        if not np.all(np.isfinite(alpha0)) or not np.all(alpha0 > 0):
            problems.append(
                f"Dirichlet prior {prior_name!r}: concentrations must be finite and > 0"
            )
            continue
        likelihood = claim.layout.likelihood
        concentrations[likelihood] = alpha0
        consumed.update((likelihood, prior_name))
        if claim.is_goal:
            goals[likelihood] = alpha0
    orphan_priors = [name for name in drafts.prior_names if name not in consumed]
    if orphan_priors:
        problems.append(
            "dirichlet factors are parameter priors and must face a supported "
            f"mean-field likelihood; unpaired: {orphan_priors}"
        )
    return concentrations, goals, problems


def pack_dirichlet(draft, concentrations):
    return {
        draft.arrays.destinations[0].name: np.stack(
            [concentrations[member] for member in draft.plan.members]
        )
    }


def pack_gfe(draft, goals, goal_priors, resolved):
    group = draft.plan
    ko, ki = group.cards
    sources = {
        (source.factor, source.role): resolved[source.blob]
        for source in draft.arrays.inputs
    }
    la, lc = [], []
    for obs_name, goal_name in zip(group.members, goals, strict=True):
        a = sources[obs_name, "matrix"]
        _expect(obs_name, a, (ko, ki))
        if group.learned:
            if not np.all(a > 0):
                raise CompileError(f"factor {obs_name!r}: Dirichlet counts must be > 0")
            la.append(a)
        else:
            _stochastic(obs_name, a)
            la.append(_log_of(a))
        if group.goal_learned:
            c_alpha0 = goal_priors[goal_name]
            _expect(goal_name, c_alpha0, (ko,))
            lc.append(c_alpha0)
        else:
            c = sources[goal_name, "statistic"]
            _expect(goal_name, c, (ko,))
            _stochastic(goal_name, c[:, None])
            lc.append(_log_of(c))
    a_key, c_key = (destination.name for destination in draft.arrays.destinations)
    return {a_key: np.stack(la), c_key: np.stack(lc)}


def pack_transition_gfe(draft, resolved):
    group = draft.plan
    ko, ki, ku = group.cards
    sources = {source.factor: resolved[source.blob] for source in draft.arrays.inputs}
    stacked = []
    for member in group.members:
        arr = sources[member]
        _expect(member, arr, (ku, ko, ki))
        if group.learned:
            if not np.all(arr > 0):
                raise CompileError(f"factor {member!r}: Dirichlet counts must be > 0")
            stacked.append(arr)
        else:
            for u in range(ku):
                _stochastic(f"{member}[u={u}]", arr[u])
            stacked.append(_log_of(arr))
    return {draft.arrays.destinations[0].name: np.stack(stacked)}


def pack_hybrid_gfe(template, draft, resolved):
    group = draft.plan
    sources = requirement_arrays(draft.arrays, resolved)
    packed = _stack_mix_params(
        template,
        group.type,
        group.dims,
        group.k_sel,
        group.members,
        sources,
    )
    goals = pack_gaussian_parameters(
        draft.arrays.auxiliary_kernels["goal"],
        template,
        group.goal_members,
        sources,
    )
    packed.update(goal_m=goals["m"], goal_W=goals["W"])
    return declared_parameters(draft.arrays, packed)


def pack_mixture(template, draft, resolved):
    group = draft.plan
    packed = _stack_mix_params(
        template,
        group.type,
        group.dims,
        group.k_sel,
        group.members,
        requirement_arrays(draft.arrays, resolved),
    )
    return declared_parameters(draft.arrays, packed)


def pack_recurrence(template, draft, resolved):
    group = draft.plan
    packed = _stack_recurrence_params(
        template,
        group.cards[0],
        group.dim,
        group.members,
        requirement_arrays(draft.arrays, resolved),
    )
    return declared_parameters(draft.arrays, packed)


def pack_precision(template, draft, resolved):
    packed = {
        "m": _stack_prec_means(
            template,
            draft.plan.members,
            requirement_arrays(draft.arrays, resolved),
        )
    }
    return declared_parameters(draft.arrays, packed)


def _stack_mix_params(
    t: Template,
    ftype: str,
    dims: tuple[int, ...],
    k_sel: int,
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-member hybrid parameters, stacked [M, K, ...] over the K regimes
    (cff-pksr). Blobs are per-regime stacks: mean [K, d]/cov [K, d, d] for
    gaussian_mixture, matrix [K, do, di]/offset [K, do]/cov [K, do, do] for
    gaussian_mixture_linear. Each cov[k] is validated symmetric + SPD (the
    Cholesky is the check) and stored as a precision W = V⁻¹, the form every
    mixg rule consumes."""
    out: dict[str, list[np.ndarray]] = {}
    for m in members:
        f = t.factors[m]
        blobs = {role: resolved[f.param_blobs[role]] for role in sorted(f.param_blobs)}
        cov = blobs["cov"]
        if ftype == "gaussian_mixture":
            (d,) = dims
            _expect(m, blobs["mean"], (k_sel, d))
            _expect(m, cov, (k_sel, d, d))
            out.setdefault("m", []).append(blobs["mean"])
        else:  # gaussian_mixture_linear
            do, di = dims
            _expect(m, blobs["matrix"], (k_sel, do, di))
            _expect(m, blobs["offset"], (k_sel, do))
            _expect(m, cov, (k_sel, do, do))
            out.setdefault("A", []).append(blobs["matrix"])
            out.setdefault("b", []).append(blobs["offset"])
        w_regimes = []
        for k in range(k_sel):
            ck = cov[k]
            if not np.allclose(ck, ck.T, atol=1e-12):
                raise CompileError(f"factor {m!r}: covariance not symmetric")
            try:
                np.linalg.cholesky(ck)
            except np.linalg.LinAlgError:
                raise CompileError(
                    f"factor {m!r}: covariance not positive definite"
                ) from None
            w_regimes.append(np.linalg.inv(ck))
        out.setdefault("W", []).append(np.stack(w_regimes))
    return {k: np.stack(v) for k, v in out.items()}


def _stack_prec_means(
    t: Template, members: list[str], resolved: dict[str, np.ndarray]
) -> np.ndarray:
    """Stack per-member gaussian_prec means [M, 1]. The mean is the scalar m in
    N(x; m, τ⁻¹); the out edge is dim-1, so any size-1 blob is accepted and
    reshaped to a length-1 vector (the form gauss.prec_to_x consumes)."""
    means = []
    for m in members:
        arr = resolved[t.factors[m].param_blobs["mean"]]
        if arr.size != 1:
            raise CompileError(
                f"factor {m!r}: parameter 'mean' must be a scalar, got shape {arr.shape}"
            )
        means.append(arr.reshape(1))
    return np.stack(means)


def _stack_recurrence_params(
    t: Template,
    card: int,
    drive_dim: int,
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Stack source-specific stick weights and offsets for one plan."""
    weights = []
    offsets = []
    for member in members:
        factor = t.factors[member]
        member_weights = resolved[factor.param_blobs["weights"]]
        member_offsets = resolved[factor.param_blobs["offsets"]]
        _expect(member, member_weights, (card, card - 1, drive_dim))
        _expect(member, member_offsets, (card, card - 1))
        if not (
            np.all(np.isfinite(member_weights)) and np.all(np.isfinite(member_offsets))
        ):
            raise CompileError(
                f"factor {member!r}: recurrence weights and offsets must be finite"
            )
        weights.append(member_weights)
        offsets.append(member_offsets)
    return {"weights": np.stack(weights), "offsets": np.stack(offsets)}


def _log_of(arr: np.ndarray) -> np.ndarray:
    """Elementwise log with exact zeros mapped to NEG, the engine's finite
    log-zero (semantically zero: exp(NEG) == 0, and NEG - NEG == 0).

    The floor under the log branch is the dtype's smallest normal, so log(0)
    is never evaluated even for the entries the where-mask discards — a
    literal 1e-300 floor underflows to exactly 0 under float32 and both warns
    and returns -inf (cff-ey4p). Positive normal values are unchanged in any
    float dtype; positive subnormals clamp to log(tiny), which is finite.
    """
    dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float64
    return np.where(arr > 0.0, np.log(np.maximum(arr, np.finfo(dtype).tiny)), NEG)


def _expect(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise CompileError(
            f"factor {name!r}: parameter shape {arr.shape} != expected {shape}"
        )


def _stochastic(name: str, arr: np.ndarray) -> None:
    sums = arr.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise CompileError(
            f"factor {name!r}: conditional not column-stochastic (col sums {sums}); "
            "true-FE semantics require normalized factors"
        )
