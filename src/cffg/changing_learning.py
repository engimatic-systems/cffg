"""Bind the real-event clock and immutable support of declared changing transition tables."""

import numpy as np


def validate_learning_state(plan, family, params):
    if plan.change_hazard is None:
        family.validate_prior(plan.name, params, plan.shapes)
        return
    if set(params) != {"alpha"}:
        raise ValueError(
            f"learnable {plan.name!r}: changing transitions need exactly alpha"
        )
    alpha = np.asarray(params["alpha"])
    if alpha.dtype not in (np.float32, np.float64):
        raise ValueError(
            f"learnable {plan.name!r}: changing transition counts need float32 or float64"
        )
    allowed = np.asarray(plan.transition_support)
    if (
        alpha.shape != allowed.shape
        or not np.isfinite(alpha).all()
        or not np.all(np.where(allowed, alpha > 0, alpha == 0))
    ):
        raise ValueError(
            f"learnable {plan.name!r}: concentration shape or transition support mismatch"
        )
    check_count_representation(plan.name, alpha, allowed, family.axis(plan.ftype))


def check_count_representation(name, values, allowed, axis):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        total = values.sum(axis=axis, keepdims=True)
        means = values / total
    # XLA can flush a positive subnormal even when NumPy can store its bits.
    tiny = np.finfo(values.dtype).tiny
    if (
        not np.isfinite(total).all()
        or not np.all(np.where(allowed, values >= tiny, values == 0))
        or not np.all(np.where(allowed, means >= tiny, means == 0))
    ):
        raise ValueError(
            f"learnable {name!r}: transition counts unrepresentable in {values.dtype}"
        )


def initial_count_errors(arena):
    """Find defaults that would lose support or a finite total when installed."""
    errors = []
    for plan in arena.learn_plans:
        if plan.change_hazard is None:
            continue
        for target in plan.targets:
            if target.form != "counts":
                continue
            value = np.asarray(arena.log_params[target.key][target.index])
            for dtype in {value.dtype, np.dtype(np.float32)}:
                try:
                    with np.errstate(over="ignore", under="ignore"):
                        check_count_representation(
                            plan.name,
                            value.astype(dtype),
                            np.asarray(plan.transition_support),
                            1 if plan.ftype == "transition_mixture" else 0,
                        )
                except ValueError as error:
                    errors.append((target.key, dtype.str, str(error)))
    return tuple(errors)


def projected(plan, initial, committed, counts):
    """Pool one physical event before applying its hazard to each column."""
    from cffg.transition_change import host_columns

    axis = 1 if plan.ftype == "transition_mixture" else 0
    arrays = (initial["alpha"], committed["alpha"], counts, plan.transition_support)
    prior, carry, evidence, allowed = (
        np.moveaxis(np.asarray(x), axis, -1) for x in arrays
    )
    result = host_columns(prior, carry, evidence, allowed, plan.change_hazard)
    if not np.all(result.valid):
        raise ValueError(
            f"learnable {plan.name!r}: invalid transition update on its declared support"
        )
    return {"alpha": np.moveaxis(result.concentration, -1, axis)}


def predict(plan, initial, committed):
    if plan.change_hazard is None or plan.change_hazard == 0:
        return committed
    return projected(plan, initial, committed, np.zeros_like(committed["alpha"]))


def expected_log(plan, family, params):
    if plan.change_hazard is None or np.all(np.asarray(plan.transition_support)):
        return family.rollover(params, plan.ftype)
    from scipy.special import digamma
    from cffg.compiled import NEG

    axis = family.axis(plan.ftype)
    allowed = np.asarray(plan.transition_support)
    alpha = params["alpha"]
    values = digamma(np.where(allowed, alpha, 1)) - digamma(
        alpha.sum(axis=axis, keepdims=True)
    )
    return {plan.targets[0].role: np.where(allowed, values, NEG)}
