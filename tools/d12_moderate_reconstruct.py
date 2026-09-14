"""Rebuild D12 report relationships from binary32 payloads, without a solver.

Keep the operation boundaries of the native local kernels, especially rounded
physical covariance versus retained high/low covariance differences. Binary64
is used only to recover exact products of two binary32 words for compensated
affine accumulation; this is not promoted inference or universal exactness.
Cross-implementation comparisons use the frozen local arithmetic allowance.
"""

import numpy as np

from d12_compile_fixture import fixture_arrays


def fields(arrays, prefix):
    return {
        key.removeprefix(prefix + "."): value
        for key, value in arrays.items()
        if key.startswith(prefix + ".")
    }


def checked_sum(a, b, *, subtract=False):
    """Use the native affine kernel's normal-word/cancellation support guard."""
    result = a - b if subtract else a + b
    left, right, bits = (
        np.asarray(value, np.float32).view(np.uint32) for value in (a, b, result)
    )
    sign = np.uint32(1 << 31)
    normal = np.uint32(1 << 23)
    lm, rm, magnitude = (value & ~sign for value in (left, right, bits))
    unsupported = ((lm != 0) & (lm < normal)) | ((rm != 0) & (rm < normal))
    different = ((left ^ right) & sign) != 0
    cancellation = ~different if subtract else different
    lost = cancellation & (lm != rm) & (magnitude < normal)
    return np.where(unsupported | lost, np.float32(np.nan), result)


def two_sum(a, b):
    total = checked_sum(a, b)
    virtual = checked_sum(total, a, subtract=True)
    error = checked_sum(
        checked_sum(a, checked_sum(total, virtual, subtract=True), subtract=True),
        checked_sum(b, virtual, subtract=True),
    )
    return total, error


def affine(weights, values, *offsets):
    """Replay the short same-dtype compensated dot, retaining product errors."""
    weights, values = np.broadcast_arrays(weights, values)
    exact = weights.astype(np.float64) * values.astype(np.float64)
    product = exact.astype(np.float32)
    remainder = exact - product.astype(np.float64)
    error = remainder.astype(np.float32)
    tiny = np.finfo(np.float32).tiny
    supported = (
        np.isfinite(exact)
        & np.isfinite(product)
        & np.isfinite(error)
        & ((exact == 0) | (abs(product) >= tiny))
        & ((remainder == 0) | (abs(error) >= tiny))
    )
    product, error = (
        np.where(supported, word, np.float32(np.nan)) for word in (product, error)
    )
    total = np.zeros(product.shape[:-1], np.float32)
    correction = np.zeros_like(total)
    for index in range(product.shape[-1]):
        total, remainder = two_sum(total, product[..., index])
        correction = checked_sum(correction, checked_sum(remainder, error[..., index]))
    for offset in offsets:
        total, remainder = two_sum(total, offset)
        correction = checked_sum(correction, remainder)
    result = checked_sum(total, correction)
    return np.where(np.isfinite(result), result, np.float32(np.nan))


def combine(base, increment):
    ones = np.ones_like(base)[..., None]
    high = affine(ones, increment[..., None], base)
    low = affine(ones, increment[..., None], base, -high)
    return high, low


def symmetric(value):
    row, col = np.indices(value.shape[-2:])
    return np.where(row <= col, value, value.swapaxes(-1, -2))


def project(region, side):
    """Conditional mixture, with the native reference-covariance rounding steps."""
    d = region["displacement"].shape[-1] // 2
    block = slice(side * d, (side + 1) * d)
    axis = 2 - side
    p = region["probability"]
    centers = region["center"][..., block]
    center = np.take(centers, 0, axis=axis)
    relative = region["displacement"][..., block] + (
        centers - np.expand_dims(center, axis)
    )
    probability = p.sum(axis=axis, dtype=np.float32)
    denominator = np.where(probability > 0, probability, np.float32(1))
    weights = np.moveaxis(p, axis, -1)
    mean = (
        affine(weights[..., None, :], np.moveaxis(relative, axis, -1))
        / denominator[..., None]
    )
    centered = relative - np.expand_dims(mean, axis)
    covariance = region["covariance"][..., block, block]
    low = region["covariance_correction"][..., block, block]
    indices = np.expand_dims(np.argmax(p > 0, axis=axis), axis)[..., None, None]
    base = np.take_along_axis(covariance, indices, axis=axis).squeeze(axis)
    difference = covariance - np.expand_dims(base, axis)
    variation = affine(
        centered[..., :, None, None], centered[..., None, :, None], difference, low
    )
    increment = affine(weights[..., None, None, :], np.moveaxis(variation, axis, -1))
    increment = increment / denominator[..., None, None]
    high, low = (symmetric(word) for word in combine(base, increment))
    active = probability > 0
    return dict(
        probability=probability,
        displacement=np.where(active[..., None], mean, np.float32(0)),
        center=center,
        covariance=np.where(active[..., None, None], high, np.float32(0)),
        covariance_correction=np.where(active[..., None, None], low, np.float32(0)),
    )


def physical_covariance(belief):
    # This once-rounded view is used by scales and by the native auxiliary rule.
    return belief["covariance"] + belief["covariance_correction"]


def pg_mean(xi):
    small = xi <= np.sqrt(np.finfo(np.float32).eps)
    safe = np.sqrt(np.where(small, np.float32(1), xi))
    series_xi = np.where(small, xi, np.float32(0))
    series = np.float32(0.25) - series_xi / 48 + series_xi * series_xi / 480
    return np.where(small, series, np.tanh(safe / 2) / (2 * safe))


def suffix(value):
    return np.cumsum(value[:, :, ::-1], axis=2, dtype=np.float32)[:, :, ::-1][:, :, :-1]


def risk_moments(p, mean, covariance):
    """Reverse conditional mixture recurrence, preserving tiny source exposure."""
    exposure = np.zeros(p.shape[:2], np.float32)
    old_mean = np.zeros(mean.shape[:2] + mean.shape[3:], np.float32)
    old_covariance = np.zeros(covariance.shape[:2] + covariance.shape[3:], np.float32)
    means, covariances = [], []
    for j in reversed(range(p.shape[2])):
        mass, m, cov = p[:, :, j], mean[:, :, j], covariance[:, :, j]
        total = exposure + mass
        denominator = np.where(total > 0, total, np.float32(1))
        old_fraction, new_fraction = exposure / denominator, mass / denominator
        delta = m - old_mean
        mixed_mean = old_mean + new_fraction[..., None] * delta
        mixed_mean = np.where((exposure > 0)[..., None], mixed_mean, m)
        mixed_covariance = old_fraction[..., None, None] * old_covariance
        mixed_covariance += new_fraction[..., None, None] * cov
        mixed_covariance += (
            (old_fraction * new_fraction)[..., None, None]
            * delta[..., :, None]
            * delta[..., None, :]
        )
        exposure, old_mean, old_covariance = total, mixed_mean, mixed_covariance
        means.append(mixed_mean)
        covariances.append(mixed_covariance)
    return (
        np.stack(means[::-1], axis=2)[:, :, :-1],
        np.stack(covariances[::-1], axis=2)[:, :, :-1],
    )


def physical_learning(linear, precision, center):
    physical_linear = linear.copy()
    physical_linear[..., :-1] += center[:, :, None] * linear[..., -1:]
    d = center.shape[-1]
    transform = np.broadcast_to(
        np.eye(d + 1, dtype=np.float32), center.shape[:-1] + (d + 1, d + 1)
    ).copy()
    transform[..., :d, d] = center
    physical_precision = np.einsum(
        "tiac,tilcd,tibd->tilab", transform, precision, transform, optimize=True
    )
    return dict(physical_linear=physical_linear, physical_precision=physical_precision)


def learning(region, parameters, xi):
    """Source-conditioned D12.8 and D11.1 values from the saved pair belief."""
    weights, offsets = (
        np.asarray(parameters[name], np.float32) for name in ("weights", "offsets")
    )
    d = weights.shape[-1]
    p = region["probability"]
    centers = region["center"][..., :d]
    center = centers[:, :, 0]
    mean = region["displacement"][..., :d] + (centers - center[:, :, None])
    covariance = physical_covariance(region)[..., :d, :d]
    second = covariance + mean[..., :, None] * mean[..., None, :]
    augmented = np.concatenate((mean, np.ones(p.shape + (1,), np.float32)), axis=-1)
    moments = np.zeros(p.shape + (d + 1, d + 1), np.float32)
    moments[..., :d, :d] = second
    moments[..., :d, d] = mean
    moments[..., d, :d] = mean
    moments[..., d, d] = 1
    first = p[..., None] * augmented
    exposure = suffix(p)
    accumulated = suffix(p[..., None, None] * moments)
    conditional_mean, conditional_covariance = risk_moments(p, mean, covariance)
    origin = np.broadcast_to(center[:, :, None], conditional_mean.shape)
    logit_mean = affine(
        np.concatenate((weights, weights), axis=-1),
        np.concatenate((origin, conditional_mean), axis=-1),
        offsets,
    )
    variance = np.einsum(
        "...a,...ab,...b->...", weights, conditional_covariance, weights, optimize=True
    )
    target = np.where(
        exposure > 0, np.maximum(variance, 0) + logit_mean**2, np.float32(0)
    )
    linear = first[:, :, :-1] - suffix(first) / 2
    precision = pg_mean(xi)[..., None, None] * accumulated
    return dict(
        target=target,
        residual=xi - target,
        exposure=exposure,
        linear=linear,
        precision=precision,
        center=center,
        **physical_learning(linear, precision, center),
    )


def packed_messages(banks):
    d = banks[0]["linear"].shape[-1]
    row, col = np.triu_indices(d)
    constants = [(b["constant"][:, 1:] - b["constant"][:, :1]).ravel() for b in banks]
    return np.concatenate(
        (
            *constants,
            *(b["linear"].ravel() for b in banks),
            *(b["precision"][..., row, col].ravel() for b in banks),
        )
    )


def scaled_blocks(arrays, case, parameters=None):
    """Reconstruct each residual and scale; never read saved residual/scales."""
    if parameters is None:
        parameters, _ = fixture_arrays(case["T"])
    region = fields(arrays, "state.evaluation.local.regions")
    separators = fields(arrays, "state.evaluation.local.separators")
    aux = learning(region, parameters, arrays["xi"])
    current = packed_messages(
        [fields(arrays, f"messages.{side}") for side in ("left", "right")]
    )
    target = packed_messages(
        [fields(arrays, f"state.evaluation.local.targets.{side}") for side in (0, 1)]
    )
    scale = lambda a, b: np.maximum(np.float32(1), np.maximum(abs(a), abs(b)))
    mass, mean, covariance, mean_scale, covariance_scale = [], [], [], [], []
    row, col = np.triu_indices(case["d"])
    for side in (0, 1):
        projection = project(region, side)
        incident = {
            name: value[side : side + case["T"] - 1]
            for name, value in separators.items()
        }
        mass.append((projection["probability"] - incident["probability"]).ravel())
        pm = projection["displacement"] + (projection["center"] - incident["center"])
        mean.append((pm - incident["displacement"]).ravel())
        mean_scale.append(scale(pm, incident["displacement"]).ravel())
        difference = affine(
            np.ones_like(projection["covariance"])[..., None],
            projection["covariance"][..., None],
            -incident["covariance"],
            projection["covariance_correction"],
            -incident["covariance_correction"],
        )
        covariance.append(difference[..., row, col].ravel())
        covariance_scale.append(
            scale(physical_covariance(projection), physical_covariance(incident))[
                ..., row, col
            ].ravel()
        )
    blocks = [
        target - current,
        np.concatenate(mass),
        np.concatenate(mean),
        np.concatenate(covariance),
        aux["residual"].ravel(),
    ]
    scales = [
        scale(target, current),
        np.ones_like(blocks[1]),
        np.concatenate(mean_scale),
        np.concatenate(covariance_scale),
        (1 + abs(arrays["xi"])).ravel(),
    ]
    f, dual = (
        arrays[f"state.evaluation.local.{name}"]
        for name in ("free_energy", "dual_energy")
    )
    certificates = [
        np.max(abs(block / denominator), initial=0)
        for block, denominator in zip(blocks, scales, strict=True)
    ]
    certificates.append(abs(f - dual) / (1 + abs(f)))
    return (
        np.concatenate(blocks),
        np.concatenate(scales),
        np.asarray(certificates, np.float32),
        aux,
    )
