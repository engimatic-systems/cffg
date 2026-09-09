"""Test-only joint Gaussian, independently implemented in canonical form.

Three unequal roles and a conditioned middle role exercise a signature not
present in the built-ins. Test oracles use covariance conditioning and KL,
not these Schur-complement message/energy implementations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from cffg import gauss
from cffg.core import (
    BlobRef,
    FactorSpec,
    FactorType,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.kernel_bindings import GaussianBinding, GaussianConditioning, ParameterSpec
from cffg.kernels import FactorKernel, HostileInputs

CUSTOM = "joint_field_sensor_offset"
IMPL = "cffgtests.joint_field_sensor_offset"
ROLES = ("field", "sensor", "offset")
PARAM_KEYS = ("joint_precision", "joint_information", "log_scale")
HOSTILE = HostileInputs(
    "Provably N/A: a proper continuous joint Gaussian has full real support.",
    "Vacuous cavities are neutral; invalid required precision yields NaN messages and infinite energy.",
    "Proper-precision guards use the existing dtype-relative predicate; no ridge or absolute floor.",
    "Dimension100 beside small groups; flat arena and exact state-byte checks.",
)


def role_indices(dims):
    starts = np.cumsum((0, *dims))
    return tuple(np.arange(starts[r], starts[r + 1]) for r in range(len(dims)))


def combine(eta, precision, dims):
    return (
        jnp.concatenate([eta[r, :d] for r, d in enumerate(dims)]),
        block_diag(*[precision[r, :d, :d] for r, d in enumerate(dims)]),
    )


def local_messages(eta, precision, joint, information, dims):
    cavity_eta, cavity_precision = combine(eta, precision, dims)
    indices = role_indices(dims)
    outputs = []
    for r, own in enumerate(indices):
        other = np.concatenate([idx for s, idx in enumerate(indices) if s != r])
        cross = joint[np.ix_(own, other)]
        q = joint[np.ix_(other, other)] + cavity_precision[np.ix_(other, other)]
        rhs = information[other] + cavity_eta[other]
        out_e = information[own] - cross @ jnp.linalg.solve(q, rhs)
        out_p = joint[np.ix_(own, own)] - cross @ jnp.linalg.solve(q, cross.T)
        out_p = (out_p + out_p.T) / 2
        proper = gauss.is_proper_precision(joint) & gauss.is_proper_precision(q)
        outputs.append(
            (jnp.where(proper, out_e, jnp.nan), jnp.where(proper, out_p, jnp.nan))
        )
    return tuple(outputs)


def reduced(joint, information, constant, dims, observed_role, y, mask):
    roles = tuple(r for r in range(len(dims)) if r != observed_role)
    indices = role_indices(dims)
    kept, observed = np.concatenate([indices[r] for r in roles]), indices[observed_role]
    joo, juo = joint[np.ix_(observed, observed)], joint[np.ix_(kept, observed)]
    juu, hu, ho = joint[np.ix_(kept, kept)], information[kept], information[observed]
    conditioned = (juu, hu - juo @ y, constant + ho @ y - 0.5 * y @ joo @ y)
    solved = jnp.linalg.solve(joo, ho)
    marginalized = (
        juu - juo @ jnp.linalg.solve(joo, juo.T),
        hu - juo @ solved,
        constant
        + 0.5 * ho @ solved
        + 0.5 * len(observed) * jnp.log(2 * jnp.pi)
        - 0.5 * jnp.linalg.slogdet(joo)[1],
    )
    return tuple(
        jnp.where(mask > 0, c, m) for c, m in zip(conditioned, marginalized)
    ), roles


def message(eta, precision, params, dims, observed_role, observation, mask):
    def one(e, p, joint, information, constant, y, present):
        outputs = list(local_messages(e, p, joint, information, dims))
        if observed_role is not None:
            (jr, hr, _), roles = reduced(
                joint, information, constant, dims, observed_role, y, present
            )
            remaining_dims = tuple(dims[r] for r in roles)
            remaining = local_messages(
                e[np.array(roles)], p[np.array(roles)], jr, hr, remaining_dims
            )
            for role, value in zip(roles, remaining):
                outputs[role] = value
        return tuple(outputs)

    if observed_role is None:
        observation, mask = (
            jnp.zeros((eta.shape[0], 0), eta.dtype),
            jnp.zeros(eta.shape[0], eta.dtype),
        )
    return jax.vmap(one)(
        eta, precision, *(params[key] for key in PARAM_KEYS), observation, mask
    )


def belief(eta, precision, joint, information, constant, dims, observed_role, y, mask):
    original_joint = joint
    if observed_role is not None:
        (joint, information, constant), roles = reduced(
            joint, information, constant, dims, observed_role, y, mask
        )
        eta, precision = eta[np.array(roles)], precision[np.array(roles)]
        dims = tuple(dims[r] for r in roles)
    cavity_eta, cavity_precision = combine(eta, precision, dims)
    total = joint + cavity_precision
    mean, covariance = (
        jnp.linalg.solve(total, information + cavity_eta),
        jnp.linalg.inv(total),
    )
    entropy = 0.5 * (
        len(mean) * (1 + jnp.log(2 * jnp.pi)) - jnp.linalg.slogdet(total)[1]
    )
    energy = (
        -entropy
        - constant
        - information @ mean
        + 0.5 * (mean @ joint @ mean + jnp.trace(joint @ covariance))
    )
    proper = (
        gauss.is_proper_precision(original_joint)
        & gauss.is_proper_precision(total)
        & jnp.isfinite(energy)
    )
    return (
        jnp.where(proper, energy, jnp.inf),
        jnp.where(proper, mean, jnp.nan),
        jnp.where(proper, covariance, jnp.nan),
        proper,
    )


def statistics(eta, precision, params, dims, observed_role, observation, mask):
    if observed_role is None:
        observation, mask = (
            jnp.zeros((eta.shape[0], 0), eta.dtype),
            jnp.zeros(eta.shape[0], eta.dtype),
        )
    values = jax.vmap(
        lambda e, p, j, h, c, y, m: belief(e, p, j, h, c, dims, observed_role, y, m)
    )(
        eta,
        precision,
        *(params[key] for key in PARAM_KEYS),
        observation,
        mask,
    )
    return {
        "energy": values[0],
        "mean": values[1],
        "covariance": values[2],
        "proper": values[3],
    }


def energy(*args):
    return statistics(*args)["energy"]


def pack(template, members, resolved):
    rows = []
    for name in members:
        factor = template.factors[name]
        mean, root = (
            resolved[factor.param_blobs[key]] for key in ("center", "precision_root")
        )
        if not np.isfinite(mean).all() or not np.isfinite(root).all():
            raise ValueError("joint Gaussian parameters must be finite")
        joint = root @ root.T
        try:
            np.linalg.cholesky(joint)
        except np.linalg.LinAlgError as err:
            raise ValueError(
                "joint Gaussian precision must be positive definite"
            ) from err
        information = joint @ mean
        constant = (
            0.5 * np.linalg.slogdet(joint)[1]
            - 0.5 * len(mean) * np.log(2 * np.pi)
            - 0.5 * mean @ information
        )
        rows.append((joint, information, constant))
    return {key: np.stack([row[i] for row in rows]) for i, key in enumerate(PARAM_KEYS)}


def validate_sensor(value, dimension):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (dimension,) or not np.isfinite(arr).all():
        raise ValueError(f"sensor requires a finite vector of shape {(dimension,)}")
    return arr


def declaration():
    return FactorKernel(
        FactorType(CUSTOM, ROLES, ("center", "precision_root"), family="gaussian"),
        IMPL,
        "1.3.0",
        hostile=HOSTILE,
        binding=GaussianBinding(
            ROLES,
            {
                "center": ParameterSpec(lambda s: (sum(s.values()),)),
                "precision_root": ParameterSpec(
                    lambda s: (sum(s.values()), sum(s.values()))
                ),
            },
            PARAM_KEYS,
            pack,
            message,
            energy,
            GaussianConditioning("sensor", validate_sensor),
            statistics,
            statistics,
        ),
    )


def model(dims=(3, 2, 1), observed=False):
    n = sum(dims)
    rng = np.random.default_rng(31)
    root = 1.2 * np.eye(n) + np.triu(rng.normal(size=(n, n)) * 0.04 / np.sqrt(n))
    arrays = {"center": np.linspace(-0.4, 0.6, n), "precision_root": root}
    t = Template(
        name="registered_joint_gaussian",
        variables={role: VariableSpec(role, dim=d) for role, d in zip(ROLES, dims)},
        factors={
            "joint": FactorSpec(
                "joint",
                CUSTOM,
                dict(zip(ROLES, ROLES)),
                param_blobs={key: key for key in arrays},
            )
        },
        ports={
            role: PortSpec(
                role,
                role,
                PortRole.DATA if observed and role == "sensor" else PortRole.GLUE,
            )
            for role in ROLES
        },
        blobs={key: BlobRef(key, f"{key}.npy") for key in arrays},
    )
    return t, arrays


def covariance_oracle(arrays, dims, observation=None, conditioned=False):
    mean = arrays["center"]
    covariance = np.linalg.inv(arrays["precision_root"] @ arrays["precision_root"].T)
    if not conditioned:
        return mean, covariance, 0.0
    indices = role_indices(dims)
    observed, kept = indices[1], np.concatenate((indices[0], indices[2]))
    if observation is None:
        return mean[kept], covariance[np.ix_(kept, kept)], 0.0
    voo, vuo = (
        covariance[np.ix_(observed, observed)],
        covariance[np.ix_(kept, observed)],
    )
    delta = np.asarray(observation) - mean[observed]
    posterior_mean = mean[kept] + vuo @ np.linalg.solve(voo, delta)
    posterior_cov = covariance[np.ix_(kept, kept)] - vuo @ np.linalg.solve(voo, vuo.T)
    evidence = 0.5 * (
        len(observed) * np.log(2 * np.pi)
        + np.linalg.slogdet(voo)[1]
        + delta @ np.linalg.solve(voo, delta)
    )
    return posterior_mean, posterior_cov, evidence


def chain_model(n=16):
    _, arrays = model((3, 2, 3))
    t = Template(
        name="registered_gaussian_chain",
        blobs={key: BlobRef(key, f"{key}.npy") for key in arrays},
    )
    degrees = {}
    observations = {}
    for i in range(n):
        edges = {"field": f"z{i}", "sensor": f"sensor{i}", "offset": f"z{i + 1}"}
        name = f"f{i:03}"
        t.factors[name] = FactorSpec(
            name, CUSTOM, edges, param_blobs={key: key for key in arrays}
        )
        for role, dim in zip(ROLES, (3, 2, 3)):
            var = edges[role]
            t.variables[var] = VariableSpec(var, dim=dim)
            degrees[var] = degrees.get(var, 0) + 1
        observations[edges["sensor"]] = np.array([0.1 * i, -0.2]) if i % 3 else None
    for var, degree in degrees.items():
        if degree == 1:
            t.ports[var] = PortSpec(
                var, var, PortRole.DATA if var in observations else PortRole.GLUE
            )
    return t, arrays, observations
