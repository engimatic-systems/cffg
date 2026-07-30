"""Executable factor registration (cff-ri8m): the FactorKernel contract.

`core.ir.register_factor_type` records a *structural* signature. Until this
module, the executable half — parameter packing, message updates, node
energies, observation conditioning — lived as hard-coded type-string branches
in compiler.py and engine.py, so every new factor meant central surgery (the
adversarial finding behind cff-ri8m). A `FactorKernel` declares that
executable half in one typed record; the compiler, VM, ledger, and
serialization dispatch from the registry instead of branching.

v1 binding scope: kernels attach to the scalar-EF arena (natural-parameter
messages, `ef_eta [n_slots, n_params]`). That is the representative extension
surface: a new conjugate observation factor registers from user code with
zero edits to cffg compiler/engine/validation source and round-trips through
template JSON and .cffgz via its namespaced (impl_id, version) — see
tests/test_kernels.py. The discrete, Gaussian, and hybrid families still run
on their central paths and migrate incrementally (cff-209l); `equality`
stays structural infrastructure (family-polymorphic sum-minus-self), not a
kernel.

Registration is refused without a `HostileInputs` declaration — the
numerical-robustness contract's planning-time checklist made machine-checked
— and without a namespaced impl id. Serialization records (impl_id, version)
for non-`cffg.` kernels so a template that needs an external kernel fails
loudly when it is missing (core.serialize).

Compatibility policy (pre-1.0, per cff-ri8m AC6): `register_kernel` and the
`FactorKernel` field set are **internal-unstable** — they may change in any
minor release while built-ins migrate onto the contract. The *serialized*
surface is stable earlier: factor `type` names, the `kernels` manifest keys
(`impl`, `version`), and the loud-failure guarantee for missing
implementations hold from the first release that writes them. External
kernels must namespace impl ids with their own prefix (`cffg.` is reserved);
bump MAJOR on any change to a kernel's message/energy semantics or parameter
schema — templates check impl identity and MAJOR compatibility at load.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import numpy as np

from cffg import ef
from cffg.core.ir import (
    EF_EDGE_FAMILIES,
    FACTOR_TYPES,
    ConstraintKind,
    FactorType,
    Template,
    register_factor_type,
)


@dataclass(frozen=True)
class HostileInputs:
    """Planning-time answers to the hostile-inputs checklist (AGENTS.md).

    One short statement of behavior per probe — or "provably N/A: <why>".
    Registration refuses blank answers: a rule whose failure modes were never
    stated cannot be reviewed."""

    sparse_support: str  # exact zeros / empty support
    improper_messages: str  # vacuous or improper incoming messages
    float32: str  # consumer-default dtype (the suite forces x64)
    scale: str  # 10-100x the fixture's domain size


@dataclass(frozen=True)
class FactorKernel:
    """One factor type's executable contract, declared in one place.

    The jnp callables run inside the jitted step/BFE with the group static —
    they must be trace-safe and dtype-polymorphic (derive dtype from `inc`,
    never from literals). `message`/`energy`/`readout` all take
    `(inc, params, obs, mask)`: `inc = ef_eta[gather]` is [M, R, P] incoming
    messages on `message_roles`, `params` maps each of `param_keys` to its
    stacked [M, ...] array, and `obs`/`mask` are [M] observation rows (None
    when `observed_out` is None)."""

    factor_type: FactorType  # structural roles/domains (core registry record)
    impl_id: str  # namespaced identity, e.g. "cffg.ef.gamma" / "mypkg.myfactor"
    version: str  # "MAJOR.MINOR.PATCH"; serialization records + checks MAJOR
    arena: str  # message representation this kernel binds to ("ef" in v1)
    family: str  # EF family of the message edges (key into ef.FAMILIES)
    message_roles: tuple[str, ...]  # structural roles that carry messages
    edge_families: Mapping[str, str]  # role -> required VariableSpec.ef tag
    message: Callable  # (inc, params, obs, mask) -> outgoing [M, R, P]
    energy: Callable  # (inc, params, obs, mask) -> node energies [M]
    hostile: HostileInputs
    param_keys: tuple[str, ...] = ()  # params-dict keys the runtime passes through
    pack_params: Callable | None = None  # (template, members, resolved) -> {key: [M, ...]}
    observed_out: str | None = None  # obs-family tag when "out" is a conditioned DATA port
    validate_observation: Callable | None = None  # raw value -> float, raises ValueError
    constraint_modes: frozenset[str] = frozenset()  # ConstraintKind values this node accepts
    readout: Callable | None = None  # optional (inc, params, obs, mask) -> {name: [M, ...]}


KERNELS: dict[str, FactorKernel] = {}
_KERNEL_REGISTERED_TYPES: set[str] = set()  # FactorTypes we added (for unregister)

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_NODE_MODES = frozenset(k.value for k in ConstraintKind)


def register_kernel(k: FactorKernel) -> FactorKernel:
    """Validate and register an executable kernel (and its FactorType).

    Loud by design: every structural or declarative inconsistency is a
    registration-time ValueError, never a compile- or run-time surprise."""
    name = k.factor_type.name
    problems: list[str] = []
    if name in KERNELS:
        raise ValueError(f"kernel for factor type {name!r} already registered")
    if k.arena != "ef":
        problems.append(
            f"arena {k.arena!r} has no kernel binding yet — discrete/Gaussian/"
            "hybrid dispatch migrates under cff-209l"
        )
    if "." not in k.impl_id or any(c.isspace() for c in k.impl_id) or not k.impl_id:
        problems.append(f"impl_id {k.impl_id!r} must be namespaced ('<namespace>.<name>')")
    if not _VERSION_RE.match(k.version):
        problems.append(f"version {k.version!r} is not MAJOR.MINOR.PATCH")
    if k.family not in ef.FAMILIES:
        problems.append(f"family {k.family!r} not in ef.FAMILIES {sorted(ef.FAMILIES)}")
    for probe in ("sparse_support", "improper_messages", "float32", "scale"):
        if not str(getattr(k.hostile, probe, "")).strip():
            problems.append(
                f"hostile-input declaration {probe!r} is blank — the numerical "
                "robustness contract requires an answer (or 'provably N/A: why')"
            )
    if not set(k.message_roles) <= set(k.factor_type.edge_roles):
        problems.append(
            f"message_roles {k.message_roles} not a subset of edge roles "
            f"{k.factor_type.edge_roles}"
        )
    if not set(k.edge_families) <= set(k.factor_type.edge_roles):
        problems.append(
            f"edge_families keys {sorted(k.edge_families)} not a subset of edge "
            f"roles {k.factor_type.edge_roles}"
        )
    if not set(k.constraint_modes) <= _NODE_MODES:
        problems.append(
            f"constraint_modes {sorted(set(k.constraint_modes) - _NODE_MODES)} "
            f"not in {sorted(_NODE_MODES)}"
        )
    if (k.pack_params is None) != (len(k.param_keys) == 0):
        problems.append("pack_params and param_keys must be declared together")
    if k.observed_out is not None:
        if "out" not in k.factor_type.edge_roles or "out" in k.message_roles:
            problems.append(
                "observed_out requires an 'out' edge role that carries no messages"
            )
        if k.validate_observation is None:
            problems.append("observed_out requires validate_observation")
        if k.observed_out in ef.FAMILIES:
            problems.append(
                f"observed_out {k.observed_out!r} is a message family — observation "
                "tags mark conditioned DATA-only edges"
            )
    existing = FACTOR_TYPES.get(name)
    if existing is not None and existing != k.factor_type:
        problems.append(
            f"factor type {name!r} already registered with a different structural "
            "signature"
        )
    recorded = EF_EDGE_FAMILIES.get(name)
    if recorded is not None and dict(recorded) != dict(k.edge_families):
        problems.append(
            f"EF_EDGE_FAMILIES already pins {name!r} to {recorded}, kernel declares "
            f"{dict(k.edge_families)}"
        )
    if problems:
        raise ValueError(
            f"cannot register kernel {k.impl_id!r} for factor type {name!r}:\n  - "
            + "\n  - ".join(problems)
        )
    if existing is None:
        register_factor_type(k.factor_type)
        _KERNEL_REGISTERED_TYPES.add(name)
    if recorded is None and k.edge_families:
        EF_EDGE_FAMILIES[name] = dict(k.edge_families)
    KERNELS[name] = k
    return k


def unregister_kernel(name: str) -> None:
    """Remove a kernel (tests/tooling). Also removes the FactorType and edge-
    family pins *if this registry added them* — built-ins stay put."""
    KERNELS.pop(name, None)
    if name in _KERNEL_REGISTERED_TYPES:
        FACTOR_TYPES.pop(name, None)
        EF_EDGE_FAMILIES.pop(name, None)
        _KERNEL_REGISTERED_TYPES.discard(name)


def kernel_for(factor_type: str) -> FactorKernel | None:
    return KERNELS.get(factor_type)


def conditioned_out_types() -> frozenset[str]:
    """Factor types whose 'out' edge is a conditioned DATA-only port."""
    return frozenset(n for n, k in KERNELS.items() if k.observed_out is not None)


def observation_families() -> frozenset[str]:
    """Legal DATA-only `VariableSpec.ef` tags: built-ins plus registered."""
    return ef.OBS_FAMILIES | frozenset(
        k.observed_out for k in KERNELS.values() if k.observed_out is not None
    )


def observation_validator(fam_tag: str) -> Callable | None:
    """The registered validator for one observation-family tag, if any."""
    for k in KERNELS.values():
        if k.observed_out == fam_tag:
            return k.validate_observation
    return None


def manifest(factor_types: Iterable[str]) -> dict[str, dict[str, str]]:
    """(impl, version) requirements for the *external* kernels among
    `factor_types`. Built-ins (`cffg.` namespace) are omitted: they ship with
    the package, and omitting them keeps existing template JSON and structure
    hashes byte-stable."""
    out: dict[str, dict[str, str]] = {}
    for t in sorted(set(factor_types)):
        k = KERNELS.get(t)
        if k is not None and not k.impl_id.startswith("cffg."):
            out[t] = {"impl": k.impl_id, "version": k.version}
    return out


# --- Built-in EF kernels ------------------------------------------------------
# The jnp math stays in cffg.efrules; imports are deferred into the callables
# so this module (used by the JAX-free compiler and core.validate) never
# imports JAX at module scope.


def _prior_pack(family: str, p_roles: tuple[str, str]):
    fam = ef.FAMILIES[family]

    def pack(t: Template, members, resolved) -> dict[str, np.ndarray]:
        """Stack per-member prior natural parameters {"eta": [M, 2]}. Priors
        must be proper — an improper prior makes every downstream belief and
        ledger term undefined, so it fails at pack time, not as runtime +inf."""
        etas = []
        for m in members:
            vals = []
            for role in p_roles:
                arr = resolved[t.factors[m].param_blobs[role]]
                if arr.size != 1:
                    raise ValueError(
                        f"factor {m!r}: parameter {role!r} must be a scalar, got "
                        f"shape {arr.shape}"
                    )
                vals.append(float(arr.reshape(())))
            if not all(v > 0 for v in vals):
                raise ValueError(
                    f"factor {m!r}: improper {family} prior {tuple(vals)}; "
                    "parameters must be > 0"
                )
            etas.append(fam.to_natural(*vals))
        return {"eta": np.stack(etas)}

    return pack


def _prior_message(inc, params, obs, mask):
    return params["eta"].astype(inc.dtype)[:, None, :]


def _prior_energy(family: str):
    def energy(inc, params, obs, mask):
        from cffg import efrules

        return efrules.prior_energy(
            efrules.JFAMILIES[family], inc[:, 0], params["eta"].astype(inc.dtype)
        )

    return energy


_PRIOR_HOSTILE = HostileInputs(
    sparse_support=(
        "provably N/A: EF beliefs have full support on the family domain; the "
        "only zero message is the vacuous eta=0"
    ),
    improper_messages=(
        "improper incoming sums make prior_energy +inf (loud), never NaN "
        "(efrules propriety guard)"
    ),
    float32="constant eta cast to the arena dtype; closed forms are dtype-polymorphic",
    scale="scalar nodes; member count only batches the leading axis",
)


def _obs_hostile(what: str) -> HostileInputs:
    return HostileInputs(
        sparse_support=(
            "provably N/A: the likelihood increment is a constant in T-space; "
            "no support pruning exists on EF edges"
        ),
        improper_messages=(
            f"improper rate/probability beliefs make {what} energy +inf (loud); "
            "mask 0 drops the node (vacuous message, energy telescopes)"
        ),
        float32="increment built from the observation row's dtype; no literals",
        scale="scalar observations; member count only batches the leading axis",
    )


def _poisson_message(inc, params, obs, mask):
    from cffg import efrules

    return efrules.poisson_likelihood_message(obs, mask)[:, None, :]


def _poisson_energy(inc, params, obs, mask):
    from cffg import efrules

    return efrules.poisson_obs_energy(inc[:, 0], obs, mask)


def _validate_poisson(value) -> float:
    val = float(value)
    if not (val >= 0 and val == int(val)):
        raise ValueError(f"Poisson counts are integers >= 0, got {value}")
    return val


def _bernoulli_message(inc, params, obs, mask):
    from cffg import efrules

    return efrules.bernoulli_likelihood_message(obs, mask)[:, None, :]


def _bernoulli_energy(inc, params, obs, mask):
    from cffg import efrules

    return efrules.bernoulli_obs_energy(inc[:, 0], obs, mask)


def _validate_bernoulli(value) -> float:
    val = float(value)
    if val not in (0.0, 1.0):
        raise ValueError(f"Bernoulli flips are 0 or 1, got {value}")
    return val


register_kernel(
    FactorKernel(
        factor_type=FACTOR_TYPES["gamma"],
        impl_id="cffg.ef.gamma",
        version="1.0.0",
        arena="ef",
        family="gamma",
        message_roles=("out",),
        edge_families={"out": "gamma"},
        message=_prior_message,
        energy=_prior_energy("gamma"),
        hostile=_PRIOR_HOSTILE,
        param_keys=("eta",),
        pack_params=_prior_pack("gamma", ("shape", "rate")),
    )
)

register_kernel(
    FactorKernel(
        factor_type=FACTOR_TYPES["beta"],
        impl_id="cffg.ef.beta",
        version="1.0.0",
        arena="ef",
        family="beta",
        message_roles=("out",),
        edge_families={"out": "beta"},
        message=_prior_message,
        energy=_prior_energy("beta"),
        hostile=_PRIOR_HOSTILE,
        param_keys=("eta",),
        pack_params=_prior_pack("beta", ("a", "b")),
    )
)

register_kernel(
    FactorKernel(
        factor_type=FACTOR_TYPES["poisson_obs"],
        impl_id="cffg.ef.poisson_obs",
        version="1.0.0",
        arena="ef",
        family="gamma",
        message_roles=("rate",),
        edge_families={"out": "poisson", "rate": "gamma"},
        message=_poisson_message,
        energy=_poisson_energy,
        hostile=_obs_hostile("Poisson"),
        observed_out="poisson",
        validate_observation=_validate_poisson,
    )
)

register_kernel(
    FactorKernel(
        factor_type=FACTOR_TYPES["bernoulli_obs"],
        impl_id="cffg.ef.bernoulli_obs",
        version="1.0.0",
        arena="ef",
        family="beta",
        message_roles=("prob",),
        edge_families={"out": "bernoulli", "prob": "beta"},
        message=_bernoulli_message,
        energy=_bernoulli_energy,
        hostile=_obs_hostile("Bernoulli"),
        observed_out="bernoulli",
        validate_observation=_validate_bernoulli,
    )
)
