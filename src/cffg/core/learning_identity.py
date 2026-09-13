"""Serialized identities for posterior updates and their sufficient statistics.

These defaults describe new declarations. Legacy migration is deliberately
separate: an unidentified recurrence posterior cannot be assigned D9 semantics.
"""

from copy import deepcopy
import re
from types import MappingProxyType


LEARNING_IDENTITIES = MappingProxyType({
    "dirichlet": "cffg.learning.dirichlet/1.0.0",
    "niw": "cffg.learning.niw/1.0.0",
    "mniw": "cffg.learning.mniw/1.0.0",
    "component_mniw": "cffg.learning.component-mniw/1.0.0",
    "pg_weights": "cffg.recurrence.d9-structured-pair/1.0.0",
})

LEGACY_IDENTITIES = MappingProxyType({
    key: value for key, value in LEARNING_IDENTITIES.items() if key != "pg_weights"
})


def model_only_declarations(document: dict) -> dict:
    """Select current learning rules for a verified model-only archive.

    Point parameters carry no posterior provenance. These declarations allow
    inference and learning from explicitly supplied fresh priors; they never
    identify the statistics in an old checkpoint. Posterior restoration must
    still pass its separate semantic-identity checks.
    """
    migrated = deepcopy(document)

    def visit(fragment):
        if fragment.get("learning_schema_version") is None:
            for spec in fragment.get("learnables", {}).values():
                identity = LEARNING_IDENTITIES.get(spec["family"])
                if "semantic_identity" not in spec and identity is not None:
                    spec["semantic_identity"] = identity
        for plate in fragment.get("plates", {}).values():
            visit(plate["fragment"])

    visit(migrated)
    return migrated


def check_scoped_name(name: str) -> None:
    """Reject path ambiguity while allowing nested scopes and encoded punctuation."""
    if (
        not isinstance(name, str)
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or "\\" in name
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise ValueError(f"unsafe or ambiguous scoped name {name!r}")


def check_identity(name: str, identity: str) -> None:
    if not isinstance(identity, str) or not re.fullmatch(
        r"[a-zA-Z][a-zA-Z0-9_-]*(?:\.[a-zA-Z0-9_-]+)+/[0-9]+\.[0-9]+\.[0-9]+",
        identity,
    ):
        raise ValueError(f"learnable {name!r}: missing or invalid semantic identity {identity!r}")


def declaration_identity(name: str, family: str, spec: dict, version) -> str:
    """Migrate only historically unchanged declarations; never infer D9 evidence."""
    if version is None and "semantic_identity" not in spec:
        if family not in LEGACY_IDENTITIES:
            raise ValueError(
                f"learnable {name!r}: unversioned {family!r} declaration has no "
                "safe semantic migration; unidentified recurrence may contain D8 "
                "state and cannot be resumed as D9"
            )
        return LEGACY_IDENTITIES[family]
    identity = spec.get("semantic_identity")
    check_identity(name, identity)
    expected = LEARNING_IDENTITIES.get(family)
    if expected is not None and identity != expected:
        raise ValueError(
            f"learnable {name!r}: unknown or incompatible semantic identity "
            f"{identity!r}; supported identity is {expected!r}"
        )
    return identity
