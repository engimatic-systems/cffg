"""Resolve declared blob sources without choosing a factor's mathematical meaning."""

from pathlib import Path

import numpy as np

from cffg.compiler_errors import CompileError
from cffg.core.ir import Template
from cffg.registered.capabilities import snapshot_registry
from cffg.registered.requirements import source_requirements
from cffg.specialized.discovery import initializer_fallbacks


def infer_blob_shape(blob_name, sources, fallbacks):
    """Run registered providers first, then the historical first-role fallback."""
    declared = set()
    sizes_by_factor = {}
    for source in sources:
        if source.blob != blob_name or source.shape is None:
            continue
        if source.factor not in sizes_by_factor:
            sizes_by_factor[source.factor] = dict(source.shape.role_sizes)
        declared.add(
            tuple(source.shape.parameter.shape(sizes_by_factor[source.factor]))
        )
    if len(declared) > 1:
        raise CompileError(
            f"blob {blob_name!r}: conflicting registered shapes {sorted(declared)}"
        )
    if declared:
        return declared.pop()
    for source in sources:
        if source.blob != blob_name:
            continue
        shape = source.initializer_fallback
        if shape is None:
            shape = fallbacks.get((source.factor, source.role))
        if shape is not None:
            return shape
    return None


def resolve_arrays(
    t: Template,
    arrays: dict[str, np.ndarray] | None,
    root: str | Path | None,
    *,
    sources=None,
    fallbacks=None,
) -> dict[str, np.ndarray]:
    """Resolve overrides > existing files > initializers, retaining source dtype checks.

    Direct compatibility calls build declarations; compiler orchestration passes
    the already captured requirements and never repeats family discovery here.
    """
    if sources is None:
        sources = source_requirements(t, snapshot_registry(t))
    if fallbacks is None:
        fallbacks = initializer_fallbacks(t)
    needed = {source.blob for source in sources}
    covariance_blobs = {source.blob for source in sources if source.covariance}
    out: dict[str, np.ndarray] = {}
    for name in sorted(needed):
        path = Path(root) / t.blobs[name].path if root is not None else None
        if arrays is not None and name in arrays:
            value = np.asarray(arrays[name])
        elif path is not None and path.exists():
            value = _load_blob(path, name)
        elif t.blobs[name].init:
            from cffg import inits

            shape = infer_blob_shape(name, sources, fallbacks)
            if shape is None:
                raise CompileError(f"blob {name!r}: cannot infer shape for init spec")
            try:
                value = np.asarray(inits.from_spec(t.blobs[name].init, shape))
            except ValueError as error:
                raise CompileError(
                    f"blob {name!r}: init spec failed: {error}"
                ) from None
        else:
            raise CompileError(
                f"blob {name!r}: no array override, no file"
                + (f" at {path}" if path is not None else " (no root directory)")
                + ", and no init spec"
            )
        if name in covariance_blobs:
            from cffg.params import _validate_covariance

            try:
                # Check source roundoff before host packing widens the dtype.
                _validate_covariance(value, f"blob {name!r}: covariance")
            except ValueError as error:
                raise CompileError(str(error)) from error
        out[name] = np.asarray(value, dtype=np.float64)
    return out


def _blob_shape(t: Template, blob_name: str) -> tuple[int, ...] | None:
    """Compatibility entry; stage consumers already hold these declarations."""
    return infer_blob_shape(
        blob_name,
        source_requirements(t, snapshot_registry(t)),
        initializer_fallbacks(t),
    )


def _load_blob(path: Path, name: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path))
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        tensors = load_file(path)
        key = (
            name
            if name in tensors
            else next(iter(tensors))
            if len(tensors) == 1
            else None
        )
        if key is None:
            raise CompileError(
                f"blob {name!r}: {path} holds {sorted(tensors)}, need {name!r}"
            )
        return np.asarray(tensors[key])
    raise CompileError(f"blob {name!r}: unsupported format {path.suffix!r}")
