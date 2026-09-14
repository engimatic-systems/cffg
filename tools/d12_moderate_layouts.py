"""Complete abstract descriptor contracts, bound to their producer and runtime.

Regenerate prospectively with ``uv run python tools/d12_moderate_layouts.py``.
The fresh native-float32 child only traces shapes: no D12 lowering, compilation,
optimizer, or memory qualification. Deterministic gzip keeps the complete leaf
inventory small; ``gzip.open(PATH, 'rt')`` exposes the ordinary JSON document.
"""

from functools import lru_cache
import gzip
import json
from pathlib import Path
import subprocess
import sys

from d12_compile_scaling import child_environment, digest
from d12_moderate_plan import ROOT

PATH = Path(__file__).with_suffix(".json.gz")
RUNTIME_KEYS = (
    "python",
    "executable",
    "platform",
    "machine",
    "jax",
    "jaxlib",
    "numpy",
    "backend",
    "devices",
    "environment",
    "trace_config",
)


def producer_sources():
    paths = sorted((ROOT / "src/cffg").glob("hybrid_*.py"))
    paths += [
        ROOT / name
        for name in (
            "src/cffg/__init__.py",
            "tools/d12_compile_fixture.py",
            "tools/d12_compile_scaling.py",
            "tools/d12_moderate_workload.py",
            "tools/d12_moderate_runner.py",
            "tools/d12_moderate_layouts.py",
            "uv.lock",
            "pyproject.toml",
        )
    ]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def trace_config(jax):
    # Logging does not alter abstract buffers; include all other JAX settings,
    # including promotion, dot precision, transforms, backend and dtype policy.
    ignored = {"jax_log_compiles", "jax_explain_cache_misses"}
    return {
        name: (
            value.name
            if hasattr(value, "name")
            else value
            if isinstance(value, (str, int, float, bool, type(None)))
            else str(value)
        )
        for name, value in jax.config.values.items()
        if name not in ignored
    }


def runtime_binding(runtime):
    result = {name: runtime[name] for name in RUNTIME_KEYS}
    # uv and pytest may select python/python3 symlinks to the same interpreter.
    result["executable"] = str(Path(result["executable"]).resolve())
    return result


def live_binding():
    import jax
    from d12_moderate_runner import runtime_identity

    return dict(
        sources=producer_sources(), runtime=runtime_binding(runtime_identity(jax))
    )


@lru_cache(maxsize=4)
def read_contract(file_hash):
    with gzip.open(PATH, "rt") as stream:
        result = json.load(stream)
    if result["schema"] != "cffg.d12-logical-contract.v1":
        raise ValueError("unknown logical descriptor contract")
    return result


def contract():
    from d12_moderate_plan import load_spec

    load_spec()  # Verifies the prospectively registered descriptor artifact hash.
    result = read_contract(digest(PATH))
    if result["binding"]["sources"] != producer_sources():
        raise ValueError(
            "logical descriptor producer changed; regenerate the prospective contract"
        )
    return result


def expected_inventory(shape, binding):
    reference = contract()
    if binding != reference["binding"]:
        raise ValueError("logical inventory source/runtime binding mismatch")
    return next(
        item["inventory"]
        for item in reference["cases"]
        if item["shape"] == {key: shape[key] for key in ("T", "K", "d")}
    )


def generate():
    from d12_compile_scaling import native_jax

    native_jax()
    from d12_compile_fixture import context
    from d12_moderate_workload import logical_inventory, wide_context

    binding = live_binding()
    cases = []
    for t, k, d in ((32, 4, 2), (100, 4, 2), (100, 20, 10)):
        ctx = context(t) if k == 4 else wide_context(t, k, d)
        cases.append(dict(shape=dict(T=t, K=k, d=d), inventory=logical_inventory(*ctx)))
    if live_binding() != binding:
        raise ValueError("descriptor producer/runtime changed during tracing")
    result = dict(
        schema="cffg.d12-logical-contract.v1",
        binding=binding,
        cases=cases,
        complete_program_calls=dict(lower=0, compile=0, execute=0),
    )
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    PATH.write_bytes(gzip.compress(encoded, mtime=0))
    print(
        json.dumps(
            dict(
                path=str(PATH),
                sha256=digest(PATH),
                leaves=[
                    sum(len(v) for v in c["inventory"]["groups"].values())
                    for c in cases
                ],
                complete_program_calls=result["complete_program_calls"],
            )
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        generate()
    elif sys.argv[1:]:
        raise SystemExit("usage: uv run python tools/d12_moderate_layouts.py")
    else:
        subprocess.run(
            [sys.executable, __file__, "--child"],
            cwd=ROOT,
            env=child_environment(),
            check=True,
        )
