"""`.cffgz` bundles: full-restore round-trips and loud integrity failures."""

import json
import zipfile

import numpy as np
import pytest

from cffg.bundle import BundleError, load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine
from cffg.models.gridworld import build_gridworld
from cffg.models.tmaze import build_tmaze


def test_round_trip_reproduces_inference(tmp_path):
    template, arrays = build_tmaze()
    path = tmp_path / "tmaze.cffgz"
    save_bundle(path, template, arrays)

    loaded_t, loaded_arrays, manifest = load_bundle(path)
    assert structure_hash(loaded_t) == structure_hash(template)
    assert manifest["structure_hash"] == structure_hash(template)

    a = Engine(compile_template(template, arrays=arrays))
    b = Engine(compile_template(loaded_t, arrays=loaded_arrays))
    sa = a.run(a.init_state(), 400, tol=1e-13, damp=0.5).require_converged()
    sb = b.run(b.init_state(), 400, tol=1e-13, damp=0.5).require_converged()
    np.testing.assert_allclose(a.marginals(sa)["u1"], b.marginals(sb)["u1"], atol=1e-12)


def test_bundle_is_self_contained(tmp_path):
    template, arrays = build_gridworld(k=5)
    path = tmp_path / "gw.cffgz"
    save_bundle(path, template, arrays)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    assert "model.json" in names and "manifest.json" in names
    assert {f"blobs/{b}.npy" for b in template.blobs} <= names


def test_tampered_blob_rejected(tmp_path):
    template, arrays = build_gridworld(k=5)
    path = tmp_path / "gw.cffgz"
    save_bundle(path, template, arrays)

    # rewrite one blob with different bytes, keep everything else
    with zipfile.ZipFile(path) as zf:
        contents = {n: zf.read(n) for n in zf.namelist()}
    import io

    buf = io.BytesIO()
    np.save(buf, np.zeros((5,)))
    contents["blobs/d.npy"] = buf.getvalue()
    with zipfile.ZipFile(path, "w") as zf:
        for n, raw in contents.items():
            zf.writestr(n, raw)

    with pytest.raises(BundleError, match="sha256 mismatch"):
        load_bundle(path)


def test_manifest_structure_hash_checked(tmp_path):
    template, arrays = build_gridworld(k=5)
    path = tmp_path / "gw.cffgz"
    save_bundle(path, template, arrays)
    with zipfile.ZipFile(path) as zf:
        contents = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(contents["manifest.json"])
    manifest["structure_hash"] = "0" * 64
    contents["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as zf:
        for n, raw in contents.items():
            zf.writestr(n, raw)
    with pytest.raises(BundleError, match="structure hash mismatch"):
        load_bundle(path)


def test_missing_arrays_refused(tmp_path):
    template, arrays = build_gridworld(k=5)
    del arrays["Bs"]
    with pytest.raises(BundleError, match="missing for blobs"):
        save_bundle(tmp_path / "x.cffgz", template, arrays)
