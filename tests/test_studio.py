"""The graph accepted by Studio must survive persistence, concurrency and replay."""

from io import BytesIO
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import pytest

from cffg.studio.editing import apply_commands
from cffg.studio.library import Conflict, Library
from cffg.studio.models import Document, starter


def test_portable_rslds_recovers_ties_and_resizes_learning_state():
    original = starter("rslds", "Switching observations")
    model = Document.exchange(original.certify())
    apply_commands(
        model,
        [
            {"action": "rename_variable", "name": "y0", "to": "position"},
            {
                "action": "resize_variable",
                "name": "position",
                "size": 2,
                "policy": "reinitialize",
            },
        ],
        "test",
        "",
    )
    copy = Document.load(model.certify())
    assert copy.template.ports["y0"].variable == "position"
    assert all(
        copy.template.variables[p.variable].dim == 2
        for p in copy.template.ports.values()
    )
    assert copy.arrays["emission_matrix"].shape == (2, 1)
    assert len(copy.learning.params) == 4
    for owner in ("recurrence", "initial_regime", "dynamics"):
        for key, value in original.learning.params[owner].items():
            np.testing.assert_array_equal(copy.learning.params[owner][key], value)


def test_map_latent_coordinates_and_keep_rectangular_noise_proper():
    d = starter("kalman", "Resize", latent=2, observed=3)
    d.arrays["m0"] = np.array([3.0, 7.0], dtype=np.float32)
    apply_commands(
        d,
        [
            {
                "action": "resize_variable",
                "name": "x0",
                "size": 3,
                "policy": "map",
                "mapping": [1, None, 0],
            }
        ],
        "tester",
        "",
    )
    np.testing.assert_equal(d.arrays["m0"], [7.0, 0.0, 3.0])
    assert np.linalg.eigvalsh(d.arrays["P0"]).min() > 0
    Document.load(d.certify())


def test_studio_head_is_atomic_and_request_retry_returns_original_revision(tmp_path):
    library = Library(tmp_path / "models")
    saved = library.create(starter("blank", "Blank"), "Alice")
    model, base = saved["id"], saved["revision"]
    doc = library.get(model)
    apply_commands(
        doc, [{"action": "add_channel", "name": "y", "size": 2}], "Bob", base
    )
    valid = doc.certify()
    token = uuid4().hex
    rev = library.publish(
        model, base, doc, valid, "Bob", "Add channel", token, "fingerprint"
    )
    assert (
        library.publish(
            model, base, doc, valid, "Bob", "Add channel", token, "fingerprint"
        )
        == rev
    )
    with pytest.raises(Conflict):
        library.publish(
            model, base, doc, valid, "Alice", "Stale edit", uuid4().hex, "other"
        )
    apply_commands(
        doc,
        [{"action": "set_array", "name": "y_factor.cov", "values": [[0, 0], [0, 0]]}],
        "Bob",
        rev,
    )
    with pytest.raises(ValueError):
        doc.certify()
    assert library.revision(model) == rev
    assert library.get(model, base).template.variables == {}


def test_project_exchange_preserves_review_anchor_after_rename():
    doc = starter("kalman", "Review")
    doc.certify()
    apply_commands(
        doc,
        [
            {"action": "annotate", "ref": "var:x0", "text": "Check the initial state"},
            {"action": "rename_variable", "name": "x0", "to": "initial"},
        ],
        "Alice",
        "rev",
    )
    b = doc.certify()
    assert doc.meta["notes"][0]["anchor"] == doc.meta["anchors"]["var:initial"]
    private = doc.project(b)
    loaded = Document.exchange(private)
    assert loaded.meta["notes"] == doc.meta["notes"]
    assert loaded.meta["anchors"] == doc.meta["anchors"]
    assert "source" not in loaded.meta
    with ZipFile(BytesIO(b)) as z:
        assert "studio.json" not in z.namelist()


def test_studio_api_rejects_stale_agent_and_replays_model_history(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from cffg.studio.server import create_app

    app = create_app(tmp_path / "state", tmp_path / "repo", roots=[tmp_path / "empty"])
    client = TestClient(app)
    assert client.get("/v1/models").status_code == 401
    token = app.state.access.manage("issue", "Sketch agent", "edit")["token"]
    client.headers["authorization"] = "Bearer " + token
    created = client.post("/v1/models", json={"name": "Blank"})
    assert created.status_code == 200
    model, base = created.json()["id"], created.json()["revision"]
    path = "/v1/models/" + model
    tx = {
        "base": base,
        "request_id": uuid4().hex,
        "commands": [{"action": "add_channel", "name": "sensor", "size": 1}],
    }
    assert client.post(path + "/transactions/preview", json=tx).status_code == 200
    first = client.post(path + "/transactions", json=tx).json()
    assert first["revision"] != base
    assert (
        client.post(path + "/transactions", json=tx).json()["revision"]
        == first["revision"]
    )
    tx["request_id"] = uuid4().hex
    assert client.post(path + "/transactions", json=tx).status_code == 409
    assert client.get(path + "/revisions/" + base).json()["template"]["variables"] == {}
    assert client.get(path + "/history").json()[0]["revision"] == first["revision"]
    assert client.post("/v1/search/scan", json={"roots": ["/"]}).status_code == 403


def test_plate_edit_preserves_per_copy_values_and_scoped_annotations():
    from cffg.core import Expose, PlateSpec, Template
    from test_plates import pixel_fragment

    host = Template(
        "Camera",
        plates={
            "pixels": PlateSpec(
                pixel_fragment(), (2,), (Expose(),), per_copy_blobs=("d",)
            )
        },
    )
    graph = Document(
        host,
        {
            "pixels/0/d": np.array([0.1, 0.2, 0.7]),
            "pixels/1/d": np.array([0.5, 0.3, 0.2]),
        },
    )
    graph.certify()
    before = {name: value.copy() for name, value in graph.arrays.items()}
    anchor = graph.fragment("pixels").meta["anchors"]["var:z"]
    apply_commands(
        graph,
        [
            {
                "action": "edit_plate",
                "plate": "pixels",
                "commands": [
                    {"action": "rename_variable", "name": "z", "to": "brightness"},
                    {
                        "action": "annotate",
                        "ref": "var:brightness",
                        "text": "Shared latent domain",
                    },
                ],
            }
        ],
        "Cooper",
        "rev",
    )
    copy = Document.exchange(graph.project(graph.certify()))
    fragment = copy.fragment("pixels")
    assert fragment.meta["notes"][0]["anchor"] == anchor
    assert fragment.template.factors["obs"].edges["in"] == "brightness"
    assert fragment.meta["anchors"]["var:brightness"] == anchor
    for name, value in before.items():
        np.testing.assert_array_equal(copy.arrays[name], value)


def test_discovery_reconciles_root_change_and_same_mtime_rewrite(tmp_path, monkeypatch):
    import json
    import os
    from threading import Event
    from cffg.studio.discovery import Discovery

    state, first, second = (tmp_path / key for key in ("state", "first", "second"))
    for directory in (state, first, second):
        directory.mkdir()

    def save(path, name):
        path.write_text(
            json.dumps(
                {"name": name, "schema_version": 1, "variables": {}, "factors": {}}
            )
        )

    for i in range(125):
        save(first / f"{i}.cffg", "A match")
    sample = second / "example.cffg"
    save(sample, "Z match")
    bundled = starter("kalman", "Spring tracker")
    embedded = first / "unrelated-filename.cffgstudio"
    embedded.write_bytes(bundled.project(bundled.certify()))
    service = Discovery(state, [first, second])

    def idle():
        assert service.lock.acquire(timeout=5)
        service.lock.release()
        assert service.error is None

    idle()
    assert service.find("Spring")[0]["name"] == "Spring tracker"
    old = sample.stat()
    save(sample, "X match")
    os.utime(sample, ns=(old.st_atime_ns, old.st_mtime_ns))
    started, resume = Event(), Event()
    scan = service._scan

    def delayed():
        started.set()
        assert resume.wait(5)
        scan()

    monkeypatch.setattr(service, "_scan", delayed)
    service.scan()
    assert started.wait(5)
    service.configure([second])
    assert [row["path"] for row in service.find("match")] == [str(sample)]
    assert [row["name"] for row in service.find("")] == ["Z match"]
    resume.set()
    idle()
    assert [row["name"] for row in service.find("match")] == ["X match"]
    sample.unlink()
    service.scan()
    idle()
    assert service.find("match") == []
