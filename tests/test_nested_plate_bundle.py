"""Two-level plate array paths and learned ownership survive a bundle round trip."""

import io
import json
import zipfile

import numpy as np
import pytest

from cffg import Engine, compile_template
from cffg.bundle import BundleError, load_bundle, save_bundle
from cffg.core import BlobRef, Expose, FactorSpec, LearnableSpec, PlateSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.plates import expand_plates
from cffg.core.serialize import structure_hash
from cffg.learn import Learnables


def rooms(owner="implicit", *, learned=False):
    sensor = Template(
        "sensor",
        variables={"z": VariableSpec("z", cardinality=2), "x": VariableSpec("x", cardinality=2)},
        factors={
            "prior": FactorSpec("prior", "categorical", {"out": "z"}, param_blobs={"statistic": "d"}),
            "emission": FactorSpec("emission", "matrix_cat", {"in": "z", "out": "x"},
                                   param_blobs={"matrix": "A"}),
        },
        ports={"x": PortSpec("x", "x", PortRole.DATA)},
        blobs={name: BlobRef(name, f"source/{name}.npy") for name in ("A", "d")},
        learnables={
            "prior_counts": LearnableSpec("prior_counts", "dirichlet", ("d",)),
            "emission_counts": LearnableSpec("emission_counts", "dirichlet", ("A",)),
        } if learned else {},
    )
    room = Template(
        "room", plates={"sensor": PlateSpec(sensor, (2,), (Expose(),), per_copy_blobs=("d",))},
        blobs={} if owner == "implicit" else {"A": sensor.blobs["A"]},
    )
    scope = "school" if owner == "outer-tied" else ""
    private = ("A",) if owner == "outer-private" else ()
    model = Template("school", plates={"room": PlateSpec(room, (2, 2), (Expose(),), private, scope)})

    arrays, observations, posterior = {}, {}, {}
    log_evidence = 0.
    for floor in range(2):
        for unit in range(2):
            prefix = f"room/{floor}/{unit}"
            emission_key = "school/A" if owner == "outer-tied" else f"{prefix}/A"
            shift = 0.06 * (floor * 2 + unit) if owner != "outer-tied" else 0.
            emission = np.array([[.8 - shift, .21 + shift], [.2 + shift, .79 - shift]])
            arrays[emission_key] = emission
            for device in range(2):
                sensor_prefix = f"{prefix}/sensor/{device}"
                p = .12 + .09 * (floor * 4 + unit * 2 + device)
                prior = np.array([p, 1 - p])
                arrays[f"{sensor_prefix}/d"] = prior
                observation = (floor + device) % 2
                observations[f"{sensor_prefix}/x"] = observation
                joint = prior * emission[observation]
                posterior[f"{sensor_prefix}/z"] = joint / joint.sum()
                log_evidence += np.log(joint.sum())
    return model, arrays, observations, posterior, log_evidence


@pytest.mark.parametrize(("owner", "emission_path"), [
    ("implicit", "blobs/room/{index:0}/A.npy"),
    ("outer-private", "blobs/room/{index:0}/A.npy"),
    ("outer-tied", "blobs/school/A.npy"),
])
def test_nested_paths_match_multidimensional_inventory(tmp_path, owner, emission_path):
    model, arrays, *_ = rooms(owner)
    path = tmp_path / "nested.cffgz"
    save_bundle(path, model, arrays)
    saved = load_bundle(path)
    room = saved.template.plates["room"].fragment
    sensor = room.plates["sensor"].fragment
    assert sensor.blobs["A"].path == emission_path
    assert sensor.blobs["d"].path == "blobs/room/{index:0}/sensor/{index:1}/d.npy"
    if owner != "implicit":
        assert room.blobs["A"].path == emission_path
    assert structure_hash(saved.template) == structure_hash(model) == saved.manifest["structure_hash"]
    expected_paths = {name: f"blobs/{name}.npy" for name in arrays}
    assert saved.manifest["blob_paths"] == expected_paths
    assert {name: blob.path for name, blob in expand_plates(saved.template).blobs.items()} == expected_paths
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"model.json", "manifest.json", *expected_paths.values()}
    for name, array in saved.arrays.items():
        np.testing.assert_array_equal(array, arrays[name], strict=True)


def test_nested_literal_index_token_in_plate_name_is_encoded(tmp_path):
    model, arrays, *_ = rooms()
    name = "sensor:{index:0}"
    room = model.plates["room"].fragment
    room.plates[name] = room.plates.pop("sensor")
    renamed = {key.replace("/sensor/", f"/{name}/"): value for key, value in arrays.items()}
    path = tmp_path / "literal-index.cffgz"
    save_bundle(path, model, renamed)
    saved = load_bundle(path)
    sensor = saved.template.plates["room"].fragment.plates[name].fragment
    literal = "sensor%3A%7Bindex%3A0%7D"
    assert sensor.blobs["d"].path == f"blobs/room/{{index:0}}/{literal}/{{index:1}}/d.npy"
    assert saved.manifest["blob_paths"][f"room/0/1/{name}/0/d"] == f"blobs/room/0/1/{literal}/0/d.npy"


@pytest.mark.parametrize("owner", ["implicit", "outer-tied"])
def test_nested_bundle_restores_inference_and_scoped_learners(tmp_path, owner):
    model, arrays, observations, posterior, log_evidence = rooms(owner, learned=True)
    priors = {}
    sources = {}
    for name in observations:
        sensor = name.removesuffix("/x")
        room = sensor.split("/sensor/")[0]
        private = sensor + "/prior_counts"
        shared = ("school" if owner == "outer-tied" else room) + "/emission_counts"
        priors[private] = np.array([2., 3.])
        priors[shared] = np.array([[2., 3.], [4., 5.]])
        sources[private] = [sensor + "/prior"]
        sources.setdefault(shared, []).append(sensor + "/emission")
    before = compile_template(model, arrays=arrays)
    engine = Engine(before)
    initial = Learnables(engine, priors=priors).initial_params
    updated = {name: {"alpha": value["alpha"] + i}
               for i, (name, value) in enumerate(sorted(initial.items()), 1)}
    learned = Learnables.from_persistent_state(engine, initial, updated)

    path = tmp_path / "learned.cffgz"
    save_bundle(path, model, arrays, learnables=learned)
    saved = load_bundle(path)
    assert set(saved.learnables.plans) == set(priors)
    assert saved.learnables.persistent_descriptors() == learned.persistent_descriptors()
    for name in priors:
        assert saved.learnables.plans[name].sources == tuple(sorted(sources[name]))
        np.testing.assert_array_equal(saved.learnables.params[name]["alpha"], updated[name]["alpha"], strict=True)
        np.testing.assert_array_equal(saved.learnables.initial_params[name]["alpha"], priors[name], strict=True)

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extracted)
    after = compile_template(saved.template, root=extracted)
    assert after.member_of == before.member_of
    assert after.data_slots == before.data_slots
    for arena in (before, after):
        e = Engine(arena)
        bound = e.bind_data(e.init_state(), observations)
        state = e.run(bound, 20, tol=1e-12).require_converged()
        marginals = e.marginals(state)
        for name, expectation in posterior.items():
            np.testing.assert_allclose(marginals[name], expectation, atol=1e-12, rtol=0)
        assert e.bfe(state) == pytest.approx(-log_evidence, abs=1e-12)


@pytest.mark.parametrize("mutation", ["missing", "tampered-blob", "swapped-indices"])
def test_nested_archive_rejects_missing_corrupt_or_misaddressed_arrays(tmp_path, mutation):
    model, arrays, *_ = rooms()
    path = tmp_path / "integrity.cffgz"
    save_bundle(path, model, arrays)
    member = "blobs/room/0/1/sensor/0/d.npy"
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if mutation == "missing":
        del members[member]
        error = "missing blob file"
    elif mutation == "tampered-blob":
        buffer = io.BytesIO()
        np.save(buffer, np.array([.5, .5]), allow_pickle=False)
        members[member] = buffer.getvalue()
        error = "sha256 mismatch"
    else:
        document = json.loads(members["model.json"])
        room = document["plates"]["room"]["fragment"]
        sensor = room["plates"]["sensor"]["fragment"]
        sensor["blobs"]["d"]["path"] = "blobs/room/{index:1}/sensor/{index:0}/d.npy"
        members["model.json"] = json.dumps(document).encode()
        error = "blob paths disagree"

    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    with pytest.raises(BundleError, match=error):
        load_bundle(path)


@pytest.mark.parametrize("mutation", ["missing", "scope-collision"])
def test_invalid_nested_inventory_does_not_replace_destination(tmp_path, mutation):
    model, arrays, *_ = rooms()
    path = tmp_path / "existing.cffgz"
    path.write_bytes(b"previous bytes")
    name = "room/1/0/sensor/1/d"
    if mutation == "missing":
        del arrays[name]
        error = "arrays missing for blobs"
    else:
        model.blobs[name] = BlobRef(name, "redundant.npy")
        error = "blobs name clash"
    with pytest.raises(ValueError, match=f"{error}.*{name}"):
        save_bundle(path, model, arrays)
    assert path.read_bytes() == b"previous bytes"
