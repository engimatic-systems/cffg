"""Factor-graph plates, phase 1 (cff-cpxj): the four validation use cases plus
serialization/backward-compat/edit coverage. Plates declare "duplicate this
fragment N times and wire the copies" and expand to a flat template in the
compiler's grounding phase; the SPEC (fragment + dims + bonds) is what
serializes and what the structure hash covers, never the expansion.
"""

import dataclasses
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cffg import inits
from cffg.compiler import compile_template
from cffg.core import (
    BlobRef,
    Broadcast,
    Chain,
    Expose,
    FactorSpec,
    ModelError,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
    expand_plates,
    plate_observations,
    validate,
)
from cffg.core.compose import compose
from cffg.core.edits import SetPlateDims, apply_edits, bundle_from_json, bundle_to_json
from cffg.core.serialize import from_json, structure_hash, to_dict, to_json
from cffg.engine import Engine
from cffg.viz import render_svg

K, M = 3, 2  # latent / observation cardinalities


# --------------------------------------------------------------------------
# Fragments


def pixel_fragment(seed: int = 1) -> Template:
    """One pixel: a prior over a latent z and a categorical emission to x
    (the observed pixel). A and the prior d are tied across the whole grid."""
    return Template(
        name="pixel",
        variables={"z": VariableSpec("z", K), "x": VariableSpec("x", M)},
        factors={
            "prior": FactorSpec(
                "prior", "categorical", edges={"out": "z"}, param_blobs={"statistic": "d"}
            ),
            "obs": FactorSpec(
                "obs", "matrix_cat", edges={"out": "x", "in": "z"}, param_blobs={"matrix": "A"}
            ),
        },
        ports={"x": PortSpec("x", "x", PortRole.DATA)},
        blobs={
            "d": BlobRef("d", "d.npy", inits.spec("uniform")),
            "A": BlobRef("A", "A.npy", inits.spec("random_stochastic", seed=seed)),
        },
    )


def selector_fragment() -> Template:
    """A mixture-bank member: a categorical whose output edge is exposed to be
    broadcast into the shared selector. The weights are per-copy."""
    return Template(
        name="member",
        variables={"s": VariableSpec("s", K)},
        factors={
            "cat": FactorSpec(
                "cat", "categorical", edges={"out": "s"}, param_blobs={"statistic": "w"}
            )
        },
        ports={"s": PortSpec("s", "s", PortRole.GLUE)},
        blobs={"w": BlobRef("w", "w.npy", inits.spec("uniform"))},
    )


def sensor_fragment(d: int = 1) -> Template:
    """One sensor: a linear-Gaussian emission y = state (observed), the state
    edge exposed to be broadcast into the shared state hub."""
    return Template(
        name="sensor",
        variables={"y": VariableSpec("y", dim=d), "state": VariableSpec("state", dim=d)},
        factors={
            "emit": FactorSpec(
                "emit",
                "gaussian_linear",
                edges={"out": "y", "in": "state"},
                param_blobs={"matrix": "SA", "offset": "Sb", "cov": "Scov"},
            )
        },
        ports={
            "y": PortSpec("y", "y", PortRole.DATA),
            "st": PortSpec("st", "state", PortRole.GLUE),
        },
        blobs={
            "SA": BlobRef("SA", "sa.npy", inits.spec("eye")),
            "Sb": BlobRef("Sb", "sb.npy", inits.spec("zeros")),
            "Scov": BlobRef("Scov", "sc.npy", inits.spec("eye", scale=2.0)),
        },
    )


# --------------------------------------------------------------------------
# (A) pixel-grid observation layer: 2-D plate, tied A, aggregated image bind


def test_pixel_grid_matches_single_pixel(viz_gallery):
    host = Template(
        name="grid", plates={"pix": PlateSpec(pixel_fragment(), (4, 4), (Expose(),))}
    )
    engine = Engine(compile_template(host))
    state = engine.init_state()

    rng = np.random.default_rng(0)
    image = rng.integers(0, M, size=(4, 4))
    obs = plate_observations("pix", image, "x")
    assert set(obs) == {f"pix/{r}/{c}/x" for r in range(4) for c in range(4)}

    state = engine.bind_data(state, obs)
    state = engine.run(state, 25).state
    marg = engine.marginals(state)

    # One pixel's latent posterior must equal a hand-built single-pixel model.
    single = Engine(compile_template(pixel_fragment()))
    for (r, c), _ in np.ndenumerate(image):
        ss = single.bind_data(single.init_state(), {"x": int(image[r, c])})
        ss = single.run(ss, 25).state
        np.testing.assert_allclose(
            marg[f"pix/{r}/{c}/z"], single.marginals(ss)["z"], atol=1e-9
        )
    viz_gallery(host, tag="4x4 pixel observation plate (tied A)")


# --------------------------------------------------------------------------
# (B) mixture bank broadcasting into a selector: equality-tree synthesis


def _broadcast_weights():
    return [
        np.array([0.6, 0.3, 0.1]),
        np.array([0.2, 0.5, 0.3]),
        np.array([0.1, 0.2, 0.7]),
    ]


def test_broadcast_matches_hand_expanded_reference(viz_gallery):
    weights = _broadcast_weights()
    n = len(weights)
    host = Template(
        name="bank",
        plates={
            "b": PlateSpec(
                selector_fragment(), (n,), (Broadcast("s", "sel"),), per_copy_blobs=("w",)
            )
        },
    )
    arrays = {f"b/{i}/w": weights[i] for i in range(n)}
    engine = Engine(compile_template(host, arrays=arrays))
    state = engine.run(engine.init_state(), 20).state
    hub = engine.marginals(state)["b/sel"]

    # Hand-expanded reference: the same equality tree, built without plates.
    ref = Template(
        name="ref",
        variables={
            "hub": VariableSpec("hub", K),
            "s0": VariableSpec("s0", K),
            "s1": VariableSpec("s1", K),
            "s2": VariableSpec("s2", K),
            "link0": VariableSpec("link0", K),
        },
        factors={
            "c0": FactorSpec("c0", "categorical", edges={"out": "s0"}, param_blobs={"statistic": "w0"}),
            "c1": FactorSpec("c1", "categorical", edges={"out": "s1"}, param_blobs={"statistic": "w1"}),
            "c2": FactorSpec("c2", "categorical", edges={"out": "s2"}, param_blobs={"statistic": "w2"}),
            "eq0": FactorSpec("eq0", "equality", edges={"a": "hub", "b": "s0", "c": "link0"}),
            "eq1": FactorSpec("eq1", "equality", edges={"a": "link0", "b": "s1", "c": "s2"}),
        },
        ports={"hub": PortSpec("hub", "hub", PortRole.GLUE)},
        blobs={f"w{i}": BlobRef(f"w{i}", f"w{i}.npy") for i in range(n)},
    )
    ref_engine = Engine(compile_template(ref, arrays={f"w{i}": weights[i] for i in range(n)}))
    ref_state = ref_engine.run(ref_engine.init_state(), 20).state

    np.testing.assert_allclose(hub, ref_engine.marginals(ref_state)["hub"], atol=1e-10)
    # And the analytic anchor: normalized elementwise product of the weights.
    prod = weights[0] * weights[1] * weights[2]
    np.testing.assert_allclose(hub, prod / prod.sum(), atol=1e-10)
    viz_gallery(host, tag="mixture bank broadcasting into a selector")


def test_broadcast_n1_and_n2_edge_cases():
    weights = _broadcast_weights()
    for n in (1, 2):
        host = Template(
            name="bank",
            plates={
                "b": PlateSpec(
                    selector_fragment(), (n,), (Broadcast("s", "sel"),), per_copy_blobs=("w",)
                )
            },
        )
        arrays = {f"b/{i}/w": weights[i] for i in range(n)}
        engine = Engine(compile_template(host, arrays=arrays))
        state = engine.run(engine.init_state(), 20).state
        hub = engine.marginals(state)["b/sel"]
        prod = np.ones(K)
        for i in range(n):
            prod = prod * weights[i]
        np.testing.assert_allclose(hub, prod / prod.sum(), atol=1e-10)


# --------------------------------------------------------------------------
# (C) N-sensor array into one Gaussian state: broadcast + information addition


def test_sensor_array_information_addition(viz_gallery):
    n = 4
    ys = [0.5, 1.0, 2.0, -0.5]
    host = Template(
        name="array",
        plates={"s": PlateSpec(sensor_fragment(), (n,), (Broadcast("st", "state"),))},
    )
    engine = Engine(compile_template(host))
    state = engine.bind_data(
        engine.init_state(), {f"s/{i}/y": [ys[i]] for i in range(n)}
    )
    state = engine.run(state, 30).state
    mean, cov = engine.gaussian_marginals(state)["s/state"]

    # Hand posterior: product of N sensors, A=I, b=0, W=1/2 (cov=2). Information
    # form adds: Lambda = sum W = n/2, eta = sum W*y = sum(y)/2.
    lam = n / 2.0
    eta = sum(ys) / 2.0
    np.testing.assert_allclose(mean, [eta / lam], atol=1e-9)
    np.testing.assert_allclose(cov, [[1.0 / lam]], atol=1e-9)
    viz_gallery(host, tag="N-sensor array into one Gaussian state")


# --------------------------------------------------------------------------
# (D) plate resize edit: structure hash changes, recompiles, runs


def test_set_plate_dims_edit_recompiles():
    host = Template(
        name="grid", plates={"pix": PlateSpec(pixel_fragment(), (2, 2), (Expose(),))}
    )
    before = structure_hash(host)
    resized = apply_edits(host, [SetPlateDims("pix", (3, 3))])
    after = structure_hash(resized)
    assert before != after
    assert resized.plates["pix"].dims == (3, 3)

    engine = Engine(compile_template(resized))
    obs = plate_observations("pix", np.zeros((3, 3), dtype=int), "x")
    state = engine.bind_data(engine.init_state(), obs)
    state = engine.run(state, 10).state
    assert "pix/2/2/z" in engine.marginals(state)


def test_set_plate_dims_codec_round_trips():
    actions = [SetPlateDims("pix", (5, 5))]
    back = bundle_from_json(bundle_to_json(actions))
    assert back == actions


def test_set_plate_dims_unknown_plate_raises():
    host = Template(name="grid", plates={"pix": PlateSpec(pixel_fragment(), (2,), (Expose(),))})
    with pytest.raises(Exception):
        apply_edits(host, [SetPlateDims("nope", (3,))])


# --------------------------------------------------------------------------
# Serialization: spec, not expansion


def test_serialization_round_trips_with_plate():
    host = Template(
        name="grid",
        plates={"pix": PlateSpec(pixel_fragment(), (4, 4), (Expose(),))},
    )
    back = from_json(to_json(host))
    assert to_dict(back) == to_dict(host)
    validate(back)
    assert back.plates["pix"].dims == (4, 4)
    assert isinstance(back.plates["pix"].bonds[0], Expose)


def test_spec_size_independent_of_copy_count():
    # Same digit-width dims, wildly different copy counts: the serialized SPEC
    # length is identical (it never materializes the copies).
    small = Template(name="g", plates={"p": PlateSpec(pixel_fragment(), (4, 4), (Expose(),))})
    big = Template(name="g", plates={"p": PlateSpec(pixel_fragment(), (9, 9), (Expose(),))})
    assert len(to_json(small)) == len(to_json(big))  # 16 vs 81 copies
    assert structure_hash(small) != structure_hash(big)


def test_structure_hash_covers_dims_not_fragment_blob_paths():
    base = Template(name="g", plates={"p": PlateSpec(pixel_fragment(), (2, 2), (Expose(),))})
    # A blob-path move inside the fragment must NOT change the hash.
    frag = pixel_fragment()
    frag.blobs["A"] = dataclasses.replace(frag.blobs["A"], path="elsewhere/A.npy")
    moved = Template(name="g", plates={"p": PlateSpec(frag, (2, 2), (Expose(),))})
    assert structure_hash(moved) == structure_hash(base)
    # But resizing dims does.
    resized = Template(name="g", plates={"p": PlateSpec(pixel_fragment(), (3, 2), (Expose(),))})
    assert structure_hash(resized) != structure_hash(base)


def test_no_plates_backward_compatible(hmm_slice):
    # A plateless template serializes byte-identically to before (no "plates").
    d = to_dict(hmm_slice)
    assert "plates" not in d
    assert to_dict(from_json(to_json(hmm_slice))) == d


# --------------------------------------------------------------------------
# plate_observations helper


def test_plate_observations_1d_and_2d():
    o1 = plate_observations("p", [7, 8, 9], "x")
    assert o1 == {"p/0/x": 7, "p/1/x": 8, "p/2/x": 9}
    o2 = plate_observations("p", np.array([[0, 1], [1, 0]]), "x")
    assert o2 == {"p/0/0/x": 0, "p/0/1/x": 1, "p/1/0/x": 1, "p/1/1/x": 0}


# --------------------------------------------------------------------------
# Validation of the SPEC


def test_reject_3d_dims():
    host = Template(name="g", plates={"p": PlateSpec(pixel_fragment(), (2, 2, 2), (Expose(),))})
    with pytest.raises(ModelError, match="1-D or 2-D"):
        validate(host)


def test_reject_plate_nested_inside_fragment_before_grounding():
    middle = Template(
        name="middle",
        plates={"inner": PlateSpec(pixel_fragment(), (2,), (Expose(),))},
    )
    host = Template(
        name="host",
        plates={"outer": PlateSpec(middle, (2,), (Expose(),))},
    )
    message = (
        r"plate 'outer': fragment declares nested plates \['inner'\]; "
        r"nested plate grounding is not supported in phase 1"
    )

    with pytest.raises(ModelError, match=message):
        validate(host)
    with pytest.raises(ModelError, match=message):
        expand_plates(host)


def test_chain_endpoints_must_be_glue():
    # 'x' is a DATA port, not glue — chaining on it is rejected.
    host = Template(
        name="g", plates={"p": PlateSpec(pixel_fragment(), (3,), (Chain("x", "x"),))}
    )
    with pytest.raises(ModelError, match="must be a glue port"):
        validate(host)


def test_bond_references_real_ports():
    host = Template(
        name="g", plates={"p": PlateSpec(sensor_fragment(), (3,), (Broadcast("nope", "hub"),))}
    )
    with pytest.raises(ModelError, match="not a fragment port"):
        validate(host)


# --------------------------------------------------------------------------
# Viz: plate boxes with count badges, from the SPEC


def test_plate_viz_renders_box_and_badge(viz_gallery):
    host = Template(
        name="grid", plates={"pix": PlateSpec(pixel_fragment(), (4, 4), (Expose(),))}
    )
    svg = render_svg(host, level=4)
    ET.fromstring(svg)  # well-formed
    assert "×4×4" in svg  # count badge from the spec
    assert "plate:pix" in svg  # plate box back-reference
    # Independent of copy count: the SVG does not grow with dims (drawn once).
    big = Template(
        name="grid", plates={"pix": PlateSpec(pixel_fragment(), (9, 9), (Expose(),))}
    )
    assert len(render_svg(big, level=4)) == len(render_svg(host, level=4).replace("×4×4", "×9×9"))
    viz_gallery(svg, tag="plate box with x4x4 count badge")


# --------------------------------------------------------------------------
# 1-D chain plate == unrolling (the unroll-vs-chain question, reported)


def test_chain_plate_reproduces_unrolled_hmm(hmm_slice):
    """A 1-D chain plate of the HMM slice chains exactly like unroll: bonding
    each copy's tail to the next copy's head. Compiled + run against a hand
    forward-backward smoother, the smoothed marginals match."""
    T = 3
    rng = np.random.default_rng(7)

    def col_stochastic(rows, cols):
        m = rng.uniform(0.1, 1.0, size=(rows, cols))
        return m / m.sum(axis=0, keepdims=True)

    D = rng.dirichlet(np.ones(K))
    B = col_stochastic(K, K)
    A = col_stochastic(M, K)

    host = Template(
        name="hmm", plates={"c": PlateSpec(hmm_slice, (T,), (Chain("tail", "head"),))}
    )
    chained = expand_plates(host)  # flat template; first copy's head still exposed

    # Terminate the open head with a state prior (compose onto the plate output).
    prior = Template(
        name="p",
        variables={"z0": VariableSpec("z0", K)},
        factors={"d": FactorSpec("d", "categorical", edges={"out": "z0"}, param_blobs={"statistic": "d"})},
        ports={"attach": PortSpec("attach", "z0", PortRole.GLUE)},
        blobs={"d": BlobRef("d", "d.npy")},
    )
    model = compose(prior, chained, {"attach": "c/0/head"}, name="chain", scope_a="pr", scope_b="ch")

    # A and B are TIED (one shared blob each), so compose scopes them once.
    arrays = {"pr/d": D, "ch/A": A, "ch/B": B}
    engine = Engine(compile_template(model, arrays=arrays))

    obs = [0, 1, 0]
    state = engine.bind_data(
        engine.init_state(), {f"ch/c/{t}/sense": obs[t] for t in range(T)}
    )
    state = engine.run(state, 60, tol=1e-10).require_converged()
    marg = engine.marginals(state)

    # Hand forward-backward: z_0 prior-only; x_t emitted from z_t (t>=1).
    em = [np.ones(K)] + [A[o] for o in obs]
    alphas, log_cs = [D.copy()], []
    for t in range(T):
        a = em[t + 1] * (B @ alphas[-1])
        alphas.append(a / a.sum())
    betas = [np.ones(K)]
    for t in reversed(range(T)):
        b = B.T @ (em[t + 1] * betas[0])
        betas.insert(0, b / b.sum())
    smoothed = np.array([a * b for a, b in zip(alphas, betas)])
    smoothed /= smoothed.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(marg["pr/z0"], smoothed[0], atol=1e-6)
    for t in range(T):
        np.testing.assert_allclose(marg[f"ch/c/{t}/z_mid"], smoothed[t + 1], atol=1e-6)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 49, 784])
def test_balanced_broadcast_wiring(n):
    from collections import Counter
    from math import ceil, log2

    plate = Template("balanced", plates={"p": PlateSpec(selector_fragment(), (n,), (Broadcast("s", "hub"),))})
    flat = expand_plates(plate)
    validate(flat)
    assert to_dict(flat) == to_dict(expand_plates(from_json(to_json(plate))))
    nodes = {
        f.edges["a"]: (f.edges["b"], f.edges["c"]) for f in flat.factors.values() if f.type == "equality"
    }
    assert len(nodes) == n - 1
    assert len([v for v in flat.variables if "/_bcast/" in v]) == max(n - 2, 0)
    degrees = Counter(v for f in flat.factors.values() for v in f.edges.values())
    assert max(degrees.values()) <= 2

    def descend(edge, depth):
        if edge not in nodes:
            return [(edge, depth)]
        left, right = (descend(child, depth + 1) for child in nodes[edge])
        assert abs(len(left) - len(right)) <= 1
        return left + right

    leaves = descend("p/hub", 0)
    expected = ["p/hub"] if n == 1 else [f"p/{i}/s" for i in range(n)]
    assert [name for name, _ in leaves] == expected
    assert max(depth for _, depth in leaves) <= ceil(log2(n))


def test_balanced_broadcast_rejects_name_collision():
    from cffg.core.compose import ComposeError

    plate = Template(
        "collision", plates={"p": PlateSpec(selector_fragment(), (3,), (Broadcast("s", "hub"),))}
    )
    path = "p/_bcast/hub/eq1"
    plate.variables["outside"] = VariableSpec("outside", K)
    plate.blobs["old"] = BlobRef("old", "old.npy", inits.spec("uniform"))
    plate.factors[path] = FactorSpec(
        path, "categorical", {"out": "outside"}, param_blobs={"statistic": "old"}
    )
    plate.ports["existing"] = PortSpec("existing", "outside", PortRole.GLUE)
    before = to_json(plate)
    with pytest.raises(ComposeError, match=path):
        expand_plates(plate)
    assert to_json(plate) == before


def _linear_reference(plate, n, branch):
    flat = expand_plates(plate)
    for i in range(n - 1):
        name = f"p/_bcast/hub/eq{i}"
        inlet = "p/hub" if i == 0 else f"p/_bcast/hub/link{i - 1}"
        continuation = f"p/{i + 1}/{branch}" if i == n - 2 else f"p/_bcast/hub/link{i}"
        flat.factors[name] = FactorSpec(
            name, "equality", {"a": inlet, "b": f"p/{i}/{branch}", "c": continuation}
        )
    validate(flat)
    return flat


@pytest.mark.parametrize("semiring", ["sum", "max"])
def test_balanced_broadcast_matches_chain_and_flooding(semiring):
    n = 5
    plate = Template(
        "sparse",
        plates={"p": PlateSpec(selector_fragment(), (n,), (Broadcast("s", "hub"),), per_copy_blobs=("w",))},
    )
    weights = np.array(
        [[0.7, 0.3, 0], [1 / 3, 1 / 3, 1 / 3], [0.4, 0.4, 0.2], [0.1, 0.8, 0.1], [0.4, 0.5, 0.1]]
    )
    arrays = {f"p/{i}/w": w for i, w in enumerate(weights)}
    expected = weights.prod(axis=0)
    expected /= expected.sum()
    values = []
    for template, schedule in [
        (plate, "auto"),
        (plate, "flood"),
        (_linear_reference(plate, n, "s"), "flood"),
    ]:
        eng = Engine(compile_template(template, arrays=arrays, semiring=semiring), schedule=schedule)
        if schedule == "auto":
            assert eng.schedule.kind == "two_pass"
        state = eng.run(eng.init_state(), 32, tol=1e-10).require_converged()
        if semiring == "sum":
            for name in ["p/hub", *(f"p/{i}/s" for i in range(n))]:
                belief = eng.marginals(state)[name]
                np.testing.assert_allclose(belief, expected, atol=1e-10, rtol=0)
                assert belief[2] == 0
            values.append(float(eng.bfe(state)))
        else:
            assert set(eng.decode(state).values()) == {int(expected.argmax())}
            values.append(float(eng.objective(state)))
    np.testing.assert_allclose(values, values[0], atol=1e-10, rtol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_balanced_gaussian_broadcast_matches_information_sum(dtype):
    import jax

    n, dim = 5, 32
    plate = Template(
        "continuous", plates={"p": PlateSpec(sensor_fragment(dim), (n,), (Broadcast("st", "hub"),))}
    )
    # The last sensor is unbound and contributes zero information.
    readings = np.arange((n - 1) * dim, dtype=dtype).reshape(n - 1, dim) / dim
    observations = {f"p/{i}/y": y for i, y in enumerate(readings)}
    atol = 64 * np.finfo(dtype).eps if dtype is np.float32 else 1e-10
    with jax.enable_x64(dtype == np.float64):
        energies = []
        for template in (plate, _linear_reference(plate, n, "state")):
            eng = Engine(compile_template(template), schedule="flood")
            state = eng.run(eng.bind_data(eng.init_state(), observations), 32, tol=atol).require_converged()
            beliefs = eng.gaussian_marginals(state)
            for name in ["p/hub", *(f"p/{i}/state" for i in range(n))]:
                mean, cov = beliefs[name]
                np.testing.assert_allclose(mean, readings.mean(axis=0), atol=atol, rtol=atol)
                np.testing.assert_allclose(cov, 2 * np.eye(dim) / (n - 1), atol=atol, rtol=atol)
            energies.append(float(eng.bfe(state)))
        np.testing.assert_allclose(energies, energies[0], atol=atol, rtol=atol)


def test_frozen_broadcast_json_and_bundle(tmp_path):
    import json

    from cffg.bundle import save_bundle, load_bundle
    from cffg.schedule import ScanStage, build_schedule
    from collections import Counter

    # The pre-balancing schema-1 compact template and its canonical structure hash.
    old = {
        "schema_version": "1",
        "name": "bank",
        "variables": {},
        "factors": {},
        "constraints": [],
        "ports": {},
        "blobs": {},
        "plates": {
            "b": {
                "fragment": {
                    "schema_version": "1",
                    "name": "leaf",
                    "variables": {"v": {"cardinality": 2}},
                    "factors": {
                        "p": {
                            "type": "categorical",
                            "edges": {"out": "v"},
                            "param_edges": {},
                            "param_blobs": {"statistic": "p"},
                        }
                    },
                    "constraints": [],
                    "ports": {"v": {"variable": "v", "role": "glue"}},
                    "blobs": {"p": {"path": "p.npy"}},
                },
                "dims": [3],
                "bonds": [{"kind": "broadcast", "port": "v", "hub": "c"}],
                "per_copy_blobs": [],
            }
        },
    }
    t = from_json(json.dumps(old))
    assert to_dict(t) == old
    assert structure_hash(t) == "57689e3ef8a045c77b74269b21f84542f587122adf34ebf97cc548e2744067c0"
    arrays = {"p": np.array([0.1, 0.9])}
    save_bundle(tmp_path / "b.cffgz", t, arrays)
    loaded, restored, _ = load_bundle(tmp_path / "b.cffgz")
    arena = compile_template(t, arrays=arrays)
    schedules = [build_schedule(arena), build_schedule(compile_template(loaded, arrays=restored))]
    firings = Counter()
    for stage in schedules[0].stages:
        for batch in stage.elems if isinstance(stage, ScanStage) else (stage,):
            members = arena.groups[batch.group].members
            firings.update(members[i] for i in np.ravel(batch.rows))
    assert {name: n for name, n in firings.items() if "/_bcast/" in name} == {
        "b/_bcast/c/eq0": 2,
        "b/_bcast/c/eq1": 2,
    }
    assert schedules[0].kind == schedules[1].kind == "two_pass"
    assert schedules[0].fingerprint == schedules[1].fingerprint


def test_balanced_forest_mixed_schedule():
    from cffg.core import ConstraintSpec, ConstraintKind
    from cffg.core.compose import merge

    t = Template("bank", plates={"p": PlateSpec(selector_fragment(), (7,), (Broadcast("s", "hub"),))})
    bead = pixel_fragment()
    bead.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "obs"))
    model = merge(expand_plates(t), bead, name="forest")
    values = []
    for schedule in ("auto", "flood"):
        engine = Engine(compile_template(model), schedule=schedule)
        if schedule == "auto":
            assert engine.schedule.kind == "mixed"
        result = engine.run(engine.bind_data(engine.init_state(), {"x": 1}), 80, tol=1e-10)
        state = result.require_converged()
        values.append((engine.marginals(state), float(engine.bfe(state))))
    for name in model.variables:
        np.testing.assert_allclose(values[0][0][name], values[1][0][name], atol=1e-10, rtol=0)
    np.testing.assert_allclose(values[0][1], values[1][1], atol=1e-10, rtol=0)


def test_balanced_broadcast_reports_support_failure():
    from cffg.engine import RunStatus, _replace

    t = Template(
        "bank",
        plates={"p": PlateSpec(selector_fragment(), (4,), (Broadcast("s", "hub"),), per_copy_blobs=("w",))},
    )
    arrays = {f"p/{i}/w": np.array([1.0, 0.0, 0.0]) if i % 2 else np.array([0.0, 1.0, 0.0]) for i in range(4)}
    eng = Engine(compile_template(t, arrays=arrays), schedule="flood")
    state = eng.init_state()
    assert eng.run(state, 32, tol=1e-10).status is RunStatus.INFEASIBLE
    invalid = _replace(state, messages=state.messages.at[0].set(np.nan))
    assert eng.run(invalid, 32, tol=1e-10).status is RunStatus.DIVERGED


@pytest.mark.slow
@pytest.mark.parametrize("domain", ["categorical", "gaussian"])
def test_broadcast_compiles_784_leaves_at_domain_size_32(domain):
    fragment = selector_fragment() if domain == "categorical" else sensor_fragment(32)
    port = "s" if domain == "categorical" else "st"
    if domain == "categorical":
        fragment.variables["s"] = VariableSpec("s", 32)
    t = Template("large", plates={"p": PlateSpec(fragment, (28, 28), (Broadcast(port, "hub"),))})
    arena = compile_template(t)
    groups = (*arena.groups, *arena.gauss_groups)
    assert sum(name.startswith("p/_bcast/hub/eq") for group in groups for name in group.members) == 783
    assert len([name for name in arena.edges if name.startswith("p/_bcast/hub/link")]) == 782
