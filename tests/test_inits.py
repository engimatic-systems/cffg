"""Initializer templates (cff-wx28): the warm-start vocabulary.

Property checks per kind, the counts-strength knob, serialization/hash
semantics, and the end-to-end promise: a template whose blobs carry init
specs compiles and runs with no arrays and no files.
"""

import numpy as np
import pytest

from cffg import inits
from cffg.compiler import CompileError, compile_template
from cffg.core.serialize import from_json, structure_hash, to_json


def cols_stochastic(m):
    np.testing.assert_allclose(m.sum(axis=-2), 1.0, atol=1e-12)


def test_uniform_and_one_hot():
    cols_stochastic(inits.make("uniform", (4, 4)))
    v = inits.make("one_hot", (5,), index=3)
    assert v[3] == 1.0 and v.sum() == 1.0
    m = inits.make("one_hot", (5, 5), index=2)
    cols_stochastic(m)
    assert (m[2] == 1.0).all()


def test_sticky_and_banded():
    m = inits.make("sticky", (6, 6), p_stay=0.8)
    cols_stochastic(m)
    np.testing.assert_allclose(np.diag(m), 0.8)
    b = inits.make("banded", (7, 7), sigma=1.2)
    cols_stochastic(b)
    # mass decays monotonically away from the diagonal
    assert b[0, 0] > b[1, 0] > b[2, 0] > b[3, 0]
    ring = inits.make("banded", (7, 7), sigma=1.2, wrap=True)
    np.testing.assert_allclose(ring[6, 0], ring[1, 0])  # wrap: distance 1 both ways


def test_shift():
    m = inits.make("shift", (5, 5), offset=1, p_move=0.85)
    cols_stochastic(m)
    assert m[1, 0] == 0.85 and m[0, 0] == pytest.approx(0.15)
    assert m[4, 4] == 1.0  # clamped at the edge: move + stay coincide
    w = inits.make("shift", (5, 5), offset=1, p_move=1.0, wrap=True)
    assert w[0, 4] == 1.0


def test_stacked_leading_axes():
    m = inits.make("sticky", (3, 6, 6), p_stay=0.7)  # transition_mixture layout
    assert m.shape == (3, 6, 6)
    for u in range(3):
        cols_stochastic(m[u])
        np.testing.assert_allclose(m[u], m[0])


def test_random_kinds_are_seeded():
    a = inits.make("random_stochastic", (4, 4), seed=7, concentration=0.5)
    b = inits.make("random_stochastic", (4, 4), seed=7, concentration=0.5)
    c = inits.make("random_stochastic", (4, 4), seed=8, concentration=0.5)
    np.testing.assert_array_equal(a, b)
    assert np.abs(a - c).max() > 1e-3
    cols_stochastic(a)
    s = inits.make("random_spd", (2, 3, 3), seed=1)
    for i in range(2):
        np.testing.assert_allclose(s[i], s[i].T)
        assert np.linalg.eigvalsh(s[i]).min() > 0


def test_strength_turns_templates_into_counts():
    counts = inits.make("banded", (5, 5), sigma=1.0, strength=20.0)
    assert (counts > 0).all()  # floored: valid Dirichlet counts
    recovered = counts / counts.sum(axis=0, keepdims=True)
    soft = inits.make("banded", (5, 5), sigma=1.0)
    assert np.abs(recovered - soft).max() < 0.06  # floor blurs slightly
    with pytest.raises(ValueError, match="not a probability template"):
        inits.make("eye", (3, 3), strength=5.0)
    with pytest.raises(ValueError, match="floor"):
        inits.make("uniform", (3,), strength=5.0, floor=0.0)


def test_spec_round_trips_through_json_and_hash():
    from cffg.core.ir import BlobRef

    from conftest import make_hmm_slice

    t = make_hmm_slice()
    plain_hash = structure_hash(t)
    t.blobs["B"] = BlobRef("B", "blobs/B.safetensors", inits.spec("sticky", p_stay=0.9))
    back = from_json(to_json(t))
    assert back.blobs["B"].init == t.blobs["B"].init
    arr = inits.from_spec(back.blobs["B"].init, (3, 3))
    np.testing.assert_allclose(np.diag(arr), 0.9)
    # init specs are parameter data: the structure hash must not move
    assert structure_hash(t) == plain_hash


def test_compile_from_init_specs_alone(tmp_path):
    """The self-containment promise: no arrays, no files — the template's own
    init specs produce a runnable model, and a file at the blob path wins
    over the spec when present (checkpoint-over-warm-start)."""
    from cffg.core.ir import BlobRef
    from cffg.engine import Engine

    from conftest import make_hmm_slice
    from cffg.core.compose import compose
    from cffg.core.ir import FactorSpec, Template, VariableSpec

    t = make_hmm_slice()
    t.blobs["A"] = BlobRef("A", "A.npy", inits.spec("random_stochastic", seed=3))
    t.blobs["B"] = BlobRef("B", "B.npy", inits.spec("banded", sigma=0.8))

    prior = Template(
        name="prior",
        variables={"z": VariableSpec("z", 3)},
        factors={"p": FactorSpec("p", "categorical", edges={"out": "z"}, param_blobs={"statistic": "d"})},
        blobs={"d": BlobRef("d", "d.npy", inits.spec("uniform"))},
    )
    from cffg.core.ir import PortRole, PortSpec

    prior.ports["attach"] = PortSpec("attach", "z", PortRole.GLUE)
    model = compose(prior, t, {"attach": "head"}, name="m", scope_a="p", scope_b="s")

    engine = Engine(compile_template(model, root=tmp_path))
    state = engine.bind_data(engine.init_state(), {"s/sense": 1})
    state, _ = engine.run(state, 20, tol=1e-12)
    q = engine.marginals(state)["s/z_mid"]
    assert np.isfinite(engine.bfe(state)) and abs(q.sum() - 1.0) < 1e-9

    # A file at the blob path now overrides the init spec.
    banded = inits.make("banded", (3, 3), sigma=0.8)
    np.save(tmp_path / "B.npy", np.roll(np.eye(3), 1, axis=0))  # different dynamics
    engine2 = Engine(compile_template(model, root=tmp_path))
    st2 = engine2.bind_data(engine2.init_state(), {"s/sense": 1})
    st2, _ = engine2.run(st2, 20, tol=1e-12)
    assert np.abs(engine2.marginals(st2)["s/z_mid"] - q).max() > 1e-3
    del banded


def test_missing_everything_errors_clearly(tmp_path):
    from conftest import make_hmm_slice
    from cffg.core.ir import PortRole

    t = make_hmm_slice()
    for p in list(t.ports.values()):
        if p.role is PortRole.GLUE:
            pass
    with pytest.raises(CompileError, match="no init spec"):
        compile_template(t, root=tmp_path)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown initializer"):
        inits.make("mystery", (3,))
    with pytest.raises(ValueError, match="unknown initializer"):
        inits.spec("mystery")
