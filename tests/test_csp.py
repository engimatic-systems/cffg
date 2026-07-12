"""Constraint / optimization tier (cff-e8zk): min-sum semiring + the hard-
constraint factor library (alldifferent, exactly_k/at_most_k, table). Every
target is brute-forceable so BP/max-product is pinned against enumeration:

  (a) Viterbi identity on trees      (e) exactly_k marginals + MAP
  (b) 4-rooks / alldifferent         (f) mixed soft+hard: MAP + BFE = -log Z
  (c) LatinSquare table constraint   (g) ledger API: bfe raises in max mode
  (d) 4x4 mini-Sudoku via decimation (h) bipartite matching exactness
                                     (i) f32 smoke + 100-var chain vs Viterbi DP
"""

import itertools

import numpy as np
import pytest

from cffg import csp
from cffg.compiler import CompileError, compile_template
from cffg.core import FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.ir import BlobRef
from cffg.engine import Engine, RunStatus


# --- builders ---------------------------------------------------------------


def _cat(name, out, blob):
    return FactorSpec(name, "categorical", edges={"out": out}, param_blobs={"statistic": blob})


def priors_plus_constraints(n, cards, priors, constraint, extra_blobs=None):
    """n edges x0..x{n-1} each carrying a categorical prior, tied into one
    constraint factor (degree 2 per edge). `cards` is per-edge cardinality."""
    if isinstance(cards, int):
        cards = [cards] * n
    variables = {f"x{i}": VariableSpec(f"x{i}", cards[i]) for i in range(n)}
    factors = {f"p{i}": _cat(f"p{i}", f"x{i}", f"d{i}") for i in range(n)}
    factors["c"] = constraint
    blobs = {f"d{i}": BlobRef(f"d{i}", f"d{i}.npy") for i in range(n)}
    if extra_blobs:
        blobs.update(extra_blobs)
    t = Template(name="cspmodel", variables=variables, factors=factors, blobs=blobs)
    arrays = {f"d{i}": priors[i] for i in range(n)}
    return t, arrays


def enumerate_marginals(n, cards, priors, feasible):
    """Exact marginals + log Z over the assignments passing `feasible`."""
    if isinstance(cards, int):
        cards = [cards] * n
    tot = [np.zeros(cards[i]) for i in range(n)]
    Z = 0.0
    for a in itertools.product(*(range(cards[i]) for i in range(n))):
        if not feasible(a):
            continue
        w = float(np.prod([priors[i][a[i]] for i in range(n)]))
        Z += w
        for i in range(n):
            tot[i][a[i]] += w
    return [t / Z for t in tot], np.log(Z)


def brute_map(n, cards, priors, feasible):
    if isinstance(cards, int):
        cards = [cards] * n
    best, best_w = None, -np.inf
    for a in itertools.product(*(range(cards[i]) for i in range(n))):
        if not feasible(a):
            continue
        w = float(np.prod([priors[i][a[i]] for i in range(n)]))
        if w > best_w:
            best_w, best = w, a
    return list(best), -np.log(best_w)


# --- (a) Viterbi identity on trees ------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_viterbi_identity_on_tree(seed):
    """Random categorical/matrix_cat tree, max mode: decode == brute-force MAP
    and objective == -log f at the MAP exactly."""
    rng = np.random.default_rng(seed)
    K = 3
    d0 = rng.dirichlet(np.ones(K))

    def colstoch():
        m = rng.uniform(0.1, 1.0, (K, K))
        return m / m.sum(0, keepdims=True)

    T1, T2, T3 = colstoch(), colstoch(), colstoch()
    # tree: prior on z0; equality branches z0 into z0a/z0b; z0a -> z1 (T1),
    # z0b -> z2 (T2), z2 -> z3 (T3). Branching goes through an equality node so
    # every edge keeps FFG degree <= 2.
    variables = {f"z{i}": VariableSpec(f"z{i}", K) for i in range(4)}
    variables["z0a"] = VariableSpec("z0a", K)
    variables["z0b"] = VariableSpec("z0b", K)
    factors = {
        "p": _cat("p", "z0", "d0"),
        "eq": FactorSpec("eq", "equality", edges={"a": "z0", "b": "z0a", "c": "z0b"}),
        "t1": FactorSpec("t1", "matrix_cat", edges={"out": "z1", "in": "z0a"}, param_blobs={"matrix": "T1"}),
        "t2": FactorSpec("t2", "matrix_cat", edges={"out": "z2", "in": "z0b"}, param_blobs={"matrix": "T2"}),
        "t3": FactorSpec("t3", "matrix_cat", edges={"out": "z3", "in": "z2"}, param_blobs={"matrix": "T3"}),
    }
    ports = {
        "b1": PortSpec("b1", "z1", PortRole.DATA),
        "b3": PortSpec("b3", "z3", PortRole.DATA),
    }
    t = Template("vtree", variables=variables, factors=factors, ports=ports,
                 blobs={b: BlobRef(b, b) for b in ("d0", "T1", "T2", "T3")})
    arrays = {"d0": d0, "T1": T1, "T2": T2, "T3": T3}
    eng = Engine(compile_template(t, arrays=arrays, semiring="max"))
    st = eng.run(eng.init_state(), 20, harden_last=0).state
    dec = eng.decode(st)

    best, best_w = None, -np.inf
    for a in itertools.product(range(K), repeat=4):
        w = d0[a[0]] * T1[a[1], a[0]] * T2[a[2], a[0]] * T3[a[3], a[2]]
        if w > best_w:
            best_w, best = w, a
    assert [dec[f"z{i}"] for i in range(4)] == list(best)
    np.testing.assert_allclose(eng.objective(st), -np.log(best_w), atol=1e-12)


# --- (b) 4-rooks: alldifferent permutation ----------------------------------


def _alldiff(n):
    return FactorSpec("c", "alldifferent", edges={f"x{i}": f"x{i}" for i in range(n)})


def test_alldifferent_marginals_and_map():
    """4 vars, domain 4 (256 states): sum-mode marginals match enumeration; the
    max-mode decode is a valid permutation; distinct solutions across restarts."""
    n = K = 4
    feasible = lambda a: len(set(a)) == n
    seen = set()
    for seed in range(6):
        rng = np.random.default_rng(seed)
        priors = [rng.dirichlet(np.ones(K)) for _ in range(n)]
        t, arrays = priors_plus_constraints(n, K, priors, _alldiff(n))

        eng = Engine(compile_template(t, arrays=arrays))
        st, deltas = eng.run(
            eng.init_state(), 40, tol=1e-13
        ).require_converged_with_trace()
        marg, logZ = enumerate_marginals(n, K, priors, feasible)
        for i in range(n):
            np.testing.assert_allclose(eng.marginals(st)[f"x{i}"], marg[i], atol=1e-9)
        np.testing.assert_allclose(eng.bfe(st), -logZ, atol=1e-9)

        engm = Engine(compile_template(t, arrays=arrays, semiring="max"))
        stm = engm.run(engm.init_state(), 40, harden_last=0).state
        dec = tuple(engm.decode(stm)[f"x{i}"] for i in range(n))
        assert len(set(dec)) == n  # a valid permutation (rooks non-attacking)
        assert dec == tuple(brute_map(n, K, priors, feasible)[0])
        seen.add(dec)
    assert 1 < len(seen) <= 24  # several distinct optima, at most 4! permutations


# --- (c) LatinSquare2-style table constraint --------------------------------


def test_table_constraint_latin_square():
    """A tiny 4-variable table (weighted) CSP: the MAP equals brute force.

    Vendored allowed-tuple set inline — the structure of toulbar2's LatinSquare2
    tutorial instance (a 2x2 Latin square read row-major as x0..x3, each row and
    column a permutation of {0,1}); source: toulbar2 CFN tutorial, LatinSquare2
    (github.com/toulbar2/toulbar2 web/TUTORIALS)."""
    n = 4
    cards = [2, 2, 2, 2]
    # 2x2 Latin squares (rows [x0 x1], [x2 x3]); rows and cols each a permutation
    allowed = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
    rng = np.random.default_rng(11)
    priors = [rng.dirichlet(np.ones(2)) for _ in range(n)]
    t, arrays = priors_plus_constraints(
        n, cards, priors,
        FactorSpec("c", "table", edges={f"x{i}": f"x{i}" for i in range(n)},
                   param_blobs={"table": "tab"}),
        extra_blobs={"tab": BlobRef("tab", "tab")},
    )
    arrays["tab"] = allowed
    feasible = lambda a: any(np.array_equal(a, row) for row in allowed)

    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.run(eng.init_state(), 20, tol=1e-13).require_converged()
    marg, logZ = enumerate_marginals(n, cards, priors, feasible)
    for i in range(n):
        np.testing.assert_allclose(eng.marginals(st)[f"x{i}"], marg[i], atol=1e-9)
    np.testing.assert_allclose(eng.bfe(st), -logZ, atol=1e-9)

    engm = Engine(compile_template(t, arrays=arrays, semiring="max"))
    stm = engm.run(engm.init_state(), 20, harden_last=0).state
    dec = [engm.decode(stm)[f"x{i}"] for i in range(n)]
    assert dec == brute_map(n, cards, priors, feasible)[0]


# --- (d) 4x4 mini-Sudoku via BP + decimation --------------------------------


def _sudoku_template(givens):
    """16 cells (domain 4): each cell fans through two equality factors into its
    row, column, and box alldifferent constraints, plus a `g` DATA port used to
    clamp givens. `givens` maps (r, c) -> value in 0..3."""
    K = 4
    variables, factors, ports, blobs = {}, {}, {}, {}
    row_edges = {r: [] for r in range(4)}
    col_edges = {c: [] for c in range(4)}
    box_edges = {b: [] for b in range(4)}
    for r in range(4):
        for c in range(4):
            for tag in ("r", "cc", "b", "tmp", "g"):
                v = f"{tag}_{r}_{c}"
                variables[v] = VariableSpec(v, K)
            factors[f"eq1_{r}_{c}"] = FactorSpec(
                f"eq1_{r}_{c}", "equality",
                edges={"a": f"r_{r}_{c}", "b": f"cc_{r}_{c}", "c": f"tmp_{r}_{c}"})
            factors[f"eq2_{r}_{c}"] = FactorSpec(
                f"eq2_{r}_{c}", "equality",
                edges={"a": f"tmp_{r}_{c}", "b": f"b_{r}_{c}", "c": f"g_{r}_{c}"})
            ports[f"g_{r}_{c}"] = PortSpec(f"g_{r}_{c}", f"g_{r}_{c}", PortRole.DATA)
            row_edges[r].append(f"r_{r}_{c}")
            col_edges[c].append(f"cc_{r}_{c}")
            box_edges[(r // 2) * 2 + c // 2].append(f"b_{r}_{c}")
    for group, edges in [("row", row_edges), ("col", col_edges), ("box", box_edges)]:
        for key, es in edges.items():
            factors[f"{group}{key}"] = FactorSpec(
                f"{group}{key}", "alldifferent",
                edges={f"x{i}": e for i, e in enumerate(es)})
    t = Template("sudoku4", variables=variables, factors=factors, ports=ports, blobs=blobs)
    obs = {f"g_{r}_{c}": givens.get((r, c)) for r in range(4) for c in range(4)}
    return t, obs


def _brute_sudoku(givens):
    cells = [(r, c) for r in range(4) for c in range(4)]

    def ok(g):
        for i in range(4):
            if len(set(g[i, :])) < 4 or len(set(g[:, i])) < 4:
                return False
        for br in (0, 2):
            for bc in (0, 2):
                if len(set(g[br:br + 2, bc:bc + 2].flatten())) < 4:
                    return False
        return True

    sols = []

    def bt(g, idx):
        if idx == 16:
            if ok(g):
                sols.append(g.copy())
            return
        r, c = cells[idx]
        if (r, c) in givens:
            g[r, c] = givens[(r, c)]
            bt(g, idx + 1)
        else:
            for v in range(4):
                g[r, c] = v
                bt(g, idx + 1)
        g[r, c] = -1

    bt(-np.ones((4, 4), dtype=int), 0)
    return sols


def test_mini_sudoku_decimation():
    solution = np.array([[0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2], [3, 2, 1, 0]])
    givens = {(0, 0): 0, (0, 2): 2, (1, 1): 3, (1, 3): 1,
              (2, 0): 1, (2, 2): 3, (3, 1): 2, (3, 3): 0}
    sols = _brute_sudoku(givens)
    assert len(sols) == 1 and np.array_equal(sols[0], solution)  # unique in-test

    t, obs = _sudoku_template(givens)
    eng = Engine(compile_template(t))
    st = eng.bind_data(eng.init_state(), obs)
    st, decoded = csp.decimate(eng, st, n_sweeps=30, tol=1e-10)
    grid = np.array([[eng.decode(st)[f"g_{r}_{c}"] for c in range(4)] for r in range(4)])
    assert np.array_equal(grid, solution)


# --- (e) exactly_k ----------------------------------------------------------


def test_exactly_k_marginals_and_map():
    n, k = 6, 3
    rng = np.random.default_rng(21)
    priors = [rng.dirichlet(np.ones(2)) for _ in range(n)]
    t, arrays = priors_plus_constraints(
        n, 2, priors,
        FactorSpec("c", "exactly_k", edges={f"x{i}": f"x{i}" for i in range(n)},
                   param_blobs={"k": "kk"}),
        extra_blobs={"kk": BlobRef("kk", "kk")},
    )
    arrays["kk"] = np.array(k)
    feasible = lambda a: sum(a) == k

    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.run(eng.init_state(), 40, tol=1e-13).require_converged()
    marg, logZ = enumerate_marginals(n, 2, priors, feasible)
    for i in range(n):
        np.testing.assert_allclose(eng.marginals(st)[f"x{i}"], marg[i], atol=1e-9)
    np.testing.assert_allclose(eng.bfe(st), -logZ, atol=1e-9)

    engm = Engine(compile_template(t, arrays=arrays, semiring="max"))
    stm = engm.run(engm.init_state(), 40, harden_last=0).state
    assert [engm.decode(stm)[f"x{i}"] for i in range(n)] == brute_map(n, 2, priors, feasible)[0]


def test_at_most_k_marginals():
    n, k = 5, 2
    rng = np.random.default_rng(22)
    priors = [rng.dirichlet(np.ones(2)) for _ in range(n)]
    t, arrays = priors_plus_constraints(
        n, 2, priors,
        FactorSpec("c", "at_most_k", edges={f"x{i}": f"x{i}" for i in range(n)},
                   param_blobs={"k": "kk"}),
        extra_blobs={"kk": BlobRef("kk", "kk")},
    )
    arrays["kk"] = np.array(k)
    feasible = lambda a: sum(a) <= k
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.run(eng.init_state(), 40, tol=1e-13).require_converged()
    marg, logZ = enumerate_marginals(n, 2, priors, feasible)
    for i in range(n):
        np.testing.assert_allclose(eng.marginals(st)[f"x{i}"], marg[i], atol=1e-9)
    np.testing.assert_allclose(eng.bfe(st), -logZ, atol=1e-9)


# --- (f) mixed soft + hard: MAP and BFE = -log Z ----------------------------


def test_mixed_soft_hard_tree():
    """Soft categorical priors + one alldifferent hub (a tree: the constraint
    node is the single cycle-free hub). MAP matches enumeration and the sum-mode
    BFE equals -log Z of the constrained model exactly."""
    n = K = 4
    rng = np.random.default_rng(31)
    priors = [rng.dirichlet(np.ones(K)) for _ in range(n)]
    t, arrays = priors_plus_constraints(n, K, priors, _alldiff(n))
    feasible = lambda a: len(set(a)) == n

    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.run(eng.init_state(), 40, tol=1e-13).require_converged()
    marg, logZ = enumerate_marginals(n, K, priors, feasible)
    np.testing.assert_allclose(eng.bfe(st), -logZ, atol=1e-9)

    engm = Engine(compile_template(t, arrays=arrays, semiring="max"))
    stm = engm.run(engm.init_state(), 40, harden_last=0).state
    _, map_energy = brute_map(n, K, priors, feasible)
    np.testing.assert_allclose(engm.objective(stm), map_energy, atol=1e-12)


# --- (g) ledger API ---------------------------------------------------------


def test_bfe_raises_in_max_mode():
    n = K = 4
    rng = np.random.default_rng(41)
    priors = [rng.dirichlet(np.ones(K)) for _ in range(n)]
    t, arrays = priors_plus_constraints(n, K, priors, _alldiff(n))
    engm = Engine(compile_template(t, arrays=arrays, semiring="max"))
    stm = engm.run(engm.init_state(), 20, harden_last=0).state
    with pytest.raises(RuntimeError, match="objective value, not a free energy"):
        engm.bfe(stm)
    # objective is meaningful in both semirings — sum mode decodes too.
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.run(eng.init_state(), 20, tol=1e-13).require_converged()
    assert isinstance(eng.objective(st), float)


def test_max_mode_rejects_vmp_and_gaussian():
    from cffg.models.kalman import lgssm

    tk, ar = lgssm(2, np.eye(1), np.eye(1), np.eye(1), np.eye(1), np.zeros(1), np.eye(1))
    with pytest.raises(CompileError, match="Gaussian"):
        compile_template(tk, arrays=ar, semiring="max")


# --- (h) bipartite matching exactness (Bayati-Shah-Sharma) ------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_bipartite_matching_exact(seed):
    """K_{4,4} with random distinct weights as a mixed model (soft unary weights
    + alldifferent): max-product decode == the optimal assignment (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    n = K = 4
    rng = np.random.default_rng(100 + seed)
    W = rng.uniform(0.0, 5.0, (n, K))  # W[i, j] = weight of matching left i to right j
    # unary priors ∝ exp(W): max Π priors[i][x_i] == max Σ W subject to permutation
    priors = [np.exp(W[i]) / np.exp(W[i]).sum() for i in range(n)]
    t, arrays = priors_plus_constraints(n, K, priors, _alldiff(n))

    engm = Engine(compile_template(t, arrays=arrays, semiring="max"))
    stm = engm.run(engm.init_state(), 40, harden_last=0).state
    dec = [engm.decode(stm)[f"x{i}"] for i in range(n)]
    _, opt = linear_sum_assignment(-W)  # max-weight assignment
    assert dec == list(opt)


# --- (i) scale + dtype ------------------------------------------------------


def test_f32_smoke_alldifferent():
    """Runs under JAX's default float32 without NaNs (the _norm_pad / NEG floor
    survive single precision on a constraint node)."""
    import jax

    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        n = K = 4
        rng = np.random.default_rng(7)
        priors = [rng.dirichlet(np.ones(K)).astype(np.float32) for _ in range(n)]
        t, arrays = priors_plus_constraints(n, K, priors, _alldiff(n))
        eng = Engine(compile_template(t, arrays=arrays))
        st = eng.run(eng.init_state(), 30).state
        for i in range(n):
            assert np.all(np.isfinite(eng.marginals(st)[f"x{i}"]))
        assert np.isfinite(eng.bfe(st))
    finally:
        jax.config.update("jax_enable_x64", was)


def test_100_variable_chain_vs_viterbi_dp():
    """A length-100 categorical/matrix_cat chain at max mode decodes to the same
    path as a plain numpy Viterbi DP (scale case, tree, exact)."""
    n, K = 100, 4
    rng = np.random.default_rng(3)
    d0 = rng.dirichlet(np.ones(K))
    Ts = [rng.uniform(0.1, 1.0, (K, K)) for _ in range(n - 1)]
    Ts = [T / T.sum(0, keepdims=True) for T in Ts]

    variables = {f"z{i}": VariableSpec(f"z{i}", K) for i in range(n)}
    factors = {"p": _cat("p", "z0", "d0")}
    blobs = {"d0": BlobRef("d0", "d0")}
    for i in range(n - 1):
        factors[f"t{i}"] = FactorSpec(
            f"t{i}", "matrix_cat", edges={"out": f"z{i + 1}", "in": f"z{i}"},
            param_blobs={"matrix": f"T{i}"})
        blobs[f"T{i}"] = BlobRef(f"T{i}", f"T{i}")
    ports = {"last": PortSpec("last", f"z{n - 1}", PortRole.DATA)}
    t = Template("chain100", variables=variables, factors=factors, ports=ports, blobs=blobs)
    arrays = {"d0": d0, **{f"T{i}": Ts[i] for i in range(n - 1)}}
    eng = Engine(compile_template(t, arrays=arrays, semiring="max"))
    result = eng.run(eng.init_state(), 2 * n, tol=1e-12, harden_last=0)
    assert result.status is RunStatus.CONVERGED
    st = result.state
    dec = [eng.decode(st)[f"z{i}"] for i in range(n)]

    # numpy Viterbi in log space
    logd = np.log(d0)
    logT = [np.log(T) for T in Ts]
    delta = logd.copy()
    back = []
    for i in range(n - 1):
        scores = logT[i] + delta[None, :]  # [K_next, K_prev]
        back.append(np.argmax(scores, axis=1))
        delta = scores.max(axis=1)
    path = [int(np.argmax(delta))]
    for i in range(n - 2, -1, -1):
        path.append(int(back[i][path[-1]]))
    path = path[::-1]
    assert dec == path


# --- viz --------------------------------------------------------------------


def test_constraint_glyphs_and_diagram(viz_gallery):
    from cffg.viz import derive
    from cffg.viz.diagram import GLYPHS

    assert GLYPHS["alldifferent"] == "!="
    n = K = 4
    rng = np.random.default_rng(0)
    priors = [rng.dirichlet(np.ones(K)) for _ in range(n)]
    t, _ = priors_plus_constraints(n, K, priors, _alldiff(n))
    d = derive(t)
    assert d.nodes["c"].glyph == "!="
    viz_gallery(t, tag="alldifferent over four categorical priors")
