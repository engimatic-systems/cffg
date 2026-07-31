"""Hybrid GFE epistemics: closed forms + decision experiments (cff-dfpi, D5).

The research spike behind DERIVATIONS.md D5. Everything here is standalone
NumPy implementing the derived formulas rather than engine plumbing. These
functions are the independent executable specification against which the
P-substituted hybrid mixture composite is tested.

Objects (D5.2 notation):

  regime belief      rho[s],                    s = 1..K   (discrete)
  state beliefs      q(z|s) = N(ms[s], Ss[s])              (Gaussian, d_z)
  emission           f(x|z,s) = N(x; As[s] z + bs[s], Vs[s])   (d_x)
  goal               p~(x) = N(g, Gamma)

  predictive component s:  xbar[s] = As[s] ms[s] + bs[s]
                           P[s]    = As[s] Ss[s] As[s]' + Vs[s]
  predictive           q_w(x) = sum_s rho[s] N(x; xbar[s], P[s])   (mixture)

The single non-closed-form quantity in the substituted node objective is the
mixture entropy H[q_w]. The surrogate used throughout is the moment-matched
(collapsed) Gaussian entropy — an upper bound on H[q_w], hence the reported
GFE lower-bounds (is optimistic about) the exact substituted objective. The
bound direction and the full sandwich are pinned in
tests/test_hybrid_epistemics.py against grid quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOG_2PI = float(np.log(2.0 * np.pi))


# --- predictive mixture ------------------------------------------------------


@dataclass
class Predictive:
    """The substituted observation predictive: a labelled Gaussian mixture."""

    rho: np.ndarray  # [K]
    xbar: np.ndarray  # [K, d]
    P: np.ndarray  # [K, d, d]


def predictive(rho, ms, Ss, As, bs, Vs) -> Predictive:
    """Push per-regime state beliefs through per-regime linear emissions."""
    rho, ms, Ss = np.asarray(rho), np.asarray(ms), np.asarray(Ss)
    As, bs, Vs = np.asarray(As), np.asarray(bs), np.asarray(Vs)
    xbar = np.einsum("sij,sj->si", As, ms) + bs
    P = np.einsum("sij,sjk,slk->sil", As, Ss, As) + Vs
    return Predictive(rho=rho, xbar=xbar, P=P)


def gauss_entropy(P) -> float:
    """H[N(., P)] = 0.5 logdet(2*pi*e*P)."""
    P = np.atleast_2d(np.asarray(P))
    d = P.shape[-1]
    sign, logdet = np.linalg.slogdet(P)
    if np.any(sign <= 0):
        return float("inf")  # improper belief: loud, matching engine convention
    return float(0.5 * (logdet + d * (LOG_2PI + 1.0)))


def collapse(pred: Predictive):
    """Moment-matched single Gaussian (the D3 collapse of q_w)."""
    m = pred.rho @ pred.xbar
    scatter = np.einsum("s,si,sj->ij", pred.rho, pred.xbar, pred.xbar) - np.outer(m, m)
    return m, np.einsum("s,sij->ij", pred.rho, pred.P) + scatter


# --- mixture entropy: surrogate and sandwich (D5.2 Prop. 3) ------------------


def entropy_collapse_upper(pred: Predictive) -> float:
    """H[q_w] <= H[N(collapse)]: Gaussian is max-entropy at fixed moments."""
    return gauss_entropy(collapse(pred)[1])


def entropy_joint_upper(pred: Predictive) -> float:
    """H[q_w] <= H(rho) + sum_s rho_s H[component s]  (H[x] <= H[x,s])."""
    return discrete_entropy(pred.rho) + float(
        sum(r * gauss_entropy(P) for r, P in zip(pred.rho, pred.P))
    )


def entropy_conditional_lower(pred: Predictive) -> float:
    """H[q_w] >= H[x|s]: conditioning reduces entropy."""
    return float(sum(r * gauss_entropy(P) for r, P in zip(pred.rho, pred.P)))


def entropy_pairwise_lower(pred: Predictive) -> float:
    """Hershey-Olsen: H[q_w] >= -sum_s rho_s log sum_t rho_t N(xbar_s; xbar_t, P_s+P_t)."""
    K, d = pred.xbar.shape
    out = 0.0
    for s in range(K):
        z = 0.0
        for t in range(K):
            z += pred.rho[t] * _gauss_pdf(pred.xbar[s], pred.xbar[t], pred.P[s] + pred.P[t])
        out -= pred.rho[s] * np.log(z)
    return float(out)


def entropy_quadrature(pred: Predictive, span: float = 12.0, n: int = 20001) -> float:
    """Grid quadrature of H[q_w] for d_x = 1 — the test oracle, never the engine."""
    assert pred.xbar.shape[1] == 1
    sd = np.sqrt(pred.P[:, 0, 0])
    lo = float(np.min(pred.xbar[:, 0] - span * sd))
    hi = float(np.max(pred.xbar[:, 0] + span * sd))
    x = np.linspace(lo, hi, n)
    dens = np.zeros_like(x)
    for r, m, v in zip(pred.rho, pred.xbar[:, 0], pred.P[:, 0, 0]):
        dens += r * np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
    mask = dens > 0
    return float(-np.trapezoid(dens[mask] * np.log(dens[mask]), x[mask]))


def _gauss_pdf(x, m, P):
    d = x.shape[0]
    sign, logdet = np.linalg.slogdet(P)
    r = x - m
    return np.exp(-0.5 * (d * LOG_2PI + logdet + r @ np.linalg.solve(P, r)))


def discrete_entropy(rho) -> float:
    rho = np.asarray(rho)
    nz = rho[rho > 0]
    return float(-np.sum(nz * np.log(nz)))


# --- the two epistemic channels (D5.2 Prop. 2) -------------------------------


def mi_state_given_regime(Ss, As, Vs) -> np.ndarray:
    """I(x; z | s) = 0.5 (logdet P_s - logdet V_s), exact, per regime."""
    Ss, As, Vs = np.asarray(Ss), np.asarray(As), np.asarray(Vs)
    P = np.einsum("sij,sjk,slk->sil", As, Ss, As) + Vs
    sp, ldp = np.linalg.slogdet(P)
    sv, ldv = np.linalg.slogdet(Vs)
    # improper covariances are loud (+inf), never a signed-logdet artefact
    return np.where((sp > 0) & (sv > 0), 0.5 * (ldp - ldv), np.inf)


def mi_regime_hat(pred: Predictive) -> float:
    """Surrogate I^(x; s) = H[collapse] - H[x|s]  (>= exact I(x; s) >= 0)."""
    return entropy_collapse_upper(pred) - entropy_conditional_lower(pred)


def mi_regime_bounds(pred: Predictive):
    """(lower, upper) sandwich on the exact I(x; s); upper includes the H(rho) cap."""
    hx_s = entropy_conditional_lower(pred)
    lower = max(0.0, entropy_pairwise_lower(pred) - hx_s)
    upper = min(mi_regime_hat(pred), discrete_entropy(pred.rho))
    return lower, upper


def grad_collapse_entropy(pred: Predictive) -> np.ndarray:
    """d H[N(collapse)] / d rho_s (unconstrained partials; softmax absorbs the shift)."""
    m, Pbar = collapse(pred)
    inv = np.linalg.inv(Pbar)
    out = np.empty(len(pred.rho))
    for s in range(len(pred.rho)):
        dP = (
            pred.P[s]
            + np.outer(pred.xbar[s], pred.xbar[s])
            - np.outer(pred.xbar[s], m)
            - np.outer(m, pred.xbar[s])
        )
        out[s] = 0.5 * np.trace(inv @ dP)
    return out


# --- goal term (exact) and the assembled objective (D5.2 Prop. 1) ------------


def goal_cross_entropy(pred: Predictive, g, Gamma) -> float:
    """E_{q_w}[-log N(x; g, Gamma)] — exact for a mixture (quadratic in x)."""
    g, Gamma = np.atleast_1d(g), np.atleast_2d(Gamma)
    d = g.shape[0]
    inv = np.linalg.inv(Gamma)
    quad = 0.0
    for r, m, P in zip(pred.rho, pred.xbar, pred.P):
        e = m - g
        quad += r * (e @ inv @ e + np.trace(inv @ P))
    return float(0.5 * (d * LOG_2PI + np.linalg.slogdet(Gamma)[1] + quad))


@dataclass
class GfeTerms:
    """Substituted node-local objective, term by term (D5.2 Prop. 1-2).

    total = -(mi_regime_hat + mi_state) - (h_regime + h_state) + goal_risk

    mi_regime_hat is the one surrogate (collapse upper bound on I(x; s), so
    `total` lower-bounds the exact substituted objective); every other term is
    exact. mi_regime_exact_bounds carries the diagnostic sandwich.
    """

    mi_regime_hat: float
    mi_state: float
    h_regime: float
    h_state: float
    goal_risk: float
    mi_regime_exact_bounds: tuple[float, float]

    @property
    def epistemic(self) -> float:
        return self.mi_regime_hat + self.mi_state

    @property
    def total(self) -> float:
        return -self.epistemic - self.h_regime - self.h_state + self.goal_risk


def gfe_hat(rho, ms, Ss, As, bs, Vs, g, Gamma) -> GfeTerms:
    """The derived node-local GFE of the P-substituted hybrid composite."""
    pred = predictive(rho, ms, Ss, As, bs, Vs)
    mi_z = float(np.asarray(rho) @ mi_state_given_regime(Ss, As, Vs))
    h_state = float(sum(r * gauss_entropy(S) for r, S in zip(rho, np.asarray(Ss))))
    return GfeTerms(
        mi_regime_hat=mi_regime_hat(pred),
        mi_state=mi_z,
        h_regime=discrete_entropy(rho),
        h_state=h_state,
        goal_risk=goal_cross_entropy(pred, g, Gamma),
        mi_regime_exact_bounds=mi_regime_bounds(pred),
    )


def gfe_exact_quadrature(rho, ms, Ss, As, bs, Vs, g, Gamma) -> float:
    """Exact substituted objective for d_x = 1 (mixture entropy by quadrature)."""
    pred = predictive(rho, ms, Ss, As, bs, Vs)
    h_qw = entropy_quadrature(pred)
    h_x_given_zs = float(sum(r * gauss_entropy(V) for r, V in zip(rho, np.asarray(Vs))))
    h_state = float(sum(r * gauss_entropy(S) for r, S in zip(rho, np.asarray(Ss))))
    return (
        -h_qw
        + h_x_given_zs
        - discrete_entropy(rho)
        - h_state
        + goal_cross_entropy(pred, g, Gamma)
    )


# --- stationarity over the regime simplex (D5.2 Prop. 4) ---------------------


def regime_log_potential(rho, ms, Ss, As, bs, Vs, g, Gamma) -> np.ndarray:
    """r_s(rho): the substituted node's log-potential over regimes.

    Stationarity of  G^[rho] - sum_s rho_s mu_e(s)  on the simplex is
    rho = softmax(r(rho) + mu_e), with
        r_s = dH^/drho_s - h_s + H[q(z|s)] - c_s ,
    h_s the emission entropy and c_s the per-regime goal quadratic.
    """
    pred = predictive(rho, ms, Ss, As, bs, Vs)
    g, Gamma = np.atleast_1d(g), np.atleast_2d(Gamma)
    inv = np.linalg.inv(Gamma)
    h_emit = np.array([gauss_entropy(V) for V in np.asarray(Vs)])
    h_z = np.array([gauss_entropy(S) for S in np.asarray(Ss)])
    c = np.array(
        [
            0.5 * ((m - g) @ inv @ (m - g) + np.trace(inv @ P))
            for m, P in zip(pred.xbar, pred.P)
        ]
    )
    return grad_collapse_entropy(pred) - h_emit + h_z - c


def solve_regime_fixed_point(
    mu_e, ms, Ss, As, bs, Vs, g, Gamma, damp: float = 0.5, iters: int = 500, tol: float = 1e-12
):
    """Damped fixed-point iteration for rho = softmax(r(rho) + mu_e).

    K is tiny (regimes/actions); damped iteration suffices for the spike. The
    engine version inherits the D1 Newton solver with its status contract.
    Returns (rho, residual, converged).
    """
    mu_e = np.asarray(mu_e, dtype=float)
    rho = np.full(len(mu_e), 1.0 / len(mu_e))
    resid = np.inf
    for _ in range(iters):
        logits = regime_log_potential(rho, ms, Ss, As, bs, Vs, g, Gamma) + mu_e
        target = _softmax(logits)
        resid = float(np.max(np.abs(target - rho)))
        rho = (1 - damp) * target + damp * rho
        if resid < tol:
            break
    return rho, resid, bool(resid < tol)


def _softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


# --- negative control: additive-control LGSSM (D5.2 Prop. 5) -----------------


def lgssm_policy_terms(F, Bmat, Q, C, R, m0, S0, controls, g, Gamma) -> list[GfeTerms]:
    """Per-slice GFE terms for one policy in an additive-control LGSSM.

    z' = F z + B u + w,  x = C z' + e — evaluated through the *general* hybrid
    formulas with K = 1, so the policy-independence of every non-goal term is
    exercised through the same code path as the mixture case, not assumed.
    """
    F, Bmat, Q = np.asarray(F), np.asarray(Bmat), np.asarray(Q)
    C, R = np.asarray(C), np.asarray(R)
    m, S = np.asarray(m0, dtype=float), np.asarray(S0, dtype=float)
    out = []
    for u in np.asarray(controls, dtype=float):
        m = F @ m + Bmat @ u
        S = F @ S @ F.T + Q
        out.append(
            gfe_hat(
                np.array([1.0]),
                m[None],
                S[None],
                C[None],
                np.zeros(C.shape[0])[None],
                R[None],
                g,
                Gamma,
            )
        )
    return out


# --- the behavioural experiment (D5.3): probe-then-commit --------------------
#
# A plant is in one of two calibration regimes s* (prior 1/2 each); its scalar
# latent state z ~ N(0, S0) with *known* linear dynamics. Step 1 the agent may
# PROBE (emission separates the regimes: x = a z +/- 1 + small noise) or IDLE
# (emission identical across regimes: no regime information). Step 2 it must
# COMMIT to a regime; the outcome reads N(+1, sr^2) if correct, N(-1, sr^2) if
# wrong, against goal N(+1, sg^2). Probing carries slightly *higher* step-1
# risk than idling, so a KL-control (no-MI) scorer never probes: any probing
# is purposeful exploration, attributable to the epistemic term alone.


@dataclass
class ProbeWorld:
    a: float = 0.3  # emission gain on z
    S0: float = 0.25  # prior state variance
    v_probe: float = 0.05**2
    v_idle: float = 0.5  # idle risk strictly below probe risk: probing is never
    # the KL-control choice, so any probing is attributable to the MI term
    b_probe: tuple[float, float] = (1.0, -1.0)
    sr: float = 0.1  # commit-outcome noise sd
    sg: float = 0.2  # goal sd on the outcome
    g1: float = 0.0  # broad step-1 goal
    gamma1: float = 10.0**2

    def emission(self, action: str):
        """Per-regime (A, b, V) arrays for the step-1 observation."""
        b = np.array(self.b_probe) if action == "probe" else np.zeros(2)
        v = self.v_probe if action == "probe" else self.v_idle
        As = np.full((2, 1, 1), self.a)
        return As, b[:, None], np.full((2, 1, 1), v)

    def state_belief(self):
        return np.zeros((2, 1)), np.full((2, 1, 1), self.S0)


def step1_scores(world: ProbeWorld, rho, epistemic: bool = True) -> dict[str, float]:
    """Two-slice policy scores. The commit slice is mean-field (no contingency,
    LIMITATIONS H5), so its terms are identical across step-1 actions and only
    the step-1 slice separates the policies — included anyway for honesty."""
    ms, Ss = world.state_belief()
    scores = {}
    commit = commit_risk(world, rho)  # action-invariant under mean-field
    for action in ("probe", "idle"):
        As, bs, Vs = world.emission(action)
        t = gfe_hat(rho, ms, Ss, As, bs, Vs, world.g1, world.gamma1)
        scores[action] = (t.total if epistemic else t.goal_risk) + commit
    return scores


def commit_risk(world: ProbeWorld, rho) -> float:
    """Best-commit expected goal cross-entropy under the current regime belief."""
    risks = []
    for c in (0, 1):
        mu = np.where(np.arange(2) == c, 1.0, -1.0)
        quad = sum(r * ((m - 1.0) ** 2 + world.sr**2) for r, m in zip(rho, mu))
        risks.append(0.5 * (LOG_2PI + np.log(world.sg**2) + quad / world.sg**2))
    return float(min(risks))


def posterior_regime(world: ProbeWorld, rho, action: str, x: float) -> np.ndarray:
    """Exact Bayes update of the regime belief from the step-1 observation."""
    ms, Ss = world.state_belief()
    As, bs, Vs = world.emission(action)
    pred = predictive(rho, ms, Ss, As, bs, Vs)
    like = np.array(
        [_gauss_pdf(np.array([x]), m, P) for m, P in zip(pred.xbar, pred.P)]
    )
    post = rho * like
    return post / post.sum()


def run_probe_experiment(n_episodes: int = 200, seed: int = 0, epistemic: bool = True):
    """Perception-action loop; returns probe rate, success rate, satiation trace."""
    world = ProbeWorld()
    rng = np.random.default_rng(seed)
    probes = successes = 0
    post_probe_mi = []
    for _ in range(n_episodes):
        s_true = int(rng.integers(2))
        rho = np.array([0.5, 0.5])
        scores = step1_scores(world, rho, epistemic=epistemic)
        action = min(scores, key=scores.get)
        if action == "probe":
            probes += 1
            z = rng.normal(0.0, np.sqrt(world.S0))
            As, bs, Vs = world.emission(action)
            x = float(
                As[s_true, 0, 0] * z + bs[s_true, 0] + rng.normal(0, np.sqrt(Vs[s_true, 0, 0]))
            )
            rho = posterior_regime(world, rho, action, x)
            ms, Ss = world.state_belief()
            post_probe_mi.append(mi_regime_hat(predictive(rho, ms, Ss, As, bs, Vs)))
        commit = int(np.argmax(rho))
        successes += int(commit == s_true)
    return {
        "probe_rate": probes / n_episodes,
        "success_rate": successes / n_episodes,
        "post_probe_mi": post_probe_mi,
    }
