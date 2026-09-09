# Derivations

Formal derivations of message updates and free-energy terms that go beyond
the papers, each paired with a plain-English companion for readers who know
active inference practically (e.g. via pymdp) but don't want to fight the
notation. One section per derivation; update in place; every claim that can
be executable is pinned by a named test.

---

## D1 — Dynamics curiosity: P-substitution on the transition node (cff-fk2u)

### D1.1 The plain-English version

**What pymdp does.** When you enable `pB` learning in pymdp, policies get an
extra additive score — the "B-novelty" bonus (the `spm_wnorm` term): policies
that visit state-action pairs whose transition counts are low get a bump.
It works, but it is *bolted on*: a separate quantity computed outside the
inference machinery and added to the expected free energy by hand.

**Why CFFG can't just bolt it on.** This engine's contract is that every
number comes out of one variational objective (the constrained Bethe free
energy) and its message-passing fixed points. A bonus term with no place in
the objective would corrupt the FE ledger — the quantity we use for
convergence, model comparison, and policy scoring.

**The papers' one trick, in plain terms.** For *future observations*, the
LAIF papers use P-substitution. Ordinary inference treats a variable's
posterior as something to be inferred from evidence — but a future
observation has no evidence yet, and pretending it does means smoothing your
plan against imagined data. P-substitution instead pins the node's belief
about that variable to *whatever the model itself predicts*:
q(x|z) := p(x|z). That single constraint changes the node's free-energy
contribution from "surprise about x" into "**expected** surprise", and
expected surprise provably splits into two pieces: how far predictions sit
from goals (the pragmatic/risk term) *minus* how much the observation would
teach you (the epistemic term). Exploration is not added; it *falls out* of
the substituted objective. With a Dirichlet belief over the observation
model A, a second channel falls out too: observations that would update
uncertain columns of A are worth seeking (parameter novelty).

**What we did.** We applied the identical constraint to the *transition*
node p(z′|z,u) with a per-action Dirichlet belief over B. Running the same
three-step algebra the papers run for the observation node, the node's local
free energy becomes (§D1.2, Prop. 1–2):

    G_tm = − H[q(z,u)] − I(z′ ; (z,u), B)

The mutual-information term is the epistemic value, and it splits by the
chain rule into two channels:

1. **State-transition information** I(z′; (z,u)) — actions with distinctive
   outcomes carry information about the trajectory;
2. **Parameter novelty** I(z′; B | z,u) — and this term is *exactly* (not
   approximately) the expected size of tomorrow's Bayesian update of the
   B-column you'd touch: the expected KL between your B-posterior after
   observing the transition and before it (Prop. 3, by Dirichlet
   conjugacy). Low counts ⇒ big expected update ⇒ attractive. This is the
   quantity pymdp's W-term linearizes — here it appears exactly, in nats,
   inside the ledger.

**Why the machinery came for free.** Solving the substituted node's
stationarity conditions (Prop. 4) produces a fixed-point equation of exactly
the same shape as the papers' Corollary 1 — just over the *joint*
(state, action) belief, with the incoming message from the future standing
where the goal prior stood. So the existing Newton solver runs it verbatim,
and the observation composite's local solve is a special case of the
transition one. There is one crucial interior-edge difference: the transition
composite must send a *cavity* `log q_w - mu_down`, because the downstream
factor remains in the graph. Their product, not either message alone, is the
substituted next-state marginal.

**The honest caveat.** z′ is a latent state — you never observe it directly;
you observe y through the sensor model. The channel prices information as if
z′ were going to be revealed. By data processing (Prop. 6) the *realizable*
gain is at most the scored gain: the term is an **optimistic bound**, tight
when sensors pin the state down, loose when they don't. (pymdp's B-novelty
has the same semantics; it is just rarely stated.)

**What it looks like in behavior** (all pinned in
`tests/test_b_epistemics.py`): the agent's first act probes the unfamiliar
action; the preference decays like 1/counts as experience accumulates; and
the ablation without the composite is not merely indifferent but
ambiguity-*averse* — the digamma point-estimate E[log B] penalizes low-count
columns, so plain BP actively avoids exactly the actions it most needs to
learn. The composite corrects a bias, not just an absence.

### D1.2 The formal derivation

**Setting.** Discrete CFFG node a with factor f(z′, z, u) = p(z′|z, u, B),
z′, z ∈ [K], u ∈ [U]; mean-field parameter belief
q(B) = ∏_{u,z} Dir(B_u[:,z] | α_u[:,z]) held fixed within a sweep (as for
the A-composite). Incoming log-messages: μ_in(z), μ_e(u), μ_down(z′).
Notation: B̄ = E_q(B)[B]; for a column α with α₀ = Σα,

    ĥ(z,u) := − Σ_{z′} E_{q(B)}[ B log B ]_{z′,z,u}
             = − Σ_{z′} (α/α₀)(ψ(α+1) − ψ(α₀+1))     (closed form)

**Definition (substituted joint).** P-substitution on 𝒫 = {z′} with
parameter uncertainty constrains node a's belief to

    q̃(z′, z, u, B) := q(z,u) · q(B) · p(z′|z,u,B),

and replaces the substituted variable's log-belief inside the entropy by its
predicted marginal: log q_a(z′,z,u) → log q_w(z′) + log q(z,u), where
q_w(z′) := Σ_{z,u} q(z,u) B̄_u[z′,z]. (This is precisely the recipe the
learned-A composite implements: mean weights B̄ outside the log, coupled
expectation E[B log B] inside.) The node-local objective becomes

    G_tm[q(z,u)] = E_q̃[ log q_w(z′) + log q(z,u) − log p(z′|z,u,B) ].

**Proposition 1 (closed form).**
G_tm = −H[q_w] + E_{q(z,u)}[ĥ] − H[q(z,u)].
*Proof.* Three terms of the definition. (i) E_q̃[log q_w(z′)]: the
z′-marginal of q̃ is Σ_{z,u} q E_B[p(z′|z,u,B)] = q_w, giving −H[q_w].
(ii) −E_q̃[log p] = −Σ_{z,u} q(z,u) Σ_{z′} E_B[B log B] = +E_q[ĥ].
(iii) E_q̃[log q(z,u)] = −H[q(z,u)]. ∎

**Proposition 2 (epistemic decomposition).**
G_tm = −I_q̃(z′ ; (z,u), B) − H[q(z,u)], and
I(z′; (z,u), B) = I(z′; (z,u)) + E_{q(z,u)}[ I(z′; B | z,u) ].
*Proof.* I(z′; (z,u), B) = H[z′] − H[z′ | (z,u), B] = H[q_w] − E_q[ĥ],
since H[z′|(z,u),B] under q̃ is E_{q(z,u)}E_{q(B)}[H[p(·|z,u,B)]] = E_q[ĥ].
Substitute into Prop. 1; the split is the chain rule for mutual
information. ∎

**Proposition 3 (novelty = expected Bayesian update).** For the Dirichlet
column α at (z,u): I(z′; B | z,u) = H[B̄_u[:,z]] − ĥ(z,u), and

    I(z′; B | z,u) = E_{z′ ~ B̄}[ KL( Dir(α + e_{z′}) ‖ Dir(α) ) ]  ≥ 0.

*Proof.* First equality: I = H[z′|z,u] − H[z′|z,u,B] with H[z′|z,u] =
H[B̄-column] under q̃. Second: mutual information equals expected
posterior-to-prior KL, I(z′;B) = E_{z′}[KL(q(B|z′) ‖ q(B))], and by
Dirichlet–categorical conjugacy q(B|z′) = Dir(α + e_{z′}). Nonnegativity is
Jensen (the "uncertain columns have lower expected entropy" gap). ∎
*Executable:* `test_novelty_identity` verifies the two sides to 1e-8 for
random α. This is the exact quantity pymdp's `spm_wnorm` B-novelty
approximates.

**Proposition 4 (stationarity ⇒ fixed point and messages).** Adding the
Lagrangian couplings of the incoming messages (μ_down enters through the
node's z′-marginal, which the substitution fixes to q_w), stationarity of

    L = G_tm − E_q[μ_in + μ_e] − Σ_{z′} q_w(z′) μ_down(z′) + λ(Σq − 1)

with respect to q(z,u) gives log q(z,u) = ρ(z,u) + μ_in(z) + μ_e(u) + const:

    q̄ = softmax( ρ(q̄) + μ_in ⊕ μ_e ),
    ρ(z,u) = Σ_{z′} B̄_u[z′,z] ( μ_down(z′) − log q_w(z′) ) − ĥ(z,u).

*Proof.* Differentiate: ∂(−H[q_w])/∂q(z,u) = Σ_{z′} B̄(log q_w + 1) (chain
rule through q_w, which is linear in q); ∂E_q[ĥ]/∂q = ĥ; ∂(−H[q])/∂q =
log q + 1; the message couplings contribute −μ_in − μ_e − Σ B̄ μ_down.
Collect and exponentiate. ∎
Outgoing messages drop each edge's own incoming term from log q̄ (it is
additive and constant along the marginalized axes, so it cancels exactly):
to the selector, logsumexp_z(ρ̄ + μ_in); to the in-edge,
logsumexp_u(ρ̄ + μ_e). The substituted out-edge is interior, so its
outgoing message must divide out the facing message:

    μ_to_out(z′) = log q_w(z′) − μ_down(z′) + const,
    softmax(μ_to_out + μ_down) = q_w.

On exact-zero support the subtraction is masked: q_w=0 remains a zero cavity;
q_w>0 against μ_down=−∞ is infeasible. The node-local FE contribution is
Proposition 1 evaluated directly,

    G_tm(q̄) = Σ q_w log q_w + Σ q̄ ĥ + Σ q̄ log q̄.

It is not −q̄ᵀρ̄−H[q̄]: because ρ contains μ_down, that stationary
Lagrangian value equals G_tm−E_qw[μ_down]. Scoring it as node energy would
count the downstream coupling once here and again in the downstream node plus
edge correction.
*Executable:* `test_fixed_point_satisfies_its_equation`,
`test_out_edge_cavity_reconstructs_substituted_marginal`,
`test_energy_is_direct_node_local_gfe`, and
`test_two_composite_chain_matches_constrained_objective` (the latter solves
the coupled objective independently and checks beliefs, messages, and ledger).

**Proposition 5 (consistency).** (a) Existence of a fixed point: T(q) =
softmax(ρ(q) + log d) is continuous on the (K·U)-simplex, so Brouwer
applies (uniqueness is not claimed, matching Part II). (b) The observation
composite of Part II (Corollary 1) is the special case s = z (drop u),
μ_down = log c (an absorbed goal leaf), B̄ → Ā: identical local equation
and solver, but not identical exterior/interior cavity bookkeeping.
(c) As α₀ → ∞, ψ(α+1) − ψ(α₀+1) → log(α/α₀), so ĥ → H[B̄-column] and the
novelty channel vanishes: fully-learned dynamics generate no curiosity.
*Executable:* `test_novelty_channel_prefers_unexplored_action` (decay),
`test_agent_curiosity_decays_with_experience` (in the loop).

**Proposition 6 (optimistic-bound semantics).** Given (z,u), the actual
evidence about B arrives through observations y with B → z′ → y a Markov
chain, so by data processing I(y; B | z,u) ≤ I(z′; B | z,u): the channel
scores an upper bound on the single-step realizable information gain, with
equality iff y determines z′. Recorded in LIMITATIONS S1c; the term is a
GFE contribution (no −log Z identity is claimed for models containing it).

**Cost.** The Newton solve is over K·U logits: O((K·U)³) per member per
sweep. Fine at planning scale; a structure-exploitation ticket if colonies
ever need it at hundreds of states × many actions.

---

## D2 — The exponential-family message contract (cff-8s6a)

### D2.1 The plain-English version

Every message family this engine already runs is secretly the same trick.
Discrete messages are vectors of log-probabilities: multiplying two of them
means *adding* the vectors. Gaussian messages are (η, Λ) information pairs:
multiplying means *adding* the pairs. Both are instances of one fact: for
any exponential-family distribution p(x|η) = h(x)·exp(ηᵀT(x) − A(η)),
products of densities add their natural parameters η. And everything else
message passing needs is algebra on A(η): the moments any VMP rule or FE
term wants are the gradient ∇A(η) = E[T(x)]; entropy and KL are one-line
combinations of A and ∇A. So "adding a new distribution" should never mean
new engine machinery — it means a *table entry*: the family's T, A, ∇A, and
domain. `cffg/ef.py` is that table (Gamma and Beta first — rates/precisions
and probabilities), NumPy-pure, with every closed form pinned against
numerical quadrature, and the conjugate-update identity (natural addition ==
exact Bayes: Gamma+Poisson counts, Beta+Bernoulli flips) tested exactly.
Conditioned observations work like the Gaussian convention from cff-4e5z:
an observed count/flip contributes a closed-form energy term and a constant
likelihood message — never a δ-clamp.

### D2.2 Contract and integration plan (formal part is the table itself)

**Contract.** A family is (T, h, A) with: message = η; product = η₁+η₂;
vacuous = 0; moments = ∇A(η); H = A(η) − ηᵀ∇A(η) − E[log h];
KL(η₁‖η₂) = (η₁−η₂)ᵀ∇A(η₁) − A(η₁) + A(η₂); propriety = η in the domain of
A. Gamma: T=(log x, x), η=(α−1, −β), A=lnΓ(α)−α ln β. Beta:
T=(log x, log(1−x)), η=(a−1, b−1), A=lnB(a,b). Executable anchors:
`tests/test_ef.py` (gradient identity, quadrature moments/entropy/KL,
conjugate addition, observation energies).

**Integration (landed 2026-07-09, the engine-facing half of cff-8s6a):**
1. `VariableSpec` carries a tagged domain (discrete k | gaussian d | ef
   family name); serialization is additive (`{"ef": "gamma"}`), mirroring
   cff-4e5z. Message families (gamma, beta) live in `ef.FAMILIES`;
   observation families (poisson, bernoulli) in `ef.OBS_FAMILIES` — the
   latter tag edges that exist only to hang a conditioned DATA port on
   (their unobserved predictives leave the exponential family, so no
   message ever flows there).
2. The third arena is `ef_eta [n_efslots, 2]` with per-group family tags
   (`EfGroupPlan`); vacuous = zeros; same slot discipline as the Gaussian
   arena, independent numbering. The entire update vocabulary: priors and
   observed likelihoods send constants, equality sends sum-minus-self —
   conjugate BP converges in one sweep per graph diameter.
3. Factor vocabulary v1: `gamma`/`beta` prior leaves (scalar shape/rate,
   a/b blobs; propriety is a compile error, not a runtime surprise);
   `poisson_obs`/`bernoulli_obs` conditioned likelihoods (the out edge must
   be a DATA port; missing data drops the node exactly as in the Gaussian
   convention). The ledger is exact and loud: closed-form node energies and
   interior-edge entropies (`cffg/efrules.py`), improper beliefs → +inf
   (never NaN), and `tests/test_ef_engine.py` pins BFE = −log evidence
   (negative-binomial / Beta-Bernoulli marginals) to 1e-12. The mixed
   `gaussian_prec` node (Gaussian with a Gamma precision edge) belongs to
   cff-7c5o as the abstraction's first mixed client.
4. Cat/Dir/Gaussian re-expression landed under cff-l6i5: their dedicated
   arenas remain intact, while `ef.py` / `efrules.py` family entries own
   natural addition, log-partition gradients, and entropy algebra. HMM,
   Kalman, and T-maze messages and BFE values remain bit-identical to the
   pre-refactor implementation.

---

## D3 — Gaussian mixtures: hybrid collapse rules and the bounded ledger (cff-pksr)

### D3.1 The plain-English version

A switching Kalman filter has a discrete "which regime are we in?" variable
steering a continuous state: x' = A_s x + b_s + noise, with s categorical.
The node connecting them, `gaussian_mixture_linear(out, in, selector)`, is
the continuous sibling of `transition_mixture`. Exact inference is
exponential (every regime *sequence* spawns its own Kalman filter), so every
practical algorithm — IMM, assumed-density filtering, RxInfer's switching
models — does the same thing: wherever a message would be a *mixture* of
Gaussians, replace it with the single Gaussian that has the mixture's mean
and covariance ("moment matching", "collapse"). Three subtleties make the
difference between folklore and a correct implementation:

1. **The message to the selector is exact and free.** "How much does each
   regime explain the data?" is a per-regime Gaussian integral — the joint
   log-partition of (node × both Gaussian neighbours' messages). No
   approximation, closed form.
2. **The collapse weights are direction-dependent.** Collapsing the
   *forward* message (to `out`), the components' masses are identical, so
   the weights are exactly the incoming selector probabilities. Collapsing
   the *backward* message (to `in`), each component carries a different
   evidence scale c̃_s·Z_s — engine messages are deliberately scale-free,
   so these factors must be rebuilt inside the rule or the weights are
   silently wrong.
3. **The free-energy ledger stays *almost* exact.** The node belief
   q(y, x, s) is a labelled mixture whose joint entropy is closed form
   (H(s-marginal) + Σ_s ρ_s·H[Gaussian_s]) — node energies are exact for
   the beliefs the engine holds. The single approximation in the reported
   BFE: interior Gaussian *edge* entropies score the collapsed Gaussian,
   and a Gaussian is the max-entropy distribution for given moments, so
   each such +H term is an over-estimate and the reported BFE is a **lower
   bound** (optimistic) relative to the mixture-Bethe value at the same
   moments. Same optimistic-bound semantics as the GFE composite
   (LIMITATIONS S1c); pinned by a bound-direction test.

### D3.2 Rules (f_s(y,x) = N(y; A_s x + b_s, V_s), W_s = V_s⁻¹, K regimes)

Incoming: μ_o = (η_o, Λ_o) on out, μ_i = (η_i, Λ_i) on in (information form,
scale-free), μ_e(s) selector log-message. Z(η, Λ) := (2π)^{d/2}
det(Λ)^{−1/2} exp(½ηᵀΛ⁻¹η) is the unnormalized-Gaussian mass; all propriety
guards below fall back to the vacuous message mid-flooding and to +inf in
ledger terms (the engine-wide loudness convention).

**→ selector (exact).** Per regime, the joint over (y, x) of node × μ_o × μ_i
has (same blocks as `gauss.linear_energy`):

    Λ_J(s) = [[W_s+Λ_o, −W_sA_s], [−A_sᵀW_s, A_sᵀW_sA_s+Λ_i]],
    η_J(s) = [W_s b_s + η_o, −A_sᵀW_s b_s + η_i]

    log μ(s) = log Z(η_J(s), Λ_J(s)) − ½ b_sᵀW_s b_s + ½ log det(W_s/2π·I…)
             = log Z(η_J, Λ_J) − ½ b_sᵀW_s b_s + ½ log det W_s − ½ d_y log 2π.

Guard: Λ_J(s) ≻ 0 for all s, else uniform. (Prior variant f_s(y) = N(y; m_s,
V_s): Λ_J = W_s + Λ_o ≻ 0 always; log μ(s) = log Z(W_s m_s + η_o, W_s + Λ_o)
− ½ m_sᵀW_s m_s + ½ log det W_s − ½ d log 2π.)

**→ out (collapsed forward).** Component s: the scale-free forward
predictive (η_f(s), Λ_f(s)) = `linear_forward`(μ_i; A_s, b_s, W_s). Masses
∫∫ f_s μ_i dy dx = ∫ μ_i are s-independent, so weights w_s = softmax of the
incoming selector log-message alone. Moment match:

    m̄ = Σ w_s m_s,   C̄ = Σ w_s (C_s + m_s m_sᵀ) − m̄ m̄ᵀ,
    (m_s, C_s) = (Λ_f⁻¹η_f, Λ_f⁻¹);   message = (C̄⁻¹ m̄, C̄⁻¹).

Guard: every Λ_f(s) ≻ 0 (needs Λ_i ≻ 0), else vacuous.

**→ in (collapsed backward, the weight trap).** Component s: scale-free
(η_b(s), Λ_b(s)) = `linear_backward`(μ_o; A_s, b_s, W_s), plus the scale the
engine's convention dropped:

    log c̃_s = log ∫ N(y; b_s, V_s) μ_o(y) dy
             = log Z(W_s b_s + η_o, W_s + Λ_o) − ½ b_sᵀW_s b_s
               + ½ log det W_s − ½ d_y log 2π.

Weights w̃_s ∝ exp(μ_e(s)) · c̃_s · Z(η_b(s), Λ_b(s)); moment match as above.
Guard: every Λ_b(s) ≻ 0 (needs Λ_o ≻ 0 and rank(A_s) = d_x), else vacuous.

**Node energy (exact for the held beliefs).** ρ_s = softmax over s of
[μ_e(s) + log μ(s)-integrand above]; q(y,x|s) is Gaussian with (η_J, Λ_J):

    F_a = −H(ρ) − Σ_s ρ_s H[q(y,x|s)] − Σ_s ρ_s E_{q|s}[log f_s],

with E_{q|s}[log f_s] = ½ log det W_s − ½ d_y log 2π − ½ (r_sᵀW_s r_s +
tr(W_s J_s Σ_J J_sᵀ)), r_s = J_s m_J − b_s, J_s = [I, −A_s] — the
`gauss.linear_energy` residual algebra per component. The selector edge and
the Gaussian edges keep their standard interior +H corrections; the Gaussian
ones use collapsed beliefs, which is the ledger's one bounded term (D3.1.3).

Checks: K = 1 reduces every rule to the plain `gaussian_linear` rules and
the energy to `linear_energy`. Vacuous μ_e ⇒ w_s uniform in the forward
collapse. A_s ≡ A, b_s ≡ b, V_s ≡ V across s ⇒ collapse is exact (mixture
of identical Gaussians) and F_a reduces to linear_energy + H-cancellation
on the selector edge. Executable anchors: `tests/test_mixg.py` (quadrature
moments/energy, weight-trap regression, bound direction),
`tests/test_switching_kf.py` (enumeration + independent collapsed-smoother
parity).

---

## D4 — Gaussian conjugate learning: sufficient stats, NIW/MNIW/Gamma, gaussian_prec (cff-7c5o)

### D4.1 The plain-English version

The continuous mirror of Dirichlet A-learning. Discrete learning already
works like this: run inference, read each learned node's *belief*, add its
expected sufficient statistics to the conjugate prior's counts, and roll the
posterior's expected-log parameters (`agent._elog`) back into the next
trial's messages. Gaussians follow the identical shape with three castings:

1. **Sufficient statistics are node-belief moments.** A gaussian(_linear)
   node's belief is itself Gaussian (the joint (η_J, Λ_J) the energies
   already build), so E[x], E[xxᵀ], E[yx̃ᵀ], E[x̃x̃ᵀ] (x̃ = [x; 1] to catch
   the offset) are closed-form reads — `Engine.gaussian_node_stats`.
2. **Conjugate updates are count bookkeeping.** NIW for a prior node's
   (mean, cov); matrix-normal-inverse-Wishart (MNIW) for a linear node's
   ([A b], V); Gamma for a scalar precision. Soft statistics (beliefs, not
   samples) enter exactly like soft Dirichlet counts.
3. **What rolls forward is the *expected natural parameter*.** Dirichlet
   rolls E[log A] because log A is what multiplies the sufficient statistic;
   for Gaussians that role is played by the precision, so the rollover is
   E[W] = ν_n Ψ_n⁻¹ (and Ā = M_n), NOT the naive Ψ_n/(ν_n−d−1) point
   covariance. Messages built from E[W] are exactly the VMP geometric-mean
   messages; the remaining difference from full VMP is (a) the FE constant
   E[log det W] − log det E[W] and (b) an x-dependent trace term coupling
   A's posterior covariance into state inference. v1 ships the rollover
   (empirical-Bayes EM, matching the Dirichlet loop) and defers (b) —
   documented, not silently approximated (LIMITATIONS S1e).

`gaussian_prec` is the exception where we do full in-loop VMP, because it is
scalar and exactly tractable: a Gaussian node whose precision lives on an
`ef:gamma` edge — the EF abstraction's first mixed client (cff-8s6a's
promise). Mean-field q(x)q(τ) makes both VMP messages closed-form and the
free energy the exact mean-field VFE, so sweeps must not increase it — the
monotonicity acceptance test.

### D4.2 Rules and updates

**gaussian_prec** f(x; m, τ) = N(x; m, τ⁻¹), x scalar Gaussian edge, τ on an
ef:gamma edge (natural η = (α−1, −β)), m a blob. With q(τ) the τ-edge
marginal (α, β from η_q) and q(x) the x-edge marginal (mean x̄, var v):

    → x edge:  exp(E_τ[log f]) ∝ N(x; m, E[τ]⁻¹),  info form (E[τ]·m, E[τ]),
               E[τ] = α/β.
    → τ edge:  exp(E_x[log f]) ∝ τ^{1/2} e^{−τ·s/2},  gamma-natural
               increment η = (1/2, −s/2),  s = E[(x−m)²] = (x̄−m)² + v.
    node energy:  −E_q[log f] = −½(ψ(α) − ln β) + ½ ln 2π + ½·(α/β)·s.

Both messages read *marginals* (both slots), not cavity messages — the
VMP convention, same as the discrete vmp groups. Guards: improper q(x) or
q(τ) ⇒ vacuous messages / +inf energy (engine-wide loudness). Vacuous τ-edge
belief never arises after one sweep when a proper gamma prior terminates the
edge (its message is a proper constant).

**NIW update** (gaussian node, prior (μ0, κ0, Ψ0, ν0), trial stats
(x̄_t, S_t = E[xxᵀ]_t), n trials):

    κn = κ0 + n,  νn = ν0 + n,  μn = (κ0 μ0 + Σ x̄_t) / κn,
    Ψn = Ψ0 + Σ (S_t − x̄_t x̄_tᵀ) + Σ x̄_t x̄_tᵀ + κ0 μ0 μ0ᵀ − κn μn μnᵀ.

(The first sum is within-trial covariance E[xxᵀ] − x̄x̄ᵀ — soft stats enter
here; with point observations it vanishes into the scatter term.)
Rollover: mean μn, precision E[W] = νn Ψn⁻¹.

**MNIW update** (gaussian_linear node, x̃ = [x; 1], prior (M0 [dy,di+1],
Λ0 [di+1,di+1] column precision, Ψ0, ν0), per-trial stats Syy = E[yyᵀ],
Syx̃ = E[yx̃ᵀ], Sx̃x̃ = E[x̃x̃ᵀ], summed over the trial's node occurrences,
n = total occurrences):

    Λn = Λ0 + Σ Sx̃x̃,  Mn = (M0 Λ0 + Σ Syx̃) Λn⁻¹,  νn = ν0 + n,
    Ψn = Ψ0 + Σ Syy + M0 Λ0 M0ᵀ − Mn Λn Mnᵀ.

Rollover: [Ā b̄] = Mn, E[W] = νn Ψn⁻¹. Ψn ≻ 0 is guaranteed in exact
arithmetic (Schur complement of a PSD block matrix); the implementation
asserts it rather than ridging — a violation means broken stats, not noise.

**Gamma precision update** (gaussian_prec nodes sharing a τ edge):
αn = α0 + n/2, βn = β0 + Σ s_t / 2 — but note this one usually needs no
trial loop at all: it is the exact in-loop conjugate posterior the τ edge
already carries at the VMP fixed point (prior η + Σ node increments), which
is precisely why gaussian_prec is the mixed client and not a rollover client.

Checks: NIW/MNIW with hard stats and n → ∞ concentrate at the empirical
MLE; E_q[stats] with vague priors recover generating LGSSM parameters
(acceptance test); gaussian_prec fixed point equals the exact
Gamma-posterior when x is observed (conjugacy, zero collapse error).
Executable anchors: `tests/test_conj.py`, `tests/test_gauss_vmp.py`.

---

## D5 — Hybrid GFE epistemics: where continuous curiosity lives (cff-dfpi)

### D5.1 The plain-English version

**The open problem, stated honestly.** Koudahl, Kouw & de Vries (Entropy
23:1565, 2021) proved that in a linear-Gaussian state-space model with
*additive* controls, the epistemic terms of the expected free energy are the
same for every policy — a Kalman agent scored by EFE/GFE has nothing to be
curious *about*, because no action changes what it would learn. This is not
an implementation gap; it is a property of the model class. The same paper,
and Collis et al. (arXiv:2409.01066, 2024), point at the door left open:
controls that enter *multiplicatively* — switching dynamics or emissions —
do carry policy-dependent information.

**The position this derivation takes (the cff-dfpi decision).** CFFG does
not promise a "curious Kalman agent", because none can exist (Prop. 5 below
re-proves the negative result inside our own objective, so it is pinned by a
test rather than folklore). Instead, **continuous epistemics live on the
discrete regime/selector variables of hybrid models** — exactly the
variables the switching-KF factors (D3) already carry. The recipe is the
same single trick as everywhere else in this engine: put the P-substitution
bead on a *hybrid mixture* observation composite. The substituted objective
then splits into

1. a **regime channel** I(x; s) — "which mode is the world in?" — the term
   Koudahl's negative result says is missing from pure Gaussian models, and
   the one that makes probing/sensing actions valuable;
2. a **state channel** I(x; z | s) — a per-regime log-det ratio, closed form,
   which prices sensor precision (and is constant across policies exactly
   when the negative result says it must be);
3. the **goal term** — the cross-entropy of the predictive mixture against a
   Gaussian goal, which stays *exactly* closed form (a mixture against a
   quadratic log-density integrates term by term).

**The one approximation, declared.** The regime channel needs the entropy of
a Gaussian mixture — the known non-closed-form (LIMITATIONS S1d's sibling).
We score it with the moment-matched collapse the D3 rules already compute.
The collapsed Gaussian is the max-entropy density for its moments, so the
surrogate *over*-states information gain and the reported GFE *lower-bounds*
the exact substituted objective — the same optimistic-bound stance as S1c
and S1d, with the direction pinned by quadrature tests. Two exact
consistency anchors: coincident components ⇒ the surrogate is exact and the
channel is exactly zero (so the negative control is *inside* the formula,
not a special case); and the exact channel can never exceed H[q(s)] ≤ log K
— a cap the diagnostics report alongside the surrogate, because for
well-separated regimes the surrogate exceeds it (measured: 1.86 nats scored
vs log 2 = 0.69 realizable in the probe experiment).

**What the experiment shows** (all in `tests/test_hybrid_epistemics.py`,
formulas in `cffg/experiments/hybrid_epistemics.py`): a plant with an
unknown binary regime and a Gaussian latent state, where a PROBE action is
strictly *worse* than IDLE on risk alone. The GFE scorer probes in 100% of
episodes and commits correctly in 100%; probing collapses the regime channel
to ~0 (satiation) so a second probe is never valued. The KL-control ablation
(same formulas, MI terms dropped) never probes and scores 44.5% — a coin
flip. Purposeful exploration, attributable to the derived term and nothing
else. A second experiment re-runs the *same* general formulas on an
additive-control LGSSM and finds every epistemic/entropy term identical
across random policies to 1e-10 — Koudahl 2021, reproduced through our own
code path.

**What this is not.** The exterior fixed-Gaussian-goal composite is executable
in `cffg.hybrid_gfe`; this remains neither a claim about arbitrary nonlinear
MI nor exact mixture entropies (impossible in closed form — any future
nonlinear channel enters through declared quadrature/MC bounds per H8), and
not contingent planning (the commit slice is mean-field, H5 as always).

### D5.2 The formal derivation

**Setting.** Hybrid composite: emission node f(x|z,s) = N(x; A_s z + b_s, V_s)
with discrete selector s ∈ [K] (regime or control), Gaussian state
z ∈ ℝ^{d_z}, substituted future observation x ∈ ℝ^{d_x}, facing an absorbed
Gaussian goal p̃(x) = N(g, Γ). Beliefs: the labelled-mixture family the D3
rules hold, q(z,s) = q(s) q(z|s) with q(s) = ρ, q(z|s) = N(m_s, S_s).
Per-regime predictive: x̄_s = A_s m_s + b_s, P_s = A_s S_s A_sᵀ + V_s;
substituted marginal q_w(x) = Σ_s ρ_s N(x; x̄_s, P_s) — a mixture.
Notation: 𝒩H(P) := ½ log det(2πe P); h_s := 𝒩H(V_s).

**Definition (substituted objective).** P-substitution on {x} constrains the
node belief to q̃(x,z,s) = q(z,s) f(x|z,s) and replaces log q(x) by
log q_w(x) inside the expectation (papers Part II Eq. 6–8, as in D1):

    G[q] = E_q̃[ log q_w(x) + log q(z,s) − log f(x|z,s) − log p̃(x) ].

**Proposition 1 (closed form up to one term).**

    G = −H[q_w] + Σ_s ρ_s h_s − H(ρ) − Σ_s ρ_s 𝒩H(S_s) + C(q_w, p̃),
    C(q_w, p̃) = ½ log det(2πΓ)
              + ½ Σ_s ρ_s [ (x̄_s−g)ᵀΓ⁻¹(x̄_s−g) + tr(Γ⁻¹ P_s) ].

*Proof.* (i) the x-marginal of q̃ is q_w, giving −H[q_w]; (ii)
−E[log f] = Σ_s ρ_s E_{q(z|s)} H[N(A_s z+b_s, V_s)] = Σ ρ_s h_s (the Gaussian
entropy is mean-independent); (iii) E[log q(z,s)] = −H(ρ) − Σρ_s 𝒩H(S_s);
(iv) −E_{q_w}[log p̃] is the expectation of a quadratic under each mixture
component — exact. Every term except H[q_w] is closed form. ∎

**Proposition 2 (two epistemic channels).**
G = −I(x; z,s) − H[q(z,s)] + C(q_w, p̃), with the chain rule split

    I(x; z,s) = I(x; s) + Σ_s ρ_s I(x; z|s),
    I(x; z|s) = ½( log det P_s − log det V_s )          (exact),
    I(x; s)   = H[q_w] − Σ_s ρ_s 𝒩H(P_s)               (mixture entropy).

*Proof.* I(x; z,s) = H[q_w] − E_{q(z,s)}H[f] = H[q_w] − Σρ_s h_s; substitute
into Prop. 1. The split is the MI chain rule; the conditional term is the
log-det ratio of a joint Gaussian. ∎
Unlike D1's transition channel, x here is the variable that will actually be
observed — there is **no data-processing gap** (no S1c-style optimism in the
channel itself); the only approximation is Prop. 3's surrogate.

**Proposition 3 (mixture-entropy surrogate and sandwich).** Let
(x̄̄, P̄) = Σρ_s(x̄_s, ·) be the moment-matched collapse (D3), and
Ĥ := 𝒩H(P̄). Then

    max( Σ_s ρ_s 𝒩H(P_s),  −Σ_s ρ_s log Σ_t ρ_t N(x̄_s; x̄_t, P_s+P_t) )
      ≤ H[q_w] ≤ min( Ĥ,  H(ρ) + Σ_s ρ_s 𝒩H(P_s) ),

so the surrogate regime channel Î(x;s) := Ĥ − Σρ_s 𝒩H(P_s) satisfies
0 ≤ I(x;s) ≤ Î(x;s), and the exact channel obeys I(x;s) ≤ H(ρ) ≤ log K.
Scoring G with Ĥ gives Ĝ ≤ G: the reported GFE is a lower bound
(optimistic), the S1c/S1d stance. Tight iff the components coincide, where
Î = I = 0 exactly.
*Proof.* Upper-left: Gaussian is max-entropy at fixed moments. Upper-right:
H[x] ≤ H[x,s] = H(ρ) + Σρ_s H[x|s]. Lower-left: conditioning reduces
entropy. Lower-right: Jensen inside each component (Hershey–Olsen), with
E_{N_s}[q_w] = Σ_t ρ_t N(x̄_s; x̄_t, P_s+P_t). Î ≥ 0 because
P̄ ⪰ Σρ_s P_s and log det is Loewner-monotone and concave. I(x;s) ≤ H(ρ)
because MI with a discrete variable is capped by its entropy. ∎
*Executable:* `test_entropy_sandwich_against_quadrature`,
`test_sandwich_strict_when_separated_tight_when_coincident`,
`test_gfe_hat_lower_bounds_exact_objective`.

**Proposition 4 (stationarity over the regime simplex ⇒ same solver shape).**
Hold the per-regime Gaussians (m_s, S_s) at their D3 message-passing values
(decision: continuous edges keep BP/collapse messages — see remark below)
and add the selector's incoming log-message μ_e. Stationarity of
Ĝ[ρ] − Σ_s ρ_s μ_e(s) on the simplex is the fixed point

    ρ̄ = σ( r(ρ̄) + μ_e ),
    r_s(ρ) = ∂Ĥ/∂ρ_s − h_s + 𝒩H(S_s) − c_s,
    ∂Ĥ/∂ρ_s = ½ tr( P̄⁻¹ [ P_s + x̄_s x̄_sᵀ − x̄_s x̄̄ᵀ − x̄̄ x̄_sᵀ ] ),
    c_s = ½[ (x̄_s−g)ᵀΓ⁻¹(x̄_s−g) + tr(Γ⁻¹P_s) ].

Same shape as D1 Prop. 4 / Part II Corollary 1: a K-logit solve the existing
Newton machinery runs unchanged; T(ρ) = σ(r(ρ)+μ_e) is continuous on the
simplex so Brouwer gives existence (uniqueness not claimed, matching Part
II). Outgoing selector message: μ_out(s) = r_s(ρ̄) (own incoming dropped,
the D1 convention). Node ledger term: Ĝ(ρ̄) evaluated directly — not the
stationary Lagrangian value (the cff-hp3y lesson: message couplings are not
node energy). The substituted x-edge is exterior with the goal absorbed in
v1; an interior variant must send the collapsed cavity
collapse(q_w) ⊘ μ_down in information form (D1's cavity rule + D3's
collapse), which remains a separately derived follow-up.
*Proof.* ∂/∂ρ_s of Prop. 1 with Ĥ: the goal and entropy terms are linear or
−Σρ log ρ in ρ; the Ĥ gradient is the chain rule through (x̄̄, P̄), which
are linear/quadratic in ρ. Exponentiate the multiplier equation. ∎
*Executable:* `test_collapse_entropy_gradient_matches_finite_differences`,
`test_fixed_point_satisfies_its_equation`,
`test_fixed_point_is_constrained_minimum` (dense simplex search oracle).

**Proposition 5 (the negative result, GFE form — Koudahl 2021 restated).**
Additive-control LGSSM slice: z′ = F z + B u + w, w ~ N(0,Q), emission
x = C z′ + e, e ~ N(0,R), deterministic policy u_{1:T}, prior N(m₀, S₀).
This is the K = 1 case of Prop. 1–2, where the covariance recursion
S_t = F S_{t−1} Fᵀ + Q contains no u, hence at every slice

    I(x_t; z_t) = ½( log det(C S_t Cᵀ + R) − log det R ),   and all of
    I(x;s) = 0,  H-terms,  tr(Γ⁻¹P_t)

are identical for every policy; only the goal quadratic
(C m_t(u) − g)ᵀΓ⁻¹(C m_t(u) − g) varies. argmin over policies of the GFE
therefore equals the KL-control optimum: **no purposeful exploration exists
in this class**, at any horizon. (Remark: with *stochastic* q(u) entering
the mean, Var(u) inflates S_t — the agent is rewarded for keeping its own
controls entropic, which is noise-seeking, not information-seeking; the
per-policy statement is the meaningful one.) Any of C, R, F, Q made
s-dependent through a discrete selector (multiplicative/switching control,
action-dependent sensing) re-opens the regime channel of Prop. 2 —
policy-dependent again. Nonlinear observation maps and state-dependent
(heteroscedastic) noise are the non-closed-form generalizations; the regime
mixture is their tractable discretization.
*Executable:* `test_additive_lgssm_epistemics_constant_across_policies`
(runs the *general* hybrid formulas per policy — the constancy is measured,
not assumed), `test_switching_emission_restores_policy_dependent_epistemics`.

**Proposition 6 (parameter novelty, the continuous mirror of D1 Prop. 3).**
If a regime's emission mean is uncertain, p(x|s,μ_s) = N(x; μ_s, V_s) with
conjugate q(μ_s) = N(n_s, U_s) (D4), the substituted objective acquires

    I(x; μ_s | s) = ½( log det(V_s + U_s) − log det V_s )
                  = E_x[ KL( q(μ_s | x) ‖ q(μ_s) ) ]  ≥ 0,

exact and closed form: the expected size of tomorrow's Bayesian update of
that regime's mean. As evidence accumulates U_s → 0 and the channel
vanishes — the same satiation law as the Dirichlet channel (D1 Prop. 5c).
With full NIW uncertainty the predictive is Student-t: its entropy and
E_IW[log det V] are still closed form (digamma family), so the channel stays
exact — recorded here for a parameter-novelty follow-up, not exercised by the
current fixed-parameter composite.
*Proof.* Gaussian–Gaussian MI is the log-det ratio; MI = expected
posterior-to-prior KL always. ∎

**Cost.** The regime solve is K logits — O(K³) Newton per composite per
sweep, plus one d_x×d_x collapse (already computed by the D3 rules). Cheaper
than the discrete GFE-TM composite (K·U logits); nothing new to exploit.

### D5.3 The decision and the experiment

**Decision (acceptance criterion of cff-dfpi):** CFFG adopts the
**discrete-regime epistemic mechanism**: P-substitution composites attach to
hybrid mixture nodes and score Î(x;s) + Σρ_s I(x;z|s) with the collapse
surrogate (declared optimistic bound, H(ρ) cap reported as a diagnostic);
continuous edges keep their exact/collapsed BP messages and no
Gaussian-only GFE message family is added — Prop. 5 says there is nothing
for one to express. Purposeful continuous exploration is modelled by making
the relevant uncertainty discrete (regimes, sensor choices, operating
modes), which is also what the post-2023 literature converged on (Collis et
al. 2024's rSLDS abstractions).

**Probe-then-commit experiment** (`cffg/experiments/hybrid_epistemics.py`):
binary regime (prior ½/½), scalar latent state z ~ N(0, 0.25), step-1 choice
between PROBE (emission x = 0.3z ± 1, σ = 0.05 — separates regimes; slightly
*higher* risk than idling) and IDLE (regime-blind emission), step-2 forced
commit scored N(+1, 0.2²) if correct / N(−1, 0.2²) if wrong. Measured
(200 episodes, seed 0): GFE scorer — probe rate 1.0, success 1.0, post-probe
Î < 1e-5 (satiation); KL-control ablation (identical formulas, MI dropped) —
probe rate 0.0, success 0.445. Scored margins: probe wins by ≈3.0 nats with
the epistemic terms, loses by ≈0.0025 nats without. The surrogate's optimism
is visible and bounded exactly as Prop. 3 predicts: Î = 1.857 nats scored,
exact channel log 2 ≈ 0.693, sandwich [0.540, 0.693].
*Executable:* `test_probe_experiment_gfe_explores_and_wins`,
`test_probe_experiment_ablation_never_explores`, plus the float32 /
degenerate-belief / improper-covariance contract tests.

## D6a — VMP Dirichlet parameter edges and their VFE ledger (cff-l29q)

### D6a.1 The plain-English version

A categorical table can be an uncertain variable in the graph instead of a
fixed blob. The useful conjugacy is specifically **mean-field VMP
conjugacy**. A likelihood node sees the current beliefs over its ordinary
categorical edges and sends their outer product to the table as fractional
counts. A proper Dirichlet prior plus all such counts gives another proper
Dirichlet; in the other direction, the likelihood uses the posterior expected
log table, not the log posterior mean.

This is the same algebra used by the out-of-graph `Learnables` fold, but that
does not make the two procedures interchangeable without conditions. They
agree when they start from the same proper prior, integrate the same
mean-field node beliefs exactly once, use the same parameter tying and axis
layout, and compare at the same inference fixed point. The in-graph form can
feed the updated expected log table back into inference during a trial;
`Learnables` normally folds a converged trial afterwards.

Ordinary sum-product BP does **not** supply this message. Marginalizing a
latent categorical assignment gives a weighted sum of table entries, while a
Dirichlet-shaped message is a product of powers of those entries. They agree
only when all roles selecting a table cell are hard (or, equivalently, every
assignment with nonzero incoming mass selects the same cell). The Part II
`mu3(A)` planning message is also not Dirichlet-shaped and is deliberately
absent from this contract.

The free-energy ledger must retain the parameter belief. A likelihood node
contributes its expected negative log likelihood, the prior contributes its
cross-entropy, and the parameter belief contributes its negative entropy.
In the node-local CFFG spelling both adjacent nodes initially contain the
parameter negative entropy; the parameter-edge correction removes one copy.
Dropping that correction, or dropping the parameter entropy altogether,
changes the VFE and therefore changes both convergence and policy scores.

### D6a.2 Formal VMP updates

**Layout and notation.** Every conditional table normalizes over its output
axis. Let `c` index one output cell and `j` the remaining column indices:

    categorical:        θ[c]
    matrix_cat:         A[c, i]
    transition_mixture: B[u, c, i]     (serialized/blob parameter layout).

Each column has an independent Dirichlet prior
`p(θ_.j | α⁰_.j) = Dir(α⁰_.j)`, with every concentration finite and strictly
positive. Define

    L[c,j] := E_q[log θ[c,j]]
            = ψ(α[c,j]) − ψ(Σ_d α[d,j]).

There is no numeric floor in this definition. Positive concentrations may
produce very negative finite expected logs, especially in float32; exact
zero support belongs to categorical beliefs and contributes an exact zero
count.

**Factor-to-parameter message.** For
`f(s,θ) = θ[cell(s)]`, the mean-field constraint
`q_f(s,θ) = q(θ) ∏_r q_r(s_r)` gives

    log μ_{f→θ}(θ)
      = E_{∏q_r}[log f(s,θ)] + const
      = Σ_{c,j} n[c,j] log θ[c,j] + const,

so, under the Dirichlet natural convention
`log Dir(θ|α) = const + Σ(α−1)log θ`, the message natural parameter is the
fractional count `η_f = n` (not `n−1`). The three sufficient statistics are

    categorical:        n[c]     = q_out(c)
    matrix_cat:         n[c,i]   = q_out(c) q_in(i)
    transition_mixture: n[u,c,i] = q_sel(u) q_out(c) q_in(i).

For clamped variables the corresponding `q` is a one-hot vector, so
impossible cells receive exactly zero. Shapes above are parameter layout;
the engine's transition node-belief layout `[c,i,u]` is transposed to
`[u,c,i]`.

With likelihood nodes `a = 1…N`, the parameter marginal is

    q(θ) = Dir(α),                 α = α⁰ + Σ_a n_a.             (D6a.1)

The parameter-to-likelihood cavity message excludes that likelihood:

    η_{θ→a} = (α⁰ − 1) + Σ_{b≠a} n_b.

The VMP update *from* likelihood `a` to any structural role uses the complete
current marginal `q(θ)` and replaces the fixed `log θ` table in the ordinary
geometric rule by `L = E_q log θ`. For example,

    log μ_{f→out}(c) = Σ_i q_in(i) L[c,i] + const,

with the analogous two- or three-way contractions for the other roles.
This use of the complete marginal is the coordinate-ascent VMP update; the
cavity expression above defines edge-message bookkeeping, not a directive to
discard `a` while taking `E_q log θ`.

**Why BP is not closed.** Exact sum-product instead sends

    μ^BP_{f→θ}(θ) = Σ_s w(s) θ[cell(s)],

where `w` is the product of incoming structural messages. A Dirichlet-shaped
message would be `C ∏_{c,j} θ[c,j]^{η[c,j]}`. The former has nonzero mixed
curvature in `log μ = log Σ_s w(s)θ[cell(s)]`; the latter is affine in
`log θ`. They cannot agree on an open subset when two distinct cells have
positive weight. If all nonzero weight selects one cell `k`, the sum reduces
to `Cθ_k`, the hard-count special case `η_k=1`. Thus a latent-state BP
parameter edge is rejected rather than silently approximated.
*Executable:* `test_exact_bp_parameter_message_is_not_dirichlet_closed`.

### D6a.3 Parameter VFE and the complete ledger

For one column define the multivariate beta function

    log B(α) = Σ_c log Γ(α_c) − log Γ(Σ_c α_c)

and `ℓ_c = ψ(α_c) − ψ(α_+)`. The three parameter-related terms are

    Q term (negative entropy):
      E_q log q = −log B(α) + Σ_c (α_c−1)ℓ_c

    prior energy:
      −E_q log p(θ|α⁰) = log B(α⁰) − Σ_c (α⁰_c−1)ℓ_c

    likelihood energy:
      −E_q log f = −Σ_c n_c ℓ_c.

They sum over columns for matrix and transition tables. At the coordinate
optimum `α = α⁰+n`, all expected-log terms cancel:

    F_θ,prior,lik = log B(α⁰) − log B(α⁰+n).                    (D6a.2)

The **complete** mean-field VFE also includes the ordinary state-belief terms
and all other factors. For this likelihood block, with no other terms shown,

    F_block = Σ_r Σ_{s_r} q_r(s_r) log q_r(s_r)
              + E_q log q(θ) − E_q log p(θ|α⁰)
              − E_{q(θ)∏q_r} log f(s,θ).                       (D6a.3)

Zeros in `q_r` use the continuous convention `0 log 0 = 0`.

In node-local CFFG form, the prior and likelihood node beliefs each contain
the parameter marginal:

    F_prior-node = E_q log q(θ) − E_q log p(θ|α⁰)
    F_lik-node   = E_q log q(θ) + Σ_r E log q_r − E log f.

The parameter-edge correction is `−E_q log q(θ)` (equivalently `−` the
negative entropy), leaving exactly (D6a.3). Equality trees used to share one
parameter repeat this accounting mechanically: every internal edge removes
the one extra marginal-entropy copy introduced by its incident node terms.
*Executable:* `test_categorical_fractional_counts_and_total_vfe`,
`test_matrix_cat_fractional_counts_and_total_vfe`,
`test_transition_mixture_fractional_counts_and_total_vfe`.

**Conjugate-equivalence theorem.** An in-graph posterior (D6a.1) equals an
out-of-graph `Dirichlet.fold` posterior iff:

1. both start from the identical proper `α⁰`;
2. every source likelihood carries the declared mean-field constraint;
3. both consume the same converged structural marginals, whose product is
   the VMP node belief, once per factor instance;
4. blob/edge tying, source restrictions, and the output-axis layout route
   every statistic to the same posterior column; and
5. no update is interleaved on one side before the compared structural fixed
   point, and no extra functional message such as `mu3` is present.

Under these assumptions both sides are definitionally
`α⁰ + Σ_a E_q[t_a]`; equality follows entrywise. Without (2), the exact BP
node belief can carry correlations and its parameter message has the
non-closure above. Without (3–4), a statistic is duplicated, omitted, or
transposed. Without (5), the two algorithms are being compared at different
coordinate-ascent iterates, not at the same posterior.

**Hostile-input contract.** Missing, non-finite, zero, or negative prior
concentrations are not vacuous messages; they make the Dirichlet base
improper and compilation must fail loudly. Sparse *positive* concentrations
remain valid without a literal floor. Exact-zero state mass remains an exact
zero statistic. Expected logs must stay finite at float32, and contractions
must operate at category/action sizes 10–100 times the tiny anchors without
dimension-dependent guards. These cases are pinned by
`test_dirichlet_vmp_hostile_inputs_and_scale`.

## D6b — The functional `mu3(A)` message and importance sampling (cff-almk)

### D6b.1 The plain-English version

The planning message back to an uncertain observation matrix is a function,
not a familiar probability distribution. Given a possible matrix `A`, a state
belief `z`, and an observation goal `c`, it scores how well that matrix makes
the substituted future-observation objective. The score contains predictive
entropy and `A log A` terms, so it cannot be reduced to fractional Dirichlet
counts. The proper Dirichlet posterior supplied by perception is instead the
**base distribution**; the planning function tilts that base.

CFFG evaluates the tilted belief with importance sampling. It draws matrices
from the proper base posterior, evaluates every future slice's functional
log-message on the same draw, adds those log-messages, and normalizes the
weights in log space. Sharing a draw is what combines future slices as one
posterior rather than pretending each slice contributed a conjugate count.
The result supplies weighted expectations of `A`, column entropy, the
backward-message function, and the parameter part of the free-energy ledger.

Randomness is explicit and replayable. One inference attempt draws its
particles from a caller-provided JAX key and reuses them until the conditional
message fixed point is reached. A fresh sampling epoch splits the returned key
and draws a new set. Reusing particles gives a falsifiable convergence test;
averaging independent post-burn-in policy scores reduces Monte Carlo noise but
does not turn an unconverged or collapsed inference attempt into a converged
one.

The serialized graph stores a registered evaluator identity and the graph
sources of `A`, `z`, and `c`; particles and weights are derived runtime state.
For the built-in observation composite the identity is
`cffg.mu3_is.v1`. A P-substitution constraint names it through
`parameter_message=mu3_is_v1`; `A` is the likelihood's matrix parameter
blob/base posterior, `z` its `in` edge, and `c` the facing categorical goal
blob. Multiple composite instances reading the same `A` blob form one
functional-message group. This representation survives JSON and `.cffgz`
round trips without serializing a Python closure or a sample array.

### D6b.2 Functional message

Let `A ∈ [0,1]^{M×K}` be column-stochastic,
`zbar ∈ Δ^{K−1}`, and `c ∈ Δ^{M−1}`. Let
`Abar=E_q(A)[A]` be the current matrix marginal mean and
`pbar=Abar zbar` the current posterior-predictive observation belief. The
P-substituted observation model is `p(x|z,A)=Cat(x|Az)`. Part II §5.2,
Fig. 5 and Appendix C.6 define

    h(A) := −diag(Aᵀ log A)
    r(Abar,zbar,c) := log c − log(Abar zbar)
    ξ(A;Abar,zbar,c) := Aᵀ r(Abar,zbar,c) − h(A)
    log μ3(A;Abar,zbar,c) := zbarᵀ ξ(A;Abar,zbar,c).           (D6b.1)

The overbars are essential and are easy to lose in PDF text extraction: the
denominator is the current marginal predictive `Abar zbar`, not the
sample-specific `Azbar`. During one coordinate update it is held fixed while
`μ3` is evaluated as a function of `A`; the outer message iteration brings
`Abar` to self-consistency. A scalar form is

    φ(A;Abar,zbar,c) := log μ3(A;Abar,zbar,c)
      = Σ_{x,k} zbar_k A_{xk} log A_{xk}
        + Σ_x (Azbar)_x [log c_x − log pbar_x].                (D6b.2)

Every `0 log 0` and zero-times-log-ratio term is exactly zero. If a sampled
predictive `(Azbar)_x>0` while `c_x=0`, the score is `−∞`, correctly assigning
zero functional weight. A positive sampled predictive with `pbar_x=0` makes
the score `+∞` and exposes an incoherent cavity instead of being floored.
When `(Azbar)_x=0`, that coordinate contributes zero even at a support
boundary. No probability floor changes support.

**Non-closure.** A Dirichlet natural log-message is
`const + Σ_{x,k} η_{xk} log A_{xk}`, affine in `log A`. Equation (D6b.2)
contains both linear `A_{xk}` and `A_{xk} log A_{xk}` terms; the effective
coefficient of `log A_{xk}` therefore varies with `A`. Its Hessian cannot
equal that of a Dirichlet log monomial on an open subset. A column with
`zbar_k=0` is the vacuous exception, but there is no Dirichlet family that
represents the general message. It must not enter the D6a concentration/count
slots or be silently moment-matched.
*Executable:* `test_mu3_log_density_matches_expanded_formula_and_is_not_dirichlet`.

For planning slices `t=1…T` that share `A`,

    Φ(A;Abar) := Σ_t φ(A; Abar,zbar_t,c_t),                    (D6b.3)
    q*(A) := q0(A) exp Φ(A;Abar) / Z,
    Z := E_{q0}[exp Φ(A;Abar)],   Abar=E_q*[A].                (D6b.4)

where the base
`q0(A)=∏_k Dir(A_.k|α_.k)` must be finite and strictly proper. A missing,
vacuous, non-finite, zero, or negative base concentration is a hard error.
The same draw `A_s` is evaluated against every slice before weights are
normalized; normalizing slice-by-slice would define a different posterior.

Equation (D6b.4) is a fixed-point definition: a coordinate update evaluates
weights with the previous `Abar`, and repeated updates with reused particles
must make both weights and `Abar` stationary.

The Figure 5 GFE composite needs only two primitive weighted moments:

    Abar       = E_q*[A]
    hbar       = −E_q*[diag(Aᵀ log A)]
    r_t        = log c_t − log(Abar zbar_t)
    rhobar_t   = Abarᵀ r_t − hbar
    Ephi_t     = zbar_tᵀ rhobar_t
    EΦ         = Σ_t Ephi_t.                                  (D6b.5)

`Abar zbar_t` is the posterior predictive observation belief. `rhobar_t`
drives the state-side fixed point/message and gives the Figure 5 average
energy `U_x,t=−zbar_tᵀrhobar_t`. There is no `E[log A]` in this local
composite; it belongs to the separate Dirichlet base/prior ledger. Because
(D6b.4) is the coordinate-optimal parameter belief for the held cavity,

    KL(q* || q0) = E_q*[Φ(A;Abar)] − log Z,
    KL(q* || q0) − E_q*[Φ(A;Abar)] = −log Z.                  (D6b.6)

Thus the complete parameter ledger is one prior/posterior KL correction plus
all slice energies, or equivalently `−log Z`; it is not just `−EΦ`.

### D6b.3 Importance-sampling and status contract

**Proposal and weights.** Draw `S` independent matrices
`A_s ~ q0(A)` in canonical column order. The proposal is exactly the proper
base, so no proposal-density subtraction remains:

    ℓ_s = Φ(A_s;Abar)
    log ŵ_s = ℓ_s − logsumexp(ℓ_1,…,ℓ_S)
    ŵ_s = exp(log ŵ_s)
    log Zhat = logsumexp(ℓ_1,…,ℓ_S) − log S.                  (D6b.7)

`−∞` scores have exactly zero weight. NaN or `+∞` scores are divergence. If
every score is `−∞`, normalization is undefined and the result is divergence,
not a uniform fallback. `Abar` and `hbar` in (D6b.5) are weighted sums with
`ŵ_s`; the resulting `Abar` is the next fixed-point iterate. The
finite-sample ledger uses
`KLhat = Σ_s ŵ_s ℓ_s − log Zhat` and total parameter contribution
`−log Zhat`, preserving the identity in (D6b.6) for the empirical base
measure.

**Diagnostics and collapse.**

    ESS := 1 / Σ_s ŵ_s²,        ESS fraction := ESS/S.         (D6b.8)

The runtime configuration supplies a dimensionless minimum ESS fraction.
Below it, the attempt is `budget_exhausted`: diagnostics and particles may
be inspected, but its beliefs/score cannot be required as converged.
Resampling weighted particles would duplicate existing information without
improving the importance estimate, so the built-in evaluator does not hide
collapse behind SIR. The response is to increase `S`, improve the proposal in
a future registered evaluator, or start an independent refresh epoch; a
refresh consumes a new split key and replaces all particles.

**PRNG and reuse.** An attempt receives one explicit key. At each refresh,
exactly

    next_key, sample_key = jax.random.split(key)

is performed; `jax.random.split(sample_key, K)` then assigns one subkey to
each of the `K` columns in canonical order and each subkey draws `S`
Dirichlet columns. `next_key` is returned. The particles and weights are reused through
every deterministic message sweep of that attempt (common random numbers).
Identical evaluator metadata, arrays, configuration, and input key must replay
bitwise. Samples/weights are runtime state, not template/blob state; a runtime
checkpoint may store them for mid-attempt replay, while a fresh reconstruction
needs only the template, blobs, configuration, and key.

For the rolling trial lifecycle, the proper base is the exact raw A posterior
written by `Learnables` before preparation. `RollingAgent` owns a small planning
runtime `(epoch, key_cursor, RunResult, invalidation flags)`. A solved unchanged
trial returns the same result and consumes no key. Perception accumulates the A
sufficient statistic once, then invalidates planning; exact equality between
the next written count rows and stored `base_alpha` is required before any
particle reuse. The `trial` policy draws a new set at that boundary. Under the
explicit `posterior` policy a context-only rollover reconstructs a fresh
message state with the stored samples and base moments, then solves the same
sample objective without another split. Checkpoints store the full prepared
state rather than regenerating samples, so continuation does not depend on a
later JAX sampler implementation.

*Executable:* `test_trial_loop_matches_direct_engines_and_checkpoint_replays`,
`test_metadata_key_and_fixed_epoch_are_explicit`, and
`test_float32_ten_slice_agent_keeps_one_shared_evaluator_finite`;
`test_rolling_agent_two_plans_match_binary_quadrature_and_parameter_ledger`
supplies the independent two-plan quadrature and ledger anchor.

**Reference-implementation boundary.** Part II requires a functional message,
importance-sampled expectations, a stochastic GFE trace, and post-burn-in
policy averaging; it does not prescribe the proposal, particle count, ESS
threshold, resampling, reuse, or RNG protocol. LAIF commit `9453782`
represents `mu3` as a Julia closure, requests 20 `LeftProposal` particles,
and relies on RxInfer/BayesBase's global-RNG bootstrap sampler and
linear-space weights. BayesBase reuses particles while multiplying further
factors and resamples below ESS `S/10`, but neither ESS nor weight-collapse
status reaches the LAIF result. Those are reference choices, not paper
requirements. CFFG keeps the useful left/base proposal and particle reuse,
but makes the evaluator serializable, normalization log-space, the key
explicit, and collapse loud; it does not copy the reference's hidden
resampling/global-RNG behavior.

**Termination and policy scores.**

- `converged`: with one fixed particle set, the ordinary FE/message
  stationarity checks pass, every required expectation and FE is finite, and
  the ESS threshold passes.
- `budget_exhausted`: the deterministic sweep budget ends before stationarity,
  or weights are finite but ESS collapses below the configured threshold.
- `diverged`: the base is improper; inputs, deltas, expectations, or FE become
  non-finite; a score is NaN/`+∞`; or all scores are `−∞`.

A policy-score estimator may run independent refresh epochs, discard a
configured burn-in prefix, and average `log Zhat`/complete FE over the
remaining **converged** epochs. Burn-in length is estimation policy (Part II
uses ten iterations), never a convergence criterion. Any diverged epoch makes
the enclosing result diverged; a budget-exhausted/collapsed epoch prevents a
converged label even if an average looks stable.

**Hostile-input contract.** Proper sparse Dirichlet bases are accepted without
floors. Boundary samples with exact-zero `A` entries use (D6b.2)'s exact
support conventions. A goal zero yields either a zero contribution when the
sample predictive is zero or `−∞` weight when it is positive; a zero marginal
predictive facing positive sampled mass is incoherent and diverges. Float32
uses dtype-derived validity checks, and matrix/state dimensions 10–100 times
the binary oracle must remain finite.
*Executable:* `test_mu3_importance_sampling_matches_binary_quadrature`,
`test_mu3_replays_from_explicit_key_and_combines_slices`,
`test_mu3_hostile_support_weight_collapse_float32_and_scale`.

## D6c — The conjugate `mu1(c)` learned-goal message (cff-gjjk)

### D6c.1 The plain-English version

A fixed goal vector says which future observations are desirable. A learned
goal instead carries a proper Dirichlet belief over that vector. One future
GFE composite predicts observation mass `n=Abar zbar`; that mass is one soft
categorical count, so its message to the goal is an ordinary Dirichlet
message. Composites sharing the same serialized goal prior add their soft
counts before forming one posterior.

Inference sweeps refine that one trial's prediction. They do not represent
new trials: every sweep replaces `n`, while the incoming `alpha0` remains
fixed. Only an explicit completed-trial boundary may carry the converged
posterior forward as the next prior.

### D6c.2 Message, state coordinate, and shared goals

For future slice `t`, let

    n_t := Abar_t zbar_t,                 Σ_x n_t,x = 1.

Then Part II Fig. 5/App. C.3 gives

    log mu1,t(c) = n_t^T log c + const,
    mu1,t(c) proportional to Dir(c | n_t + 1).                (D6c.1)

The `+1` describes the message distribution itself. Multiplication by the
proper prior `q0(c)=Dir(alpha0)` adds natural parameters, so for every slice
sharing that goal prior,

    n := Σ_t n_t,             q(c)=Dir(alpha),  alpha=alpha0+n. (D6c.2)

There is no additional `+1` in `alpha`. The GFE state coordinate consumes

    ell := E_q[log c] = digamma(alpha)-digamma(Σ_x alpha_x),
    rho_t = Abar_t^T[ell-log(Abar_t zbar_t)]-hbar_t.           (D6c.3)

It is `E[log c]`, not `log E[c]`. In the serialized graph, each categorical
goal connects its `statistic` role through a parameter edge to a Dirichlet
prior, carries an explicit mean-field bead, and its facing P-substitution
declares `goal_uncertainty=dirichlet`. Reusing one prior blob name ties the
composites even when their hidden-state cardinalities put them in different
execution groups. Equal prior values under distinct blob names do not tie them.
One `GoalParameterPlan` collects original predictive contributions across all
groups and writes `n` back to every derived carrier edge. Its runtime owner is
`goal_parameters.py`. Every local solve reads the pre-sweep shared coordinate;
pooling follows all those solves and precedes damping. The trust-region chart
also gives each shared message side one coordinate, copied to every alias in
trial states. The template and prior blob are source truth;
posterior messages are runtime state.

### D6c.3 Complete ledger and rollover boundary

The goal-specific complete contribution is

    KL[Dir(alpha)||Dir(alpha0)] - n^T ell.                    (D6c.4)

Combining likelihood, prior, posterior entropy, and parameter-edge correction
before evaluation gives the stable form

    log B(alpha0)-log B(alpha)
      +(alpha-alpha0-n)^T ell.                                (D6c.5)

At the coordinate update (D6c.2), the expected-log coefficient is exactly
zero. This is the same cancellation as D6a and is essential for sparse
float32 priors. The remaining per-slice GFE terms—predictive entropy,
`hbar(A)`, state entropy, and ordinary state-factor/edge terms—remain in the
structural ledger; `-n_t^T ell` must not be counted a second time.

Away from that coordinate relation, retain the entire coefficient in D6c.5.
The ledger holds the evaluated state's `alpha` fixed and recomputes its local
predictions. Their sum is not generally the stored count message, nor the
stored predictive output after damping. `shared_goal_energy` charges this
full expression once across shape groups.

`Engine.goal_prior_updates` reads the converged posterior keyed by serialized
prior blob. `Engine.with_goal_priors` accepts it only on a fresh state, making
the trial boundary explicit and refusing within-trial accumulation.
`RollingAgent` stages the latest converged planning posterior and rolls it
once in `reset_episode`; checkpoints preserve both current and staged values.

This implements the generic unobserved Fig. 5 message. The particular Part II
section 7.2 reference experiment instead rolls the final *observed* outcome
message at trial completion. CFFG does not silently equate those protocols:
observed categorical learning remains the ordinary D6a path, while predicted
goal rollover is this explicit API.

Exact-zero predicted mass contributes exactly zero count. Proper positive
`alpha0` keeps the coordinate proper; missing/improper priors are compile
errors. No absolute probability floor is introduced, and float32 plus
observation cardinality tens of times larger than the tiny oracle must remain
finite.
*Executable:* `test_mu1_shared_goal_posterior_and_complete_fe_match_direct_algebra`,
`test_sweeps_replace_mu1_and_explicit_rollover_adds_one_new_trial`,
`test_float32_sparse_zero_mass_and_uncomfortable_observation_scale`.
Cross-group authorities in `test_shared_goals.py`:
`test_mixed_shape_step_pools_one_shared_goal`,
`test_off_coordinate_energy_charges_one_shared_parameter_term`,
`test_outer_trial_coordinates_keep_goal_aliases_equal`, and
`test_float32_sparse_support_with_32_shared_composites`. The off-coordinate
oracle deliberately changes both count and stored predictive messages; its
constant channels make the expected complete energy independent of the
production local solver.

---

## D7 — Mixed-GFE outer stationarity and globalization (cff-wafy, cff-weff)

### D7.1 The plain-English version

The Newton solve inside one GFE composite answers a local question: given all
of its incoming messages, which local belief satisfies the P-substituted
stationarity equation? A graph with several such composites asks a second,
coupled question: are all of those local answers, the ordinary factor
messages, and the beliefs on their shared edges mutually consistent? Repeating
one damped sweep is only a fixed-point iteration for that second question.
Part II proves what its fixed points mean, but deliberately does not prove
that the iteration reaches one.

This section turns the complete undamped mixed sweep into an explicit
nonlinear residual. Categorical message scales are removed, exact zeros stay
on a named support face, and Gaussian and exponential-family messages use
minimal natural coordinates. On a proper fixed-support chart, a zero residual
is exactly a stationary point of the original constrained Bethe/generalised
free energy, subject to the assumptions below. The residual does not replace
that free energy; it is a numerical representation of its stationarity
equations.

The familiar convergent-Bethe double loop does not cover the complete
supported GFE objective. Its entropy bound is valid, but learned/facing
beliefs and parameter mean fields introduce multilinear P-substitution terms
that can remain non-convex after every concave Bethe edge entropy is bounded.
Route A is therefore rejected for the general engine contract. Route B is a
gauge-reduced trust-region least-squares solve of the original stationarity
residual. Trust-region globalization guarantees progress toward a stationary
point of the numerical merit under standard assumptions; only a zero residual
is inference convergence. A nonzero stationary merit is a solver failure, not
an approximate belief and never a policy score.

### D7.2 The complete constrained problem

Let \(\mathcal V\) contain every node-local belief and \(\mathcal E\) every
interior edge belief represented by a sum/GFE plan. This includes ordinary
factor beliefs, P-substituted composite beliefs, mean-field parameter beliefs,
and Gaussian or other minimal exponential-family beliefs. Write the unchanged
ledger assembled by `_make_bfe` as

    F(b) = sum_{a in V} F_a(b_a) + sum_{e in E} H_e(b_e).       (D7.1)

`F_a` is the node-local term already registered for the factor: an ordinary
Bethe/VMP term, the direct Part-II or D1 P-substituted term, or a parameter
prior/posterior term from D6. The positive edge entropy is the Forney
overcounting correction. Equation (D7.1) is notation for the existing
complete ledger, not a new objective or a replacement decomposition.

Let \(h(b)=0\) collect an independent basis of:

- normalization and node-to-edge marginal consistency;
- Gaussian/EF moment consistency in the family's minimal sufficient
  statistics;
- P-substitution, mean-field, parameter-tying, observed-data, and active hard
  constraints; and
- the exact BP-island boundary conditions represented by scheduled messages.

Structural zeros and forbidden hard-factor configurations are removed from
the domain. Any remaining probability inequality is written \(b_i\geq0\).
The complete Lagrangian is

    L(b, lambda, nu) = F(b) + lambda^T h(b) - nu^T b,
    h(b)=0, b>=0, nu>=0, nu_i b_i=0.                           (D7.2)

This includes the node and edge equations whose multipliers are represented
by messages. It also makes the support qualification explicit: removing a
structural zero is an equality-domain choice, while setting an otherwise
feasible probability to zero is an active inequality and needs the
corresponding complementary multiplier.

#### Fixed-support charts

Choose one support stratum \(\Omega_S\). Every allowed categorical node and
edge belief is strictly positive relative to \(S\), every required
normalizer and entropy is finite, and every aggregate continuous-family
belief is proper. A categorical message \(m\) with support \(S_m\) has one
irrelevant additive scale. With a deterministic reference
\(r_m\in S_m\), pack

    c_m(m) = (m_i - m_{r_m})_{i in S_m \ {r_m}}.               (D7.3)

Coordinates outside \(S_m\) are absent, not floored. Thus
`c_m(m + kappa)=c_m(m)` exactly. Gaussian messages pack their information
vector and the independent entries of the symmetric precision matrix; the
aggregate edge precision, not every individual cavity, must be positive
definite. Other EF messages pack only the natural coordinates of a minimal
family whose aggregate parameter lies in the interior of its proper domain.
Parameter-message and fixed-particle `mu3` coordinates that change during an
outer sweep are part of the state too; immutable model parameters are not.

For numerical scaling, the continuous blocks are first made dimensionless.
A Gaussian block uses a fixed affine whitening from the proper initial edge
belief. A minimal EF block uses the Fisher metric at a fixed proper reference
belief. Categorical log ratios are already dimensionless (nats). These
changes are invertible coordinate choices; they do not alter a belief or
stationarity equation.

Let \(C\) concatenate these reduced coordinates and let \(C^{-1}\) choose the
canonical representative of every message class. The admissible chart
\(\mathcal Y_S\) contains exactly the states for which every local rule,
ledger term, and scheduled propagation is defined and proper. Marginal
consistency need not hold away from a root.

#### One complete outer map

For fixed model parameters, fixed particle samples, fixed support, and a fixed
hard-constraint phase, define \(T\) as:

1. solve every Part-II/D1 local P-substituted belief to its reported
   stationary solution;
2. execute every other non-eliminated iterative factor/parameter update;
3. refresh scheduled BP islands (a complete two-pass on each forest and one
   colored sweep on each cyclic island); and
4. put every message and auxiliary outer coordinate in its canonical gauge.

There is no damping in \(T\). Damping changes the path but not a fixed point
when it is a finite convex combination; it is not part of the equations being
solved. A stochastic `mu3` refresh defines a new sample objective and hence a
new \(T\); one residual solve reuses one particle set as required by D6b.
Define

    R(y) = C(T(C^-1(y))) - y.                                  (D7.4)

Forest BP messages may be absent from \(y\) only when their conditional
fixed point is unique up to gauge and the schedule reconstructs it exactly.
Cyclic BP messages remain coordinates. A composed schedule is not allowed to
hide a block whose intermediate equation is unsatisfied.

### D7.3 Zero-set equivalence

**Assumptions.**

1. **Fixed relative interior.** \(\Omega_S\) is constant in a neighborhood,
   all allowed categorical beliefs are positive on it, no positive predicted
   mass faces an exact-zero likelihood/goal, and all normalizers are finite.
   Structural support is nonempty.
2. **Proper minimal continuous families.** Aggregate Gaussian precisions are
   positive definite; aggregate EF natural parameters lie in the interior of
   a minimal proper family; the required log partitions and moments are
   twice continuously differentiable.
3. **Regular constraints.** The selected equalities in (D7.2) are an
   independent local basis (LICQ after normalization and multiplier gauges
   are removed), and `F` and those constraints are \(C^2\) on the chart.
4. **Complete local equations.** Every registered update used by \(T\) is
   algebraically equivalent in both directions to its block stationarity and
   consistency equations. Its conditional stationary belief is unique on
   the chosen branch and depends smoothly on the incoming coordinates. For
   the fixed-channel Part-II and D1 discrete composites this follows from the
   positive tangent Hessian

       diag(1/q) + P^T diag(1/(Pq)) P;

   `_gfe_solve_result` selects that gauge-reduced solution. Ordinary positive
   sum-product updates and proper minimal-family conditional updates have the
   analogous unique normalized solution. A future rule that cannot establish
   branch completeness must retain its local belief as an unknown and add its
   direct stationarity equation instead of hiding it behind a selected solve.
5. **Faithful scheduling.** One application of \(T\) covers every
   non-eliminated block. An eliminated forest has a unique conditional BP
   solution and is reconstructed by the exact directed pass. Fixed points of
   a colored or flooding sweep are the simultaneous ordinary update
   equations.
6. **Determinism and exact local status.** Particles, data, active constraints,
   and parameters are fixed during the solve. Every local inner status is
   `converged`; an exhausted/diverged local solve leaves \(T\) undefined for
   purposes of a convergence claim.

**Theorem 1 (fixed point iff constrained stationarity on
\(\Omega_S\)).** Under Assumptions 1–6,

    R(y)=0

if and only if the beliefs reconstructed from \(C^{-1}(y)\) and their message
multipliers satisfy every local Part-II, D1, ordinary-factor, parameter, and
edge consistency equation, and therefore the first-order KKT equations
\(\nabla_b L=0,\ h(b)=0\) of (D7.2) relative to \(\Omega_S\).

*Proof.* Because \(C\) is a bijection on message gauge classes, \(R(y)=0\)
means \(T(m)\) and \(m\) differ only by harmless categorical scales. By
faithful scheduling, every non-eliminated outgoing message is therefore a
fixed point of its registered update and every eliminated forest message is
its unique conditional BP solution. Assumption 4 converts each of those
relations, in both directions, into the corresponding node-local
stationarity equation and its outgoing marginal/moment consistency
equations. The cavity messages are precisely the independent Lagrange
multipliers for those consistencies; P-substitution and mean-field updates
supply their declared constraint equations. Collecting all blocks gives
\(\nabla_bL=0\) and \(h(b)=0\). Conversely, KKT stationarity gives every
registered local relation by the Part-II/D1 and ordinary message
derivations. Conditional uniqueness makes each update return the same belief
and outgoing multiplier class, while exact forest elimination reconstructs
the same island solution. Hence \(T(m)\) is gauge-equivalent to \(m\), so
\(R(y)=0\). ∎

The fixed-support qualification is material. If a zero coordinate is an
optimizer-chosen active inequality rather than a structural zero, Theorem 1
proves only tangent stationarity on that face. Full KKT additionally requires
the excluded-coordinate dual inequalities and complementarity in (D7.2).
Support changes therefore use an active-set certificate and restart with a
new exact chart; they never add a probability floor. A hardening-phase change
likewise creates a new active problem and restarts \(T\).

The theorem is first order only. With an independent constraint Jacobian
\(A\), let \(Z\) span its tangent/null space. A negative eigenvalue of
\(Z^T\nabla^2_{bb}LZ\) rules out a local minimum; positive definiteness is a
sufficient second-order condition for a strict constrained local minimum.
Positive semidefiniteness alone is only necessary. A stationary or strict
local point is not a global minimum of the generally non-convex GFE. Global
optimality needs a separate convexity or comparison proof.

### D7.4 Route A: why the available double-loop proof does not close

For an ordinary Kikuchi objective, the non-convex entropy allocation can be
written

    F_K = sum_a (E_a(b_a) - H(b_a)) - sum_e c_e H(b_e),

where every potential energy \(E_a\) is linear and the constraints are
linear marginal consistencies. Whenever a counting number makes
\(+|c_e|H(b_e)\) concave, KL nonnegativity supplies

    H(b) <= -sum_i b_i log b_i^(old).                          (D7.5)

This affine cross-entropy bound touches in value and first feasible
variation: the omitted `+1` multiplies `sum_i delta b_i=0`. Replacing all such
concave entropies gives the convex inner problem used by convergent
Bethe/Kikuchi double loops. Exact zeros already require a fixed support,
because crossing a zero of the reference belief makes (D7.5) infinite.

The fixed-channel, fixed-goal P-substituted discrete term is not itself the
obstruction. With \(\pi\) the state-side belief, \(r=P\pi\), and fixed
conditional entropy/goal terms, its nonlinear part is

    -H(pi) - H(r),

whose Hessian is the positive matrix in Assumption 4. The ordinary positive
edge entropies could still be bounded by (D7.5).

The complete supported Part-II objective can vary both facing beliefs and
parameter mean fields. With
\(\pi(a)=q(z)q(\theta)\), \(\tau(b)=q(w)q(\phi)\),
\(r=P\pi\), and \(C(x,b)=-\log\tilde p(x|b)\), the local form contains

    G(pi,tau) = -H(pi) - H(tau) - I_pi(a;x) + r^T C tau.       (D7.6)

The last term is bilinear in \(r,\tau\), while \(r\) is multiaffine in the
actual mean-field coordinates. It need not be convex after all concave edge
entropies are bounded. A binary counterexample makes \(x=a\) deterministic
and uses a binary facing channel that matches \(x=b\) with probability
\(1-\epsilon\). At \(\pi=\tau=(1/2,1/2)\), scalar simplex coordinates give

    Hessian(G) = [[8, Delta_C], [Delta_C, 4]],
    Delta_C = 2 log(epsilon/(1-epsilon)).

For `epsilon=0.01`, `Delta_C^2 > 32`, so this node-local Hessian is
indefinite. Lifting \(a,b\) into one region belief makes the energy linear
only by retaining the non-convex independence/P-substitution constraint;
dropping it changes the variational family. Freezing the recursive
predictive marginal also has the wrong direction:

    sum r log r - sum r log r_old = KL(r || r_old) >= 0,

so the cross term is a lower bound on `-H(r)`, not the needed upper bound.

Therefore the Heskes/Zoeter construction does not prove a tight convex
majorizer for the unchanged general supported objective, and no cited result
supplies a support-safe Gaussian/EF factor-local replacement. This is the
precise failed assumption for Route A; it is not an impossibility theorem for
every future quadratic or Bregman majorizer on a restricted chart. Route B is
the selected production contract for cff-weff.

### D7.5 Route B: gauge-reduced trust-region residual solve

Let the dimensionless reduced coordinates be divided into natural blocks.
At the start \(y_0\), freeze a positive diagonal residual scale \(D\). For
block \(j\),

    d_j = max(rms(R_j(y_0)), sqrt(finfo(dtype).eps) * u_j),    (D7.7)

where \(u_j\) is one unit in the block's dimensionless chart. The second term
only prevents division by zero at the dtype's resolution; it is not a model
ridge or probability floor. Recomputing \(D\) after each step would silently
change the merit and invalidate the standard theorem. Define

    f(y) = D^-1 R(y),       phi(y) = 0.5 ||f(y)||_2^2.         (D7.8)

Every `R=0` point is a global minimum of this numerical merit and, by
Theorem 1, a stationary point of the original constrained GFE. The converse
for stationary merit points is false:

    grad phi = J_f^T f = 0

allows a nonzero `f` orthogonal to the range of \(J_f\). Consequently `phi`
never enters `_make_bfe`, a node ledger, a learned parameter, or a policy
score. It is only step-acceptance machinery.

At iteration \(k\), approximately solve the Gauss--Newton trust-region model

    min_s 0.5 ||f_k + J_k s||^2
    subject to ||S_k s|| <= Delta_k,                           (D7.9)

where \(S_k\) is a uniformly positive dimensionless coordinate scaling. Use
the standard ratio of actual to predicted merit reduction to accept/reject
the trial and shrink or grow `Delta`. Levenberg--Marquardt is the equivalent
normal equation

    (J_k^T J_k + lambda S_k^T S_k) s = -J_k^T f_k,

with `lambda` chosen as the trust-region multiplier, not by an independent
ridge schedule. Truncated Steihaug CG needs only `Jv` and `J^T v`; it must
return at least a uniform fraction of the Cauchy decrease. A dense global
Jacobian, normal matrix, or Hessian is outside this contract.

The local Corollary-1/D1 Newton loop must not be differentiated by unrolling
its iteration count. Its gauge-reduced stationarity equation supplies an
implicit JVP solve, and the transposed local system supplies the VJP. This
keeps global products matrix-free, preserves the exact local equation, and
batches the existing small factor-local linear solves under `vmap`.

**Theorem 2 (honest globalization claim).** Suppose the initial merit
sublevel set is compact and remains a positive distance inside one admissible
\(\mathcal Y_S\); \(f\) is continuously differentiable there with a bounded,
Lipschitz (uniformly continuous is sufficient for weaker forms) Jacobian; the
trust-region scaling is uniformly bounded and positive; and every subproblem
step achieves the required Cauchy decrease. Then the standard trust-region
iteration from any admissible initial point drives
`||grad phi||` to zero (or has `liminf=0` under the weaker hypotheses), and
every accumulation point is first-order stationary for `phi`.

This is globalization from an admitted start, not global minimization. It
implies `R -> 0` only with an additional error bound, for example a uniform
positive lower bound on the smallest singular value of \(J_f\) at limit
points (a nonsingular square residual Jacobian is sufficient). Near a
zero-residual root with a Lipschitz full-rank Jacobian, Gauss--Newton/LM has
its usual fast local convergence. Without that rank/error-bound condition, a
nonzero stationary merit, rejected-step radius collapse, or exhausted Krylov
budget is `budget_exhausted` with a diagnostic such as
`stationary_nonroot`; it is never `converged`.

Numerical convergence requires all of:

- the existing dtype-scaled componentwise fixed-point certificate for `R`;
- a finite, stationary original `_make_bfe` ledger at the same accepted
  state;
- every local inner status `converged`; and
- support/propriety and predicted-reduction checks passing.

A non-finite current residual, derivative product, belief, or original FE is
`diverged`. An empty/inconsistent structural support is `infeasible`.
Improper/vacuous input for which a registered local rule is undefined is
`diverged` unless the compiler can prove structural infeasibility. A finite
trial that leaves the proper chart is rejected and shrinks the trust region;
repeated rejection to a dtype-derived radius floor is exhaustion, not
success. Root, gradient, rank, predicted-reduction, and radius floors use the
active dtype, chart units, block norms, and problem dimension. A relative
rank threshold has the form
`finfo(dtype).eps * max(m,n) * operator_scale`; no literal numerical ridge is
introduced.

### D7.6 Runtime realization and executable anchors

The cff-weff production implementation follows the
**Runtime numerical or status invariant** row of `ARCHITECTURE.md`.

- `rules._gfe_solve_result` keeps its Part-II/D1 equations and truthful local
  status. Its returned solution has a custom implicit JVP through the reduced
  local stationarity system; JAX transposes that rule for the VJP. Newton
  iterations are stopped before the implicit solution is attached, so their
  iteration count is not unrolled into the outer derivative. On an exact
  support face, supported incoming logits are normalized before the
  dtype-scaled finite workspace guard is applied; the guard therefore cannot
  turn a categorical gauge shift into a different local root or derivative.
- `_make_step` and the mixed schedule define the pure undamped \(T\).
  `_make_bfe` remains the sole scientific objective. `Engine.run` owns the
  trust-region controller, selects it by default for every GFE plan, and
  rejects nonzero `damp` unless `outer_solver="picard"` explicitly requests
  the historical diagnostic.
- `RunResult` retains the original FE trace/residual and local status, and
  exposes outer residual/merit, merit-gradient norm, trust radius,
  accepted/rejected steps, outer-map/trial/Krylov counts, solver compilations,
  and a termination reason distinguishing root, stationary-nonroot/stagnation,
  trust/Krylov budget exhaustion, rejected improper trials, infeasibility, and
  divergence. `require_converged` continues to reject every non-root outcome.
- Host/JIT implementations share one residual, packing, support, and
  acceptance kernel. Flooding and mixed schedules have different maps but
  the same fixed-point equations under Theorem 1; both need parity tests.
  Exact two-pass schedules are N/A because they contain no iterative GFE
  structure. A hardening phase freezes its active constraints and restarts
  the solve.
- Packing shapes and support masks are static per active-set solve. A
  compiled Steihaug/JVP/VJP kernel accepts per-instance coordinates, scales,
  radii, Krylov masks, parameters, and observations under `vmap`. Engine-local
  caches key those kernels by exact support, dtype, continuous chart metric,
  and hardening phase; changed runtime values remain dynamic arguments. The
  current host controller owns each instance's active-set/status transitions;
  cff-wbhu can put that same carry in `lax.scan` rather than inventing another
  numerical method. Checkpoint decoding is N/A for this change.
- Before compiling a trust kernel, at most 128 half-damped Picard steps, and
  never more than half the caller's budget, may propose a root on the current
  exact support face. Mu3 plans skip this phase because their total budget
  must cover repeated outer-root and particle-moment block alternations. For
  other plans this is only a candidate phase: success still requires an
  independently evaluated undamped residual, converged inner status, feasible
  unchanged support, finite stationary `_make_bfe` ledger, and the matrix-free
  merit-gradient diagnostic. Failure to certify transfers the remaining
  iteration budget and latest valid state to the unchanged trust-region
  controller.
- Forest factor-written messages are eliminated only on exact conditional
  islands; external half-edges remain fixed, and cyclic BP messages remain in
  the chart. Hard δ support is frozen once for its phase; multiple δ variables
  activate in deterministic sequence so a later decision conditions on the
  earlier fixed support. Fixed-particle `mu3` moments are refreshed as an exact
  factor-local block between fixed-scale trust solves, and convergence
  additionally requires that block to stop changing at the requested
  tolerance.
- A bounded Anderson/Picard shadow sequence is isolated from the production
  iterate. Its nonroot candidates enter only through the same Cauchy/model and
  actual-reduction safeguards as a trust step. A shadow state may be selected
  directly only after it independently passes the complete raw-root, local,
  support, and unchanged-ledger convergence certificate.
- The horizon-8 cycle fixture has 24-logit GFE-TM-scale members but is caused
  by eight ordinary observation-GFE composites coupled through a learned
  non-epistemic transition. The frozen state is an exact 8-position by
  2-mode corridor with beliefs `position=[1,0,...]`, `mode=[0,1]`, and one
  conjugate increment over a `6x8x8` Dirichlet prior:
  `alpha[5,0,0] += 1`. It contains no novelty schedule or behavioral result.
  The existing `32x4` D1 hostile case remains the larger local-rule anchor.

Executable anchors:

- `test_packed_outer_residual_is_gauge_invariant_at_checked_fixed_points`
  checks the chart, gauge invariance, and zero residual on the analytical
  T-maze and D1 fixed points.
- `test_two_composite_chain_matches_constrained_objective` independently
  optimizes the complete simplex objective, checks KKT consistency and the
  ledger, and classifies its selected point with the projected Hessian.
- `test_one_update_corridor_is_globalized_without_changing_the_equations`
  freezes the reusable one-update state, certifies its trust-region root, and
  retains explicit Picard mode as the bounded-cycle reproducer.
- `test_inner_solution_uses_implicit_stationarity_derivative` compares the
  local implicit JVP with an independent central difference.
- `test_matrix_free_trust_kernel_has_single_and_vmap_parity` checks that the
  compiled per-instance kernel preserves steps and diagnostics under `vmap`.

References: Part II Sections 4.4--4.5 and Appendix B; Heskes, Albers, and
Kappen, *Approximate Inference and Constrained Optimization* (2012);
Heskes and Zoeter, *Generalized belief propagation for approximate inference
in hybrid Bayesian networks* (2003); Yedidia, Freeman, and Weiss,
*Constructing Free-Energy Approximations and Generalized Belief Propagation
Algorithms* (2005); Moré, *The Levenberg--Marquardt Algorithm:
Implementation and Theory* (1977); Conn, Gould, and Toint,
*Trust Region Methods*; and Steihaug, *The Conjugate Gradient Method and
Trust Regions in Large Scale Optimization* (1983).

## D8 — Recurrent stick-breaking transitions via Pólya–Gamma VMP (cff-k9ws)

### D8.1 The plain-English version

An rSLDS needs a regime transition whose probabilities change with a
continuous state. A dense transition table cannot express that dependence
without discretizing the state. The recurrence link instead gives each
previous regime its own sequence of logistic sticks: reject sticks until one
is accepted, with the last regime receiving the remaining probability.

Logistic factors are not conjugate to a Gaussian drive. Pólya–Gamma
augmentation makes each stick conditionally Gaussian; replacing its auxiliary
variable by the best independent factor gives the familiar
Jaakkola–Jordan bound. The resulting coordinate updates are closed-form
categorical log scores and Gaussian information parameters. Because the bound
is below the exact expected log transition, the recurrence energy is above
the exact variational energy at the same beliefs. This is conservative in the
opposite direction from the optimistic hybrid-GFE approximation in D5.

### D8.2 Model and bound

Let previous regime `i`, destination `j`, and stick `l` range over
`i,j in {0,...,K-1}` and `l in {0,...,K-2}`. For Gaussian drive `x`,

    psi_il(x) = R_il^T x + r_il,
    b_jl = 1[j >= l],        a_jl = 1[j = l],
    kappa_jl = a_jl - b_jl / 2.                         (D8.1)

Only sticks with `b_jl = 1` are at risk. The final destination rejects every
stick. The conditional therefore has log probability

    log p(j | i,x) = sum_l b_jl log sigma((2 a_jl - 1) psi_il). (D8.2)

The exact Pólya–Gamma identity introduces an auxiliary and makes the exponent
quadratic in `psi_il`:

    exp(a_jl psi_il) / (1 + exp(psi_il))^b_jl
      = 2^-b_jl exp(kappa_jl psi_il)
          integral exp(-omega psi_il^2 / 2)
                   p_PG(omega | b_jl,0) d omega.              (D8.3)

Under `q(i) q(j) q(x) product_il q(omega_il)`, let

    alpha_i = q(i), beta_j = q(j), q(x) = N(m,S),
    mu_il = R_il^T m + r_il,
    xi_il^2 = E_q[psi_il^2] = mu_il^2 + R_il^T S R_il.        (D8.4)

For each at-risk Bernoulli trial the coordinate-optimal tilted auxiliary is
`PG(1, xi_il)`. Its mean per at-risk trial is

    omega_bar_il = tanh(xi_il / 2) / (2 xi_il),               (D8.5)

with analytic limit `1/4` at zero. The expected at-risk mass
`B_l = sum_{j>=l} beta_j` scales this mean in the Gaussian message. Eliminating
the auxiliary gives the Jaakkola–Jordan stick score

    s_il^+ = log sigma(xi_il) + (mu_il - xi_il) / 2,
    s_il^- = log sigma(xi_il) + (-mu_il - xi_il) / 2.         (D8.6)

Thus the expected bounded log transition for `(i,j)` is the sum of failure
scores before `j`, plus the success score at `j` when `j < K-1`. A prefix sum
constructs all scores in `O(K^2)` rather than materializing a
destination-by-stick-by-source tensor.

### D8.3 Coordinate messages and energy

Write `T_ij` for that bounded score. The categorical messages are

    log m_to_out(j) = sum_i alpha_i T_ij + constant,
    log m_to_in(i)  = sum_j beta_j T_ij + constant.           (D8.7)

With `kappa_bar_l = beta_l - B_l/2`, the Gaussian information message is

    Lambda_x = sum_il alpha_i B_l omega_bar_il R_il R_il^T,
    eta_x    = sum_il alpha_i
                 (kappa_bar_l - B_l omega_bar_il r_il) R_il. (D8.8)

These equations are one coordinate-ascent VMP block. A global mixed or flood
sweep alternates them with the rest of the graph; no hidden per-stick
iterative solver exists.

Let `L_rec^JJ = sum_ij alpha_i beta_j T_ij`. The node contribution stored in
the constrained Bethe ledger is

    F_rec^JJ = -H[q(out)] - H[q(in)] - H[q(x)] - L_rec^JJ.    (D8.9)

The ordinary interior-edge entropy corrections remove duplicate entropies
globally. Since `L_rec^JJ <= E_q[log p(out | in,x)]`,

    F_rec^JJ >= F_rec^exact.                                  (D8.10)

This statement compares objectives at held beliefs; it does not claim the two
objectives have the same optimizer.

### D8.4 Between-trial Gaussian learning of recurrence weights

The same augmentation makes the stick coefficients conjugate. Augment the
drive and coefficients as

    phi = [x; 1],       theta_il = [R_il; r_il],
    h = E[phi] = [m; 1],
    M = E[phi phi^T] = [[E[xx^T], m], [m^T, 1]].             (D8.11)

For one factor belief, retain `B_l = sum_{j>=l} beta_j` and define
`kappa_bar_l = beta_l - B_l/2`. At the point coefficients used by that
inference state,

    xi_il^2 = theta_il^T M theta_il,
    omega_bar_il = tanh(xi_il/2) / (2 xi_il).                 (D8.12)

The soft source belief `alpha_i` and soft destination belief `beta_j` then
give the Gaussian natural-statistic increments

    Delta eta_il    = alpha_i kappa_bar_l h,
    Delta Lambda_il = alpha_i B_l omega_bar_il M.             (D8.13)

No regime is selected by argmax. An impossible source contributes exactly
zero to all of its sticks; a destination with zero mass contributes neither
success nor at-risk mass. Given a proper Gaussian prior
`N(theta_il; Lambda_0^-1 eta_0, Lambda_0^-1)`, independent factor occurrences
add:

    Lambda_n = Lambda_0 + sum_t Delta Lambda_t,
    eta_n    = eta_0    + sum_t Delta eta_t,
    E[theta] = Lambda_n^-1 eta_n.                             (D8.14)

The conditional posterior in (D8.14) is exact for the fixed PG coordinates.
`Engine.recurrence_weight_stats` evaluates (D8.11--D8.13) from the converged
factor belief, and `conj.PGWeights` accumulates (D8.14) between trials. The
next inference state receives the posterior mean split back into `(R,r)`.
As in D4's NIW/MNIW loop, this is an empirical-Bayes point rollover: inference
does not carry the coefficient covariance, and the next `xi` uses the point
parameters actually installed in the recurrence factor. Full joint VB over
`q(theta)q(omega)` would add coefficient uncertainty to (D8.12) and is not
claimed here.

### D8.5 Numerical contract and executable anchors

Exact categorical log-zeros are masked before normalization and never relaxed.
A wholly vacuous Gaussian drive is neutral only as a transient while a facing
prior propagates; a final vacuous or any indefinite drive is divergence. The
Pólya–Gamma mean uses its dtype-scaled series near zero, and a dtype- and
dimension-scaled tolerance admits only contraction roundoff in the nonnegative
second moment. One plan stores `O(P K^2 d)` parameters and each update costs
`O(P K^2 d^2)`; it allocates no `K^3` transition tensor.

Executable anchors in `tests/test_recurrence_link.py`:

- `test_rule_matches_independent_numpy_coordinates` checks (D8.4--D8.8).
- `test_jj_expected_log_is_a_lower_bound` checks (D8.10) with
  independent Gaussian quadrature.
- `test_engine_tracks_dense_quadrature_and_upper_bounds_evidence` compares the
  stationary posterior and global ledger against an exact scalar integral.
- `test_exact_zero_support_is_preserved`,
  `test_vacuous_and_improper_drive_are_loud_after_transient_flooding`, and
  `test_float32_k32_d16_converges_without_cubic_plan_storage` pin the hostile
  input and scale contract.
- `test_engine_weight_stats_match_independent_soft_belief_algebra` checks
  (D8.11--D8.13), while the compact `tests/test_conj.py` PG cases check
  (D8.14), loudness, and vectorized soft-regime recovery.

References: Polson, Scott, and Windle, *Bayesian Inference for Logistic Models
Using Pólya–Gamma Latent Variables* (2013); Linderman, Johnson, and Adams,
*Dependent Multinomial Models Made Easy: Stick-Breaking with the
Pólya–Gamma Augmentation* (2015); Durante and Rigon, *Conditionally Conjugate
Mean-Field Variational Bayes for Logistic Models* (2019).

---

## D9 — Structured categorical smoothing for recurrent transitions (cff-s6yy)

### D9.1 The plain-English version

D8 treats the regime before a transition and the regime after it as
independent categorical beliefs. That makes every recurrence update cheap,
but it discards the main fact a transition is meant to express: adjacent
regimes are correlated. The cff-nbo9 oracle shows that the implemented D8
messages and learning readout are correct for that independent family, while
their consecutive-state responsibilities differ materially from exact chain
smoothing.

The repair is to keep one structured belief over the complete regime path.
For a fixed set of current Gaussian moments, each recurrence factor supplies
a table of bounded expected log transition scores. Ordinary forward/backward
sum-product then gives one singleton belief per time and one consistent pair
belief per transition. Those pair beliefs, not products of singletons, supply
the at-risk and signed-success masses used by the Gaussian message and by
coefficient learning.

This does not remove the separate Pólya–Gamma/Jaakkola–Jordan approximation.
With an uncertain Gaussian drive, the transition table is still a lower bound
on the expected log transition. The categorical coordinate is exact for that
bounded table. With a fixed observed drive, the bound is tight and the
categorical coordinate reduces to ordinary exact chain smoothing.

The selected production owner is the existing set of time-local recurrence
factors, using hybrid sum-product categorical messages and a factor-local pair
belief. On an acyclic regime chain, those local pair beliefs are the globally
consistent marginals of one path belief. Their ordinary Bethe node entropies
and categorical edge corrections already equal the chain entropy. A new
monolithic chain region would encode the same junction tree a second time,
would need special boundary and counting-number suppression, and would not
improve the objective. A scheduler may fuse a chain's local messages into one
forward/backward pass, but that is an execution optimization, not a new belief
or energy owner.

### D9.2 Structured family and bounded pair potential

Let `z_0,...,z_N` be the categorical regime path and let transition `t`
connect source `z_t=i` to destination `z_{t+1}=j`. The structured inference
family is

    q(z_0:N) q_G(X) product_til q(omega_til)
                    product_il delta(theta_il - theta_hat_il).       (D9.1)

`q_G(X)` denotes the Gaussian/collapsed belief already held by the engine. It
may contain structure outside the recurrence block, but recurrence transition
`t` reads only the marginal mean `m_t` and covariance `S_t` of its drive.
Each Pólya–Gamma coordinate remains independent across time, source, and
stick and is eliminated into the JJ bound. During inference the installed
coefficient `theta_hat_il=[R_il;r_il]` is a point value. The independent
Gaussian coefficient posteriors used between trials remain outside this
within-sequence objective; their means are installed on the next coordinate,
as in D8.4. Thus the only newly structured belief in (D9.1) is the complete
categorical path.

For transition `t`, source `i`, and stick `l`, define

    mu_til = R_il^T m_t + r_il,
    c_til^2 = mu_til^2 + R_il^T S_t R_il,
    omega_bar_til = tanh(c_til/2) / (2 c_til),                (D9.2)

with the analytic `1/4` limit at zero. Reuse D8.6's success and failure
scores with `(mu_il,xi_il)` replaced by `(mu_til,c_til)`. Their decision-list
prefix gives the bounded expected log pair potential

    ell_t(i,j) = sum_l b_jl s_til^- + a_jl (s_til^+ - s_til^-),
    ell_t(i,j) <= E_qG[log p(z_{t+1}=j | z_t=i, x_t)].       (D9.3)

The algebraic form in (D9.3) says “failure for every at-risk stick, then
replace the chosen stick's failure by success.” It is the same score table D8
already computes. No generating state, regime, or parameter enters it.

Let `u_t(i)` collect the ordinary unary log potential and incoming evidence
on `z_t`. The structured categorical objective is

    F_Z^JJ[q_Z]
      = -H[q_Z]
        - sum_tij rho_t(i,j) ell_t(i,j)
        - sum_ti gamma_t(i) u_t(i),                           (D9.4)

where `gamma_t(i)=q_Z(z_t=i)` and
`rho_t(i,j)=q_Z(z_t=i,z_{t+1}=j)`. Since (D9.3) is a pointwise lower bound,

    F_Z^JJ[q_Z] >= F_Z^exact[q_Z] for every feasible q_Z,
    min_q F_Z^JJ >= min_q F_Z^exact.                          (D9.5)

The direction is conservative: the bounded negative free energy is above the
exact value. This comparison holds at the same `q_G` and installed
coefficients. It does not claim the bounded and exact objectives select the
same Gaussian beliefs.

### D9.3 Stationary messages, singleton beliefs, and pair beliefs

Stationarity of (D9.4) on the path simplex gives

    q_Z(z_0:N) = exp(
        sum_t u_t(z_t) + sum_t ell_t(z_t,z_{t+1}) - log Z_JJ
    ).                                                        (D9.6)

The normalized log-space forward and backward recursions are, up to an
additive gauge at each time,

    a_0(i)       = u_0(i),
    a_{t+1}(j)   = u_{t+1}(j) + logsumexp_i[a_t(i)+ell_t(i,j)],
    b_N(j)       = 0,
    b_t(i)       = logsumexp_j[ell_t(i,j)+u_{t+1}(j)+b_{t+1}(j)].   (D9.7)

They give

    gamma_t(i) proportional to exp(a_t(i)+b_t(i)),
    rho_t(i,j) proportional to exp(
        a_t(i)+ell_t(i,j)+u_{t+1}(j)+b_{t+1}(j)
    ).                                                        (D9.8)

Every pair obeys both marginal constraints exactly:

    sum_j rho_t(i,j)=gamma_t(i),
    sum_i rho_t(i,j)=gamma_{t+1}(j).                          (D9.9)

Equations (D9.7--D9.8) are also the time-local recurrence factor rules. If
`lambda_L(i)` and `lambda_R(j)` are the two incoming categorical cavity
messages, the factor sends

    log m_to_out(j) = logsumexp_i[lambda_L(i)+ell_t(i,j)],
    log m_to_in(i)  = logsumexp_j[ell_t(i,j)+lambda_R(j)],     (D9.10)

and owns the factor-local belief

    rho_t(i,j) proportional to
        exp(lambda_L(i)+ell_t(i,j)+lambda_R(j)).              (D9.11)

On an acyclic categorical recurrence component, the fixed point of (D9.10)
is (D9.7), and (D9.11) is the globally consistent pair marginal in (D9.8).
On a categorical loop the same rules define ordinary Bethe pseudomarginals,
not one exact path belief. The structured-chain claim therefore requires an
acyclic recurrence component; a compiler or model validation surface must
establish that condition rather than infer exactness from convergence alone.

If every drive is fixed, `S_t=0` and `c_til=|mu_til|`. The JJ success and
failure scores become the exact log logistic probabilities, so
`ell_t(i,j)=log p(j|i,x_t)`. Equations (D9.6--D9.11) then are ordinary exact
sum-product smoothing. For uncertain drives they remain exact for the JJ
potential, while (D9.5) records the remaining approximation.

### D9.4 One pair belief for Gaussian messages and learning

For each transition, source, and stick define its pair-derived at-risk and
signed-success masses

    E_til = sum_{j>=l} rho_t(i,j),
    K_til = rho_t(i,l) - E_til/2.                             (D9.12)

The recurrence message to its Gaussian drive is

    Lambda_xt = sum_il E_til omega_bar_til R_il R_il^T,
    eta_xt    = sum_il (K_til - E_til omega_bar_til r_il) R_il. (D9.13)

For augmented feature moments `h_t=E[[x_t;1]]` and
`M_t=E[[x_t;1][x_t;1]^T]`, the coefficient-statistic increments are

    Delta eta_til    = K_til h_t,
    Delta Lambda_til = E_til omega_bar_til M_t.               (D9.14)

The same `rho_t` supplies (D9.12) to both equations. There is no product
`gamma_t(i) gamma_{t+1}(j)` fallback in messages, diagnostics, checkpoints,
or learning. Exact zero pair mass gives exact zero exposure and natural
statistics. If a source/stick has zero total exposure, its between-trial
Gaussian posterior retains the prior coordinate.

### D9.5 Entropy, counting numbers, and the BFE relationship

For any consistent positive-support chain marginals, the junction-tree
entropy identity is

    H[q_Z] = sum_t H[rho_t] - sum_{t=1}^{N-1} H[gamma_t].     (D9.15)

The pair regions have entropy counting number `+1`; each repeated interior
singleton separator has entropy counting number `-1`. In free-energy signs,
the recurrence part of (D9.4) is therefore

    F_rec^JJ
      = sum_t { -H[rho_t] - sum_ij rho_t(i,j) ell_t(i,j) }
        + sum_{t=1}^{N-1} H[gamma_t].                         (D9.16)

Equation (D9.16) is exactly the existing CFFG ledger shape once a recurrence
factor's node belief changes from the D8 singleton product to `rho_t`:

- each time-local recurrence factor contributes the braced pair-region term;
- each categorical edge shared by two adjacent pair regions contributes the
  ordinary `+H[gamma_t]` interior-edge correction;
- equality branches and external unary factors telescope through their normal
  node and edge terms, so no chain-specific correction is added;
- the factor's independent Gaussian-drive entropy remains `-H[q_t(x)]`, and
  the ordinary Gaussian edge correction removes its duplicate as before.

Thus time-local factor regions already realize the chain objective. A named
chain region would have to remove the same local node and internal-edge terms
before adding `-H[q_Z]`, creating a second counting-number owner with no
mathematical benefit on a tree.

At the stationary belief (D9.6), the complete categorical objective satisfies

    F_Z^JJ[q_Z^*] = -log Z_JJ.                                (D9.17)

With fixed drives this is the exact categorical-chain BFE and exact negative
log evidence for the held unary potentials. With uncertain drives it is the
exact BFE of the JJ factor potentials and, by (D9.5), an upper bound on the
corresponding exact-transition variational objective. Other approximations in
the full rSLDS, including Gaussian-mixture collapse, retain their separately
documented bound directions; (D9.17) does not combine them into a new global
evidence guarantee.

The old D8 node term is recovered when every pair factorizes as
`rho_t=gamma_t outer gamma_{t+1}`. Away from independence, replacing `rho_t`
by that product changes both the mutual-information part of the entropy and
the expected transition energy. Changing only the learning readout would
therefore not be stationary for either D8 or D9.

### D9.6 Ownership decision and GFE boundary

Two ownership candidates were evaluated:

1. **Selected: time-local hybrid sum-product recurrence factors.** The
   recurrence rule owns (D9.10--D9.14), its pair belief, and its node energy.
   Standard categorical edges own the separator corrections. This reuses the
   existing factor type, parameter blobs, slot addresses, equality behavior,
   serialization, and Bethe ledger. A scheduled forward/backward pass may
   fuse the local updates for `O(T K^2)` execution without changing ownership.
2. **Rejected: a named monolithic chain region.** It would own a duplicate
   path belief and entropy, require compiler topology fusion across equality
   branches, suppress the existing recurrence node and edge ledger terms, and
   invent new boundary serialization. On a tree it yields the same equations
   and no stronger consistency than candidate 1.

The selected design is ordinary BFE only at the recurrence owner. A model may
still contain supported GFE/P-substituted factors elsewhere; the recurrence
terms then remain ordinary bounded BFE contributions inside the mixed global
ledger. P-substitution on a recurrence factor or any of its roles has no D9
objective or message derivation and must keep failing at compile time. It must
not reuse (D9.16) under a GFE label. Supporting recurrence-owned
P-substitution would require a separate derivation of its substituted path
belief, boundary messages, and counting numbers.

### D9.7 Implementation change path and executable anchors

The **New factor or rule family** and **BFE or GFE objective** change paths in
`docs/ARCHITECTURE.md` apply conservatively even though the serialized factor
name stays `recurrence_link`:

- authority: `cffg.recurrence` owns pair-potential, sum-product, pair-belief,
  Gaussian-statistic, and node-energy algebra;
- compiler/arena: existing recurrence parameter packing and slot addresses
  remain, while acyclic categorical-component classification and any fused
  forward/backward schedule must agree with unfused execution;
- engine: host/JIT, synchronous/scheduled, single/`vmap`, support/status,
  `_recurrence_sweep`, `_make_step`, `_batch_stage_fn`/`_scan_stage_fn`, and
  `_make_bfe` are paired surfaces;
- beliefs/learning: `gaussian_node_stats`, a structured pair readout,
  `recurrence_weight_stats`, `Learnables`, and checkpoint consumers all use
  (D9.12), never reconstructed singleton products;
- IR/serialization: no new field or factor spelling is selected, but bundles,
  executable identities, and restored runtime states must reject pre-D9
  behavior where a checkpoint could otherwise claim the new semantics;
- GFE: ordinary coexistence is applicable; recurrence-owned P-substitution is
  an explicit compile-time rejection; arbitrary new GFE messages are N/A;
- model/consumer: `cffg.models.rslds`, fit checkpointing, experiment retention,
  and heterogeneous recovery diagnostics must preserve pair identity and
  cannot infer it later from singletons.

The independent NumPy anchors remain in
`tests/test_structured_recurrence.py`. The independently callable JAX rule
authority is anchored in `tests/test_structured_recurrence_rule.py`:

- `test_structured_messages_beliefs_and_free_energy_match_enumeration` checks
  (D9.6--D9.11) and (D9.17) for K=2 and K=3.
- `test_time_local_pair_regions_are_globally_consistent` checks the selected
  ownership boundary and both constraints in (D9.9).
- `test_pair_regions_and_edge_corrections_equal_chain_entropy_and_energy` and
  `test_path_entropy_matches_literal_enumeration` check (D9.15--D9.16).
- `test_pair_responsibilities_drive_all_recurrence_statistics` checks
  (D9.12--D9.14) and rejects singleton-product responsibilities.
- `tests/test_structured_recurrence_learning.py` checks the public pair,
  Gaussian message, and pooled coefficient statistics against the same
  independent K=2/K=3 consecutive-state authority.
- `test_uncertain_drive_retains_the_jj_bound_boundary` separates structured
  categorical exactness from the PG/JJ approximation.
- the remaining tests preserve literal log-zero support, normalize vacuous
  evidence, reject empty/non-finite support, derive float32 tolerance from the
  dtype, and run K=8/T=100 with `O(T K^2)` dynamic programming rather than
  path enumeration.
- `test_fixed_drive_rules_match_exact_chain_oracle` and
  `test_pair_node_and_separator_energy_equal_negative_log_normalizer` pin the
  callable-authority equations and energy ledger to the independent oracle.
- `test_pair_masses_and_gaussian_message_use_joint_responsibility` checks that
  the Gaussian message reads the pair directly, while the hostile-input,
  transform, and K=8 tests pin the public rule's status and scaling boundary.
- `tests/test_structured_recurrence_engine.py` applies the same identities to
  compiled flooding and mixed schedules. It checks compiler acyclicity,
  factor-local pairs, Gaussian writes, the pair/separator BFE ledger,
  fixed-drive `-log Z`, loop claim boundaries, and `O(T K^2)` batch storage.
