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
4. Cat/Dir/Gaussian re-expression on the abstraction is a REFACTOR ticket
   with zero-regression acceptance, deliberately not folded into this work.

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

**What this is not.** Not engine rules yet (this is the research spike; the
composite node is a follow-up ticket), not a claim about arbitrary nonlinear
MI or exact mixture entropies (impossible in closed form — any future
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
collapse), deferred to the implementation ticket.
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
exact — recorded here for the implementation ticket, not exercised by the
spike's tests.
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
