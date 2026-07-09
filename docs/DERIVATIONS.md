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
and the observation composite becomes a special case of the transition one.

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
logsumexp_u(ρ̄ + μ_e); to the out-edge, log q_w. The FE contribution is
G_tm(q̄) = −q̄ᵀρ̄ − H[q̄], the same closed form as the A-composite.
*Executable:* `test_fixed_point_satisfies_its_equation`,
`test_energy_is_stationary_value`,
`test_goal_information_flows_backward_through_substituted_node` (the
μ_down coupling is what keeps pragmatic planning intact).

**Proposition 5 (consistency).** (a) Existence of a fixed point: T(q) =
softmax(ρ(q) + log d) is continuous on the (K·U)-simplex, so Brouwer
applies (uniqueness is not claimed, matching Part II). (b) The observation
composite of Part II (Corollary 1) is the special case s = z (drop u),
μ_down = log c (a goal leaf), B̄ → Ā: identical equation, identical solver.
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
