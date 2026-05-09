# CTM as Interpretable Implicit World Model for PO Robot Control

**Date:** 2026-05-09
**Status:** Approved for implementation

---

## 1. Core Claim

World models improve partially observable RL by explicitly training a latent state estimator with reconstruction objectives (Dreamer, TD-MPC2). We show that the Continuous Thought Machine's synchronization dynamics develop equivalent hidden-state representations purely from reward signal — providing world-model-like state estimation *and* free architectural interpretability, without prediction losses or post-hoc probing classifiers.

**What this claims:**
- CTM's sync head is a natural inspection interface: hidden physical variables (angular velocities, cart velocities) can be read off directly from the sync representation without a probing network
- This hidden-state tracking emerges from architecture alone under pure RL training
- The world models literature frames *why* this matters: in PO settings, state estimation is the bottleneck; CTM resolves it implicitly

**What this does not claim:**
- CTM beats LSTM/MLP on raw performance consistently
- CTM's sync state perfectly reconstructs hidden variables (partial correlations are modest, ~0.12–0.49)
- CTM is more sample efficient in general

---

## 2. Environments

Two tasks, each with full-obs and PO variants:

| Task | Full obs | PO variant | Masked dims | Why |
|---|---|---|---|---|
| CartPole | CartPole-v1 | CartPole-PO-v1 | cart_vel (1), pole_angvel (3) | Existing CTM result (46 > LSTM 27); established benchmark |
| Acrobot | Acrobot-v1 | Acrobot-PO-v1 | dtheta1 (4), dtheta2 (5) | Reactive MLP fails without velocity; Sakana tested it; longer episodes (up to 500 steps) |

**PO design rationale:** masking velocities forces temporal integration — an agent must differentiate position/angle signals over time to infer velocity. Reactive MLP control degrades significantly (unlike CartPole-PO-v2 where derivative control was sufficient). The hidden variables are physically meaningful, enabling interpretability analysis.

---

## 3. Baselines

All agents use PPO (discrete). No model-based baselines (Dreamer, TD-MPC2) — they use a different training paradigm and appear only in related work.

| Agent | Architecture | Conceptual role |
|---|---|---|
| PPO-MLP | 2-layer MLP (64×64) | Reactive upper bound; memory-free |
| PPO-LSTM | LSTM (64 hidden) + MLP head | Standard recurrent baseline; opaque hidden state |
| PPO-CTM | CTM (d_model=128, n_ticks=20) + sync head | Interpretable implicit world model |

**Training config:**
- 3 seeds (42, 123, 456), 300k steps each
- CTM: lr=5e-4, n_steps=512, recurrent_seq_len=100, n_epochs=1, clip=0.1, vf_coef=0.25, ent_coef=0.1→0.005
- LSTM/MLP: standard PPO defaults (n_steps=512, n_epochs=4, lr=3e-4)
- Reported metric: mean best-checkpoint return across 3 seeds (not final, given known oscillation)

---

## 4. Experiments

### Exp 1 — Performance under partial observability

**Question:** Does CTM outperform LSTM on PO tasks?

**Method:** Training curves + best-checkpoint bar chart for all agents × environments × observability conditions.

**Primary metric:** Mean peak return across 3 seeds ± std.

**Expected outcome:** CTM ≥ LSTM on PO variants; MLP strongest on full obs; MLP weakest on hardest PO (Acrobot-PO). This experiment is necessary to establish that CTM is a viable policy, not the main novelty.

---

### Exp 2 — Sync saliency reveals hidden-state tracking

**Question:** Does CTM's sync head encode hidden physical variables?

**Method:**
1. Roll out best-checkpoint policy for 1000 steps (episodes reset hidden state at termination)
2. Record at each step: masked obs (agent input), full obs (ground truth), sync vector (16-dim)
3. Compute Pearson |r| between each sync dimension and each full-obs variable → take max over sync dims per variable
4. Compute partial correlation of best sync dim with each variable, controlling for all visible obs dimensions — isolates hidden-variable signal from shared causes
5. Repeat identically for LSTM h_t (64-dim) as the representation

**Key figure:** Side-by-side bar chart of partial-r per variable, CTM sync vs LSTM h_t. Shows the *mechanism difference* (sync head is directly readable; LSTM requires probing) not a magnitude claim.

**Existing evidence:** CartPole-PO-v1 seed456: partial-r = 0.119 (cart_vel), 0.116 (pole_angvel). LunarLander-PO: partial-r up to 0.488 (vy).

---

### Exp 3 — Tick-progressive inference

**Question:** Do more internal ticks yield better hidden-state estimates?

**Method:**
1. During rollout, record sync vector after *each* of the 20 internal ticks (not just final output)
2. Compute correlation with hidden variables at each tick index 0→19
3. Plot: tick index (x) vs partial-r for each hidden variable (y)

**Hypothesis:** With n_ticks=20 (up from n_ticks=5), a monotonic increase in hidden-variable correlation over ticks becomes visible. Previous analysis with n_ticks=5 showed only marginal improvement (0.595→0.651) — insufficient tick budget for the pattern to emerge.

**Implementation note:** Requires hooking into the CTM tick loop in `ctm.py` to expose intermediate sync states, not just the final one.

**If confirmed:** "The model refines its hidden-state estimate over internal compute steps" — the chain-of-thought analogy for control.

---

### Exp 4 — Sync matrix as task-phase attention

**Question:** Does the sync matrix change meaningfully with task phase?

**Method:**
1. Compute sync matrix at each step as outer product of sync vector: M = s · sᵀ (16×16 symmetric)
2. Snapshot at representative episode phases (e.g., Acrobot: initial swing, mid-swing, near-target)
3. Visualise as heatmaps, annotated with episode phase

**Claim level:** Qualitative only. Provides visual intuition for the "attention" analogy — which neuron pairs co-activate during which behavioural phases.

---

## 5. Interpretability Methodology (shared across Exp 2–4)

**Sync representation:** The `n_synch_out=16` diagonal of the synchronisation matrix, extracted after each CTM forward pass. Architecture-native, no additional network required.

**Partial correlation procedure:**
- Let `s_j` be sync dimension j, `v_k` be the k-th full-obs variable, `V_vis` be the matrix of all visible obs variables
- Regress both `s_j` and `v_k` on `V_vis` using OLS; take residuals `ŝ_j`, `v̂_k`
- Compute Pearson r between residuals: partial-r(j, k) = corr(ŝ_j, v̂_k)
- Report max over j for each variable k

**Why partial correlation:** Raw correlation is inflated by shared temporal structure (position correlates with velocity over an episode). Partial corr isolates whether the sync state contains *genuinely new* information about the hidden variable beyond what the visible observations already predict.

---

## 6. Expected Contributions

1. **Empirical:** CTM solves Acrobot-PO and CartPole-PO-v1 competitively against LSTM under matched training budgets
2. **Interpretability:** Sync saliency analysis reveals hidden-variable tracking without probing classifiers, on two PO control tasks
3. **Mechanistic:** Tick-progressive analysis (if confirmed) shows inference refinement over internal compute steps — connecting Sakana's chain-of-thought intuition to PO state estimation
4. **Framing:** World models as conceptual lens for understanding CTM's emergent representations in model-free RL

---

## 7. Open Questions / Risks

| Risk | Mitigation |
|---|---|
| CTM still fails to learn on Acrobot-PO | n_ticks=20 + n_steps=512 is the hypothesis fix; if it fails, revert to CartPole-PO-v1 as sole PO benchmark |
| Partial corrs remain too low to make the claim | Claim becomes "interpretability mechanism" (no probing needed) rather than "hidden-state encoding quality" |
| Tick progression still flat with n_ticks=20 | Exp 3 becomes a negative finding; remove from main story, keep as appendix |
| LSTM partial corrs equal or exceed CTM | Reframe: CTM's advantage is the *inspection mechanism* (direct readout vs probing), not tracking quality |
