# Interpretable Implicit World Models for Free-Flying Spacecraft Control via Continuous Thought Machines

**Date:** 2026-05-12
**Status:** Approved for implementation — supersedes 2026-05-09-ctm-novelty-design.md
**Target venue:** IEEE RAL / IEEE TRL / AAAI (full paper)

---

## 1. Core Claim

Partially observable control of safety-critical robotic systems (spacecraft, free-flying platforms, autonomous vehicles operating with degraded sensors) requires recurrent policies, but standard recurrent baselines (LSTM, GRU) produce opaque hidden states that resist inspection. This is a deployment-blocker for systems where operators must verify what the controller has inferred — not only what action it takes.

We show that the Continuous Thought Machine's synchronization dynamics develop hidden-state representations analogous to learned world models (Dreamer, TD-MPC2), but without reconstruction objectives or post-hoc probing classifiers. The 16-dimensional sync representation is directly linearly decodable to physically meaningful hidden variables (body-frame velocity, angular rate), making the controller's implicit state estimate readable at inference time.

**Claims:**
1. **Empirical (PO control performance):** PPO-CTM matches or exceeds PPO-LSTM on PO variants of GoToPose and TrackVelocities tasks for a 2D free-flying air-bearing platform, in IsaacLab simulation.
2. **Interpretability:** CTM sync representation is linearly decodable to hidden velocities with R²_CV substantially higher than LSTM h_t under identical decoding protocol, with no probe training required.
3. **Mechanistic:** Hidden-state decodability is a function of CTM's internal tick depth (n_ticks ablation), and intermediate-tick representations expose deliberation distinct from final-tick action encoding.
4. **Sim-to-real:** IsaacLab-trained CTM policies transfer to a real air-bearing testbed; OptiTrack motion-capture data confirms the sync representation predicts true hidden velocity in the deployed system.

**Non-claims:**
- We do not claim CTM beats MLP on all metrics. Where reward shaping permits reactive control, MLP remains competitive on performance.
- We do not claim novel architecture — CTM is from Sakana AI. The contribution is the interpretability protocol and its validation in real robot control.
- We do not propose CTM as a general replacement for explicit world models in tasks where reconstruction supervision is feasible.

---

## 2. Validation chain — three layers

| Layer | Role in paper | Environment | Stack | Status |
|---|---|---|---|---|
| **L1** | Supplementary — standard-benchmark sanity check | CartPole-PO-v1 (gym) | Existing home-grown PPO | ✓ Complete (results in 2026-05-09 spec) |
| **L2** | **Main results** | IsaacLab FloatingPlatform: GoToPose-PO + TrackVelocities-PO | IsaacLab + rsl_rl + custom `ActorCriticCTM` | To build |
| **L3** | **Headline** | Air-bearing satellite emulator, real hardware | IsaacLab-trained policy deployed, OptiTrack as ground truth oracle | To execute |

Numerical comparability: all three layers compute the same interpretability metric (linear-decodability R²_CV, see §5) over the same target variables (linear+angular velocity in body frame). Results are directly comparable across layers.

---

## 3. Environments

### 3.1 Layer 1: CartPole-PO-v1 (supplementary)

Already characterized in `2026-05-09-ctm-novelty-design.md` §4.1. Carried forward unchanged. Serves as a method-validation pointer for reviewers without IsaacLab access. Confirmed results across 3 seeds: CTM 46.8 ± 3.4 vs LSTM 26.5 ± 0.9.

### 3.2 Layer 2: IsaacLab FloatingPlatform

**Platform:** 2D planar rigid body, 3 DOF (x, y, yaw), 8 binary thrusters at fixed body-frame positions. Reference: `Isaaclab_RANSv2/source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/robots_cfg/floating_platform_cfg.py` on `dev` branch.

- Thruster max force: 1.0 N, split_thrust enabled
- Action space: 8 binary thrusters (matches hardware)
- Observation: 8-dim task observation + thruster states (see task definitions below)

#### Task A — GoToPose

Reference: `tasks/go_to_pose.py`, `tasks_cfg/go_to_pose_cfg.py`.

**Full-obs (baseline):**
```
[0]   position_distance              (m, scalar)
[1,2] cos, sin of heading-to-target  (unit)
[3,4] cos, sin of target-heading-err (unit)
[5,6] vx, vy in body frame           (m/s)         ← PO mask
[7]   ω_yaw                          (rad/s)       ← PO mask
```

**PO variant:** indices [5,6,7] are zeroed at observation time. Agent must infer velocity from how the distance and heading errors evolve. Reward identical to full-obs.

#### Task B — TrackVelocities (PO formulation: Option B)

Reference: `tasks/track_velocities.py`, `tasks_cfg/track_velocities_cfg.py`.

The existing IsaacLab task formulation exposes velocity *errors* (target − current), which already encodes current velocity. We need a deliberate PO redesign:

**PO observation for TrackVelocities-PO:**
```
[0,1] target_linear_vel_x, target_linear_vel_y  (m/s, in body frame)
[2]   target_angular_vel                        (rad/s)
[3,4] body-frame position delta over the last K=4 control steps  (m)
[5]   heading delta over the last K=4 control steps              (rad)
+ thruster states (8-dim)
```

The agent receives the target velocity and a short window of position trajectory. It does *not* receive current velocity, lateral velocity, angular velocity, or precomputed errors. Reward is computed in the environment as `-‖ v_target − v_current ‖`, but `v_current` is never observable.

**Why Option B:** This is the satellite-realistic PO formulation (position from external observer like GPS/visual, no IMU-derived velocity). MLP cannot solve this task — it has no velocity information in its observation. LSTM and CTM must integrate the position-delta window to estimate velocity, then compare against target.

**Episode structure:** target velocity vector changes every N_segment steps (configurable, default 200) following a pre-sampled trajectory. Episode length 1000 steps. Reward shaping inherited from existing `TrackVelocitiesCfg`.

#### Domain randomization (sim-to-real)

Enabled from start of training in L2 (already configured in `FloatingPlatformRobotCfg`):

| Randomization | Config | Range |
|---|---|---|
| Mass | `mass_rand_cfg` | ±0.25 kg uniform |
| CoM offset | `com_rand_cfg` | ±0.05 m uniform |
| External wrench | `wrench_rand_cfg` | force (0, 0.25) N, torque (0, 0.05) Nm |
| Thruster noise | `noisy_actions_cfg` | uniform/Gaussian ±0.1 / σ=0.025 |
| Thruster rescaling | `action_rescaler_cfg` | scale (0.8, 1.0) |

### 3.3 Layer 3: Air-bearing satellite emulator (hardware)

Same robot geometry, same actuator layout, same task definitions as L2. OptiTrack motion-capture system provides full-state ground truth (position, velocity, orientation, angular velocity) for analysis only — the policy still receives PO observations exactly as in L2.

For TrackVelocities-PO, OptiTrack-derived position is the source of the position-delta window the policy observes. For interpretability analysis, OptiTrack-derived velocity is the ground-truth `v_current` against which the CTM sync representation is regressed.

---

## 4. Algorithms

All three algorithms run under `rsl_rl` PPO via IsaacLab's training scripts (`scripts/rsl_rl/train.py`). For L1 (CartPole-PO supplementary), the existing home-grown PPO pipeline is retained — results already exist.

### 4.1 Common config (L2 and L3)

Inherits from `Isaaclab_RANSv2/.../agents/rsl_rl_ppo_cfg.py`:

```
PPO:    γ=0.99, λ=0.95, clip=0.2, value_loss_coef=1.0
        ent_coef=0.005, lr=1e-3 (adaptive, desired_kl=0.01)
        num_steps_per_env=16, num_learning_epochs=5, num_mini_batches=4
        max_grad_norm=1.0
Train:  num_envs=4096 (or hardware-appropriate), max_iterations=1000
```

### 4.2 MLP baseline

`RslRlPpoActorCriticCfg`, hidden dims [64, 64], `tanh` activation. Built-in.

### 4.3 LSTM baseline

`RslRlPpoActorCriticRecurrentCfg`, `rnn_type="lstm"`, `rnn_hidden_dim=64`, `rnn_num_layers=1`. Built-in.

### 4.4 CTM (custom)

`RslRlPpoActorCriticCfg` subclass with `class_name="ActorCriticCTM"`, registered via rsl_rl's class-name lookup. Architecture parameters from `CTMConfig` in our codebase:

```
d_model=128, synapse_hidden=64, synapse_depth=2
memory_length=20, nlm_hidden=4, nlm_depth=2
n_synch_out=16, synch_window=8, synch_decay=0.9
n_ticks=20, input_hidden=128
```

**Integration contract with rsl_rl:**

`ActorCriticCTM` must implement the rsl_rl recurrent actor-critic interface:
- `forward(observations, hidden_states) → (action_dist, value, new_hidden_states)`
- `reset(env_ids)` — resets per-env hidden state on episode termination
- `get_hidden_state_shape() → dict[str, tuple]` — declares hidden state buffers

The internal tick loop (N=20 internal iterations per env step) is wrapped inside `forward()`; the trainer sees a standard `h_t = f(h_{t-1}, x_t)` recurrence. External hidden state includes pre-activation FIFO, post-activation memory list, and last action (matching our current `CTMActorCritic.init_hidden()` contract).

---

## 5. Interpretability methodology (unified across L1, L2, L3)

### 5.1 Linear decodability R²_CV (primary metric)

For each hidden physical variable `v_k ∈ {vx, vy, ω_yaw}` and each representation `z` (CTM sync, dim 16; LSTM h_t, dim 64):

1. Roll out the best-checkpoint policy for `T = 10,000` environment steps (collected across episodes; hidden state reset on termination)
2. Log `(z_t, v_k,t, v_vis,t)` at each step, where `v_vis` is the set of visible observations
3. Fit ridge regression `v_k = W z + b` with 5-fold cross-validation, regularization swept over `α ∈ {10⁻³, 10⁻², 10⁻¹, 1, 10}`
4. Report `R²_CV` (mean over folds, best α)
5. Report alongside `R²_CV(visible-only)` — ridge fit using only `v_vis` to predict `v_k`. The difference `ΔR² = R²_CV(z) − R²_CV(v_vis)` isolates the genuinely new information in `z`

### 5.2 Tick-progressive R²_CV (CTM only)

Same procedure, applied separately to the sync vector after each internal tick `i ∈ {1, ..., 20}`. Plot `R²_CV(tick i)` vs tick index. Tests whether deliberation depth provides incremental information about hidden state, and whether intermediate ticks differ from the final tick.

### 5.3 Probing comparison (quantifies "free interpretability" claim)

To formalize "no probe needed":
- **CTM direct readout:** R²_CV as computed in §5.1 (effectively a ridge probe on z)
- **LSTM with supervised probe:** train a 2-layer MLP probe `v_k = MLP(h_t)` on a labelled training split (50% of rollout), evaluate on held-out 50%
- Comparison: does CTM with simple ridge match LSTM's MLP probe? If yes, CTM's 16-dim sync is linearly decodable to hidden state, while LSTM's 64-dim h_t requires a nonlinear supervised probe — the "free interpretability" claim is concrete.

### 5.4 Hardware interpretability validation (L3)

On the air-bearing testbed:
- Deploy policy in PO mode (control inputs are PO observations)
- Log OptiTrack full state at policy rate
- Log policy's internal `z_t` (sync or h_t) alongside actions
- Compute R²_CV identically to §5.1, with OptiTrack-derived velocity as ground truth `v_k`
- Headline figure: time series of OptiTrack-measured velocity vs `W·z_t + b` (decoded velocity from sync), demonstrating that the controller's implicit state estimate tracks the true hidden variable in a real system

---

## 6. Experimental matrix

### 6.1 Main runs (L2)

| Dimension | Values | Count |
|---|---|---|
| Environment | FloatingPlatform-GoToPose-PO, FloatingPlatform-TrackVel-PO | 2 |
| Algorithm | PPO-MLP, PPO-LSTM, PPO-CTM | 3 |
| Seed | 42, 123, 456, 789, 1234 | 5 |
| **Total** | | **30 runs** |

Plus full-obs reference runs for context (same matrix, full observation): 30 runs.

Per-run compute budget: training to convergence (~1000 iterations × 4096 envs ≈ 6.5M env steps), expected wall time 15-60 min on a single GPU based on IsaacLab norms. Total L2 training: ~30-60 GPU-hours.

### 6.2 Ablation: n_ticks (L2, CTM only)

| Dimension | Values | Count |
|---|---|---|
| Environment | FloatingPlatform-GoToPose-PO | 1 |
| n_ticks | 1, 5, 10, 20 | 4 |
| Seed | 42, 123, 456 | 3 |
| **Total** | | **12 runs** |

### 6.3 Probing comparison (L2, post-hoc analysis)

No additional training runs. Reuse best checkpoints from main matrix; train supervised probes offline. ~1 day analysis.

### 6.4 Hardware runs (L3)

| Dimension | Values | Count |
|---|---|---|
| Environment | FloatingPlatform-GoToPose-PO, FloatingPlatform-TrackVel-PO | 2 |
| Algorithm | PPO-LSTM, PPO-CTM | 2 |
| Repetitions per (env, algo) | 5 episodes each | 5 |
| **Total** | | **20 deployment episodes** |

MLP is excluded from hardware tests *if* L2 confirms it fails on TrackVel-PO (expected by construction — no velocity information in the PO observation). If L2 finding is otherwise, MLP is added back at L3 as a fourth comparison.

---

## 7. Sprint plan and deliverables

The spec covers four sprints. Each sprint produces a self-contained, testable deliverable and gets its own implementation plan written separately when the previous sprint completes. Sprint 1 is on the critical path — its success determines whether the rest proceeds as-is or fails over to the fallback strategy.


### Sprint 1 (week 1): CTM → rsl_rl integration

**Deliverable:** Working `ActorCriticCTM` registered with rsl_rl, smoke-tested on `FloatingPlatform-GoToPose` (full-obs) for one seed, achieves non-trivial return (above MLP baseline within 2× wall time).

**De-risking checkpoint:** if smoke test fails by end of week 1, escalate to fallback (custom trainer interfacing IsaacLab envs directly, retaining GPU parallelism).

### Sprint 2 (weeks 2-3): IsaacLab main experiments

**Deliverable:**
- All 30 main runs (PO + 30 full-obs reference) complete
- All 12 n_ticks ablation runs complete
- Probing comparison analysis complete
- Linear-decodability R²_CV computed for all best checkpoints
- All main paper figures and tables produced (except hardware)

### Sprint 3 (weeks 4-5): Hardware deployment

**Deliverable:**
- ROS / hardware interface from policy checkpoint to thruster commands
- 20 deployment episodes recorded with OptiTrack logs
- Headline figure: OptiTrack velocity vs CTM sync decoded velocity over time
- Hardware results section drafted

### Sprint 4 (weeks 6-8): Writing

**Deliverable:**
- Full paper draft (8-page IEEE format)
- Supplementary with L1 CartPole-PO results, full ablation tables, hyperparameter listings
- Code release packaging (this repo + IsaacLab extension + analysis scripts)

**Realistic timeline with buffer:** 3 months end-to-end.

---

## 8. Paper structure (draft)

1. **Introduction.** PO control in safety-critical robotics, interpretability as a deployment requirement, contribution claims.
2. **Related work.** World models (Dreamer, TD-MPC2). Recurrent policies for PO RL. Probing classifiers in interpretability. Sim-to-real for spacecraft control.
3. **Method.** CTM architecture (compact, citing Sakana). Linear-decodability interpretability protocol. PO formulations for both tasks.
4. **Experiments — Layer 2 (IsaacLab).** Performance results (Exp 1). Linear decodability (Exp 2). Tick-progressive analysis (Exp 3). Probing comparison (Exp 4). n_ticks ablation.
5. **Experiments — Layer 3 (Hardware).** Sim-to-real performance. OptiTrack-validated interpretability (headline figure).
6. **Discussion.** Where CTM helps vs not. Implications for safety-critical deployment. Limitations.
7. **Conclusion.**
8. **Supplementary.** L1 CartPole-PO sanity check. Hyperparameter tables. Additional plots.

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| CTM → rsl_rl integration is harder than expected | High | Sprint 1 dedicated de-risking; fallback to direct IsaacLab interface |
| CTM doesn't train well under IsaacLab PPO defaults | Medium | Inherit our characterized CTM PPO hyperparameters (lr=5e-4, clip=0.1, ent annealing) as the starting point, not rsl_rl defaults |
| Sim-to-real gap larger than domain randomization can close | Medium | Iterate randomization ranges in Sprint 3; report sim-to-real performance drop as part of the contribution rather than hiding it |
| Hardware testbed unavailable when needed | Medium | Reserve slot before Sprint 3 starts; if delayed, paper becomes "IsaacLab with sim-to-sim ablation" plus hardware in revision |
| Linear decodability R²_CV is low on hardware | Low-Medium | Hardware-only fine-tuning of policy as fallback; paper still has L2 interpretability results |
| Probing comparison shows LSTM with probe matches CTM | Low | Reframe contribution as *no probe required* (architectural readout vs supervised classifier), still publishable |
| MLP solves TrackVel-PO Option B somehow | Low | Confirmed impossible by construction (no velocity in observation) — re-verify in smoke test |

---

## 10. Open items (none blocking implementation)

- Final hyperparameter sweep ranges for n_ticks ablation values (1, 5, 10, 20 is defensible; may add intermediate values if time allows)
- Whether to include sync-head ablation (replace sync with `mean(post_act)`) — time-permitting, in supplementary
- Exact OptiTrack sampling rate and policy rate matching strategy on hardware — addressed in Sprint 3 setup
