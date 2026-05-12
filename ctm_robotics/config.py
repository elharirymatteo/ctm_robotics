"""
config.py — All hyperparameters in one place.
"""
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Environments
# ─────────────────────────────────────────────────────────────

# Discrete (PPO)
ENV_FULL = "CartPole-v1"
ENV_PO   = "CartPole-PO-v1"
ENV_PO2  = "CartPole-PO-v2"   # masks pole_angle (better POMDP for memory advantage)

# Discrete (Acrobot)
ACROBOT_FULL = "Acrobot-v1"
ACROBOT_PO   = "Acrobot-PO-v1"

# Continuous (TD3)
PEND_FULL = "Pendulum-v1"
PEND_PO   = "Pendulum-PO-v1"
BIPED_FULL = "BipedalWalker-v3"
BIPED_PO   = "BipedalWalker-PO-v3"

# Legacy (kept for backwards compat)
LUNAR_FULL = "LunarLander-v3"
LUNAR_PO   = "LunarLander-PO-v3"


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    total_steps:    int   = 300_000
    seed:           int   = 42
    n_envs:         int   = 4         # Parallel envs (PPO only)
    eval_episodes:  int   = 20
    eval_every:     int   = 10_000
    log_dir:        str   = "results"
    device:         str   = "cuda" if __import__('torch').cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# PPO (discrete envs: CartPole)
# ─────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    lr:             float = 3e-4
    gamma:          float = 0.99
    gae_lambda:     float = 0.95
    clip_eps:       float = 0.2
    vf_coef:        float = 0.5
    ent_coef:       float = 0.01
    max_grad_norm:  float = 0.5
    n_steps:        int   = 512
    n_epochs:       int   = 4
    batch_size:     int   = 64
    recurrent_seq_len: int = 16  # LSTM chunk size: 32 chunks × 4 epochs = 128 updates (matches MLP)
    target_kl:         Optional[float] = None   # Early-stop update loop if approx KL exceeds this
    ent_coef_final:    Optional[float] = None   # Anneal ent_coef → this value over training (None = no annealing)
    ent_anneal_fraction: float = 1.0            # Complete annealing by this fraction of total_steps (e.g. 0.5 = first half)


# ─────────────────────────────────────────────────────────────
# TD3 (continuous envs: Pendulum, BipedalWalker)
# ─────────────────────────────────────────────────────────────

@dataclass
class TD3Config:
    lr_actor:       float = 1e-3
    lr_critic:      float = 1e-3
    gamma:          float = 0.99
    tau:            float = 0.005     # Soft target update
    policy_noise:   float = 0.2      # Target policy smoothing noise
    noise_clip:     float = 0.5      # Clamp target noise
    exploration_noise: float = 0.1   # Action noise during collection
    policy_delay:   int   = 2        # Update actor every N critic updates
    buffer_size:    int   = 200_000
    batch_size:     int   = 256
    learning_starts:int   = 10_000   # Random exploration steps
    train_freq:     int   = 1
    burn_in:        int   = 10       # Steps to replay for hidden state rebuild


# ─────────────────────────────────────────────────────────────
# SAC (legacy, kept for CartPole comparison)
# ─────────────────────────────────────────────────────────────

@dataclass
class SACConfig:
    lr:             float = 3e-4
    gamma:          float = 0.99
    tau:            float = 0.005
    alpha:          float = 0.2
    auto_alpha:     bool  = True
    buffer_size:    int   = 100_000
    batch_size:     int   = 256
    learning_starts:int   = 1_000
    train_freq:     int   = 1


# ─────────────────────────────────────────────────────────────
# Policy configs
# ─────────────────────────────────────────────────────────────

@dataclass
class MLPConfig:
    hidden_sizes: tuple = (64, 64)

@dataclass
class LSTMConfig:
    hidden_size: int = 64
    n_layers:    int = 1

@dataclass
class CPPOConfig:
    """Continuous PPO — Pendulum/BipedalWalker specific settings."""
    lr:                 float = 3e-4
    gamma:              float = 0.9
    gae_lambda:         float = 0.95
    clip_eps:           float = 0.2
    vf_coef:            float = 0.5
    ent_coef:           float = 0.01
    max_grad_norm:      float = 0.5
    n_steps:            int   = 512
    n_epochs:           int   = 10
    batch_size:         int   = 64
    recurrent_seq_len:  int   = 16   # 4*16=64 transitions/batch, 320 grad steps (matches MLP)

@dataclass
class CTMConfig:
    d_model:        int   = 128   # paper: 128 for CartPole/Acrobot
    synapse_hidden: int   = 64    # paper: 2-layer synapse for RL
    synapse_depth:  int   = 2     # paper: 2 for RL (not 3)
    memory_length:  int   = 20    # paper: 10-50 (was 4 — critical fix)
    nlm_hidden:     int   = 4     # paper: d_hidden=4 (was 32 — critical fix)
    nlm_depth:      int   = 2
    n_synch_out:    int   = 16    # paper: J_out=16 (was 32)
    synch_window:   int   = 8
    synch_decay:    float = 0.9
    n_ticks:        int   = 20
    input_hidden:   int   = 128   # paper: d_input=128
    max_grad_norm:  float = 0.5


# ─────────────────────────────────────────────────────────────
# Agent registry
# ─────────────────────────────────────────────────────────────

AGENTS = {
    # Discrete (CartPole) — PPO
    "ppo_mlp":   {"algo": "ppo",  "policy": "mlp"},
    "ppo_lstm":  {"algo": "ppo",  "policy": "lstm"},
    "ppo_ctm":   {"algo": "ppo",  "policy": "ctm"},
    # Continuous (Pendulum, BipedalWalker) — TD3 (full obs) + PPO (PO)
    "td3_mlp":   {"algo": "td3",  "policy": "mlp"},
    "td3_lstm":  {"algo": "td3",  "policy": "lstm"},
    "td3_ctm":   {"algo": "td3",  "policy": "ctm"},
    "cppo_mlp":  {"algo": "cppo", "policy": "mlp"},
    "cppo_lstm": {"algo": "cppo", "policy": "lstm"},
    "cppo_ctm":  {"algo": "cppo", "policy": "ctm"},
    # Legacy
    "sac_mlp":   {"algo": "sac",  "policy": "mlp"},
}

# Default instances
TRAIN = TrainConfig()
PPO   = PPOConfig()
CPPO  = CPPOConfig()
TD3   = TD3Config()
SAC   = SACConfig()
MLP   = MLPConfig()
LSTM  = LSTMConfig()
CTM   = CTMConfig()
