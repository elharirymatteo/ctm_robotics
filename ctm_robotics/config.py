"""
config.py — All hyperparameters in one place.
Edit this file to tune agents or change environments.
"""
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────

ENV_FULL = "CartPole-v1"           # Fully observable
ENV_PO   = "CartPole-PO-v1"       # Partially observable (vel dims masked)

# CartPole obs: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
# PO variant masks indices 1 and 3 (the velocity dims)
MASKED_OBS_INDICES = [1, 3]       # Velocity dimensions to hide in PO variant


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    total_steps:    int   = 300_000   # Total environment steps per agent
    seed:           int   = 42
    n_envs:         int   = 4         # Parallel envs (PPO only)
    eval_episodes:  int   = 20        # Episodes per evaluation
    eval_every:     int   = 10_000    # Eval frequency (in env steps)
    log_dir:        str   = "results"
    device:         str   = "cpu"     # "cpu" or "cuda"


# ─────────────────────────────────────────────────────────────
# PPO (shared across MLP, LSTM, CTM)
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
    n_steps:        int   = 512       # Steps per rollout per env
    n_epochs:       int   = 4         # PPO update epochs per rollout
    batch_size:     int   = 64        # Minibatch size


# ─────────────────────────────────────────────────────────────
# SAC
# ─────────────────────────────────────────────────────────────

@dataclass
class SACConfig:
    lr:             float = 3e-4
    gamma:          float = 0.99
    tau:            float = 0.005     # Soft target update coefficient
    alpha:          float = 0.2       # Entropy regularization (auto-tuned if None)
    auto_alpha:     bool  = True
    buffer_size:    int   = 100_000
    batch_size:     int   = 256
    learning_starts:int   = 1_000     # Collect this many steps before training
    train_freq:     int   = 1         # Train every N steps


# ─────────────────────────────────────────────────────────────
# MLP Policy (PPO-MLP and SAC-MLP)
# ─────────────────────────────────────────────────────────────

@dataclass
class MLPConfig:
    hidden_sizes: tuple = (64, 64)


# ─────────────────────────────────────────────────────────────
# LSTM Policy (PPO-LSTM)
# ─────────────────────────────────────────────────────────────

@dataclass
class LSTMConfig:
    hidden_size: int = 64    # LSTM hidden state dimension
    n_layers:    int = 1


# ─────────────────────────────────────────────────────────────
# CTM Policy (PPO-CTM)
# Faithful to SakanaAI/continuous-thought-machines models/ctm_rl.py
# ─────────────────────────────────────────────────────────────

@dataclass
class CTMConfig:
    # Core dimensions
    d_model:        int   = 64     # Number of neurons D
    # Synapse model (U-NET-style MLP connecting neurons at each tick)
    synapse_hidden: int   = 128    # Hidden dim of synapse MLP
    synapse_depth:  int   = 3      # Layers in synapse MLP

    # Neuron-level models
    memory_length:  int   = 4      # M — history window for pre-activations
    nlm_hidden:     int   = 32     # Hidden dim of each NLM
    nlm_depth:      int   = 2      # Layers in each NLM

    # Synchronization
    n_synch_out:    int   = 32     # D_out — neurons selected for output sync
    # n_synch_action is 0 in the RL variant (no cross-attention, as per paper)
    synch_window:   int   = 8      # Sliding window for sync computation
    synch_decay:    float = 0.9    # Decay weight for older activations

    # Internal ticks
    n_ticks:        int   = 5      # T — internal ticks per environment step

    # Feature extraction for obs → embedding
    input_hidden:   int   = 64     # Backbone MLP hidden dim

    # Gradient clip (CTM tends to need tighter clipping)
    max_grad_norm:  float = 0.5


# ─────────────────────────────────────────────────────────────
# Agent registry
# ─────────────────────────────────────────────────────────────

AGENTS = {
    "ppo_mlp":  {"algo": "ppo",  "policy": "mlp"},
    "sac_mlp":  {"algo": "sac",  "policy": "mlp"},
    "ppo_lstm": {"algo": "ppo",  "policy": "lstm"},
    "ppo_ctm":  {"algo": "ppo",  "policy": "ctm"},
}

# Default instances
TRAIN   = TrainConfig()
PPO     = PPOConfig()
SAC     = SACConfig()
MLP     = MLPConfig()
LSTM    = LSTMConfig()
CTM     = CTMConfig()
