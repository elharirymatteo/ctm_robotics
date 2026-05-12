"""
training/rollout_buffer.py

A unified rollout buffer for PPO that handles:
  - Stateless policies (PPO-MLP): standard flat buffer
  - Recurrent policies (PPO-LSTM, PPO-CTM): stores full sequences + hidden states

For recurrent policies the buffer groups steps by episode so the trainer
can feed complete sequences during the update (BPTT over each episode).

Key design: we keep things simple and correct rather than maximally optimized.
The buffer stores raw numpy arrays and converts to tensors on demand.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Any


class RolloutBuffer:
    """
    Stores one rollout (n_steps × n_envs transitions) for PPO.

    Supports both stateless and recurrent policies via is_recurrent flag.
    """

    def __init__(self, n_steps: int, n_envs: int,
                 obs_dim: int, action_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 is_recurrent: bool = False,
                 continuous: bool = False,
                 device: str = "cpu"):
        self.n_steps     = n_steps
        self.n_envs      = n_envs
        self.obs_dim     = obs_dim
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.is_recurrent = is_recurrent
        self.continuous   = continuous
        self.device      = device

        self.reset()

    def reset(self):
        self.observations = np.zeros((self.n_steps, self.n_envs, self.obs_dim),
                                     dtype=np.float32)
        if self.continuous:
            self.actions = np.zeros((self.n_steps, self.n_envs, self.action_dim),
                                    dtype=np.float32)
        else:
            self.actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        self.rewards      = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones        = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values       = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.log_probs    = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.returns      = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.advantages   = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.pos          = 0
        self.full         = False

        # For recurrent policies: store hidden states at step boundaries
        # So we know where to reset them during BPTT
        self.episode_starts = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        # LSTM hidden-state snapshots at chunk-start positions.
        # Key: step index (0, seq_len, 2*seq_len, …)
        # Value: (h_np, c_np) numpy arrays (n_layers, n_envs, hidden_size)
        self.hidden_snapshots: dict = {}

    def store_hidden_snapshot(self, step: int, hidden) -> None:
        """Save LSTM or CTM hidden state at a chunk-start step."""
        if not isinstance(hidden, tuple) or len(hidden) != 2:
            return
        first, second = hidden
        if isinstance(second, torch.Tensor):
            # LSTM: (h, c)
            self.hidden_snapshots[step] = (
                first.detach().cpu().numpy().copy(),
                second.detach().cpu().numpy().copy(),
            )
        elif isinstance(second, list) and all(isinstance(p, torch.Tensor) for p in second):
            # CTM: (pre_h, post_list)
            self.hidden_snapshots[step] = (
                first.detach().cpu().numpy().copy(),
                [p.detach().cpu().numpy().copy() for p in second],
            )

    def add(self, obs, action, reward, done, value, log_prob, episode_start=None):
        """
        Add one step of data (from all n_envs simultaneously).

        Args:
            obs:           (n_envs, obs_dim)
            action:        (n_envs,) int
            reward:        (n_envs,) float
            done:          (n_envs,) bool
            value:         (n_envs,) float
            log_prob:      (n_envs,) float
            episode_start: (n_envs,) bool — True if this step starts a new episode
        """
        t = self.pos
        self.observations[t] = obs
        self.actions[t]      = action
        self.rewards[t]      = reward
        self.dones[t]        = done.astype(np.float32)
        self.values[t]       = value
        self.log_probs[t]    = log_prob
        if episode_start is not None:
            self.episode_starts[t] = episode_start.astype(np.float32)

        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_values, last_dones):
        """
        Compute GAE (Generalized Advantage Estimation) returns and advantages.

        last_values: (n_envs,) — V(s_{T+1}) from the bootstrap
        last_dones:  (n_envs,) — whether s_{T+1} is terminal
        """
        last_gae_lam = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = (self.rewards[t]
                     + self.gamma * next_values * next_non_terminal
                     - self.values[t])

            last_gae_lam = (delta
                            + self.gamma * self.gae_lambda
                            * next_non_terminal * last_gae_lam)

            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values

    def get_stateless_batches(self, batch_size: int):
        """
        For stateless (MLP) policies: yield shuffled minibatches.
        Yields dicts of torch tensors.
        """
        n = self.n_steps * self.n_envs
        indices = np.random.permutation(n)

        obs_flat    = self.observations.reshape(n, self.obs_dim)
        if self.continuous:
            actions_flat = self.actions.reshape(n, self.action_dim)
        else:
            actions_flat = self.actions.reshape(n)
        log_probs_flat = self.log_probs.reshape(n)
        advantages_flat = self.advantages.reshape(n)
        returns_flat = self.returns.reshape(n)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / \
                          (advantages_flat.std() + 1e-8)

        for start in range(0, n, batch_size):
            idx = indices[start: start + batch_size]
            act_t = torch.FloatTensor(actions_flat[idx]) if self.continuous \
                    else torch.LongTensor(actions_flat[idx])
            yield {
                "obs":        torch.FloatTensor(obs_flat[idx]).to(self.device),
                "actions":    act_t.to(self.device),
                "old_log_probs": torch.FloatTensor(log_probs_flat[idx]).to(self.device),
                "advantages": torch.FloatTensor(advantages_flat[idx]).to(self.device),
                "returns":    torch.FloatTensor(returns_flat[idx]).to(self.device),
            }

    def get_recurrent_batches(self, seq_len: int):
        """
        For recurrent (LSTM, CTM) policies: yield all-env batches.

        Yields (n_envs, seq_len, ...) tensors so all environments are
        processed in a single batched forward pass, improving GPU utilization
        vs. the old per-env approach (batch=1).
        """
        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Transpose from (T, n_envs, ...) → (n_envs, T, ...)
        obs_all = self.observations.transpose(1, 0, 2)        # (n_envs, T, obs_dim)
        lp_all  = self.log_probs.T                             # (n_envs, T)
        adv_all = adv.T                                        # (n_envs, T)
        ret_all = self.returns.T                               # (n_envs, T)
        ep_all  = self.episode_starts.T                        # (n_envs, T)
        if self.continuous:
            act_all = self.actions.transpose(1, 0, 2)          # (n_envs, T, action_dim)
        else:
            act_all = self.actions.T                           # (n_envs, T)

        T = self.n_steps
        for start in range(0, T, seq_len):
            end = min(start + seq_len, T)
            act_t = torch.FloatTensor(act_all[:, start:end]) if self.continuous \
                    else torch.LongTensor(act_all[:, start:end])
            batch = {
                "obs":           torch.FloatTensor(obs_all[:, start:end]).to(self.device),
                "actions":       act_t.to(self.device),
                "old_log_probs": torch.FloatTensor(lp_all[:, start:end]).to(self.device),
                "advantages":    torch.FloatTensor(adv_all[:, start:end]).to(self.device),
                "returns":       torch.FloatTensor(ret_all[:, start:end]).to(self.device),
                "ep_starts":     torch.FloatTensor(ep_all[:, start:end]).to(self.device),
                "seq_len":       end - start,
                "hidden_state_0": self.hidden_snapshots.get(start),  # None for CTM
            }
            yield batch
