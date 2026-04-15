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
                 device: str = "cpu"):
        self.n_steps     = n_steps
        self.n_envs      = n_envs
        self.obs_dim     = obs_dim
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.is_recurrent = is_recurrent
        self.device      = device

        self.reset()

    def reset(self):
        self.observations = np.zeros((self.n_steps, self.n_envs, self.obs_dim),
                                     dtype=np.float32)
        self.actions      = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
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
        actions_flat = self.actions.reshape(n)
        log_probs_flat = self.log_probs.reshape(n)
        advantages_flat = self.advantages.reshape(n)
        returns_flat = self.returns.reshape(n)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / \
                          (advantages_flat.std() + 1e-8)

        for start in range(0, n, batch_size):
            idx = indices[start: start + batch_size]
            yield {
                "obs":        torch.FloatTensor(obs_flat[idx]).to(self.device),
                "actions":    torch.LongTensor(actions_flat[idx]).to(self.device),
                "old_log_probs": torch.FloatTensor(log_probs_flat[idx]).to(self.device),
                "advantages": torch.FloatTensor(advantages_flat[idx]).to(self.device),
                "returns":    torch.FloatTensor(returns_flat[idx]).to(self.device),
            }

    def get_recurrent_batches(self, seq_len: int):
        """
        For recurrent (LSTM, CTM) policies: yield sequences aligned with episodes.

        We split the rollout into non-overlapping sequences of length seq_len,
        keeping n_envs as the batch dimension. Each sequence starts where the
        last one ended (or at an episode boundary).

        Returns tuples of (obs_seq, actions_seq, log_probs_seq,
                           advantages_seq, returns_seq, is_first_seq)
        where is_first_seq[b,t] = True means the hidden state should be reset.
        """
        # Shape: (n_steps, n_envs, ...)
        # We treat each env as an independent trajectory, yielding sequences
        # of length seq_len from each.

        # Normalize advantages
        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for env_idx in range(self.n_envs):
            obs_e        = self.observations[:, env_idx, :]  # (T, obs_dim)
            act_e        = self.actions[:, env_idx]           # (T,)
            lp_e         = self.log_probs[:, env_idx]         # (T,)
            adv_e        = adv[:, env_idx]                    # (T,)
            ret_e        = self.returns[:, env_idx]           # (T,)
            ep_start_e   = self.episode_starts[:, env_idx]   # (T,)

            T = self.n_steps
            for start in range(0, T, seq_len):
                end = min(start + seq_len, T)
                s = end - start

                yield {
                    "obs":        torch.FloatTensor(obs_e[start:end]).unsqueeze(0).to(self.device),
                    "actions":    torch.LongTensor(act_e[start:end]).unsqueeze(0).to(self.device),
                    "old_log_probs": torch.FloatTensor(lp_e[start:end]).unsqueeze(0).to(self.device),
                    "advantages": torch.FloatTensor(adv_e[start:end]).unsqueeze(0).to(self.device),
                    "returns":    torch.FloatTensor(ret_e[start:end]).unsqueeze(0).to(self.device),
                    "ep_starts":  torch.FloatTensor(ep_start_e[start:end]).unsqueeze(0).to(self.device),
                    "seq_len":    s,
                }
