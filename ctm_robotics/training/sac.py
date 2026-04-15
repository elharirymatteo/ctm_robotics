"""
training/sac.py

Discrete Soft Actor-Critic (SAC) for CartPole.

Based on: Christodoulou (2019) "Soft Actor-Critic for Discrete Action Spaces"
https://arxiv.org/abs/1910.07207

Key differences from continuous SAC:
- Actor outputs action *probabilities* (not Gaussian parameters)
- Q-networks output Q(s,a) for ALL actions simultaneously (no action input)
- Entropy is computed exactly from the probability distribution (no reparameterization)
- Target entropy = -log(1/|A|) × 0.98 (fraction of max entropy)

Architecture:
  - Two Q-networks (for double-Q trick to reduce overestimation)
  - One actor (SACPolicy)
  - Auto-tuned temperature α
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import os
import time
from typing import List

from ..models.mlp_policy import SACQNetwork


class ReplayBuffer:
    """Simple uniform replay buffer for SAC."""

    def __init__(self, capacity: int, obs_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device   = device
        self.obs_dim  = obs_dim
        self.pos      = 0
        self.size     = 0

        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos]      = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.dones[self.pos]    = float(done)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, batch_size)
        return {
            "obs":      torch.FloatTensor(self.obs[idx]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(self.device),
            "actions":  torch.LongTensor(self.actions[idx]).to(self.device),
            "rewards":  torch.FloatTensor(self.rewards[idx]).to(self.device),
            "dones":    torch.FloatTensor(self.dones[idx]).to(self.device),
        }

    def __len__(self):
        return self.size


class SACTrainer:
    """
    Discrete SAC trainer.

    Args:
        actor:       SACPolicy instance
        q1, q2:     SACQNetwork instances (two critics for double-Q)
        env:         Single (non-vectorized) gymnasium environment
        config:      SACConfig dataclass
        train_config:TrainConfig dataclass
        agent_name:  For logging
    """

    def __init__(self, actor, q1, q2, env, config, train_config,
                 agent_name: str = "sac_mlp"):
        self.actor  = actor
        self.q1     = q1
        self.q2     = q2
        self.env    = env
        self.cfg    = config
        self.tcfg   = train_config
        self.device = torch.device(train_config.device)
        self.agent_name = agent_name

        self.actor.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)

        # Target networks — must match source network architecture exactly
        self._q_obs_dim    = q1.net[0].in_features
        self._q_action_dim = q1.net[-1].out_features
        # Reconstruct hidden sizes from the network
        hidden_sizes = tuple(
            q1.net[i].out_features
            for i in range(0, len(q1.net)-1, 2)   # every Linear layer except last
        )
        self.q1_target = SACQNetwork(
            self._q_obs_dim, self._q_action_dim, hidden_sizes=hidden_sizes
        ).to(self.device)
        self.q2_target = SACQNetwork(
            self._q_obs_dim, self._q_action_dim, hidden_sizes=hidden_sizes
        ).to(self.device)
        self.q1_target.load_state_dict(q1.state_dict())
        self.q2_target.load_state_dict(q2.state_dict())
        # Freeze target nets
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        # Optimizers
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=config.lr)
        self.q1_opt    = torch.optim.Adam(q1.parameters(),    lr=config.lr)
        self.q2_opt    = torch.optim.Adam(q2.parameters(),    lr=config.lr)

        # Auto-tuned entropy temperature
        action_dim = q1.net[-1].out_features
        self.target_entropy = -0.98 * np.log(1.0 / action_dim)
        if config.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.log_alpha = None
            self.alpha = config.alpha

        obs_dim = env.observation_space.shape[0]
        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, train_config.device)

        self.total_steps = 0
        self.eval_steps: List[int]   = []
        self.eval_returns: List[float] = []

        os.makedirs(train_config.log_dir, exist_ok=True)

    def _soft_update(self):
        tau = self.cfg.tau
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def update(self, batch):
        obs      = batch["obs"]
        next_obs = batch["next_obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"]
        dones    = batch["dones"]

        # ── Critic update ────────────────────────────────────────────────
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_obs)
            q1_next = self.q1_target(next_obs)   # (B, A)
            q2_next = self.q2_target(next_obs)
            q_next  = torch.min(q1_next, q2_next)
            # V(s') = sum_a π(a|s') [Q(s',a) - α log π(a|s')]
            v_next  = (next_probs * (q_next - self.alpha * next_log_probs)).sum(-1)
            target  = rewards + self.cfg.gamma * (1 - dones) * v_next

        q1_pred = self.q1(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2_pred = self.q2(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)

        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # ── Actor update ─────────────────────────────────────────────────
        probs, log_probs = self.actor(obs)
        with torch.no_grad():
            q1_val = self.q1(obs)
            q2_val = self.q2(obs)
            q_val  = torch.min(q1_val, q2_val)

        # J(π) = E[sum_a π(a|s) (α log π(a|s) - Q(s,a))]
        actor_loss = (probs * (self.alpha * log_probs - q_val)).sum(-1).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # ── Alpha update ─────────────────────────────────────────────────
        if self.log_alpha is not None:
            entropy = -(probs * log_probs).sum(-1).detach().mean()
            alpha_loss = self.log_alpha * (entropy - self.target_entropy)
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        self._soft_update()

        return {
            "q1_loss":    q1_loss.item(),
            "q2_loss":    q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha":      self.alpha,
        }

    def evaluate(self, eval_env, n_episodes: int = 20):
        self.actor.eval()
        returns = []
        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=ep + 9999)
            done = False; ep_ret = 0.0
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    probs, _ = self.actor(obs_t)
                    action = probs.argmax(-1).item()
                obs, reward, term, trunc, _ = eval_env.step(action)
                ep_ret += reward
                done = term or trunc
            returns.append(ep_ret)
        self.actor.train()
        return np.mean(returns), np.std(returns)

    def train(self, eval_env=None, verbose: bool = True):
        obs, _ = self.env.reset()
        last_eval  = 0
        start_time = time.time()

        while self.total_steps < self.tcfg.total_steps:
            # Collect one step
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if self.total_steps < self.cfg.learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    probs, _ = self.actor(obs_t)
                    action = torch.multinomial(probs, 1).item()

            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            self.buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs if not done else self.env.reset()[0]
            self.total_steps += 1

            # Update
            if (self.total_steps >= self.cfg.learning_starts
                    and self.total_steps % self.cfg.train_freq == 0
                    and len(self.buffer) >= self.cfg.batch_size):
                batch  = self.buffer.sample(self.cfg.batch_size)
                losses = self.update(batch)

            # Evaluate
            if (eval_env is not None
                    and self.total_steps - last_eval >= self.tcfg.eval_every):
                mean_ret, std_ret = self.evaluate(eval_env, self.tcfg.eval_episodes)
                self.eval_returns.append(mean_ret)
                self.eval_steps.append(self.total_steps)
                last_eval = self.total_steps

                if verbose:
                    elapsed = time.time() - start_time
                    print(f"[{self.agent_name}] step={self.total_steps:>7} "
                          f"eval={mean_ret:>7.1f}±{std_ret:.1f}  "
                          f"α={self.alpha:.3f}  t={elapsed:.0f}s")

        return self.eval_steps, self.eval_returns

    def save(self, path: str):
        torch.save({
            "actor":       self.actor.state_dict(),
            "q1":          self.q1.state_dict(),
            "q2":          self.q2.state_dict(),
            "eval_steps":  self.eval_steps,
            "eval_returns":self.eval_returns,
            "total_steps": self.total_steps,
        }, path)
