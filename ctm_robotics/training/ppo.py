"""
training/ppo.py

PPO trainer that works uniformly with MLPActorCritic, LSTMActorCritic,
and CTMActorCritic via a shared interface.

The key design challenge: recurrent policies need their hidden state
carried across steps and reset at episode boundaries. We handle this
with a simple flag and the rollout buffer's recurrent batch getter.

Algorithm:
  repeat:
    1. Collect n_steps × n_envs transitions (rollout phase)
    2. Compute GAE returns and advantages
    3. For n_epochs: update policy on minibatches using the clipped PPO loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import time
import os

from .rollout_buffer import RolloutBuffer


class PPOTrainer:
    """
    Proximal Policy Optimization trainer.
    Compatible with stateless (MLP) and recurrent (LSTM, CTM) actor-critics.

    Args:
        policy:    An actor-critic instance (MLPActorCritic | LSTMActorCritic | CTMActorCritic)
        env:       Vectorized gymnasium environment (SyncVectorEnv)
        config:    PPOConfig dataclass
        train_config: TrainConfig dataclass
        is_recurrent: True for LSTM and CTM policies
        agent_name:   String label for logging
    """

    def __init__(self, policy, env, config, train_config,
                 is_recurrent: bool = False,
                 agent_name: str = "agent"):
        self.policy       = policy
        self.env          = env
        self.cfg          = config
        self.tcfg         = train_config
        self.is_recurrent = is_recurrent
        self.agent_name   = agent_name
        self.device       = torch.device(train_config.device)

        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.lr,
            eps=1e-5,
        )

        n_envs  = env.num_envs
        obs_dim = env.single_observation_space.shape[0]

        self.buffer = RolloutBuffer(
            n_steps=config.n_steps,
            n_envs=n_envs,
            obs_dim=obs_dim,
            action_dim=1,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            is_recurrent=is_recurrent,
            device=train_config.device,
        )

        # Logging
        self.ep_returns  : List[float] = []
        self.eval_returns: List[float] = []
        self.eval_steps  : List[int]   = []
        self.total_steps = 0

        os.makedirs(train_config.log_dir, exist_ok=True)

    # ── Collection ────────────────────────────────────────────────────────────

    def collect_rollout(self, obs, hidden_states, episode_starts):
        """
        Collect n_steps of experience from n_envs in parallel.

        Returns updated obs, hidden_states, episode_starts for next rollout.
        """
        self.policy.eval()
        n_envs  = self.env.num_envs

        for step in range(self.cfg.n_steps):
            obs_t = torch.FloatTensor(obs).to(self.device)  # (n_envs, obs_dim)

            with torch.no_grad():
                if self.is_recurrent:
                    actions, log_probs, values, entropies, hidden_states = \
                        self.policy.get_action(obs_t, hidden_states)
                else:
                    actions, log_probs, values, entropies = \
                        self.policy.get_action(obs_t)

            actions_np = actions.cpu().numpy()
            new_obs, rewards, terminated, truncated, infos = self.env.step(actions_np)
            dones = terminated | truncated

            self.buffer.add(
                obs=obs,
                action=actions_np,
                reward=rewards,
                done=dones,
                value=values.cpu().numpy(),
                log_prob=log_probs.cpu().numpy(),
                episode_start=episode_starts,
            )

            # Track episodic return
            for i, (done, info) in enumerate(zip(dones, infos.get('final_info', [{}]*n_envs) or [{}]*n_envs)):
                if done and info and 'episode' in info:
                    self.ep_returns.append(info['episode']['r'])

            obs = new_obs
            episode_starts = dones

            # Reset hidden state for finished episodes
            if self.is_recurrent and any(dones):
                hidden_states = self._reset_hidden_for_dones(
                    hidden_states, dones, n_envs)

        # Bootstrap value for GAE
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            if self.is_recurrent:
                _, last_values, _, _, _ = self.policy.get_action(obs_t, hidden_states)
            else:
                _, _, last_values, _ = self.policy.get_action(obs_t)

        self.buffer.compute_returns_and_advantages(
            last_values.cpu().numpy(),
            dones,
        )
        self.total_steps += self.cfg.n_steps * n_envs
        return obs, hidden_states, episode_starts

    def _reset_hidden_for_dones(self, hidden_states, dones, n_envs):
        """Reset hidden state for environments that finished an episode."""
        done_indices = np.where(dones)[0]
        if len(done_indices) == 0:
            return hidden_states

        fresh = self.policy.init_hidden(n_envs, self.device)

        # Detect policy type by structure
        pre_h_or_h, second = hidden_states
        if isinstance(second, torch.Tensor):
            # LSTM: (h, c) — both are tensors (n_layers, batch, hidden)
            h, c = hidden_states
            fh, fc = fresh
            for i in done_indices:
                h[:, i, :] = fh[:, i, :]
                c[:, i, :] = fc[:, i, :]
            return (h, c)
        else:
            # CTM: (pre_h, post_list) — second element is a Python list
            pre_h, post_list = hidden_states
            fresh_pre_h, fresh_post_list = fresh
            for i in done_indices:
                pre_h[i] = fresh_pre_h[i]
                post_list[-1][i] = fresh_post_list[-1][i]
            return (pre_h, post_list)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self):
        """Run n_epochs of PPO updates on the collected rollout."""
        self.policy.train()
        clip_eps = self.cfg.clip_eps

        pg_losses, vf_losses, ent_losses = [], [], []

        for epoch in range(self.cfg.n_epochs):
            if self.is_recurrent:
                batches = self.buffer.get_recurrent_batches(
                    seq_len=self.cfg.n_steps)
            else:
                batches = self.buffer.get_stateless_batches(
                    batch_size=self.cfg.batch_size)

            for batch in batches:
                if self.is_recurrent:
                    # Recurrent: obs_seq is (1, seq_len, obs_dim)
                    obs_seq    = batch["obs"]           # (1, T, obs_dim)
                    act_seq    = batch["actions"]        # (1, T)
                    old_lp     = batch["old_log_probs"]  # (1, T)
                    adv        = batch["advantages"]     # (1, T)
                    ret        = batch["returns"]        # (1, T)
                    n_envs_b   = obs_seq.shape[0]

                    h0 = self.policy.init_hidden(n_envs_b, self.device)
                    log_probs, entropies, values = \
                        self.policy.evaluate_actions(obs_seq, act_seq, h0)

                    old_lp_flat = old_lp.reshape(-1)
                    adv_flat    = adv.reshape(-1)
                    ret_flat    = ret.reshape(-1)

                else:
                    # Stateless
                    obs     = batch["obs"]
                    actions = batch["actions"]
                    old_lp  = batch["old_log_probs"]
                    adv_flat = batch["advantages"]
                    ret_flat = batch["returns"]

                    log_probs, entropies, values = \
                        self.policy.evaluate_actions(obs, actions)
                    old_lp_flat = old_lp

                # PPO clipped surrogate loss
                ratio = torch.exp(log_probs - old_lp_flat)
                pg1   = ratio * adv_flat
                pg2   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_flat
                pg_loss = -torch.min(pg1, pg2).mean()

                # Value function loss (clipped)
                vf_loss = F.mse_loss(values, ret_flat)

                # Entropy bonus
                ent_loss = -entropies.mean()

                loss = (pg_loss
                        + self.cfg.vf_coef * vf_loss
                        + self.cfg.ent_coef * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.cfg.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        return {
            "pg_loss":  np.mean(pg_losses),
            "vf_loss":  np.mean(vf_losses),
            "ent_loss": np.mean(ent_losses),
        }

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, eval_env, n_episodes: int = 20):
        """Run n_episodes and return mean episodic return."""
        self.policy.eval()
        returns = []

        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=ep + 9999)
            done = False
            ep_ret = 0.0

            if self.is_recurrent:
                hidden = self.policy.init_hidden(1, self.device)

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if self.is_recurrent:
                        action, _, _, _, hidden = \
                            self.policy.get_action(obs_t, hidden)
                    else:
                        action, _, _, _ = self.policy.get_action(obs_t)

                obs, reward, term, trunc, _ = eval_env.step(action.item())
                ep_ret += reward
                done = term or trunc

            returns.append(ep_ret)

        return np.mean(returns), np.std(returns)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, eval_env=None, verbose: bool = True):
        """
        Full training loop.

        Returns:
            eval_steps:   list of step counts at eval points
            eval_returns: list of mean returns at eval points
        """
        obs, _ = self.env.reset()
        n_envs = self.env.num_envs
        episode_starts = np.ones(n_envs, dtype=bool)

        if self.is_recurrent:
            hidden_states = self.policy.init_hidden(n_envs, self.device)
        else:
            hidden_states = None

        last_eval = 0
        start_time = time.time()

        while self.total_steps < self.tcfg.total_steps:
            # Collect rollout
            if self.is_recurrent:
                obs, hidden_states, episode_starts = \
                    self.collect_rollout(obs, hidden_states, episode_starts)
            else:
                obs, _, episode_starts = \
                    self.collect_rollout(obs, None, episode_starts)

            # Update policy
            losses = self.update()
            self.buffer.reset()

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
                          f"pg={losses['pg_loss']:.3f} "
                          f"vf={losses['vf_loss']:.3f}  "
                          f"t={elapsed:.0f}s")

        return self.eval_steps, self.eval_returns

    def save(self, path: str):
        torch.save({
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "eval_steps": self.eval_steps,
            "eval_returns": self.eval_returns,
            "total_steps": self.total_steps,
        }, path)
        if hasattr(self, '_verbose') and self._verbose:
            print(f"  Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.eval_steps   = ckpt["eval_steps"]
        self.eval_returns = ckpt["eval_returns"]
        self.total_steps  = ckpt["total_steps"]
