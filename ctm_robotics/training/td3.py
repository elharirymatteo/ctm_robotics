"""
training/td3.py

TD3 with R2D2-style burn-in for recurrent policies.

For recurrent actors (LSTM, CTM), the replay buffer stores full episodes.
During training, we sample sequences and replay the first `burn_in` steps
to rebuild valid hidden states before computing gradients on the remaining steps.

This solves the "stale hidden state" problem that kills off-policy recurrence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import time
import os
import copy


# ---------------------------------------------------------------------------
# Replay buffers
# ---------------------------------------------------------------------------

class FlatReplayBuffer:
    """Standard replay buffer for non-recurrent (MLP) policies."""

    def __init__(self, obs_dim, action_dim, max_size=200_000):
        self.max_size = max_size
        self.pos = 0
        self.size = 0
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        i = self.pos
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device="cpu"):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.FloatTensor(self.obs[idx]).to(device),
            "action": torch.FloatTensor(self.action[idx]).to(device),
            "reward": torch.FloatTensor(self.reward[idx]).to(device),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(device),
            "done": torch.FloatTensor(self.done[idx]).to(device),
        }


class EpisodeReplayBuffer:
    """Episode-based replay buffer for recurrent policies (R2D2-style).

    Stores complete episodes. Sampling returns sequences of length
    (burn_in + 1) so the trainer can replay burn_in steps to rebuild
    hidden states before training on the final transition.
    """

    def __init__(self, max_transitions=200_000):
        self.max_transitions = max_transitions
        self.episodes: List[dict] = []
        self.total_transitions = 0
        self._current_ep: List[tuple] = []

    def add(self, obs, action, reward, next_obs, done):
        self._current_ep.append((obs.copy(), action.copy(), reward, next_obs.copy(), done))
        if done:
            self._finish_episode()

    def _finish_episode(self):
        if not self._current_ep:
            return
        ep = {
            "obs": np.array([t[0] for t in self._current_ep], dtype=np.float32),
            "action": np.array([t[1] for t in self._current_ep], dtype=np.float32),
            "reward": np.array([t[2] for t in self._current_ep], dtype=np.float32),
            "next_obs": np.array([t[3] for t in self._current_ep], dtype=np.float32),
            "done": np.array([t[4] for t in self._current_ep], dtype=np.float32),
        }
        self.episodes.append(ep)
        self.total_transitions += len(self._current_ep)
        self._current_ep = []
        # Evict oldest episodes if over budget
        while self.total_transitions > self.max_transitions and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_transitions -= len(removed["obs"])

    @property
    def size(self):
        return self.total_transitions + len(self._current_ep)

    def sample_sequences(self, batch_size, burn_in, device="cpu"):
        """Sample batch_size sequences of length (burn_in + 1).

        Returns dict with:
            burn_obs:  (batch, burn_in, obs_dim) — obs for hidden state rebuild
            obs:       (batch, obs_dim) — training transition obs
            action:    (batch, action_dim)
            reward:    (batch,)
            next_obs:  (batch, obs_dim)
            done:      (batch,)
        """
        # Only sample from completed episodes
        if not self.episodes:
            return None

        burn_obs_list, obs_list, act_list = [], [], []
        rew_list, nobs_list, done_list = [], [], []

        for _ in range(batch_size):
            # Pick random episode
            ep_idx = np.random.randint(len(self.episodes))
            ep = self.episodes[ep_idx]
            ep_len = len(ep["obs"])

            # Pick random transition index (the one we train on)
            t = np.random.randint(ep_len)

            # Extract burn-in sequence: steps [t - burn_in, ..., t - 1]
            burn_start = max(0, t - burn_in)
            burn_seq = ep["obs"][burn_start:t]  # may be shorter than burn_in

            # Pad with zeros if burn sequence is shorter than burn_in
            pad_len = burn_in - len(burn_seq)
            if pad_len > 0:
                obs_dim = ep["obs"].shape[1]
                pad = np.zeros((pad_len, obs_dim), dtype=np.float32)
                burn_seq = np.concatenate([pad, burn_seq], axis=0)

            burn_obs_list.append(burn_seq)
            obs_list.append(ep["obs"][t])
            act_list.append(ep["action"][t])
            rew_list.append(ep["reward"][t])
            nobs_list.append(ep["next_obs"][t])
            done_list.append(ep["done"][t])

        return {
            "burn_obs": torch.FloatTensor(np.array(burn_obs_list)).to(device),
            "obs": torch.FloatTensor(np.array(obs_list)).to(device),
            "action": torch.FloatTensor(np.array(act_list)).to(device),
            "reward": torch.FloatTensor(np.array(rew_list)).to(device),
            "next_obs": torch.FloatTensor(np.array(nobs_list)).to(device),
            "done": torch.FloatTensor(np.array(done_list)).to(device),
        }


# ---------------------------------------------------------------------------
# TD3 Trainer
# ---------------------------------------------------------------------------

class TD3Trainer:
    def __init__(self, actor, critic, env, config, train_config,
                 is_recurrent=False, agent_name="agent"):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.cfg = config
        self.tcfg = train_config
        self.is_recurrent = is_recurrent
        self.agent_name = agent_name
        self.device = torch.device(train_config.device)

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)

        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=config.lr_actor)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=config.lr_critic)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        self.burn_in = getattr(config, 'burn_in', 10)

        if is_recurrent:
            self.buffer = EpisodeReplayBuffer(max_transitions=config.buffer_size)
        else:
            self.buffer = FlatReplayBuffer(obs_dim, action_dim, max_size=config.buffer_size)

        self.eval_returns: List[float] = []
        self.eval_steps: List[int] = []
        self.total_steps = 0
        os.makedirs(train_config.log_dir, exist_ok=True)

    # -- Hidden state rebuild via burn-in ----------------------------------

    def _rebuild_hidden(self, burn_obs, actor=None):
        """Run actor through burn-in observations to rebuild hidden states.

        Args:
            burn_obs: (batch, burn_in, obs_dim)
            actor: which actor to use (default: self.actor)
        Returns:
            list of hidden states, one per batch element
        """
        actor = actor or self.actor
        batch_size = burn_obs.shape[0]
        burn_len = burn_obs.shape[1]

        # LSTM: batched burn-in (much faster)
        h0 = actor.init_hidden(batch_size, self.device)
        if isinstance(h0, tuple) and isinstance(h0[1], torch.Tensor):
            # LSTM — run full sequence through encoder + LSTM in one call
            with torch.no_grad():
                if hasattr(actor, 'encoder') and hasattr(actor, 'lstm'):
                    x = actor.encoder(burn_obs.view(-1, burn_obs.shape[-1]))
                    x = x.view(batch_size, burn_len, -1)
                    _, (h, c) = actor.lstm(x, h0)
                    return [(h[:, i:i+1, :].contiguous(), c[:, i:i+1, :].contiguous())
                            for i in range(batch_size)]
                # Fallback: step-by-step batched
                h = h0
                for t in range(burn_len):
                    _, h = actor(burn_obs[:, t, :], h)
                return [(h[0][:, i:i+1, :].contiguous(), h[1][:, i:i+1, :].contiguous())
                        for i in range(batch_size)]

        # CTM: batched burn-in — process all samples together.
        # pre_h/post_list have a batch dimension so CTM forward handles
        # the full batch in one call, giving ~22x speedup over per-sample loop.
        h = actor.init_hidden(batch_size, self.device)
        with torch.no_grad():
            for t in range(burn_len):
                obs_t = burn_obs[:, t, :]           # (batch, obs_dim)
                pad = obs_t.abs().sum(-1) < 1e-8    # True = zero-padded step
                if pad.all():
                    continue
                _, h_new = actor(obs_t, h)
                if pad.any():
                    # Keep old hidden for padded steps, new for real steps
                    pre_old, post_old = h
                    pre_new, post_new = h_new
                    m3 = pad.float()[:, None, None]     # (batch, 1, 1) — 1 = keep old
                    m2 = pad.float()[:, None]           # (batch, 1)
                    h = (
                        pre_old * m3 + pre_new * (1 - m3),
                        [po * m2 + pn * (1 - m2)
                         for po, pn in zip(post_old, post_new)],
                    )
                else:
                    h = h_new
        # Return single batched hidden (CTM supports batched forward natively)
        return h

    def _batch_actor_forward(self, obs_batch, hiddens, actor=None):
        """Run actor on a batch with individual hidden states."""
        actor = actor or self.actor
        if not self.is_recurrent or hiddens is None:
            return actor(obs_batch)[0]

        # CTM: hiddens is a single batched (pre_h, post_list) tuple
        if isinstance(hiddens, tuple) and isinstance(hiddens[1], list):
            return actor(obs_batch, hiddens)[0]

        # LSTM: hiddens is a list of per-sample (h, c) tuples
        first = hiddens[0]
        if isinstance(first, tuple) and isinstance(first[1], torch.Tensor):
            hs = torch.cat([h[0] for h in hiddens], dim=1)
            cs = torch.cat([h[1] for h in hiddens], dim=1)
            return actor(obs_batch, (hs, cs))[0]

        # Fallback: per-sample loop
        actions = []
        for i in range(obs_batch.shape[0]):
            a, _ = actor(obs_batch[i:i+1], hiddens[i])
            actions.append(a)
        return torch.cat(actions, dim=0)

    def _advance_hidden(self, obs, hiddens, actor=None):
        """Advance hidden state one step through obs (no gradient)."""
        actor = actor or self.actor
        batch_size = obs.shape[0]
        with torch.no_grad():
            # CTM: batched (pre_h, post_list) tuple
            if isinstance(hiddens, tuple) and isinstance(hiddens[1], list):
                _, new_hidden = actor(obs, hiddens)
                return new_hidden
            # LSTM: list of per-sample (h, c) tuples
            hs = torch.cat([h[0] for h in hiddens], dim=1)  # (n_layers, batch, hid)
            cs = torch.cat([h[1] for h in hiddens], dim=1)
            _, (new_h, new_c) = actor(obs, (hs, cs))
            return [(new_h[:, i:i+1, :].contiguous(), new_c[:, i:i+1, :].contiguous())
                    for i in range(batch_size)]

    # -- Selection ---------------------------------------------------------

    def _select_action(self, obs, hidden, noise_scale):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, new_hidden = self.actor.get_action(obs_t, hidden)
        action = action.cpu().numpy()[0]
        action += np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action, -self.max_action, self.max_action)
        return action, new_hidden

    # -- Update ------------------------------------------------------------

    def update(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # Rebuild h_{t-1} from burn-in, then advance through obs_t to get h_t
        # so that next_action is computed with the correct hidden state.
        if self.is_recurrent:
            hiddens = self._rebuild_hidden(batch["burn_obs"], actor=self.actor_target)
            hiddens = self._advance_hidden(obs, hiddens, actor=self.actor_target)
        else:
            hiddens = None

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.cfg.policy_noise
                     ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = self._batch_actor_forward(next_obs, hiddens,
                                                     actor=self.actor_target)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            tq1, tq2 = self.critic_target(next_obs, next_action)
            target_q = reward + (1 - done) * self.cfg.gamma * torch.min(tq1, tq2)

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        return critic_loss.item()

    def _is_lstm_actor(self, actor=None):
        actor = actor or self.actor
        h0 = actor.init_hidden(1, self.device)
        return isinstance(h0, tuple) and isinstance(h0[1], torch.Tensor)

    def update_actor(self, batch):
        obs = batch["obs"]

        if self.is_recurrent:
            if self._is_lstm_actor():
                # BPTT through burn-in so recurrent weights receive gradient.
                # Without this, the LSTM trains as a stateless MLP.
                burn_obs = batch["burn_obs"]
                batch_size = obs.shape[0]
                h = self.actor.init_hidden(batch_size, self.device)
                for t in range(burn_obs.shape[1]):
                    step = burn_obs[:, t, :]
                    if step.abs().sum(-1).lt(1e-8).all():
                        continue
                    _, h = self.actor(step, h)
                actor_action, _ = self.actor(obs, h)
            else:
                # CTM: single-step BPTT through 5 internal ticks is sufficient
                hiddens = self._rebuild_hidden(batch["burn_obs"], actor=self.actor)
                actor_action = self._batch_actor_forward(obs, hiddens)
        else:
            actor_action = self.actor(obs)[0]

        actor_loss = -self.critic.q1_forward(obs, actor_action).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()
        return actor_loss.item()

    def _soft_update(self):
        tau = self.cfg.tau
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

    # -- Evaluate ----------------------------------------------------------

    def evaluate(self, eval_env, n_episodes=20):
        self.actor.eval()
        returns = []
        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=ep + 9999)
            done = False
            ep_ret = 0.0
            hidden = self.actor.init_hidden(1, self.device) if self.is_recurrent else None

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, hidden = self.actor.get_action(obs_t, hidden)
                obs, reward, term, trunc, _ = eval_env.step(action.cpu().numpy()[0])
                ep_ret += reward
                done = term or trunc
            returns.append(ep_ret)
        self.actor.train()
        return np.mean(returns), np.std(returns)

    # -- Main loop ---------------------------------------------------------

    def train(self, eval_env=None, verbose=True):
        obs, _ = self.env.reset()
        hidden = self.actor.init_hidden(1, self.device) if self.is_recurrent else None
        ep_ret = 0.0
        last_eval = 0
        start_time = time.time()
        update_count = 0

        while self.total_steps < self.tcfg.total_steps:
            if self.total_steps < self.cfg.learning_starts:
                action = self.env.action_space.sample()
                if self.is_recurrent:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        _, hidden = self.actor.get_action(obs_t, hidden)
            else:
                action, hidden = self._select_action(
                    obs, hidden, self.cfg.exploration_noise * self.max_action)

            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            ep_ret += reward

            if self.is_recurrent:
                self.buffer.add(obs, action, reward, next_obs, done)
            else:
                self.buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            self.total_steps += 1

            if done:
                obs, _ = self.env.reset()
                hidden = self.actor.init_hidden(1, self.device) if self.is_recurrent else None
                ep_ret = 0.0

            # Training
            if (self.total_steps >= self.cfg.learning_starts
                    and self.total_steps % self.cfg.train_freq == 0
                    and self.buffer.size >= self.cfg.batch_size):

                if self.is_recurrent:
                    batch = self.buffer.sample_sequences(
                        self.cfg.batch_size, self.burn_in, self.device)
                    if batch is None:
                        continue
                else:
                    batch = self.buffer.sample(self.cfg.batch_size, self.device)

                critic_loss = self.update(batch)
                update_count += 1

                if update_count % self.cfg.policy_delay == 0:
                    self.update_actor(batch)
                    self._soft_update()

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
                          f"eval={mean_ret:>7.1f}\u00b1{std_ret:.1f}  "
                          f"t={elapsed:.0f}s")

        return self.eval_steps, self.eval_returns

    def save(self, path):
        torch.save({
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "actor_target_state": self.actor_target.state_dict(),
            "critic_target_state": self.critic_target.state_dict(),
            "eval_steps": self.eval_steps,
            "eval_returns": self.eval_returns,
            "total_steps": self.total_steps,
        }, path)
