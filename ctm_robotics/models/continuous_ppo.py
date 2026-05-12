"""
models/continuous_ppo.py

Continuous-action PPO actor-critics (Gaussian policy).
Mirrors the interface of the discrete PPO models (mlp_policy, lstm_policy, ctm)
but outputs Normal distributions instead of Categorical.

Interface:
    forward(obs[, hidden]) -> (action_mean, values[, new_hidden])
    get_action(obs[, hidden]) -> (action, log_prob, value, entropy[, new_hidden])
    evaluate_actions(obs, actions[, hidden, dones]) -> (log_probs, entropies, values)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .mlp_policy import build_mlp
from .ctm import SynapseModel, NeuronLevelModels, SynchronizationHead


class ContinuousMLPActorCritic(nn.Module):
    """Gaussian MLP actor-critic for continuous PPO."""

    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64),
                 max_action=1.0):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action

        self.actor_mean = build_mlp(obs_dim, hidden_sizes, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = build_mlp(obs_dim, hidden_sizes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)

    def forward(self, obs):
        mean = self.actor_mean(obs)
        values = self.critic(obs).squeeze(-1)
        return mean, values

    def get_action(self, obs):
        mean, values = self.forward(obs)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample().clamp(-self.max_action, self.max_action)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, values, entropy

    def evaluate_actions(self, obs, actions):
        mean, values = self.forward(obs)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, values


class ContinuousLSTMActorCritic(nn.Module):
    """Gaussian LSTM actor-critic for continuous PPO."""

    def __init__(self, obs_dim, action_dim, hidden_size=64, n_layers=1,
                 max_action=1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        # Separate obs-only MLP critic: VF gradient doesn't touch LSTM backbone
        self.critic = build_mlp(obs_dim, (64, 64), 1)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def forward(self, obs, hidden):
        x = self.encoder(obs).unsqueeze(1)
        out, hidden = self.lstm(x, hidden)
        out = out.squeeze(1)
        mean = self.actor_mean(out)
        values = self.critic(obs).squeeze(-1)
        return mean, values, hidden

    def get_action(self, obs, hidden):
        mean, values, hidden = self.forward(obs, hidden)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample().clamp(-self.max_action, self.max_action)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, values, entropy, hidden

    def evaluate_actions(self, obs_seq, actions_seq, hidden_state_0, dones_seq=None,
                         tbptt_k: int = 16):
        batch, seq_len, _ = obs_seq.shape
        all_means = []
        h, c = hidden_state_0

        for t in range(seq_len):
            if dones_seq is not None and dones_seq[:, t].any():
                mask = (dones_seq[:, t] > 0.5).float().view(1, batch, 1)
                h = h * (1 - mask)
                c = c * (1 - mask)
            x = self.encoder(obs_seq[:, t, :]).unsqueeze(1)
            out, (h, c) = self.lstm(x, (h, c))
            if (t + 1) % tbptt_k == 0:
                h, c = h.detach(), c.detach()
            out = out.squeeze(1)
            all_means.append(self.actor_mean(out))

        means_flat = torch.stack(all_means, dim=1).view(batch * seq_len, -1)
        values_flat = self.critic(obs_seq.reshape(-1, self.obs_dim)).squeeze(-1)
        std = self.actor_log_std.exp()
        dist = Normal(means_flat, std)
        log_prob = dist.log_prob(actions_seq.reshape(-1, self.action_dim)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, values_flat


class ContinuousCTMActorCritic(nn.Module):
    """Gaussian CTM actor-critic for continuous PPO.
    Same CTM core as the discrete version, but with Gaussian action output."""

    def __init__(self, obs_dim, action_dim,
                 d_model=64, synapse_hidden=128, synapse_depth=3,
                 memory_length=4, nlm_hidden=32, nlm_depth=2,
                 n_synch_out=32, synch_window=8, synch_decay=0.9,
                 n_ticks=5, input_hidden=64, max_action=1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.M = memory_length
        self.n_ticks = n_ticks
        self.synch_window = synch_window
        self.max_action = max_action

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, input_hidden), nn.GELU(),
            nn.Linear(input_hidden, input_hidden),
        )

        self.synapse = SynapseModel(d_model, input_hidden, synapse_hidden, synapse_depth)
        self.nlms = NeuronLevelModels(d_model, memory_length, nlm_hidden, nlm_depth)
        self.sync_head = SynchronizationHead(d_model, n_synch_out, synch_window, synch_decay)

        self.init_post_act = nn.Parameter(torch.randn(1, d_model) * 0.1)
        self.post_norm = nn.LayerNorm(d_model)

        self.actor_mean = nn.Linear(n_synch_out, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = nn.Linear(n_synch_out, 1)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

        self.last_sync_repr = None
        self.last_post_act_seq = None

    def init_hidden(self, batch_size, device):
        pre_h = torch.zeros(batch_size, self.d_model, self.M, device=device)
        post_0 = self.init_post_act.expand(batch_size, -1).detach().clone()
        return (pre_h, [post_0])

    def _detach_hidden(self, hidden):
        pre_h, post_list = hidden
        return (pre_h.detach(), [p.detach() for p in post_list])

    def forward(self, obs, hidden):
        pre_h, post_list = hidden
        obs_embed = self.backbone(obs)
        post_act = post_list[-1]

        tick_post_acts = []
        for tick in range(self.n_ticks):
            pre_act = self.synapse(post_act, obs_embed)
            pre_h = torch.cat([pre_h[..., 1:], pre_act.unsqueeze(-1)], dim=-1)
            post_act = self.post_norm(self.nlms(pre_h))
            tick_post_acts.append(post_act)

        new_post_list = post_list + tick_post_acts
        if len(new_post_list) > self.synch_window:
            new_post_list = new_post_list[-self.synch_window:]

        sync_repr = self.sync_head(new_post_list)

        self.last_sync_repr = sync_repr.detach()
        self.last_post_act_seq = tick_post_acts

        mean = self.actor_mean(sync_repr)
        values = self.critic_head(sync_repr).squeeze(-1)
        new_hidden = (pre_h, new_post_list)
        return mean, values, new_hidden

    def get_action(self, obs, hidden):
        mean, values, hidden = self.forward(obs, hidden)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample().clamp(-self.max_action, self.max_action)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, values, entropy, hidden

    def ppo_loss_chunked(self, obs_seq, actions_seq, hidden_state_0,
                         dones_seq, old_log_probs, advantages, returns,
                         clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
                         chunk_size=128):
        """TBPTT: gradient flows freely within each chunk of chunk_size steps."""
        batch, seq_len, _ = obs_seq.shape
        pg_sum = vf_sum = ent_sum = 0.0

        hidden = hidden_state_0
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_means, chunk_vals = [], []

            for t in range(chunk_start, chunk_end):
                if dones_seq is not None:
                    mask = dones_seq[:, t]
                    if mask.any():
                        pre_h, post_list = hidden
                        fresh_pre = torch.zeros_like(pre_h)
                        fresh_p0 = self.init_post_act.expand(batch, -1).detach()
                        m2 = (mask > 0.5).unsqueeze(-1)
                        m3 = m2.unsqueeze(-1)
                        pre_h = torch.where(m3, fresh_pre, pre_h)
                        post_list = [torch.where(m2, fresh_p0, p) for p in post_list]
                        hidden = (pre_h, post_list)

                obs_t = obs_seq[:, t, :]
                mean, vals, hidden = self.forward(obs_t, hidden)
                chunk_means.append(mean)
                chunk_vals.append(vals)

            hidden = self._detach_hidden(hidden)

            means_c = torch.stack(chunk_means, dim=1)   # (batch, chunk, action_dim)
            vals_c  = torch.stack(chunk_vals,  dim=1)   # (batch, chunk)
            acts_c  = actions_seq[:, chunk_start:chunk_end, :]
            std     = self.actor_log_std.exp()
            dist    = Normal(means_c.reshape(-1, self.action_dim), std)
            lp      = dist.log_prob(acts_c.reshape(-1, self.action_dim)).sum(-1)
            ent     = dist.entropy().sum(-1)

            old_lp = old_log_probs[:, chunk_start:chunk_end].reshape(-1)
            adv    = advantages[:, chunk_start:chunk_end].reshape(-1)
            ret    = returns[:, chunk_start:chunk_end].reshape(-1)

            ratio   = torch.exp((lp - old_lp).clamp(-3, 3))
            pg1     = ratio * adv
            pg2     = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            pg_loss = -torch.min(pg1, pg2).mean()
            vf_loss = F.mse_loss(vals_c.view(-1), ret)
            ent_loss = -ent.mean()

            chunk_loss = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss
            (chunk_loss * (chunk_end - chunk_start) / seq_len).backward()

            pg_sum  += pg_loss.item()  * (chunk_end - chunk_start)
            vf_sum  += vf_loss.item()  * (chunk_end - chunk_start)
            ent_sum += ent_loss.item() * (chunk_end - chunk_start)

        return {
            "pg_loss":  pg_sum  / seq_len,
            "vf_loss":  vf_sum  / seq_len,
            "ent_loss": ent_sum / seq_len,
        }

    def get_sync_saliency(self):
        return self.last_sync_repr

    def get_neural_dynamics(self):
        return self.last_post_act_seq
