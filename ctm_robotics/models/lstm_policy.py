"""
models/lstm_policy.py

LSTM-based Actor-Critic for PPO-LSTM.

Architecture:
    obs_t → Linear encoder → LSTM(h_{t-1}, c_{t-1}) → h_t → [actor | critic]

The hidden state (h_t, c_t) is carried across environment steps within an episode,
giving the agent a compressed memory of past observations.

Key difference from CTM:
- Memory = fixed-size vector (h_t ∈ R^{hidden_size})
- No notion of neuron timing or synchronization
- All neurons share the same gating mechanism
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class LSTMActorCritic(nn.Module):
    """
    LSTM Actor-Critic compatible with our PPO trainer's recurrent interface.

    Interface contract (matches CTMActorCritic):
        forward(obs, hidden_state) → (logits, values, new_hidden_state)
        get_action(obs, hidden_state) → (action, log_prob, value, entropy, new_hidden_state)
        evaluate_actions(obs_seq, actions_seq, hidden_state_0) → (log_probs, entropies, values)
        init_hidden(batch_size, device) → hidden_state tuple

    hidden_state = (h, c) where h, c ∈ (n_layers, batch, hidden_size)
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_size: int = 64, n_layers: int = 1):
        super().__init__()
        self.obs_dim     = obs_dim
        self.action_dim  = action_dim
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        # Observation encoder (linear projection before LSTM)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )

        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,   # (batch, seq, feature)
        )

        # Actor and critic heads read from LSTM hidden output
        self.actor_head  = nn.Linear(hidden_size, action_dim)
        self.critic_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Zero-initialize hidden state. Called at episode start."""
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def forward(self, obs, hidden_state):
        """
        Single-step forward pass (one environment step).

        Args:
            obs:          (batch, obs_dim)
            hidden_state: (h, c) each (n_layers, batch, hidden_size)

        Returns:
            logits:        (batch, action_dim)
            values:        (batch,)
            hidden_state:  updated (h, c)
        """
        # Encode obs, add seq dim for LSTM
        x = self.encoder(obs).unsqueeze(1)          # (batch, 1, hidden_size)
        out, hidden_state = self.lstm(x, hidden_state)  # out: (batch, 1, hidden_size)
        out = out.squeeze(1)                         # (batch, hidden_size)

        logits = self.actor_head(out)                # (batch, action_dim)
        values = self.critic_head(out).squeeze(-1)   # (batch,)
        return logits, values, hidden_state

    def get_action(self, obs, hidden_state):
        """
        Sample action for one step.

        Returns:
            action, log_prob, value, entropy, new_hidden_state
        """
        logits, values, hidden_state = self.forward(obs, hidden_state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), values, dist.entropy(), hidden_state

    def evaluate_actions(self, obs_seq, actions_seq, hidden_state_0, dones_seq=None,
                         tbptt_k: int = 16):
        """
        Evaluate a sequence of (obs, action) pairs collected from a rollout.
        Used during the PPO update step.

        Args:
            obs_seq:       (batch, seq_len, obs_dim)  — one episode per row
            actions_seq:   (batch, seq_len)
            hidden_state_0:(h0, c0) at episode start, (n_layers, batch, hidden_size)
            dones_seq:     (batch, seq_len) float — reset hidden at episode boundaries
            tbptt_k:       BPTT truncation length (default 16)

        Returns:
            log_probs:  (batch * seq_len,)
            entropies:  (batch * seq_len,)
            values:     (batch * seq_len,)
        """
        batch, seq_len, _ = obs_seq.shape
        all_logits, all_values = [], []
        h, c = hidden_state_0

        for t in range(seq_len):
            # Reset hidden state for envs starting a new episode.
            if dones_seq is not None and dones_seq[:, t].any():
                mask = (dones_seq[:, t] > 0.5).float().view(1, batch, 1)
                h = h * (1 - mask)
                c = c * (1 - mask)

            x = self.encoder(obs_seq[:, t, :]).unsqueeze(1)  # (batch, 1, hidden_size)
            out, (h, c) = self.lstm(x, (h, c))
            if (t + 1) % tbptt_k == 0:
                h, c = h.detach(), c.detach()
            out = out.squeeze(1)                              # (batch, hidden_size)
            all_logits.append(self.actor_head(out))
            all_values.append(self.critic_head(out).squeeze(-1))

        logits_flat = torch.stack(all_logits, dim=1).view(batch * seq_len, -1)
        values_flat = torch.stack(all_values, dim=1).view(-1)
        dist = torch.distributions.Categorical(logits=logits_flat)
        return dist.log_prob(actions_seq.reshape(-1)), dist.entropy(), values_flat
