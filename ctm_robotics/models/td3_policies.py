"""
models/td3_policies.py

Deterministic actors and Q-network critics for TD3.

Actors output actions directly (no distribution), squashed to [-max_action, max_action]
via tanh. Exploration noise is added externally by the trainer.

Three actor variants share the same interface:
    forward(obs, hidden_state) -> (action, new_hidden_state)
    init_hidden(batch_size, device) -> hidden_state   (MLP returns None)

Critic is always a non-recurrent MLP: Q(obs, action) -> scalar.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm import SynapseModel, NeuronLevelModels, SynchronizationHead


# ---------------------------------------------------------------------------
# Critic (shared by all actor types)
# ---------------------------------------------------------------------------

class TD3Critic(nn.Module):
    """Twin Q-networks for TD3. Each maps (obs, action) -> Q-value."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        in_dim = obs_dim + action_dim

        def _build():
            layers = []
            prev = in_dim
            for h in hidden_sizes:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        self.q1 = _build()
        self.q2 = _build()
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MLP Actor
# ---------------------------------------------------------------------------

class TD3MLPActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_sizes=(256, 256), max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Last layer small init for stable initial actions
        last = list(self.net)[-1]
        nn.init.uniform_(last.weight, -3e-3, 3e-3)
        nn.init.uniform_(last.bias, -3e-3, 3e-3)

    def init_hidden(self, batch_size, device):
        return None

    def forward(self, obs, hidden=None):
        return torch.tanh(self.net(obs)) * self.max_action, None

    def get_action(self, obs, hidden=None):
        action, h = self.forward(obs, hidden)
        return action, h


# ---------------------------------------------------------------------------
# LSTM Actor
# ---------------------------------------------------------------------------

class TD3LSTMActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_size: int = 128, n_layers: int = 1,
                 max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, action_dim)

        self._init()

    def _init(self):
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.head.bias, -3e-3, 3e-3)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def forward(self, obs, hidden):
        x = self.encoder(obs).unsqueeze(1)
        out, hidden = self.lstm(x, hidden)
        action = torch.tanh(self.head(out.squeeze(1))) * self.max_action
        return action, hidden

    def get_action(self, obs, hidden):
        return self.forward(obs, hidden)


# ---------------------------------------------------------------------------
# CTM Actor (for TD3)
# ---------------------------------------------------------------------------

class TD3CTMActor(nn.Module):
    """
    CTM-based deterministic actor for TD3.
    Same CTM core (synapse, NLM, sync_head) as CTMActorCritic,
    but outputs a deterministic action via tanh squashing.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 d_model: int = 64,
                 synapse_hidden: int = 128, synapse_depth: int = 3,
                 memory_length: int = 4,
                 nlm_hidden: int = 32, nlm_depth: int = 2,
                 n_synch_out: int = 32,
                 synch_window: int = 8, synch_decay: float = 0.9,
                 n_ticks: int = 5, input_hidden: int = 64,
                 max_action: float = 1.0):
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
        obs_embed_dim = input_hidden

        self.synapse = SynapseModel(d_model, obs_embed_dim,
                                    synapse_hidden, synapse_depth)
        self.nlms = NeuronLevelModels(d_model, memory_length,
                                      nlm_hidden, nlm_depth)
        self.sync_head = SynchronizationHead(d_model, n_synch_out,
                                             synch_window, synch_decay)

        self.init_post_act = nn.Parameter(torch.randn(1, d_model) * 0.1)
        self.post_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(n_synch_out, action_dim)

        nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.head.bias, -3e-3, 3e-3)

        self.last_sync_repr = None
        self.last_post_act_seq = None

    def init_hidden(self, batch_size, device):
        pre_h = torch.zeros(batch_size, self.d_model, self.M, device=device)
        post_0 = self.init_post_act.expand(batch_size, -1).detach().clone()
        return (pre_h, [post_0])

    def _detach_hidden(self, hidden):
        pre_h, post_list = hidden
        return (pre_h.detach(), [p.detach() for p in post_list])

    def _forward_core(self, obs, hidden):
        """Run CTM ticks and return sync_repr + new hidden."""
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

        new_hidden = (pre_h, new_post_list)
        return sync_repr, new_hidden

    def forward(self, obs, hidden):
        sync_repr, new_hidden = self._forward_core(obs, hidden)
        action = torch.tanh(self.head(sync_repr)) * self.max_action
        return action, new_hidden

    def get_action(self, obs, hidden):
        return self.forward(obs, hidden)

    def get_sync_saliency(self):
        return self.last_sync_repr

    def get_neural_dynamics(self):
        return self.last_post_act_seq
