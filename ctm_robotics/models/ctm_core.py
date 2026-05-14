"""
models/ctm_core.py

Framework-agnostic Continuous Thought Machine core.

Encapsulates the tick loop, sync head, and hidden-state management.
No actor/critic heads, no trainer-specific code. Wrap this from any
trainer (rsl_rl, home-grown PPO, etc.) by adding heads on top of the
sync representation.

Interface:
    forward(obs, hidden_state, capture_ticks=False)
        -> (sync_repr, new_hidden_state)
        or -> (sync_repr, new_hidden_state, list_of_tick_syncs)  if capture_ticks
    init_hidden(batch_size, device) -> hidden_state

hidden_state = (pre_act_history, post_act_history_list)
    pre_act_history:      (batch, d_model, memory_length)
    post_act_history_list: list of (batch, d_model), bounded at synch_window
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .ctm import SynapseModel, NeuronLevelModels, SynchronizationHead


class CTMCore(nn.Module):
    """CTM architecture without actor/critic heads — for arbitrary trainers."""

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 128,
        synapse_hidden: int = 64,
        synapse_depth: int = 2,
        memory_length: int = 20,
        nlm_hidden: int = 4,
        nlm_depth: int = 2,
        n_synch_out: int = 16,
        synch_window: int = 8,
        synch_decay: float = 0.9,
        n_ticks: int = 20,
        input_hidden: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.M = memory_length
        self.n_ticks = n_ticks
        self.synch_window = synch_window
        self.n_synch_out = n_synch_out

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, input_hidden),
            nn.GELU(),
            nn.Linear(input_hidden, input_hidden),
        )

        self.synapse = SynapseModel(d_model, input_hidden, synapse_hidden, synapse_depth)
        self.nlms = NeuronLevelModels(d_model, memory_length, nlm_hidden, nlm_depth)
        self.sync_head = SynchronizationHead(d_model, n_synch_out, synch_window, synch_decay)

        self.init_post_act = nn.Parameter(torch.randn(1, d_model) * 0.1)
        self.post_norm = nn.LayerNorm(d_model)

        self.last_sync_repr = None
        self.last_post_act_seq = None

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state at episode start."""
        pre_h = torch.zeros(batch_size, self.d_model, self.M, device=device)
        post_0 = self.init_post_act.to(device).expand(batch_size, -1).detach().clone()
        return (pre_h, [post_0] * self.synch_window)

    def forward(self, obs, hidden_state, capture_ticks: bool = False):
        """Single env step — runs n_ticks internal iterations.

        Args:
            obs:          (batch, obs_dim)
            hidden_state: (pre_act_history, post_act_history_list)
            capture_ticks: if True, also return per-tick sync vectors

        Returns:
            sync_repr:  (batch, n_synch_out)  — final tick sync
            new_hidden: updated hidden state
            tick_syncs: (only if capture_ticks=True) list of n_ticks sync vectors
        """
        pre_h, post_list = hidden_state

        obs_embed = self.backbone(obs)
        post_act = post_list[-1]

        tick_post_acts = []
        tick_syncs = [] if capture_ticks else None
        running_post_list = list(post_list) if capture_ticks else None

        for _tick in range(self.n_ticks):
            pre_act = self.synapse(post_act, obs_embed)
            pre_h = torch.cat([pre_h[..., 1:], pre_act.unsqueeze(-1)], dim=-1)
            post_act = self.post_norm(self.nlms(pre_h))
            tick_post_acts.append(post_act)

            if capture_ticks:
                running_post_list = running_post_list + [post_act]
                if len(running_post_list) > self.synch_window:
                    running_post_list = running_post_list[-self.synch_window:]
                tick_syncs.append(self.sync_head(running_post_list))

        new_post_list = post_list + tick_post_acts
        if len(new_post_list) > self.synch_window:
            new_post_list = new_post_list[-self.synch_window:]

        sync_repr = self.sync_head(new_post_list)

        self.last_sync_repr = sync_repr.detach()
        self.last_post_act_seq = tick_post_acts

        new_hidden = (pre_h, new_post_list)

        if capture_ticks:
            return sync_repr, new_hidden, tick_syncs
        return sync_repr, new_hidden

    def get_sync_saliency(self):
        return self.last_sync_repr

    def get_neural_dynamics(self):
        return self.last_post_act_seq

    @staticmethod
    def detach_hidden(hidden_state):
        """Detach hidden state from computation graph (for use between rollout chunks)."""
        pre_h, post_list = hidden_state
        return (pre_h.detach(), [p.detach() for p in post_list])
