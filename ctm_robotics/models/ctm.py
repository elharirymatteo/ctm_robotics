"""
models/ctm.py

Continuous Thought Machine Actor-Critic for PPO.
Faithful to SakanaAI/continuous-thought-machines models/ctm_rl.py.

Key architectural facts (from the DeepWiki analysis of the Sakana repo):
  - No cross-attention in the RL variant (heads=0)
  - No input/action synchronization (n_synch_action=0)
  - Synchronization computed from sliding window with diagonal mask + decay
  - Learned initial activated state trace (not zero-initialized)
  - "first-last" neuron selection for synchronization
  - History Z^t (post-activation history) PERSISTS across environment steps

Architecture flow (one environment step, T internal ticks):
  ┌── obs ──► backbone_mlp ──► obs_embedding ──────────────────────────────────┐
  │                                                                             │
  │  for tick t in 1..T:                                                        │
  │    pre_act = synapse_mlp( concat(post_act, obs_embedding) )                │
  │    pre_act_history.append(pre_act)  ← FIFO of length M                    │
  │    post_act = NLM_d( pre_act_history_d )  ← each neuron d private MLP     │
  │    post_act_history.append(post_act)                                       │
  │    S_out = sync_from_window(post_act_history, window=W, decay=γ)           │
  │                                                                             │
  │  logits = actor_head(S_out)                                                │
  │  value  = critic_head(S_out)                                               │
  └─────────────────────────────────────────────────────────────────────────────

Interface (same as LSTMActorCritic):
    forward(obs, hidden_state) → (logits, values, new_hidden_state)
    get_action(obs, hidden_state) → (action, log_prob, value, entropy, new_hidden_state)
    evaluate_actions(obs_seq, actions_seq, hidden_state_0) → (log_probs, entropies, values)
    init_hidden(batch_size, device) → hidden_state

hidden_state for CTM = (pre_act_history, post_act_history)
    pre_act_history:  (batch, D, M)     — last M pre-activations per neuron
    post_act_history: (batch, D, T_ep)  — all post-activations since ep start
                      (grows until reset; bounded by synch_window in practice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class SynapseModel(nn.Module):
    """
    U-NET-style MLP acting as the 'synapse' between neurons.
    Takes concat(post_activations, obs_embedding) → pre_activations.
    All neurons share these weights (cross-neuron interaction).

    In the Sakana code this is fθ_syn — the recurrent MLP.
    """

    def __init__(self, d_model: int, obs_embed_dim: int,
                 synapse_hidden: int, synapse_depth: int):
        super().__init__()
        in_dim = d_model + obs_embed_dim

        layers = [nn.Linear(in_dim, synapse_hidden), nn.GELU()]
        for i in range(synapse_depth - 2):
            # U-NET skip: double hidden on the way back up (simplified)
            layers += [nn.Linear(synapse_hidden, synapse_hidden), nn.GELU()]
        layers += [nn.Linear(synapse_hidden, d_model)]

        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, post_act, obs_embed):
        """
        Args:
            post_act:   (batch, D) — current post-activations
            obs_embed:  (batch, obs_embed_dim) — encoded observation
        Returns:
            pre_act:    (batch, D) — new pre-activations
        """
        x = torch.cat([post_act, obs_embed], dim=-1)
        return self.net(x)


class NeuronLevelModels(nn.Module):
    """
    D private MLPs, one per neuron. Each processes a history of M pre-activations
    to produce the next post-activation for that neuron.

    z_d^{t+1} = g_{θ_d}(A_d^t)  where A_d^t ∈ R^M

    Implementation: we use a grouped convolution trick to run all D MLPs
    simultaneously without a Python loop — each neuron's weights form an
    independent channel group.

    Alternative (simpler but slower): loop over neurons. We implement the
    vectorized version for practical speed.
    """

    def __init__(self, d_model: int, memory_length: int,
                 nlm_hidden: int, nlm_depth: int):
        super().__init__()
        self.d_model = d_model
        self.M = memory_length

        # Layer 1: (batch, D, M) → (batch, D, nlm_hidden)
        # We implement as D separate linear transforms via a 3D weight tensor
        # Shape: (D, nlm_hidden, M) — one matrix per neuron
        self.w1 = nn.Parameter(torch.empty(d_model, nlm_hidden, memory_length))
        self.b1 = nn.Parameter(torch.zeros(d_model, nlm_hidden))

        # Intermediate layers (shared — simpler, still gives per-neuron dynamics
        # because the input history is already neuron-specific)
        mid_layers = []
        for _ in range(nlm_depth - 2):
            mid_layers += [nn.Linear(nlm_hidden, nlm_hidden), nn.GELU()]
        self.mid = nn.Sequential(*mid_layers) if nlm_depth > 2 else nn.Identity()

        # Layer out: nlm_hidden → 1 (one post-activation per neuron)
        self.w_out = nn.Parameter(torch.empty(d_model, 1, nlm_hidden))
        self.b_out = nn.Parameter(torch.zeros(d_model))

        self._init()

    def _init(self):
        # w1: (D, H, M) — each neuron has an independent M→H linear transform
        # Correct fan_in is M, not H*M as kaiming computes for 3D tensors
        bound = 1.0 / math.sqrt(self.M)
        nn.init.uniform_(self.w1, -bound, bound)
        # w_out: (D, 1, H) — each neuron: H→1
        bound = 1.0 / math.sqrt(self.w_out.shape[-1])
        nn.init.uniform_(self.w_out, -bound, bound)

    def forward(self, pre_act_history):
        """
        Args:
            pre_act_history: (batch, D, M) — last M pre-activations per neuron
        Returns:
            post_act:        (batch, D) — new post-activations
        """
        batch = pre_act_history.shape[0]
        D, H, M = self.w1.shape

        # Layer 1: batched matmul over D neurons
        # (batch, D, M) × (D, M, H) → (batch, D, H)
        x = torch.einsum('bdm,dhm->bdh', pre_act_history, self.w1) + self.b1
        x = F.gelu(x)

        # Middle layers (shared weights across neurons, applied per-neuron independently)
        if not isinstance(self.mid, nn.Identity):
            x_flat = x.view(batch * D, -1)
            x_flat = self.mid(x_flat)
            x = x_flat.view(batch, D, -1)

        # Output layer: (batch, D, H) × (D, H, 1) → (batch, D, 1)
        out = torch.einsum('bdh,doh->bdo', x, self.w_out) + self.b_out.unsqueeze(-1)
        post_act = torch.tanh(out.squeeze(-1))   # (batch, D)
        return post_act


class SynchronizationHead(nn.Module):
    """
    Computes the synchronization representation S^t from the post-activation history.

    From the Sakana RL implementation:
      - Uses a sliding window of the most recent `synch_window` activations
      - Applies exponential decay weights (older = less weight)
      - Computes diagonal dot-products (neuron-pair correlations)
      - Selects D_out neuron pairs (first-last strategy)

    S^t[i,j] ≈ sum_t(  decay^(T-t) * z_i^t * z_j^t  )

    Then flattens selected pairs → linear projection → output representation.
    """

    def __init__(self, d_model: int, n_synch_out: int,
                 synch_window: int, synch_decay: float):
        super().__init__()
        self.d_model     = d_model
        self.n_synch_out = n_synch_out
        self.synch_window = synch_window

        # Exponential decay weights for window
        decays = torch.tensor(
            [synch_decay ** (synch_window - 1 - i) for i in range(synch_window)],
            dtype=torch.float32
        )
        self.register_buffer('decays', decays)  # (synch_window,)

        # "first-last" neuron pairs: first D_out//2 and last D_out//2 neurons
        half = n_synch_out // 2
        neurons = list(range(half)) + list(range(d_model - half, d_model))
        # Build all upper-triangle pairs, then evenly sample n_synch_out
        all_r, all_c = [], []
        for i, r in enumerate(neurons):
            for j, c in enumerate(neurons):
                if j >= i:
                    all_r.append(r)
                    all_c.append(c)
        n_total = len(all_r)
        step = n_total / n_synch_out
        pairs_r = [all_r[int(i * step)] for i in range(n_synch_out)]
        pairs_c = [all_c[int(i * step)] for i in range(n_synch_out)]
        self.register_buffer('pairs_r', torch.tensor(pairs_r, dtype=torch.long))
        self.register_buffer('pairs_c', torch.tensor(pairs_c, dtype=torch.long))

        actual_pairs = len(pairs_r)
        # Project synchronization pairs → output dim
        self.proj = nn.Linear(actual_pairs, n_synch_out)
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, post_act_history):
        """
        Args:
            post_act_history: list of tensors, each (batch, D),
                              OR a stacked tensor (batch, D, T_so_far)
        Returns:
            sync_repr: (batch, n_synch_out)
        """
        if isinstance(post_act_history, list):
            if len(post_act_history) == 0:
                # No history yet — return zeros
                batch = 1
                return torch.zeros(batch, self.n_synch_out,
                                   device=self.decays.device)
            hist = torch.stack(post_act_history, dim=-1)  # (batch, D, T)
        else:
            hist = post_act_history  # (batch, D, T)

        T = hist.shape[-1]
        W = min(self.synch_window, T)

        window = hist[..., -W:]          # (batch, D, W)
        decays = self.decays[-W:]        # (W,)

        # S[i,j] = sum_t( decay_t * z_i^t * z_j^t )
        # Apply decay once (to one side) to avoid squaring the decay factor
        row_vecs = window[:, self.pairs_r, :] * decays  # (batch, n_pairs, W)
        col_vecs = window[:, self.pairs_c, :]            # (batch, n_pairs, W)
        sync_raw = (row_vecs * col_vecs).sum(-1)         # (batch, n_pairs)

        return self.proj(sync_raw)   # (batch, n_synch_out)


# ─────────────────────────────────────────────────────────────────────────────
# Main CTM Actor-Critic
# ─────────────────────────────────────────────────────────────────────────────

class CTMActorCritic(nn.Module):
    """
    Continuous Thought Machine Actor-Critic.

    Interface matches LSTMActorCritic — drop-in replacement in the PPO trainer.

    hidden_state = (pre_act_history, post_act_history_list)
        pre_act_history:      (batch, D, M)  — FIFO pre-activation window
        post_act_history_list: list of (batch, D) tensors, grows during episode

    The post_act_history list is bounded at synch_window for memory efficiency.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 d_model: int = 64,
                 synapse_hidden: int = 128, synapse_depth: int = 3,
                 memory_length: int = 4,
                 nlm_hidden: int = 32,  nlm_depth: int = 2,
                 n_synch_out: int = 32,
                 synch_window: int = 8, synch_decay: float = 0.9,
                 n_ticks: int = 5,
                 input_hidden: int = 64):
        super().__init__()
        self.obs_dim     = obs_dim
        self.action_dim  = action_dim
        self.d_model     = d_model
        self.M           = memory_length
        self.n_ticks     = n_ticks
        self.synch_window = synch_window

        # ── Observation backbone ──────────────────────────────────────────
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, input_hidden),
            nn.GELU(),
            nn.Linear(input_hidden, input_hidden),
        )
        obs_embed_dim = input_hidden

        # ── CTM core ─────────────────────────────────────────────────────
        self.synapse = SynapseModel(d_model, obs_embed_dim,
                                    synapse_hidden, synapse_depth)
        self.nlms    = NeuronLevelModels(d_model, memory_length,
                                          nlm_hidden, nlm_depth)
        self.sync_head = SynchronizationHead(d_model, n_synch_out,
                                              synch_window, synch_decay)

        # ── Learned initial activated state (as in Sakana repo) ──────────
        self.init_post_act = nn.Parameter(torch.randn(1, d_model) * 0.1)

        # ── Normalize post-activations for stable sync products ────────────
        self.post_norm = nn.LayerNorm(d_model)

        # ── Actor / Critic heads read from synchronization ────────────────
        # 2-layer MLPs matching Sakana's RL implementation (std=1 throughout)
        def _make_head(out_dim):
            m = nn.Sequential(
                nn.Linear(n_synch_out, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, out_dim),
            )
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)
            return m

        self.actor_head  = _make_head(action_dim)
        self.critic_head = _make_head(1)

        # ── Saliency storage (filled during forward, readable externally) ─
        self.last_sync_repr    = None   # (batch, n_synch_out)
        self.last_post_act_seq = None   # list of (batch, D) per tick

    # ── Hidden state management ──────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device):
        """
        Initialize hidden state at episode start.
        pre_act_history: (batch, D, M) zeros
        post_act_history: [initial_post_act]  — (batch, D)
        """
        pre_h = torch.zeros(batch_size, self.d_model, self.M, device=device)
        # Initialize post_list with synch_window copies so episode resets in ppo_loss_chunked
        # (which also produce synch_window copies) are consistent with collection.
        post_0 = self.init_post_act.to(device).expand(batch_size, -1).detach().clone()
        return (pre_h, [post_0] * self.synch_window)

    def _detach_hidden(self, hidden_state):
        """Detach hidden state from computation graph (between rollout chunks)."""
        pre_h, post_list = hidden_state
        return (pre_h.detach(),
                [p.detach() for p in post_list])

    # ── Core forward ─────────────────────────────────────────────────────────

    def forward(self, obs, hidden_state):
        """
        Single environment step — runs T internal ticks.

        Args:
            obs:          (batch, obs_dim)
            hidden_state: (pre_act_history, post_act_history_list)

        Returns:
            logits:        (batch, action_dim)
            values:        (batch,)
            new_hidden:    updated hidden state
        """
        pre_h, post_list = hidden_state
        batch = obs.shape[0]
        device = obs.device

        # Encode observation (shared across all ticks)
        obs_embed = self.backbone(obs)   # (batch, obs_embed_dim)

        # Current post-activation (latest from history)
        post_act = post_list[-1]        # (batch, D)

        # Run T internal ticks
        tick_post_acts = []
        for tick in range(self.n_ticks):
            # Synapse: combine current post-act + obs → pre-act
            pre_act = self.synapse(post_act, obs_embed)   # (batch, D)

            # Update pre-activation history (FIFO of length M)
            pre_h = torch.cat([pre_h[..., 1:],           # drop oldest
                                pre_act.unsqueeze(-1)],   # add newest
                               dim=-1)                    # (batch, D, M)

            # NLMs: each neuron processes its private history
            post_act = self.post_norm(self.nlms(pre_h))   # (batch, D)
            tick_post_acts.append(post_act)

        # Update post-activation history (append all new ticks)
        new_post_list = post_list + tick_post_acts
        # Trim to synch_window to keep memory bounded
        if len(new_post_list) > self.synch_window:
            new_post_list = new_post_list[-self.synch_window:]

        # Compute synchronization from history
        sync_repr = self.sync_head(new_post_list)   # (batch, n_synch_out)

        # Store for interpretability access
        self.last_sync_repr    = sync_repr.detach()
        self.last_post_act_seq = tick_post_acts

        # Actor / Critic heads
        logits = self.actor_head(sync_repr)            # (batch, action_dim)
        values = self.critic_head(sync_repr).squeeze(-1)  # (batch,)

        new_hidden = (pre_h, new_post_list)
        return logits, values, new_hidden

    # ── Action sampling ───────────────────────────────────────────────────────

    def get_action(self, obs, hidden_state):
        """Sample action for one step."""
        logits, values, hidden_state = self.forward(obs, hidden_state)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), values, dist.entropy(), hidden_state

    # ── PPO evaluation (sequence mode) ───────────────────────────────────────

    def evaluate_actions(self, obs_seq, actions_seq, hidden_state_0,
                         dones_seq=None):
        """
        Evaluate a batch of sequences for the PPO update.

        Args:
            obs_seq:       (batch, seq_len, obs_dim)
            actions_seq:   (batch, seq_len) long
            hidden_state_0:(pre_h0, post_list_0) at episode start
            dones_seq:     (batch, seq_len) bool (unused here, episodes aligned)

        Returns:
            log_probs:  (batch * seq_len,)
            entropies:  (batch * seq_len,)
            values:     (batch * seq_len,)
        """
        batch, seq_len, _ = obs_seq.shape
        all_logits, all_values = [], []

        hidden = hidden_state_0
        for t in range(seq_len):
            # Reset hidden state at episode boundaries
            # Detach fresh values to avoid growing autograd graph at every boundary
            # (init_post_act still gets gradient from the initial h0 at sequence start)
            if dones_seq is not None:
                mask = dones_seq[:, t]
                if mask.any():
                    pre_h, post_list = hidden
                    fresh_pre = torch.zeros_like(pre_h)
                    fresh_p0 = self.init_post_act.expand(batch, -1).detach()
                    m2 = (mask > 0.5).unsqueeze(-1)              # (batch, 1)
                    m3 = m2.unsqueeze(-1)                        # (batch, 1, 1)
                    pre_h = torch.where(m3, fresh_pre, pre_h)
                    post_list = [torch.where(m2, fresh_p0, p) for p in post_list]
                    hidden = (pre_h, post_list)

            obs_t = obs_seq[:, t, :]              # (batch, obs_dim)
            logits, vals, hidden = self.forward(obs_t, hidden)
            # Truncate BPTT between env steps: gradient flows through
            # the 5 internal ticks within this step but not across steps.
            # CTM retains recurrent memory via the hidden state contents.
            hidden = self._detach_hidden(hidden)
            all_logits.append(logits)
            all_values.append(vals)

        all_logits = torch.stack(all_logits, dim=1)   # (batch, seq_len, action_dim)
        all_values = torch.stack(all_values, dim=1)   # (batch, seq_len)

        logits_flat = all_logits.view(batch * seq_len, -1)
        dist = Categorical(logits=logits_flat)
        log_probs = dist.log_prob(actions_seq.view(-1))
        entropies = dist.entropy()
        values    = all_values.view(-1)

        return log_probs, entropies, values

    # ── Chunked PPO loss (avoids graph accumulation) ────────────────────────

    def ppo_loss_chunked(self, obs_seq, actions_seq, hidden_state_0,
                         dones_seq, old_log_probs, advantages, returns,
                         clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
                         chunk_size=128):
        """
        Compute PPO loss with TBPTT (truncated BPTT over chunks of chunk_size).

        Within each chunk, gradient flows freely through hidden states.
        Hidden state is detached only at chunk boundaries, allowing the CTM
        to learn temporal memory spanning up to chunk_size env steps.

        Gradients accumulated on self.parameters(). Caller zero_grads before,
        optimizer.step after.
        """
        batch, seq_len, _ = obs_seq.shape
        pg_sum = vf_sum = ent_sum = kl_sum = 0.0

        hidden = hidden_state_0
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_logits, chunk_values = [], []

            for t in range(chunk_start, chunk_end):
                if dones_seq is not None:
                    mask = dones_seq[:, t]
                    if mask.any():
                        pre_h, post_list = hidden
                        fresh_pre = torch.zeros_like(pre_h)
                        fresh_p0 = self.init_post_act.to(obs_seq.device).expand(batch, -1).detach()
                        m2 = (mask > 0.5).unsqueeze(-1)
                        m3 = m2.unsqueeze(-1)
                        pre_h = torch.where(m3, fresh_pre, pre_h)
                        post_list = [torch.where(m2, fresh_p0, p) for p in post_list]
                        hidden = (pre_h, post_list)

                obs_t = obs_seq[:, t, :]
                logits, vals, hidden = self.forward(obs_t, hidden)
                # Do NOT detach within chunk — BPTT through chunk_size steps
                chunk_logits.append(logits)
                chunk_values.append(vals)

            # Detach at chunk boundary only
            hidden = self._detach_hidden(hidden)

            cl = torch.stack(chunk_logits, dim=1).view(-1, self.action_dim)
            cv = torch.stack(chunk_values, dim=1).view(-1)
            dist = Categorical(logits=cl)
            lp = dist.log_prob(actions_seq[:, chunk_start:chunk_end].reshape(-1))
            ent = dist.entropy()

            old_lp = old_log_probs[:, chunk_start:chunk_end].reshape(-1)
            adv = advantages[:, chunk_start:chunk_end].reshape(-1)
            ret = returns[:, chunk_start:chunk_end].reshape(-1)

            log_ratio = (lp - old_lp).clamp(-3, 3)
            ratio = torch.exp(log_ratio)
            approx_kl = ((ratio.detach() - 1) - log_ratio.detach()).mean().item()
            pg1 = ratio * adv
            pg2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            pg_loss = -torch.min(pg1, pg2).mean()
            vf_loss = F.mse_loss(cv, ret)
            ent_loss = -ent.mean()

            chunk_loss = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss
            (chunk_loss * (chunk_end - chunk_start) / seq_len).backward()

            pg_sum  += pg_loss.item() * (chunk_end - chunk_start)
            vf_sum  += vf_loss.item() * (chunk_end - chunk_start)
            ent_sum += ent_loss.item() * (chunk_end - chunk_start)
            kl_sum  += approx_kl     * (chunk_end - chunk_start)

        return {
            "pg_loss":   pg_sum  / seq_len,
            "vf_loss":   vf_sum  / seq_len,
            "ent_loss":  ent_sum / seq_len,
            "approx_kl": kl_sum  / seq_len,
        }

    # ── Interpretability helpers ──────────────────────────────────────────────

    def get_sync_saliency(self):
        """
        Returns the last synchronization representation.
        Shape: (batch, n_synch_out)
        Can be mapped to observation variables via correlation analysis.
        """
        return self.last_sync_repr

    def get_neural_dynamics(self):
        """
        Returns post-activation sequences from the last forward pass.
        List of T tensors, each (batch, D).
        Useful for visualizing how neurons evolve over internal ticks.
        """
        return self.last_post_act_seq
