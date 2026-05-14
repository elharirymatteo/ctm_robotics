# Library Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make this repo a clean importable library that the IsaacLab repo can depend on for: (a) the CTM architecture, (b) the linear-decodability R²_CV interpretability protocol. Then re-run the L1 (CartPole-PO) supplementary analysis with the new protocol to finalize Layer 1 results.

**Architecture:** Decompose `CTMActorCritic` into a `CTMCore` (architecture + tick loop + sync; framework-agnostic) and the existing `CTMActorCritic` (thin wrapper that adds actor/critic heads and our home-grown PPO methods). Add `ctm_robotics.analysis.decodability` as a reusable module implementing ridge-R²_CV. Add tests so the IsaacLab repo can trust the imports.

**Tech Stack:** PyTorch, NumPy, scikit-learn (for ridge regression with CV), pytest.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `ctm_robotics/models/ctm_core.py` | Create | Framework-agnostic CTM: tick loop, sync head, hidden-state management. Importable by any trainer (rsl_rl, our PPO, anything). |
| `ctm_robotics/models/ctm.py` | Modify | `CTMActorCritic` becomes a thin wrapper around `CTMCore` adding actor/critic heads + PPO-specific methods. Public API unchanged. |
| `ctm_robotics/models/__init__.py` | Modify | Export `CTMCore` alongside `CTMActorCritic`. |
| `ctm_robotics/analysis/__init__.py` | Modify | Export `decodability` submodule. |
| `ctm_robotics/analysis/decodability.py` | Create | `linear_decodability(z, hidden_vars, visible_vars=None) → dict` — ridge-R²_CV protocol per spec §5.1. |
| `tests/test_ctm_core.py` | Create | Smoke test: CTMCore initializes, forward pass produces correct shapes, hidden-state round-trip. |
| `tests/test_ctm_backcompat.py` | Create | Backwards compat: `CTMActorCritic` API still works, output shapes/values identical to pre-refactor on a fixed seed. |
| `tests/test_decodability.py` | Create | Unit tests for `linear_decodability`: synthetic data with known R², ΔR² semantics, edge cases. |
| `pyproject.toml` | Modify | Add `scikit-learn` to dependencies. |
| `paper/generate_l1_supplementary.py` | Create | Regenerate L1 CartPole-PO supplementary figure/table using new R²_CV protocol on existing checkpoints. |

---

## Task 1: Add scikit-learn dependency and verify clean install

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add scikit-learn to dependencies**

Edit `pyproject.toml`, change the `dependencies` block to:

```toml
dependencies = [
    "torch>=2.0",
    "gymnasium[box2d]>=0.29",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "tqdm>=4.65",
    "scikit-learn>=1.3",
    "scipy>=1.10",
]
```

- [ ] **Step 2: Install and verify**

```bash
source .venv/bin/activate
pip install -e .
python3 -c "from sklearn.linear_model import RidgeCV; from sklearn.model_selection import KFold; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Add scikit-learn dependency for decodability analysis"
```

---

## Task 2: Extract `CTMCore` from `CTMActorCritic`

The current `CTMActorCritic` mixes architecture (tick loop, sync) with PPO-specific concerns (actor/critic heads, PPO loss methods). Extract the architecture portion into `CTMCore` so it can be wrapped by any trainer.

**Files:**
- Create: `ctm_robotics/models/ctm_core.py`
- Modify: `ctm_robotics/models/ctm.py:258-409` (the `CTMActorCritic` class)
- Modify: `ctm_robotics/models/__init__.py`

- [ ] **Step 1: Write the failing test for `CTMCore` shape contract**

Create `tests/test_ctm_core.py`:

```python
"""Smoke and shape tests for CTMCore — framework-agnostic CTM module."""
import torch
import pytest


@pytest.fixture
def core():
    from ctm_robotics.models.ctm_core import CTMCore
    return CTMCore(
        obs_dim=8, d_model=64,
        synapse_hidden=64, synapse_depth=2,
        memory_length=20, nlm_hidden=4, nlm_depth=2,
        n_synch_out=16, synch_window=8, synch_decay=0.9,
        n_ticks=20, input_hidden=64,
    )


def test_forward_shapes(core):
    """forward() returns (sync_repr, new_hidden) with correct shapes."""
    batch = 4
    device = torch.device("cpu")
    obs = torch.randn(batch, 8)
    hidden = core.init_hidden(batch, device)

    sync_repr, new_hidden = core(obs, hidden)

    assert sync_repr.shape == (batch, 16), f"sync_repr {sync_repr.shape} != (4, 16)"
    pre_h, post_list = new_hidden
    assert pre_h.shape == (batch, 64, 20), f"pre_h {pre_h.shape} != (4, 64, 20)"
    assert isinstance(post_list, list)
    assert all(p.shape == (batch, 64) for p in post_list)
    assert len(post_list) == 8  # bounded by synch_window


def test_tick_sync_capture(core):
    """forward() with capture_ticks=True returns sync after every internal tick."""
    batch = 2
    device = torch.device("cpu")
    obs = torch.randn(batch, 8)
    hidden = core.init_hidden(batch, device)

    sync_repr, new_hidden, tick_syncs = core(obs, hidden, capture_ticks=True)

    assert isinstance(tick_syncs, list)
    assert len(tick_syncs) == 20
    assert all(t.shape == (batch, 16) for t in tick_syncs)
    # Final tick equals the returned sync_repr
    assert torch.allclose(tick_syncs[-1], sync_repr)


def test_hidden_round_trip(core):
    """Hidden state can be detached and passed back without shape changes."""
    batch = 3
    device = torch.device("cpu")
    obs = torch.randn(batch, 8)
    hidden = core.init_hidden(batch, device)

    for _ in range(5):
        sync_repr, hidden = core(obs, hidden)
        # Detach (as a trainer would between rollout chunks)
        pre_h, post_list = hidden
        hidden = (pre_h.detach(), [p.detach() for p in post_list])

    assert sync_repr.shape == (batch, 16)
```

- [ ] **Step 2: Run the failing test to verify it fails for the right reason**

```bash
pytest tests/test_ctm_core.py -v
```

Expected: `ImportError: cannot import name 'CTMCore' from 'ctm_robotics.models.ctm_core'` (module doesn't exist).

- [ ] **Step 3: Create `ctm_robotics/models/ctm_core.py`**

Extract the architecture from `CTMActorCritic`. The sub-modules (`SynapseModel`, `NeuronLevelModels`, `SynchronizationHead`) remain in `ctm.py` and are imported here.

Create `ctm_robotics/models/ctm_core.py`:

```python
"""
models/ctm_core.py

Framework-agnostic Continuous Thought Machine core.

Encapsulates the tick loop, sync head, and hidden-state management.
No actor/critic heads, no trainer-specific code. Wrap this from any
trainer (rsl_rl, home-grown PPO, etc.) by adding heads on top of the
sync representation.

Interface:
    forward(obs, hidden_state, capture_ticks=False)
        → (sync_repr, new_hidden_state)
        or → (sync_repr, new_hidden_state, list_of_tick_syncs)  if capture_ticks
    init_hidden(batch_size, device) → hidden_state

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

        # Saliency storage (filled during forward, readable externally)
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_ctm_core.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add ctm_robotics/models/ctm_core.py tests/test_ctm_core.py
git commit -m "Extract framework-agnostic CTMCore from CTMActorCritic"
```

---

## Task 3: Refactor `CTMActorCritic` to use `CTMCore` internally

`CTMActorCritic` should keep its public API (so the existing training scripts and tests don't break) but delegate the architecture to `CTMCore`. Heads and PPO-specific methods stay.

**Files:**
- Modify: `ctm_robotics/models/ctm.py` (specifically the `CTMActorCritic` class)
- Create: `tests/test_ctm_backcompat.py`

- [ ] **Step 1: Write the backwards-compat test (verifies behavior is preserved)**

Create `tests/test_ctm_backcompat.py`:

```python
"""Backwards-compatibility test: CTMActorCritic public API unchanged after refactor."""
import torch
import pytest


@pytest.fixture
def ac():
    from ctm_robotics.models import CTMActorCritic
    torch.manual_seed(0)
    return CTMActorCritic(
        obs_dim=4, action_dim=2,
        d_model=64, synapse_hidden=64, synapse_depth=2,
        memory_length=20, nlm_hidden=4, nlm_depth=2,
        n_synch_out=16, synch_window=8, synch_decay=0.9,
        n_ticks=20, input_hidden=64,
    )


def test_init_hidden_contract(ac):
    h = ac.init_hidden(batch_size=3, device=torch.device("cpu"))
    pre_h, post_list = h
    assert pre_h.shape == (3, 64, 20)
    assert len(post_list) == 8
    assert all(p.shape == (3, 64) for p in post_list)


def test_forward_shapes(ac):
    obs = torch.randn(2, 4)
    h = ac.init_hidden(2, torch.device("cpu"))
    logits, values, new_h = ac(obs, h)
    assert logits.shape == (2, 2)
    assert values.shape == (2,)
    pre_h, post_list = new_h
    assert pre_h.shape == (2, 64, 20)


def test_get_action_signature(ac):
    """get_action returns (action, log_prob, value, entropy, hidden) — 5-tuple."""
    obs = torch.randn(2, 4)
    h = ac.init_hidden(2, torch.device("cpu"))
    out = ac.get_action(obs, h)
    assert len(out) == 5
    action, log_prob, value, entropy, new_h = out
    assert action.shape == (2,)
    assert action.dtype == torch.long
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    assert entropy.shape == (2,)


def test_evaluate_actions_shapes(ac):
    """evaluate_actions returns flat (batch*seq_len,) tensors."""
    batch, seq_len = 2, 10
    obs_seq = torch.randn(batch, seq_len, 4)
    actions_seq = torch.randint(0, 2, (batch, seq_len))
    h0 = ac.init_hidden(batch, torch.device("cpu"))

    log_probs, entropies, values = ac.evaluate_actions(obs_seq, actions_seq, h0)
    assert log_probs.shape == (batch * seq_len,)
    assert entropies.shape == (batch * seq_len,)
    assert values.shape == (batch * seq_len,)


def test_get_sync_saliency(ac):
    """get_sync_saliency returns the last sync after a forward pass."""
    obs = torch.randn(2, 4)
    h = ac.init_hidden(2, torch.device("cpu"))
    _ = ac(obs, h)
    s = ac.get_sync_saliency()
    assert s is not None
    assert s.shape == (2, 16)


def test_existing_checkpoint_loads(tmp_path):
    """A checkpoint saved before refactor must load cleanly after refactor.

    Uses one of the committed CartPole checkpoints as a real-world example.
    """
    import os
    from ctm_robotics.models import CTMActorCritic

    ckpt_path = "results/cartpole_nticks20_s42/ppo_ctm_CartPole_PO_v1.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not present: {ckpt_path}")

    # Build the model with same config as training
    import ctm_robotics.config as C
    model = CTMActorCritic(
        obs_dim=4, action_dim=2,
        d_model=C.CTM.d_model, synapse_hidden=C.CTM.synapse_hidden,
        synapse_depth=C.CTM.synapse_depth, memory_length=C.CTM.memory_length,
        nlm_hidden=C.CTM.nlm_hidden, nlm_depth=C.CTM.nlm_depth,
        n_synch_out=C.CTM.n_synch_out, synch_window=C.CTM.synch_window,
        synch_decay=C.CTM.synch_decay, n_ticks=C.CTM.n_ticks,
        input_hidden=C.CTM.input_hidden,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["policy_state"])
```

- [ ] **Step 2: Run test before refactor — should pass (current code already works)**

```bash
pytest tests/test_ctm_backcompat.py -v
```

Expected: 6 passed (the last one will skip if checkpoint absent — should be present from earlier work).

This establishes a baseline. If any of these break after the refactor, we know we changed behavior.

- [ ] **Step 3: Refactor `CTMActorCritic` to delegate to `CTMCore`**

Edit `ctm_robotics/models/ctm.py`. Replace the `CTMActorCritic` class (lines 258 through end of its forward method, roughly through line 409) with a version that holds a `CTMCore` instance. The signature of `__init__` and all public methods stays unchanged.

Replace lines 258 through 409 of `ctm_robotics/models/ctm.py` with:

```python
class CTMActorCritic(nn.Module):
    """
    Continuous Thought Machine Actor-Critic.

    Wraps CTMCore with discrete actor + scalar critic heads on the sync repr.
    Interface preserved for backwards compatibility with the home-grown PPO trainer.

    hidden_state = (pre_act_history, post_act_history_list)
        See CTMCore for the contract.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 d_model: int = 64,
                 synapse_hidden: int = 128, synapse_depth: int = 3,
                 memory_length: int = 4,
                 nlm_hidden: int = 32, nlm_depth: int = 2,
                 n_synch_out: int = 32,
                 synch_window: int = 8, synch_decay: float = 0.9,
                 n_ticks: int = 5,
                 input_hidden: int = 64):
        super().__init__()
        # Import here to avoid circular dependency at module load
        from .ctm_core import CTMCore

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.M = memory_length
        self.n_ticks = n_ticks
        self.synch_window = synch_window

        self.core = CTMCore(
            obs_dim=obs_dim, d_model=d_model,
            synapse_hidden=synapse_hidden, synapse_depth=synapse_depth,
            memory_length=memory_length, nlm_hidden=nlm_hidden, nlm_depth=nlm_depth,
            n_synch_out=n_synch_out, synch_window=synch_window, synch_decay=synch_decay,
            n_ticks=n_ticks, input_hidden=input_hidden,
        )

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

        self.actor_head = _make_head(action_dim)
        self.critic_head = _make_head(1)

    # ── Expose core sub-modules for backward-compat (existing analysis scripts
    # ──  e.g. run_interp_analysis.py reach into .backbone, .synapse, .nlms,
    # ──  .post_norm, .sync_head, .init_post_act directly).
    @property
    def backbone(self):       return self.core.backbone
    @property
    def synapse(self):        return self.core.synapse
    @property
    def nlms(self):           return self.core.nlms
    @property
    def sync_head(self):      return self.core.sync_head
    @property
    def post_norm(self):      return self.core.post_norm
    @property
    def init_post_act(self):  return self.core.init_post_act

    def init_hidden(self, batch_size: int, device: torch.device):
        return self.core.init_hidden(batch_size, device)

    def _detach_hidden(self, hidden_state):
        return self.core.detach_hidden(hidden_state)

    def forward(self, obs, hidden_state):
        sync_repr, new_hidden = self.core(obs, hidden_state)
        logits = self.actor_head(sync_repr)
        values = self.critic_head(sync_repr).squeeze(-1)
        return logits, values, new_hidden

    def get_action(self, obs, hidden_state):
        logits, values, hidden_state = self.forward(obs, hidden_state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), values, dist.entropy(), hidden_state

    def get_sync_saliency(self):
        return self.core.get_sync_saliency()

    def get_neural_dynamics(self):
        return self.core.get_neural_dynamics()
```

Keep `evaluate_actions` and `ppo_loss_chunked` methods as they are — they call `self.forward()` which is already updated.

**Critical:** The checkpoint loading test will fail if state_dict key names change. By using a `core` attribute, parameters now live under `core.backbone.*`, `core.synapse.*`, etc., instead of `backbone.*`, `synapse.*`. This breaks loading of existing checkpoints.

Two options:
- **A. Re-save existing checkpoints** with a migration script. One-time cost.
- **B. Override `load_state_dict` and `state_dict` to translate keys.**

Use option B. Add these methods to `CTMActorCritic`:

```python
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Override to flatten core.* keys for backward compatibility."""
        sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Translate "core.backbone.0.weight" -> "backbone.0.weight", etc.
        out = {}
        for k, v in sd.items():
            new_k = k.replace(f"{prefix}core.", f"{prefix}", 1) if k.startswith(f"{prefix}core.") else k
            out[new_k] = v
        return out

    def load_state_dict(self, state_dict, strict=True):
        """Override to translate flat keys back into core.* namespace."""
        translated = {}
        # Names of submodules that now live under self.core
        core_submodules = ("backbone.", "synapse.", "nlms.", "sync_head.",
                            "post_norm.", "init_post_act")
        for k, v in state_dict.items():
            if any(k.startswith(s) for s in core_submodules):
                translated[f"core.{k}"] = v
            else:
                translated[k] = v
        return super().load_state_dict(translated, strict=strict)
```

- [ ] **Step 4: Run backwards-compat tests to verify nothing broke**

```bash
pytest tests/test_ctm_backcompat.py -v
```

Expected: 6 passed (including the existing-checkpoint load test).

- [ ] **Step 5: Run the full test suite as a regression check**

```bash
pytest tests/ -v
```

Expected: all tests pass (test_ctm_core + test_ctm_backcompat).

- [ ] **Step 6: Sanity-check by running interpretability analysis on an existing checkpoint**

```bash
source .venv/bin/activate
python run_interp_analysis.py \
  --results-dir results/cartpole_nticks20_s42 \
  --env CartPole-PO-v1 \
  --steps 200 \
  --output /tmp/sanity_check
```

Expected: runs without error, produces partial-r values comparable to the previously logged numbers for seed 42 (cart_vel CTM ~0.13, pole_angvel CTM ~0.14). Slight numeric differences are OK — same checkpoint, same data; only the model wrapping changed.

- [ ] **Step 7: Update `ctm_robotics/models/__init__.py`**

```python
from .ctm import CTMActorCritic
from .ctm_core import CTMCore
from .lstm_policy import LSTMActorCritic
from .mlp_policy import MLPActorCritic, SACPolicy, SACQNetwork
from .td3_policies import TD3MLPActor, TD3LSTMActor, TD3CTMActor, TD3Critic
from .continuous_ppo import ContinuousMLPActorCritic, ContinuousLSTMActorCritic, ContinuousCTMActorCritic

__all__ = [
    "CTMCore",
    "CTMActorCritic",
    "LSTMActorCritic",
    "MLPActorCritic",
    "SACPolicy",
    "SACQNetwork",
    "TD3MLPActor",
    "TD3LSTMActor",
    "TD3CTMActor",
    "TD3Critic",
]
```

- [ ] **Step 8: Commit**

```bash
git add ctm_robotics/models/ctm.py ctm_robotics/models/__init__.py tests/test_ctm_backcompat.py
git commit -m "Refactor CTMActorCritic to delegate architecture to CTMCore"
```

---

## Task 4: Implement `analysis/decodability.py` — linear-decodability R²_CV protocol

Implements spec §5.1. Reusable function: given representations and target hidden variables (with optional visible variables to control for), returns R²_CV per hidden variable, plus ΔR² (over visible-only baseline) and the fitted decoder for downstream use.

**Files:**
- Create: `ctm_robotics/analysis/decodability.py`
- Modify: `ctm_robotics/analysis/__init__.py`
- Create: `tests/test_decodability.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_decodability.py`:

```python
"""Tests for the linear-decodability R²_CV protocol."""
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)


def test_perfect_decoding():
    """If hidden_var is a linear function of z, R² should be ≈ 1.0."""
    from ctm_robotics.analysis.decodability import linear_decodability

    T = 1000
    z = np.random.randn(T, 16)
    # vy is a linear combination of z
    w_true = np.random.randn(16)
    vy = z @ w_true + 0.01 * np.random.randn(T)

    out = linear_decodability(z, {"vy": vy})
    assert out["vy"]["r2_cv"] > 0.95, f"Expected R² > 0.95, got {out['vy']['r2_cv']}"


def test_no_signal():
    """If hidden_var is independent of z, R² should be ≈ 0 (or slightly negative under CV)."""
    from ctm_robotics.analysis.decodability import linear_decodability

    T = 1000
    z = np.random.randn(T, 16)
    vy = np.random.randn(T)   # independent

    out = linear_decodability(z, {"vy": vy})
    assert out["vy"]["r2_cv"] < 0.1, f"Expected R² < 0.1, got {out['vy']['r2_cv']}"


def test_delta_r2_isolates_z_information():
    """Visible vars predict vy on their own; ΔR² measures z's marginal contribution."""
    from ctm_robotics.analysis.decodability import linear_decodability

    T = 2000
    visible = np.random.randn(T, 4)
    z_independent = np.random.randn(T, 16)
    # vy = function of visible + function of z + noise
    vy = visible.sum(axis=1) + z_independent[:, 0] * 2.0 + 0.1 * np.random.randn(T)

    out = linear_decodability(z_independent, {"vy": vy}, visible_vars=visible)
    # R²(visible only) should explain ~part of vy
    # R²(z + visible) should be higher
    # ΔR² should be positive
    assert out["vy"]["r2_cv_visible"] > 0.1
    assert out["vy"]["r2_cv"] > out["vy"]["r2_cv_visible"]
    assert out["vy"]["delta_r2"] > 0.05


def test_multiple_targets():
    """linear_decodability handles a dict of multiple target variables."""
    from ctm_robotics.analysis.decodability import linear_decodability

    T = 500
    z = np.random.randn(T, 16)
    w1 = np.random.randn(16)
    w2 = np.random.randn(16)
    targets = {
        "vx": z @ w1 + 0.01 * np.random.randn(T),
        "vy": z @ w2 + 0.01 * np.random.randn(T),
    }

    out = linear_decodability(z, targets)
    assert set(out.keys()) == {"vx", "vy"}
    assert out["vx"]["r2_cv"] > 0.95
    assert out["vy"]["r2_cv"] > 0.95


def test_short_input_raises():
    """Refuse to fit on too few samples (less than 2× n_features)."""
    from ctm_robotics.analysis.decodability import linear_decodability

    z = np.random.randn(10, 16)   # 10 < 2*16
    vy = np.random.randn(10)

    with pytest.raises(ValueError, match="too few samples"):
        linear_decodability(z, {"vy": vy})
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_decodability.py -v
```

Expected: `ModuleNotFoundError: No module named 'ctm_robotics.analysis.decodability'`.

- [ ] **Step 3: Create `ctm_robotics/analysis/decodability.py`**

```python
"""
analysis/decodability.py

Linear-decodability R²_CV protocol — the unified interpretability metric
for the CTM-spacecraft paper (spec 2026-05-12-ctm-spacecraft-ral.md §5.1).

Given an arbitrary representation z (CTM sync, LSTM h_t, or anything else)
and a set of hidden physical variables, fit ridge regression with 5-fold CV
and report R²_CV per hidden variable. Optionally controls for visible
observation dimensions and reports ΔR² (the marginal information about
the hidden variable carried by z beyond what visible obs already provide).

Used identically at all three layers (gym, IsaacLab, hardware) so numbers
are directly comparable.
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


_DEFAULT_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _r2_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, alphas=_DEFAULT_ALPHAS):
    """5-fold CV ridge regression; returns mean R² over folds and best α per fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold_r2 = []
    best_alphas = []
    for train_idx, test_idx in kf.split(X):
        model = RidgeCV(alphas=alphas, scoring="r2", cv=3)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_r2.append(r2_score(y[test_idx], pred))
        best_alphas.append(model.alpha_)
    return float(np.mean(fold_r2)), float(np.std(fold_r2)), best_alphas


def linear_decodability(
    z: np.ndarray,
    hidden_vars: dict[str, np.ndarray],
    visible_vars: np.ndarray | None = None,
    n_splits: int = 5,
    alphas: tuple = _DEFAULT_ALPHAS,
) -> dict[str, dict]:
    """Linear decodability of hidden variables from a representation.

    Args:
        z:            (T, D_z) representation matrix
        hidden_vars:  {name: (T,) array of ground-truth hidden variable}
        visible_vars: optional (T, D_vis) matrix of visible observations.
                      If provided, ΔR² is computed as the marginal information
                      in z over visible-only baseline.
        n_splits:     number of folds for cross-validation (default 5)
        alphas:       ridge regularization grid

    Returns:
        {name: {"r2_cv":              float,  -- R²_CV(z → v_k)
                "r2_cv_std":          float,  -- std across folds
                "r2_cv_visible":      float,  -- R²_CV(visible → v_k)  (or None)
                "r2_cv_visible_std":  float,  -- (or None)
                "delta_r2":           float,  -- r2_cv - r2_cv_visible (or None)
                "best_alphas":        list,   -- best α per fold
               }}
    """
    z = np.asarray(z)
    T, D = z.shape
    if T < 2 * D:
        raise ValueError(
            f"too few samples: T={T} < 2*D={2*D}. Increase rollout length."
        )

    out = {}
    for name, v_k in hidden_vars.items():
        v_k = np.asarray(v_k).reshape(-1)
        if len(v_k) != T:
            raise ValueError(f"target '{name}' length {len(v_k)} != z length {T}")

        # R²_CV from z (with visible variables concatenated, if provided)
        if visible_vars is not None:
            X_full = np.concatenate([z, visible_vars], axis=1)
        else:
            X_full = z
        r2_cv, r2_cv_std, best_alphas = _r2_cv(X_full, v_k, n_splits, alphas)

        # Optional: R²_CV using ONLY visible variables (baseline)
        if visible_vars is not None:
            r2_visible, r2_visible_std, _ = _r2_cv(visible_vars, v_k, n_splits, alphas)
            delta = r2_cv - r2_visible
        else:
            r2_visible = None
            r2_visible_std = None
            delta = None

        out[name] = {
            "r2_cv":             r2_cv,
            "r2_cv_std":         r2_cv_std,
            "r2_cv_visible":     r2_visible,
            "r2_cv_visible_std": r2_visible_std,
            "delta_r2":          delta,
            "best_alphas":       best_alphas,
        }
    return out
```

- [ ] **Step 4: Update `ctm_robotics/analysis/__init__.py`**

Check current content first:

```bash
cat ctm_robotics/analysis/__init__.py
```

Then ensure decodability is exported. If the file is empty or just imports, set its content to:

```python
from . import decodability

__all__ = ["decodability"]
```

- [ ] **Step 5: Run the decodability tests**

```bash
pytest tests/test_decodability.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add ctm_robotics/analysis/decodability.py ctm_robotics/analysis/__init__.py tests/test_decodability.py
git commit -m "Add linear-decodability R²_CV protocol (analysis/decodability)"
```

---

## Task 5: Generate L1 supplementary results with the new R²_CV protocol

Re-run the interpretability analysis on existing CartPole-PO checkpoints (3 seeds) using the new `linear_decodability` function instead of max-partial-correlation. Produce the supplementary table and figure for the paper.

**Files:**
- Create: `paper/generate_l1_supplementary.py`
- Output (gitignored): `paper/figures/l1_cartpole_decodability.png`
- Output (committed): `paper/l1_supplementary_results.json`

- [ ] **Step 1: Ensure paper directory exists**

```bash
mkdir -p paper/figures
test -d paper/figures && echo "paper/figures exists"
```

- [ ] **Step 2: Write `paper/generate_l1_supplementary.py`**

```python
"""
paper/generate_l1_supplementary.py

Regenerate L1 CartPole-PO supplementary results using the unified
linear-decodability R²_CV protocol (spec §5.1).

Reads existing checkpoints from results/cartpole_nticks20_s{42,123,456}/,
rolls out for 10k steps each, applies linear_decodability to both
CTM sync and LSTM h_t representations, with visible observations as control.

Outputs:
  paper/figures/l1_cartpole_decodability.png
  paper/l1_supplementary_results.json
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ctm_robotics.envs  # noqa — triggers PO registration
import ctm_robotics.config as C
from ctm_robotics.models import CTMActorCritic, LSTMActorCritic
from ctm_robotics.envs.cartpole_po import PartialObsCartPole
from ctm_robotics.analysis.decodability import linear_decodability


CARTPOLE_PO_META = {
    "obs_names":    ["cart_pos", "cart_vel", "pole_ang", "pole_angvel"],
    "visible_idx":  [0, 2],         # cart_pos, pole_ang
    "hidden_idx":   [1, 3],         # cart_vel, pole_angvel
    "hidden_names": ["cart_vel", "pole_angvel"],
}

SEEDS = [42, 123, 456]
N_STEPS = 10_000   # spec §5.1 specifies T = 10,000


def _collect_ctm(policy, n_steps, seed):
    """Roll out CTM, return (full_obs[T,4], sync[T,16])."""
    env = PartialObsCartPole()
    obs, _ = env.reset(seed=seed)
    device = next(policy.parameters()).device
    hidden = policy.init_hidden(1, device)

    full_obs_hist, sync_hist = [], []
    for _ in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _, hidden = policy.get_action(obs_t, hidden)
        sync = policy.get_sync_saliency()
        sync_hist.append(sync.cpu().numpy()[0])
        full_obs_hist.append(env.last_full_obs.copy())
        obs, _, term, trunc, _ = env.step(action.item())
        if term or trunc:
            obs, _ = env.reset()
            hidden = policy.init_hidden(1, device)
    env.close()
    return np.array(full_obs_hist), np.array(sync_hist)


def _collect_lstm(policy, n_steps, seed):
    """Roll out LSTM, return (full_obs[T,4], h_t[T,64])."""
    env = PartialObsCartPole()
    obs, _ = env.reset(seed=seed)
    device = next(policy.parameters()).device
    hidden = policy.init_hidden(1, device)

    full_obs_hist, h_hist = [], []
    for _ in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _, hidden = policy.get_action(obs_t, hidden)
        h, _c = hidden
        h_hist.append(h[0, 0].cpu().numpy())
        full_obs_hist.append(env.last_full_obs.copy())
        obs, _, term, trunc, _ = env.step(action.item())
        if term or trunc:
            obs, _ = env.reset()
            hidden = policy.init_hidden(1, device)
    env.close()
    return np.array(full_obs_hist), np.array(h_hist)


def _run_one_seed(seed):
    """Returns dict with per-variable decodability for both CTM and LSTM."""
    print(f"\n=== Seed {seed} ===")
    device = torch.device("cpu")

    ctm = CTMActorCritic(
        obs_dim=4, action_dim=2,
        d_model=C.CTM.d_model, synapse_hidden=C.CTM.synapse_hidden,
        synapse_depth=C.CTM.synapse_depth, memory_length=C.CTM.memory_length,
        nlm_hidden=C.CTM.nlm_hidden, nlm_depth=C.CTM.nlm_depth,
        n_synch_out=C.CTM.n_synch_out, synch_window=C.CTM.synch_window,
        synch_decay=C.CTM.synch_decay, n_ticks=C.CTM.n_ticks,
        input_hidden=C.CTM.input_hidden,
    ).to(device)
    ckpt = torch.load(
        f"results/cartpole_nticks20_s{seed}/ppo_ctm_CartPole_PO_v1.pt",
        map_location=device, weights_only=False,
    )
    ctm.load_state_dict(ckpt["policy_state"])
    ctm.eval()

    lstm = LSTMActorCritic(
        obs_dim=4, action_dim=2,
        hidden_size=C.LSTM.hidden_size, n_layers=C.LSTM.n_layers,
    ).to(device)
    ckpt = torch.load(
        f"results/cartpole_nticks20_s{seed}/ppo_lstm_CartPole_PO_v1.pt",
        map_location=device, weights_only=False,
    )
    lstm.load_state_dict(ckpt["policy_state"])
    lstm.eval()

    print(f"  Collecting {N_STEPS} steps from CTM ...")
    full_obs_ctm, sync = _collect_ctm(ctm, N_STEPS, seed)
    print(f"  Collecting {N_STEPS} steps from LSTM ...")
    full_obs_lstm, h_t = _collect_lstm(lstm, N_STEPS, seed)

    visible = full_obs_ctm[:, CARTPOLE_PO_META["visible_idx"]]
    hidden_targets = {
        name: full_obs_ctm[:, idx]
        for name, idx in zip(CARTPOLE_PO_META["hidden_names"], CARTPOLE_PO_META["hidden_idx"])
    }
    visible_lstm = full_obs_lstm[:, CARTPOLE_PO_META["visible_idx"]]
    hidden_targets_lstm = {
        name: full_obs_lstm[:, idx]
        for name, idx in zip(CARTPOLE_PO_META["hidden_names"], CARTPOLE_PO_META["hidden_idx"])
    }

    print("  Decodability — CTM sync ...")
    ctm_dec = linear_decodability(sync, hidden_targets, visible_vars=visible)
    print("  Decodability — LSTM h_t ...")
    lstm_dec = linear_decodability(h_t, hidden_targets_lstm, visible_vars=visible_lstm)

    out = {"ctm": ctm_dec, "lstm": lstm_dec}
    for var in CARTPOLE_PO_META["hidden_names"]:
        c = ctm_dec[var]
        l = lstm_dec[var]
        print(f"    {var:14s}:  CTM R²={c['r2_cv']:.3f} (Δ={c['delta_r2']:+.3f})   "
              f"LSTM R²={l['r2_cv']:.3f} (Δ={l['delta_r2']:+.3f})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", default="paper/l1_supplementary_results.json")
    p.add_argument("--out-fig",  default="paper/figures/l1_cartpole_decodability.png")
    args = p.parse_args()

    per_seed = {}
    for s in SEEDS:
        per_seed[s] = _run_one_seed(s)

    # Aggregate
    agg = {"ctm": {}, "lstm": {}}
    for var in CARTPOLE_PO_META["hidden_names"]:
        for repr_name in ("ctm", "lstm"):
            r2s = [per_seed[s][repr_name][var]["r2_cv"] for s in SEEDS]
            deltas = [per_seed[s][repr_name][var]["delta_r2"] for s in SEEDS]
            agg[repr_name][var] = {
                "r2_cv_mean":    float(np.mean(r2s)),
                "r2_cv_std":     float(np.std(r2s)),
                "delta_r2_mean": float(np.mean(deltas)),
                "delta_r2_std":  float(np.std(deltas)),
            }

    print("\n=== Aggregate (mean ± std across 3 seeds) ===")
    for var in CARTPOLE_PO_META["hidden_names"]:
        c = agg["ctm"][var]; l = agg["lstm"][var]
        print(f"  {var:14s}:  CTM R²={c['r2_cv_mean']:.3f}±{c['r2_cv_std']:.3f}   "
              f"LSTM R²={l['r2_cv_mean']:.3f}±{l['r2_cv_std']:.3f}")
        print(f"  {'':14s}    ΔR²={c['delta_r2_mean']:+.3f}±{c['delta_r2_std']:.3f}     "
              f"   ΔR²={l['delta_r2_mean']:+.3f}±{l['delta_r2_std']:.3f}")

    # Save JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    serializable = {"per_seed": {str(s): per_seed[s] for s in SEEDS}, "aggregate": agg}
    # numpy → python types
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    with open(args.out_json, "w") as f:
        json.dump(_clean(serializable), f, indent=2)
    print(f"\nSaved → {args.out_json}")

    # Figure: bar chart of CTM vs LSTM ΔR² per hidden variable, mean±std
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vars_ = CARTPOLE_PO_META["hidden_names"]
    x = np.arange(len(vars_))
    w = 0.38
    ctm_dr2 = [agg["ctm"][v]["delta_r2_mean"] for v in vars_]
    ctm_err = [agg["ctm"][v]["delta_r2_std"]  for v in vars_]
    lstm_dr2 = [agg["lstm"][v]["delta_r2_mean"] for v in vars_]
    lstm_err = [agg["lstm"][v]["delta_r2_std"]  for v in vars_]

    ax.bar(x - w/2, ctm_dr2,  w, yerr=ctm_err,  capsize=4,
           color="#228833", label="CTM sync (16-dim)", edgecolor="white")
    ax.bar(x + w/2, lstm_dr2, w, yerr=lstm_err, capsize=4,
           color="#EE6677", label="LSTM $h_t$ (64-dim)", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(vars_)
    ax.set_ylabel(r"$\Delta R^2_\mathrm{CV}$  (information about hidden var beyond visible obs)")
    ax.set_title("L1 supplementary: linear decodability on CartPole-PO-v1\n"
                  "(mean ± std, 3 seeds, T=10,000 steps each)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {args.out_fig}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script**

```bash
source .venv/bin/activate
python paper/generate_l1_supplementary.py
```

Expected: prints per-seed and aggregate ΔR² values. Saves JSON to `paper/l1_supplementary_results.json` and PNG to `paper/figures/l1_cartpole_decodability.png`. Runtime ~2-5 minutes (3 seeds × 10k steps × CPU inference).

- [ ] **Step 4: Sanity-check the aggregate numbers**

```bash
python3 -c "
import json
d = json.load(open('paper/l1_supplementary_results.json'))
agg = d['aggregate']
for var in ['cart_vel', 'pole_angvel']:
    c = agg['ctm'][var]
    l = agg['lstm'][var]
    print(f'{var}:')
    print(f'  CTM:  R²={c[\"r2_cv_mean\"]:.3f}±{c[\"r2_cv_std\"]:.3f}  ΔR²={c[\"delta_r2_mean\"]:+.3f}')
    print(f'  LSTM: R²={l[\"r2_cv_mean\"]:.3f}±{l[\"r2_cv_std\"]:.3f}  ΔR²={l[\"delta_r2_mean\"]:+.3f}')
"
```

**Expected shape:** CTM ΔR² should be positive (z carries marginal information about hidden velocities beyond what visible obs provide). LSTM ΔR² may be lower than CTM's but should also be positive in principle. If CTM ΔR² is negative or near zero, something is wrong — check that the rollout produces enough variation in hidden vars (try a fresh seed) or that the model is loaded correctly.

If CTM consistently shows higher ΔR² than LSTM on both variables, the L1 supplementary claim holds. Update `paper/paper.md` (when written) to cite these numbers.

- [ ] **Step 5: Commit the L1 supplementary results**

```bash
git add paper/generate_l1_supplementary.py paper/l1_supplementary_results.json
git commit -m "Add L1 CartPole-PO supplementary: linear decodability with new R²_CV protocol"
```

(The PNG is gitignored per `.gitignore` rules from the earlier paper plan; do not stage it.)

---

## Task 6: Verify the library import contract works end-to-end

Final check: the other repo's `ActorCriticCTM` will import like this. Simulate that integration pattern to confirm nothing's missing.

**Files:** none modified (verification only)

- [ ] **Step 1: Run an external-style import check**

```bash
source .venv/bin/activate
python3 -c "
# Simulating: from another repo, what does our package look like?
from ctm_robotics.models import CTMCore
from ctm_robotics.analysis.decodability import linear_decodability
import torch, numpy as np

# 1. Build a CTMCore as the IsaacLab wrapper would
core = CTMCore(obs_dim=8, d_model=128, n_ticks=20, n_synch_out=16)
batch = 4
h = core.init_hidden(batch, torch.device('cpu'))
obs = torch.randn(batch, 8)
sync, h2 = core(obs, h)
print(f'sync shape: {sync.shape}')   # (4, 16)
assert sync.shape == (4, 16)

# 2. Use decodability as the analysis pipeline would
T = 1000
z = np.random.randn(T, 16)
v_target = z @ np.random.randn(16)
out = linear_decodability(z, {'vy': v_target})
print(f'R²_CV(vy from z): {out[\"vy\"][\"r2_cv\"]:.3f}')
assert out['vy']['r2_cv'] > 0.9, 'decodability protocol broken'

print('Library import contract OK.')
"
```

Expected output:
```
sync shape: torch.Size([4, 16])
R²_CV(vy from z): 0.999
Library import contract OK.
```

- [ ] **Step 2: Confirm all tests still pass**

```bash
pytest tests/ -v
```

Expected: all tests pass (test_ctm_core, test_ctm_backcompat, test_decodability).

- [ ] **Step 3: Tag the library state**

```bash
git tag -a v0.2.0-library -m "Library extraction complete: CTMCore + decodability + L1 supplementary"
git log --oneline -10
```

This tag is a stable point the IsaacLab repo can pin to (`pip install -e ../ctm_robotics` or by commit hash).

---

## Self-Review

**Spec coverage** (cross-reference against `2026-05-12-ctm-spacecraft-ral.md`):
- §3.1 L1 CartPole-PO supplementary → Task 5 generates final results ✓
- §4.4 CTM architecture reusable → Task 2-3 extract `CTMCore` ✓
- §5.1 Linear-decodability R²_CV protocol → Task 4 implements it ✓
- §5 unified across layers → Task 4 protocol is intentionally framework-agnostic, callable on rollout data from any layer ✓
- §5.2 tick-progressive R²_CV → `CTMCore.forward(capture_ticks=True)` exposes per-tick syncs (Task 2). Application to L2 is deferred to Sprint 2.

**Placeholder scan:** No TBDs, no "implement later". Every step has either exact code, exact command, or exact expected output.

**Type consistency:** `linear_decodability` signature consistent across test and usage. `CTMCore.forward()` return type (with/without `capture_ticks`) consistent between tests and the generator script. Hidden state contract `(pre_h, post_list)` consistent across `CTMCore`, `CTMActorCritic`, and existing analysis scripts.

**Ambiguity check:**
- The state-dict key translation strategy in Task 3 is explicit. Existing checkpoints WILL still load.
- Decodability returns `delta_r2 = None` when `visible_vars` is None. The test in `test_delta_r2_isolates_z_information` only checks `delta_r2` when visible is provided. No ambiguity.

**Scope check:** This plan is self-contained — finishes Layer 1 of the paper, leaves the library in a state the IsaacLab repo can depend on. Three focused tasks (refactor + protocol + L1 results), three days of work. Does not overlap with Sprint 1 (CTM → rsl_rl port) which is a separate plan.

**One thing I did *not* include but considered:** publishing the package to PyPI. The IsaacLab repo can `pip install -e ../ctm_robotics` from the local checkout. PyPI publication is unnecessary while the project is single-author and pre-paper.
