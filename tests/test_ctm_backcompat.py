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
