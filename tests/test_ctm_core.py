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
