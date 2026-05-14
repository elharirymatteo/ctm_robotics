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
    "visible_idx":  [0, 2],
    "hidden_idx":   [1, 3],
    "hidden_names": ["cart_vel", "pole_angvel"],
}

SEEDS = [42, 123, 456]
N_STEPS = 10_000


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
    """Roll out LSTM, return (full_obs[T,4], h_t[T,hidden_size])."""
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

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    serializable = {"per_seed": {str(s): per_seed[s] for s in SEEDS}, "aggregate": agg}
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
