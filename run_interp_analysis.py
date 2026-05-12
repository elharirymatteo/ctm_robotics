"""
run_interp_analysis.py

Exp 2: CTM sync vs LSTM h_t partial correlation comparison
Exp 3: Tick-progressive inference — partial-r at each internal tick

Usage:
    python run_interp_analysis.py --results-dir results/lunar_nticks20_s456 \
        --env LunarLander-PO-v3 --steps 1000 --output results/interp/
"""

import argparse, os, json
import numpy as np
import torch
from scipy import stats

import ctm_robotics.envs  # noqa — triggers gym registration
import ctm_robotics.config as C
from ctm_robotics.models import CTMActorCritic, LSTMActorCritic
from ctm_robotics.envs.lunarlander_po import PartialObsLunarLander
from ctm_robotics.envs.cartpole_po import PartialObsCartPole

# ─────────────────────────────────────────────────────────────────────────────
# Env / obs metadata
# ─────────────────────────────────────────────────────────────────────────────

ENV_META = {
    "LunarLander-PO-v3": {
        "make": PartialObsLunarLander,
        "obs_dim": 8, "action_dim": 4,
        "obs_names": ["x","y","vx(hid)","vy(hid)","angle","ang_vel(hid)","left_leg","right_leg"],
        "visible_idx": [0, 1, 4, 6, 7],
        "hidden_idx":  [2, 3, 5],
    },
    "CartPole-PO-v1": {
        "make": PartialObsCartPole,
        "obs_dim": 4, "action_dim": 2,
        "obs_names": ["cart_pos","cart_vel(hid)","pole_ang","pole_angvel(hid)"],
        "visible_idx": [0, 2],
        "hidden_idx":  [1, 3],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Partial correlation helper
# ─────────────────────────────────────────────────────────────────────────────

def partial_corr_max(repr_matrix, target_vec, control_matrix):
    """
    Max partial correlation between any column of repr_matrix and target_vec,
    after controlling for control_matrix.
    Returns (max_partial_r, best_dim_idx).
    """
    A = np.c_[control_matrix, np.ones(len(control_matrix))]
    beta_r, *_ = np.linalg.lstsq(A, repr_matrix, rcond=None)
    beta_v, *_ = np.linalg.lstsq(A, target_vec,  rcond=None)
    repr_res = repr_matrix - A @ beta_r   # (T, D)
    v_res    = target_vec  - A @ beta_v   # (T,)
    corrs = [abs(stats.pearsonr(repr_res[:, j], v_res)[0])
             for j in range(repr_matrix.shape[1])]
    best = int(np.argmax(corrs))
    return corrs[best], best


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_ctm(policy, env_meta, n_steps, seed=0):
    """Roll out CTM, capturing sync at every internal tick + final."""
    env = env_meta["make"]()
    obs, _ = env.reset(seed=seed)
    device = next(policy.parameters()).device
    hidden = policy.init_hidden(1, device)

    full_obs_hist, sync_hist, tick_sync_hist = [], [], []
    # tick_sync_hist[t][tick] = sync repr at tick `tick` on step `t`

    for _ in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            # ── Per-tick sync computation ──────────────────────────────────
            pre_h, post_list = hidden
            obs_embed = policy.backbone(obs_t)
            post_act  = post_list[-1]

            tick_syncs = []
            running_post_list = list(post_list)
            for tick in range(policy.n_ticks):
                pre_act = policy.synapse(post_act, obs_embed)
                pre_h = torch.cat([pre_h[..., 1:], pre_act.unsqueeze(-1)], dim=-1)
                post_act = policy.post_norm(policy.nlms(pre_h))
                running_post_list = running_post_list + [post_act]
                if len(running_post_list) > policy.synch_window:
                    running_post_list = running_post_list[-policy.synch_window:]
                tick_sync = policy.sync_head(running_post_list)
                tick_syncs.append(tick_sync.cpu().numpy()[0])

            # Final sync (same as policy.last_sync_repr after get_action)
            action, _, _, _, hidden = policy.get_action(obs_t, hidden)

        full_obs_hist.append(env.last_full_obs.copy())
        sync_hist.append(tick_syncs[-1])           # final tick
        tick_sync_hist.append(tick_syncs)           # all ticks

        obs, _, term, trunc, _ = env.step(action.item())
        if term or trunc:
            obs, _ = env.reset()
            hidden = policy.init_hidden(1, device)

    env.close()
    return (np.array(full_obs_hist),
            np.array(sync_hist),
            np.array(tick_sync_hist))   # (T, n_ticks, n_synch_out)


def collect_lstm(policy, env_meta, n_steps, seed=0):
    """Roll out LSTM, capturing h_t at each step."""
    env = env_meta["make"]()
    obs, _ = env.reset(seed=seed)
    device = next(policy.parameters()).device
    hidden = policy.init_hidden(1, device)

    full_obs_hist, ht_hist = [], []

    for _ in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _, hidden = policy.get_action(obs_t, hidden)
        h, _ = hidden
        ht_hist.append(h[0, 0].cpu().numpy())   # (hidden_size,)
        full_obs_hist.append(env.last_full_obs.copy())
        obs, _, term, trunc, _ = env.step(action.item())
        if term or trunc:
            obs, _ = env.reset()
            hidden = policy.init_hidden(1, device)

    env.close()
    return np.array(full_obs_hist), np.array(ht_hist)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/lunar_nticks20_s456")
    p.add_argument("--env", default="LunarLander-PO-v3")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    output_dir = args.output or os.path.join(args.results_dir, "interp")
    os.makedirs(output_dir, exist_ok=True)

    env_meta = ENV_META[args.env]
    device   = torch.device("cpu")
    env_tag  = args.env.replace("-", "_")

    # ── Load CTM ──────────────────────────────────────────────────────────────
    ctm = CTMActorCritic(env_meta["obs_dim"], env_meta["action_dim"],
        d_model=C.CTM.d_model, synapse_hidden=C.CTM.synapse_hidden,
        synapse_depth=C.CTM.synapse_depth, memory_length=C.CTM.memory_length,
        nlm_hidden=C.CTM.nlm_hidden, nlm_depth=C.CTM.nlm_depth,
        n_synch_out=C.CTM.n_synch_out, synch_window=C.CTM.synch_window,
        synch_decay=C.CTM.synch_decay, n_ticks=C.CTM.n_ticks,
        input_hidden=C.CTM.input_hidden).to(device)
    ckpt_ctm = torch.load(
        f"{args.results_dir}/ppo_ctm_{env_tag}.pt",
        map_location=device, weights_only=False)
    ctm.load_state_dict(ckpt_ctm["policy_state"])
    ctm.eval()

    # ── Load LSTM ─────────────────────────────────────────────────────────────
    lstm = LSTMActorCritic(env_meta["obs_dim"], env_meta["action_dim"],
        hidden_size=C.LSTM.hidden_size, n_layers=C.LSTM.n_layers).to(device)
    ckpt_lstm = torch.load(
        f"{args.results_dir}/ppo_lstm_{env_tag}.pt",
        map_location=device, weights_only=False)
    lstm.load_state_dict(ckpt_lstm["policy_state"])
    lstm.eval()

    print(f"Collecting {args.steps} steps from CTM and LSTM on {args.env}...")

    full_ctm, sync_final, tick_syncs = collect_ctm(ctm, env_meta, args.steps, args.seed)
    full_lstm, ht = collect_lstm(lstm, env_meta, args.steps, args.seed)

    obs_names  = env_meta["obs_names"]
    vis_idx    = env_meta["visible_idx"]
    hidden_idx = env_meta["hidden_idx"]
    V_vis      = full_ctm[:, vis_idx]

    # ─────────────────────────────────────────────────────────────────────────
    # Exp 2: CTM sync vs LSTM h_t partial correlations
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Exp 2 — Partial correlation: CTM sync vs LSTM h_t")
    print(f"  env={args.env}  steps={args.steps}")
    print(f"{'='*60}")
    print(f"  {'Variable':22s}  {'CTM partial-r':>14}  {'LSTM partial-r':>14}  {'hidden?':>8}")
    print(f"  {'-'*65}")

    exp2_results = {}
    for k, name in enumerate(obs_names):
        v_k = full_ctm[:, k]
        ctm_pr,  ctm_dim  = partial_corr_max(sync_final, v_k, V_vis)
        lstm_pr, lstm_dim = partial_corr_max(ht,         v_k, V_vis)
        tag = "<-- HIDDEN" if k in hidden_idx else ""
        print(f"  {name:22s}  {ctm_pr:14.3f}  {lstm_pr:14.3f}  {tag}")
        exp2_results[name] = {"ctm": round(ctm_pr, 4), "lstm": round(lstm_pr, 4),
                               "hidden": k in hidden_idx,
                               "ctm_dim": ctm_dim, "lstm_dim": lstm_dim}

    # ─────────────────────────────────────────────────────────────────────────
    # Exp 3: Tick-progressive inference
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Exp 3 — Tick-progressive partial-r (n_ticks={ctm.n_ticks})")
    print(f"{'='*60}")

    n_ticks = ctm.n_ticks
    exp3_results = {name: [] for name in obs_names}

    for tick_i in range(n_ticks):
        sync_at_tick = tick_syncs[:, tick_i, :]   # (T, n_synch_out)
        for k, name in enumerate(obs_names):
            v_k = full_ctm[:, k]
            pr, _ = partial_corr_max(sync_at_tick, v_k, V_vis)
            exp3_results[name].append(round(pr, 4))

    print(f"\n  {'Variable':22s}  tick_1  tick_5  tick_10  tick_15  tick_20  trend")
    print(f"  {'-'*72}")
    for k, name in enumerate(obs_names):
        vals = exp3_results[name]
        t1, t5, t10, t15, t20 = vals[0], vals[4], vals[9], vals[14], vals[19]
        # Simple trend: is last half higher than first half on average?
        trend = "↑" if np.mean(vals[n_ticks//2:]) > np.mean(vals[:n_ticks//2]) else "→"
        tag   = " (hid)" if k in hidden_idx else ""
        print(f"  {name+tag:28s}  {t1:.3f}  {t5:.3f}   {t10:.3f}   {t15:.3f}   {t20:.3f}   {trend}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "env": args.env, "steps": args.steps, "n_ticks": n_ticks,
        "exp2": exp2_results,
        "exp3": {name: vals for name, vals in exp3_results.items()},
    }
    out_path = os.path.join(output_dir, "interp_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        # Exp 2 bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        names_short = [n.replace("(hid)", "*").replace("ang_vel", "ωvel") for n in obs_names]
        x = np.arange(len(obs_names))
        w = 0.35
        ctm_vals  = [exp2_results[n]["ctm"]  for n in obs_names]
        lstm_vals = [exp2_results[n]["lstm"] for n in obs_names]
        colors_ctm  = ["#e05252" if k in hidden_idx else "#7fbfff"
                       for k in range(len(obs_names))]
        colors_lstm = ["#b03030" if k in hidden_idx else "#4a8abf"
                       for k in range(len(obs_names))]
        ax.bar(x - w/2, ctm_vals,  w, label="CTM sync",  color=colors_ctm)
        ax.bar(x + w/2, lstm_vals, w, label="LSTM $h_t$", color=colors_lstm)
        ax.set_xticks(x); ax.set_xticklabels(names_short, rotation=20, ha="right")
        ax.set_ylabel("Partial correlation |r|")
        ax.set_title(f"Exp 2: Hidden-state tracking — CTM sync vs LSTM $h_t$\n"
                     f"{args.env}  (red bars = hidden variables, * = hidden)")
        ax.legend(); ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp2_partial_corr.png"), dpi=150)
        plt.close()

        # Exp 3 tick progression (hidden vars only)
        fig, ax = plt.subplots(figsize=(10, 5))
        tick_x = np.arange(1, n_ticks + 1)
        colors = ["#e05252", "#e09852", "#52b0e0"]
        for ci, k in enumerate(hidden_idx):
            name = obs_names[k]
            ax.plot(tick_x, exp3_results[name], marker="o", markersize=4,
                    label=name, color=colors[ci])
        ax.set_xlabel("Internal tick"); ax.set_ylabel("Partial correlation |r|")
        ax.set_title(f"Exp 3: Tick-progressive hidden-state inference\n"
                     f"{args.env}  (partial-r with hidden variables per tick)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp3_tick_progression.png"), dpi=150)
        plt.close()
        print(f"  Plots saved to {output_dir}/")
    except Exception as e:
        print(f"  Plot error: {e}")


if __name__ == "__main__":
    main()
