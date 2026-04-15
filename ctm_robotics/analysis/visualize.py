"""
analysis/visualize.py

Visualization tools for the CTM comparison experiment.

Produces:
  1. Training curves — mean episodic return vs steps for all agents
  2. CTM neural dynamics — heatmap of neuron activity over internal ticks
  3. CTM synchronization matrix — snapshot at a given episode step
  4. Observation saliency — correlation of sync repr with each obs dimension
  5. Per-episode comparison — episode length distribution (violin plot)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional


# ─── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "ppo_mlp":  "#5B8EF7",   # blue
    "sac_mlp":  "#F7A25B",   # orange
    "ppo_lstm": "#5BF7C2",   # teal
    "ppo_ctm":  "#C97AF6",   # purple
}

LABELS = {
    "ppo_mlp":  "PPO-MLP",
    "sac_mlp":  "SAC-MLP",
    "ppo_lstm": "PPO-LSTM",
    "ppo_ctm":  "PPO-CTM",
}

# Custom dark colormap for sync matrix
_sync_colors = ["#0a0a14", "#1a1a40", "#4040a0", "#8060e0", "#c070f0", "#f0a0ff"]
SYNC_CMAP = LinearSegmentedColormap.from_list("sync", _sync_colors)
ACT_CMAP  = LinearSegmentedColormap.from_list("act",  ["#0a0a14", "#1a4060", "#60a0f0", "#f0e060"])


def set_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0d0d18",
        "axes.facecolor":    "#0d0d18",
        "axes.edgecolor":    "#333355",
        "axes.labelcolor":   "#b0aed0",
        "axes.titlecolor":   "#e0dff0",
        "text.color":        "#b0aed0",
        "xtick.color":       "#666688",
        "ytick.color":       "#666688",
        "grid.color":        "#1e1e35",
        "grid.linewidth":    0.6,
        "legend.facecolor":  "#12121f",
        "legend.edgecolor":  "#333355",
        "font.family":       "monospace",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "figure.dpi":        120,
    })


# ─── 1. Training curves ───────────────────────────────────────────────────────

def plot_training_curves(
    results: Dict[str, Dict],   # {agent_name: {"steps": [...], "returns": [...]}}
    env_name: str = "CartPole",
    save_path: str = "results/training_curves.png",
    smooth_window: int = 3,
):
    """
    Plot eval return vs environment steps for all agents.

    Args:
        results:      dict mapping agent name → {"steps": list, "returns": list}
        env_name:     string label for the environment
        save_path:    where to write the PNG
        smooth_window: rolling average window
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(f"Training curves — {env_name}", fontsize=14, color="#e0dff0", y=1.01)

    def smooth(x, w):
        if len(x) < w:
            return np.array(x)
        return np.convolve(x, np.ones(w) / w, mode="valid")

    for agent, data in results.items():
        steps   = np.array(data["steps"])
        returns = np.array(data["returns"])
        color   = COLORS.get(agent, "#aaaaaa")
        label   = LABELS.get(agent, agent)

        # Raw (transparent)
        ax.plot(steps, returns, color=color, alpha=0.2, linewidth=0.8)

        # Smoothed
        s_ret   = smooth(returns, smooth_window)
        s_steps = steps[smooth_window - 1:] if len(steps) >= smooth_window else steps
        s_steps = s_steps[:len(s_ret)]
        ax.plot(s_steps, s_ret, color=color, linewidth=2.2, label=label)

    ax.axhline(y=500, color="#ffffff", linestyle="--", linewidth=0.8,
               alpha=0.3, label="Max return (500)")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episodic return")
    ax.legend(loc="lower right", framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves_both_envs(
    full_results: Dict[str, Dict],
    po_results:   Dict[str, Dict],
    save_path:    str = "results/training_curves_both.png",
    smooth_window: int = 3,
):
    """Side-by-side: full obs vs partial obs environments."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    fig.suptitle("PPO-MLP  ·  SAC-MLP  ·  PPO-LSTM  ·  PPO-CTM",
                 fontsize=13, color="#e0dff0")

    def smooth(x, w):
        if len(x) < w:
            return np.array(x)
        return np.convolve(x, np.ones(w) / w, mode="valid")

    for ax, results, title in [
        (axes[0], full_results, "CartPole-v1  (fully observable)"),
        (axes[1], po_results,   "CartPole-PO  (velocities hidden)"),
    ]:
        for agent, data in results.items():
            if not data["steps"]:
                continue
            steps   = np.array(data["steps"])
            returns = np.array(data["returns"])
            color   = COLORS.get(agent, "#aaaaaa")
            label   = LABELS.get(agent, agent)
            ax.plot(steps, returns, color=color, alpha=0.2, linewidth=0.8)
            s_ret = smooth(returns, smooth_window)
            # Align steps to smoothed length
            if len(steps) >= smooth_window:
                s_steps = steps[smooth_window - 1: smooth_window - 1 + len(s_ret)]
            else:
                s_steps = steps[:len(s_ret)]
            if len(s_steps) == len(s_ret):
                ax.plot(s_steps, s_ret, color=color, linewidth=2.2, label=label)

        ax.axhline(y=500, color="#ffffff", linestyle="--",
                   linewidth=0.8, alpha=0.3)
        ax.set_title(title, pad=10)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Mean episodic return")
        ax.legend(loc="lower right", framealpha=0.85)
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 2. CTM neural dynamics ───────────────────────────────────────────────────

def plot_neural_dynamics(
    post_act_seq,           # list of arrays, each (D,)  — T internal ticks
    obs: Optional[np.ndarray] = None,
    episode_step: int = 0,
    save_path: str = "results/ctm_neural_dynamics.png",
):
    """
    Heatmap of neuron post-activations over internal ticks.
    Each row = one neuron, each column = one tick.
    Reveals which neurons are active, oscillating, or synchronized.
    """
    set_style()
    T = len(post_act_seq)
    D = post_act_seq[0].shape[-1]

    # Stack: (D, T)
    matrix = np.stack([a.squeeze() if a.ndim > 1 else a
                       for a in post_act_seq], axis=-1)   # (D, T)
    if matrix.ndim == 3:
        matrix = matrix[0]  # take first batch item

    nrows = 1 if obs is None else 2
    fig = plt.figure(figsize=(max(8, T * 0.5), 5 if obs is None else 7))
    gs  = gridspec.GridSpec(nrows, 1, height_ratios=([4, 1] if obs is not None else [1]),
                             hspace=0.35)

    ax_dyn = fig.add_subplot(gs[0])
    im = ax_dyn.imshow(matrix, aspect="auto", cmap=ACT_CMAP,
                        vmin=-1, vmax=1, interpolation="nearest")
    ax_dyn.set_xlabel("Internal tick →")
    ax_dyn.set_ylabel("Neuron index")
    ax_dyn.set_title(f"CTM neural dynamics  (episode step {episode_step})")
    ax_dyn.set_xticks(range(T))
    ax_dyn.set_xticklabels([f"t{i+1}" for i in range(T)], fontsize=9)
    cb = plt.colorbar(im, ax=ax_dyn, fraction=0.02, pad=0.02)
    cb.set_label("Post-activation")

    if obs is not None:
        ax_obs = fig.add_subplot(gs[1])
        obs_labels = ["cart_pos", "cart_vel", "pole_θ", "pole_ω"][:len(obs)]
        ax_obs.bar(range(len(obs)), obs, color=COLORS["ppo_ctm"], alpha=0.7)
        ax_obs.set_xticks(range(len(obs)))
        ax_obs.set_xticklabels(obs_labels, fontsize=9)
        ax_obs.set_title("Current observation", fontsize=11)
        ax_obs.axhline(0, color="#444466", linewidth=0.8)
        ax_obs.set_ylabel("Value")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 3. Synchronization matrix ────────────────────────────────────────────────

def plot_sync_matrix(
    sync_matrices,   # list of arrays, each (D, D) — one per episode step
    save_path: str = "results/ctm_sync_matrix.png",
    n_cols: int = 4,
):
    """
    Grid of synchronization matrix snapshots over an episode.
    Each cell S[i,j] = how much neurons i and j co-activate.
    """
    set_style()
    n = len(sync_matrices)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.2, n_rows * 3))
    fig.suptitle("CTM synchronization matrix S^t over episode", fontsize=13,
                 color="#e0dff0")

    axes_flat = np.array(axes).flatten()
    for i, mat in enumerate(sync_matrices):
        if mat.ndim == 3:
            mat = mat[0]   # first batch item
        ax = axes_flat[i]
        im = ax.imshow(mat, cmap=SYNC_CMAP, aspect="auto",
                        interpolation="nearest")
        ax.set_title(f"step {i+1}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 4. Observation saliency ─────────────────────────────────────────────────

def plot_obs_saliency(
    sync_reprs: np.ndarray,   # (T_ep, n_synch_out) — sync repr over episode
    obs_history: np.ndarray,  # (T_ep, obs_dim)
    obs_names: Optional[List[str]] = None,
    save_path: str = "results/ctm_saliency.png",
):
    """
    Compute and plot correlation between sync representation dimensions
    and each observation variable over an episode.

    This is the core interpretability analysis:
    High |corr| between sync_repr[k] and obs[j] suggests the CTM's
    internal synchronization tracks observation variable j.

    For your RANS extension, obs_names = ["pos_x", "pos_y", "heading",
    "vel_x", "vel_y", "ang_vel", "goal_dist", "goal_bearing"]
    """
    set_style()
    obs_dim    = obs_history.shape[1]
    n_synch    = sync_reprs.shape[1]
    obs_names  = obs_names or [f"obs[{i}]" for i in range(obs_dim)]

    # Correlation matrix: (obs_dim, n_synch)
    corr = np.zeros((obs_dim, n_synch))
    for i in range(obs_dim):
        for j in range(n_synch):
            o = obs_history[:, i]
            s = sync_reprs[:, j]
            if o.std() > 1e-8 and s.std() > 1e-8:
                corr[i, j] = np.corrcoef(o, s)[0, 1]

    # Max abs correlation per obs dim (most strongly tracked sync dimension)
    max_corr = np.abs(corr).max(axis=1)   # (obs_dim,)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={"width_ratios": [1.5, 1]})
    fig.suptitle("CTM synchronization ↔ observation saliency", fontsize=13,
                 color="#e0dff0")

    # Left: full correlation heatmap (obs_dim × n_synch)
    ax = axes[0]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(obs_dim))
    ax.set_yticklabels(obs_names, fontsize=10)
    ax.set_xlabel("Sync repr dimension")
    ax.set_ylabel("Observation variable")
    ax.set_title("Correlation: sync repr ↔ obs")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)

    # Right: max correlation bar chart
    ax2 = axes[1]
    bars = ax2.barh(range(obs_dim), max_corr,
                    color=[COLORS["ppo_ctm"]] * obs_dim, alpha=0.8)
    ax2.set_yticks(range(obs_dim))
    ax2.set_yticklabels(obs_names, fontsize=10)
    ax2.set_xlabel("|max correlation| with any sync dim")
    ax2.set_title("Saliency per observation")
    ax2.set_xlim(0, 1)

    # Annotate values
    for bar, val in zip(bars, max_corr):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 5. Final performance summary ────────────────────────────────────────────

def plot_final_summary(
    results_full: Dict[str, Dict],
    results_po:   Dict[str, Dict],
    save_path: str = "results/final_summary.png",
    last_n: int = 5,
):
    """
    Bar chart comparing final performance (mean of last_n eval points)
    across agents and environments.
    """
    set_style()
    agents = [a for a in ["ppo_mlp", "sac_mlp", "ppo_lstm", "ppo_ctm"]
              if a in results_full or a in results_po]
    labels = [LABELS.get(a, a) for a in agents]

    def final_mean(data):
        r = data.get("returns", [])
        if not r:
            return 0.0
        return float(np.mean(r[-last_n:]))

    full_means = [final_mean(results_full.get(a, {"returns": []})) for a in agents]
    po_means   = [final_mean(results_po.get(a,   {"returns": []})) for a in agents]

    x     = np.arange(len(agents))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(x - width/2, full_means, width, label="CartPole-v1 (full obs)",
                color=[COLORS.get(a, "#aaa") for a in agents], alpha=0.9)
    b2 = ax.bar(x + width/2, po_means, width, label="CartPole-PO (partial obs)",
                color=[COLORS.get(a, "#aaa") for a in agents], alpha=0.45,
                edgecolor=[COLORS.get(a, "#aaa") for a in agents], linewidth=1.5)

    ax.axhline(500, color="#ffffff", linestyle="--", linewidth=0.8,
               alpha=0.3, label="Max (500)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean episodic return (last 5 evals)")
    ax.set_title(f"Final performance comparison  (last {last_n} evals)")
    ax.set_ylim(0, 540)
    ax.legend(framealpha=0.85)

    # Value labels
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 5,
                f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 5,
                f"{h:.0f}", ha="center", va="bottom", fontsize=9)

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")
