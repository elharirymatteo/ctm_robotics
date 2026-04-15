"""
plot_results.py

Re-generate all figures from saved result files.
Run this after run_comparison.py has finished (or for quick re-styling).

Usage:
    python plot_results.py
    python plot_results.py --results-dir my_results/
    python plot_results.py --ctm-analysis       # also re-run CTM interpretability plots
"""

import argparse
import json
import os

import numpy as np
import torch
import gymnasium as gym

import ctm_robotics.envs  # noqa — triggers CartPole-PO-v1 registration
import ctm_robotics.config as C
from ctm_robotics.envs.cartpole_po import PartialObsCartPole
from ctm_robotics.models import CTMActorCritic
from ctm_robotics.analysis.visualize import (
    plot_training_curves_both_envs,
    plot_neural_dynamics,
    plot_sync_matrix,
    plot_obs_saliency,
    plot_final_summary,
)


AGENTS = ["ppo_mlp", "sac_mlp", "ppo_lstm", "ppo_ctm"]


def load(agent, env_id, results_dir):
    path = os.path.join(results_dir, f"{agent}_{env_id.replace('-','_')}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"  [warn] missing: {path}")
    return {"steps": [], "returns": []}


def build_ctm_policy(obs_dim, action_dim):
    return CTMActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=C.CTM.d_model,
        synapse_hidden=C.CTM.synapse_hidden,
        synapse_depth=C.CTM.synapse_depth,
        memory_length=C.CTM.memory_length,
        nlm_hidden=C.CTM.nlm_hidden,
        nlm_depth=C.CTM.nlm_depth,
        n_synch_out=C.CTM.n_synch_out,
        synch_window=C.CTM.synch_window,
        synch_decay=C.CTM.synch_decay,
        n_ticks=C.CTM.n_ticks,
        input_hidden=C.CTM.input_hidden,
    )


def run_ctm_analysis(env_id, results_dir, n_steps_collect=200):
    """Load a trained CTM checkpoint and produce interpretability plots."""
    print("\n-- CTM interpretability analysis --")

    if env_id == C.ENV_PO:
        env = PartialObsCartPole()
        obs_names = ["cart_pos", "cart_vel(hidden)", "pole_ang", "pole_angvel(hidden)"]
    else:
        env = gym.make("CartPole-v1")
        obs_names = ["cart_pos", "cart_vel", "pole_ang", "pole_angvel"]

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device(C.TRAIN.device)

    policy = build_ctm_policy(obs_dim, action_dim)
    policy.to(device)

    ckpt_path = os.path.join(results_dir, f"ppo_ctm_{env_id.replace('-', '_')}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy_state"])
        print(f"  Loaded weights from {ckpt_path}")
    else:
        print("  No checkpoint found — using random weights (architecture demo only)")

    policy.eval()
    obs, _ = env.reset(seed=0)
    hidden = policy.init_hidden(1, device)

    obs_history = []
    sync_repr_history = []
    sync_matrices = []
    dynamics_snapshots = []

    for step in range(n_steps_collect):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _, hidden = policy.get_action(obs_t, hidden)

        obs_history.append(obs.copy())

        sr = policy.get_sync_saliency()
        if sr is not None:
            sync_repr_history.append(sr.cpu().numpy()[0])

        nd = policy.get_neural_dynamics()
        if nd is not None and len(nd) > 0:
            dynamics_snapshots.append([a[0].cpu().numpy() for a in nd])

        if sr is not None:
            s = sr.cpu().numpy()[0]
            sync_matrices.append(np.outer(s, s))

        obs, reward, term, trunc, _ = env.step(action.item())
        if term or trunc:
            obs, _ = env.reset()
            hidden = policy.init_hidden(1, device)

    env.close()

    obs_arr = np.array(obs_history)
    sync_arr = np.array(sync_repr_history)

    if dynamics_snapshots:
        plot_neural_dynamics(
            post_act_seq=dynamics_snapshots[min(5, len(dynamics_snapshots) - 1)],
            obs=obs_arr[min(5, len(obs_arr) - 1)],
            episode_step=min(5, len(dynamics_snapshots) - 1),
            save_path=os.path.join(results_dir, "ctm_neural_dynamics.png"),
        )

    if sync_matrices:
        snapshot_steps = [0, 4, 9, 19, 39]
        snapshots = [sync_matrices[i] for i in snapshot_steps if i < len(sync_matrices)]
        plot_sync_matrix(
            sync_matrices=snapshots,
            save_path=os.path.join(results_dir, "ctm_sync_matrix.png"),
        )

    if len(sync_arr) > 10:
        plot_obs_saliency(
            sync_reprs=sync_arr,
            obs_history=obs_arr,
            obs_names=obs_names,
            save_path=os.path.join(results_dir, "ctm_saliency.png"),
        )


def main():
    p = argparse.ArgumentParser(description="Re-plot results from saved JSON files")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--ctm-analysis", action="store_true",
                   help="Also re-run CTM interpretability plots from checkpoint")
    p.add_argument("--ctm-env", choices=["full", "po"], default="po",
                   help="Which env to use for CTM analysis (default: po)")
    p.add_argument("--ctm-steps", type=int, default=200,
                   help="Number of steps to collect for CTM analysis")
    args = p.parse_args()
    rd = args.results_dir

    full_res = {a: load(a, C.ENV_FULL, rd) for a in AGENTS}
    po_res = {a: load(a, C.ENV_PO, rd) for a in AGENTS}

    plot_training_curves_both_envs(
        full_results=full_res,
        po_results=po_res,
        save_path=os.path.join(rd, "training_curves_both.png"),
    )

    plot_final_summary(
        results_full=full_res,
        results_po=po_res,
        save_path=os.path.join(rd, "final_summary.png"),
    )

    if args.ctm_analysis:
        env_id = C.ENV_PO if args.ctm_env == "po" else C.ENV_FULL
        run_ctm_analysis(env_id, rd, n_steps_collect=args.ctm_steps)

    print("Plots saved to", rd)


if __name__ == "__main__":
    main()
