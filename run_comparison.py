"""
run_comparison.py — Main entry point.

Trains PPO-MLP, SAC-MLP, PPO-LSTM, PPO-CTM on CartPole-v1 and CartPole-PO-v1,
then saves results and plots.

Usage:
    # Train all agents on both environments
    python run_comparison.py

    # Train one agent on one env (for fast iteration)
    python run_comparison.py --agents ppo_ctm --envs po --steps 100000

    # Skip training, only plot previously saved results
    python run_comparison.py --plot-only

    # Quick smoke test (few steps, no eval plots)
    python run_comparison.py --smoke-test
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import gymnasium as gym

import ctm_robotics.envs  # noqa — triggers CartPole-PO-v1 registration
import ctm_robotics.config as C
from ctm_robotics.envs.cartpole_po import PartialObsCartPole, make_vec_env, make_env
from ctm_robotics.models import MLPActorCritic, LSTMActorCritic, CTMActorCritic
from ctm_robotics.models.mlp_policy import SACQNetwork, SACPolicy
from ctm_robotics.training.ppo import PPOTrainer
from ctm_robotics.training.sac import SACTrainer
from ctm_robotics.analysis.visualize import (
    plot_training_curves_both_envs,
    plot_neural_dynamics,
    plot_sync_matrix,
    plot_obs_saliency,
    plot_final_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

def build_ppo_mlp(obs_dim, action_dim):
    return MLPActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=C.MLP.hidden_sizes,
    )

def build_sac_mlp(obs_dim, action_dim):
    actor = SACPolicy(obs_dim, action_dim, hidden_sizes=C.MLP.hidden_sizes)
    q1    = SACQNetwork(obs_dim, action_dim, hidden_sizes=C.MLP.hidden_sizes)
    q2    = SACQNetwork(obs_dim, action_dim, hidden_sizes=C.MLP.hidden_sizes)
    return actor, q1, q2

def build_ppo_lstm(obs_dim, action_dim):
    return LSTMActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=C.LSTM.hidden_size,
        n_layers=C.LSTM.n_layers,
    )

def build_ppo_ctm(obs_dim, action_dim):
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


# ─────────────────────────────────────────────────────────────────────────────
# Training orchestration
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_agent(agent_name: str, env_id: str, total_steps: int,
                seed: int, results_dir: str, eval_every: int = None,
                eval_episodes: int = None, verbose: bool = True) -> dict:
    """
    Train a single agent on a single environment.
    Returns dict with {"steps": [...], "returns": [...]}.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_every = eval_every or C.TRAIN.eval_every
    eval_episodes = eval_episodes or C.TRAIN.eval_episodes

    # Env
    if agent_name == "sac_mlp":
        # SAC uses a single non-vectorized env
        if env_id == C.ENV_PO:
            train_env = PartialObsCartPole()
        else:
            train_env = gym.make("CartPole-v1")
        train_env.reset(seed=seed)
    else:
        n_envs = C.TRAIN.n_envs
        train_env = make_vec_env(env_id, n_envs=n_envs, seed=seed)

    # Eval env (always single)
    if env_id == C.ENV_PO:
        eval_env = PartialObsCartPole()
    else:
        eval_env = gym.make("CartPole-v1")

    obs_dim    = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.n

    print(f"\n{'─'*60}")
    print(f"  Agent : {agent_name.upper()}")
    print(f"  Env   : {env_id}")
    print(f"  Steps : {total_steps:,}")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")

    # ── Build and train ────────────────────────────────────────────────────
    t0 = time.time()
    tcfg = C.TrainConfig(total_steps=total_steps, seed=seed,
                          n_envs=C.TRAIN.n_envs,
                          eval_episodes=eval_episodes,
                          eval_every=eval_every,
                          log_dir=results_dir)

    if agent_name == "ppo_mlp":
        policy = build_ppo_mlp(obs_dim, action_dim)
        print(f"  Params: {count_params(policy):,}")
        trainer = PPOTrainer(policy, train_env, C.PPO, tcfg,
                             is_recurrent=False, agent_name=agent_name)

    elif agent_name == "sac_mlp":
        actor, q1, q2 = build_sac_mlp(obs_dim, action_dim)
        print(f"  Params: {count_params(actor):,} (actor) + {count_params(q1):,} (per Q)")
        trainer = SACTrainer(actor, q1, q2, train_env, C.SAC, tcfg,
                             agent_name=agent_name)

    elif agent_name == "ppo_lstm":
        policy = build_ppo_lstm(obs_dim, action_dim)
        print(f"  Params: {count_params(policy):,}")
        trainer = PPOTrainer(policy, train_env, C.PPO, tcfg,
                             is_recurrent=True, agent_name=agent_name)

    elif agent_name == "ppo_ctm":
        policy = build_ppo_ctm(obs_dim, action_dim)
        print(f"  Params: {count_params(policy):,}")
        # CTM uses its own grad clip
        ctm_ppo = C.PPOConfig(
            lr=C.PPO.lr, gamma=C.PPO.gamma, gae_lambda=C.PPO.gae_lambda,
            clip_eps=C.PPO.clip_eps, vf_coef=C.PPO.vf_coef,
            ent_coef=C.PPO.ent_coef, max_grad_norm=C.CTM.max_grad_norm,
            n_steps=C.PPO.n_steps, n_epochs=C.PPO.n_epochs,
            batch_size=C.PPO.batch_size,
        )
        trainer = PPOTrainer(policy, train_env, ctm_ppo, tcfg,
                             is_recurrent=True, agent_name=agent_name)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    steps, returns = trainer.train(eval_env=eval_env, verbose=verbose)

    elapsed = time.time() - t0
    final = f"{returns[-1]:.1f}" if returns else "N/A"
    print(f"  Done in {elapsed:.0f}s  |  final eval: {final}")

    # Save checkpoint
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, f"{agent_name}_{env_id.replace('-', '_')}.pt")
    trainer.save(ckpt_path)

    result = {"steps": steps, "returns": returns}

    # Save JSON result
    json_path = ckpt_path.replace(".pt", ".json")
    with open(json_path, "w") as f:
        json.dump(result, f)

    train_env.close()
    eval_env.close()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CTM interpretability analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_ctm_analysis(env_id: str, results_dir: str, n_steps_collect: int = 100):
    """
    Run a trained (or fresh) CTM policy for n_steps_collect steps,
    collecting sync matrices, neural dynamics, and obs history,
    then produce interpretability plots.
    """
    print("\n── CTM interpretability analysis ────────────────────────")

    if env_id == C.ENV_PO:
        env = PartialObsCartPole()
        obs_names = ["cart_pos", "cart_vel(hidden)", "pole_θ", "pole_ω(hidden)"]
    else:
        env = gym.make("CartPole-v1")
        obs_names = ["cart_pos", "cart_vel", "pole_θ", "pole_ω"]

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = build_ppo_ctm(obs_dim, action_dim)
    device = torch.device(C.TRAIN.device)
    policy.to(device)

    # Try to load trained weights
    ckpt_path = os.path.join(results_dir,
                              f"ppo_ctm_{env_id.replace('-', '_')}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(ckpt["policy_state"])
        print(f"  Loaded weights from {ckpt_path}")
    else:
        print("  No checkpoint found — using random weights (still useful for architecture demo)")

    policy.eval()
    obs, _ = env.reset(seed=0)
    hidden = policy.init_hidden(1, device)

    obs_history       = []
    sync_repr_history = []
    sync_matrices     = []
    dynamics_snapshots = []

    for step in range(n_steps_collect):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _, hidden = policy.get_action(obs_t, hidden)

        obs_history.append(obs.copy())

        # Collect sync repr
        sr = policy.get_sync_saliency()
        if sr is not None:
            sync_repr_history.append(sr.cpu().numpy()[0])

        # Collect neural dynamics (post-act over ticks)
        nd = policy.get_neural_dynamics()
        if nd is not None and len(nd) > 0:
            # Each element is (batch, D) → take first batch
            dynamics_snapshots.append([a[0].cpu().numpy() for a in nd])

        # Collect sync matrix (approximate: outer product of sync repr with itself)
        if sr is not None:
            s = sr.cpu().numpy()[0]
            sync_matrices.append(np.outer(s, s))

        obs, reward, term, trunc, _ = env.step(action.item())
        if term or trunc:
            obs, _ = env.reset()
            hidden = policy.init_hidden(1, device)

    env.close()

    obs_arr  = np.array(obs_history)         # (T, obs_dim)
    sync_arr = np.array(sync_repr_history)   # (T, n_synch_out)

    # Plot 1: neural dynamics at step 5
    if dynamics_snapshots:
        plot_neural_dynamics(
            post_act_seq=dynamics_snapshots[min(5, len(dynamics_snapshots)-1)],
            obs=obs_arr[min(5, len(obs_arr)-1)],
            episode_step=min(5, len(dynamics_snapshots)-1),
            save_path=os.path.join(results_dir, "ctm_neural_dynamics.png"),
        )

    # Plot 2: sync matrices at steps 1, 5, 10, 20, 40
    if sync_matrices:
        snapshot_steps = [0, 4, 9, 19, 39]
        snapshots = [sync_matrices[i] for i in snapshot_steps if i < len(sync_matrices)]
        plot_sync_matrix(
            sync_matrices=snapshots,
            save_path=os.path.join(results_dir, "ctm_sync_matrix.png"),
        )

    # Plot 3: saliency
    if len(sync_arr) > 10:
        plot_obs_saliency(
            sync_reprs=sync_arr,
            obs_history=obs_arr,
            obs_names=obs_names,
            save_path=os.path.join(results_dir, "ctm_saliency.png"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CTM vs PPO/SAC/LSTM comparison")
    p.add_argument("--agents", nargs="+",
                   default=["ppo_mlp", "sac_mlp", "ppo_lstm", "ppo_ctm"],
                   choices=["ppo_mlp", "sac_mlp", "ppo_lstm", "ppo_ctm"],
                   help="Which agents to train")
    p.add_argument("--envs", nargs="+", default=["full", "po"],
                   choices=["full", "po"],
                   help="Environments: full=CartPole-v1, po=partial-obs")
    p.add_argument("--steps", type=int, default=None,
                   help="Override total training steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip training, only replot saved results")
    p.add_argument("--no-ctm-analysis", action="store_true",
                   help="Skip CTM interpretability plots")
    p.add_argument("--smoke-test", action="store_true",
                   help="Very short run to verify everything works")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def load_existing(agent, env_id, results_dir):
    """Load a previously saved JSON result."""
    path = os.path.join(results_dir,
                        f"{agent}_{env_id.replace('-','_')}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"steps": [], "returns": []}


def main():
    args = parse_args()

    total_steps = args.steps or (5_000 if args.smoke_test else C.TRAIN.total_steps)
    eval_every  = 1_000 if args.smoke_test else C.TRAIN.eval_every
    eval_episodes = 5 if args.smoke_test else C.TRAIN.eval_episodes
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    verbose = not args.quiet

    env_map = {
        "full": C.ENV_FULL,
        "po":   C.ENV_PO,
    }
    env_ids = [env_map[e] for e in args.envs]

    print("=" * 60)
    print("  CTM vs PPO / SAC / LSTM — CartPole Comparison")
    print("=" * 60)
    print(f"  Agents : {args.agents}")
    print(f"  Envs   : {env_ids}")
    print(f"  Steps  : {total_steps:,} per agent per env")
    print(f"  Seed   : {args.seed}")
    print(f"  Results: {results_dir}/")
    if args.smoke_test:
        print("  ⚠  SMOKE TEST MODE — very short run")

    # ── Training ──────────────────────────────────────────────────────────────
    all_results = {env_id: {} for env_id in env_ids}

    for env_id in env_ids:
        for agent in args.agents:
            if args.plot_only:
                all_results[env_id][agent] = load_existing(agent, env_id, results_dir)
            else:
                result = train_agent(
                    agent_name=agent,
                    env_id=env_id,
                    total_steps=total_steps,
                    seed=args.seed,
                    results_dir=results_dir,
                    eval_every=eval_every,
                    eval_episodes=eval_episodes,
                    verbose=verbose,
                )
                all_results[env_id][agent] = result

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating plots ────────────────────────────────────")

    full_res = all_results.get(C.ENV_FULL, {})
    po_res   = all_results.get(C.ENV_PO,   {})

    # Fill in empty dicts for missing envs
    if not full_res:
        full_res = {a: {"steps": [], "returns": []} for a in args.agents}
    if not po_res:
        po_res   = {a: {"steps": [], "returns": []} for a in args.agents}

    plot_training_curves_both_envs(
        full_results=full_res,
        po_results=po_res,
        save_path=os.path.join(results_dir, "training_curves_both.png"),
    )

    plot_final_summary(
        results_full=full_res,
        results_po=po_res,
        save_path=os.path.join(results_dir, "final_summary.png"),
    )

    # CTM interpretability analysis
    if not args.no_ctm_analysis and "ppo_ctm" in args.agents:
        ana_env = C.ENV_PO if "po" in args.envs else C.ENV_FULL
        run_ctm_analysis(
            env_id=ana_env,
            results_dir=results_dir,
            n_steps_collect=50 if args.smoke_test else 200,
        )

    print("\n── Done! ───────────────────────────────────────────────")
    print(f"  All outputs written to: {results_dir}/")
    print("  Files:")
    for f in sorted(os.listdir(results_dir)):
        size = os.path.getsize(os.path.join(results_dir, f))
        print(f"    {f:45s}  {size//1024:>5} KB")


if __name__ == "__main__":
    main()
