"""
run_comparison.py — Main entry point.

Trains agents on CartPole (PPO, discrete) and Pendulum/BipedalWalker (TD3, continuous).
Each environment has a fully-observable and partially-observable variant.

Usage:
    python run_comparison.py                          # all agents, all envs
    python run_comparison.py --agents td3_ctm --envs pend_full
    python run_comparison.py --smoke-test             # quick sanity check
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import gymnasium as gym

import ctm_robotics.envs  # noqa — triggers PO variant registration
import ctm_robotics.config as C
from ctm_robotics.envs.cartpole_po import make_vec_env, make_env
from ctm_robotics.models import (
    MLPActorCritic, LSTMActorCritic, CTMActorCritic,
    TD3MLPActor, TD3LSTMActor, TD3CTMActor, TD3Critic,
    ContinuousMLPActorCritic, ContinuousLSTMActorCritic, ContinuousCTMActorCritic,
)
from ctm_robotics.models.mlp_policy import SACQNetwork, SACPolicy
from ctm_robotics.training.ppo import PPOTrainer
from ctm_robotics.training.sac import SACTrainer
from ctm_robotics.training.td3 import TD3Trainer
from ctm_robotics.analysis.visualize import (
    plot_training_curves_both_envs,
    plot_neural_dynamics,
    plot_sync_matrix,
    plot_obs_saliency,
    plot_final_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

ENV_MAP = {
    "full":          C.ENV_FULL,
    "po":            C.ENV_PO,
    "po2":           C.ENV_PO2,
    "acrobot_full":  C.ACROBOT_FULL,
    "acrobot_po":    C.ACROBOT_PO,
    "pend_full":     C.PEND_FULL,
    "pend_po":       C.PEND_PO,
    "biped_full":    C.BIPED_FULL,
    "biped_po":      C.BIPED_PO,
    # Legacy
    "lunar_full":    C.LUNAR_FULL,
    "lunar_po":      C.LUNAR_PO,
}

# PO wrapper classes (lazy import to avoid circular deps)
def _get_po_wrappers():
    from ctm_robotics.envs.cartpole_po import PartialObsCartPole
    from ctm_robotics.envs.lunarlander_po import PartialObsLunarLander
    from ctm_robotics.envs.pendulum_po import PartialObsPendulum
    from ctm_robotics.envs.bipedal_po import PartialObsBipedalWalker
    from ctm_robotics.envs.acrobot_po import PartialObsAcrobot
    return {
        C.ENV_PO: PartialObsCartPole,
        C.LUNAR_PO: PartialObsLunarLander,
        C.PEND_PO: PartialObsPendulum,
        C.BIPED_PO: PartialObsBipedalWalker,
        C.ACROBOT_PO: PartialObsAcrobot,
    }

OBS_NAMES = {
    C.ENV_FULL:      ["cart_pos", "cart_vel", "pole_ang", "pole_angvel"],
    C.ENV_PO:        ["cart_pos", "cart_vel(hid)", "pole_ang", "pole_angvel(hid)"],
    C.ENV_PO2:       ["cart_pos", "cart_vel", "pole_ang(hid)", "pole_angvel"],
    C.ACROBOT_FULL:  ["cos_th1", "sin_th1", "cos_th2", "sin_th2", "dtheta1", "dtheta2"],
    C.ACROBOT_PO:    ["cos_th1", "sin_th1", "cos_th2", "sin_th2", "dtheta1(hid)", "dtheta2(hid)"],
    C.LUNAR_FULL:  ["x", "y", "vx", "vy", "angle", "ang_vel", "left_leg", "right_leg"],
    C.LUNAR_PO:    ["x", "y", "vx(hid)", "vy(hid)", "angle", "ang_vel(hid)", "left_leg", "right_leg"],
    C.PEND_FULL:   ["cos(th)", "sin(th)", "ang_vel"],
    C.PEND_PO:     ["cos(th)", "sin(th)", "ang_vel(hid)"],
    C.BIPED_FULL:  ["hull_ang", "hull_angvel", "vel_x", "vel_y",
                    "hip1_ang", "hip1_spd", "knee1_ang", "knee1_spd",
                    "hip2_ang", "hip2_spd", "knee2_ang", "knee2_spd",
                    "leg1_gnd", "leg2_gnd"] + [f"lidar_{i}" for i in range(10)],
    C.BIPED_PO:    ["hull_ang", "hull_angvel(hid)", "vel_x(hid)", "vel_y(hid)",
                    "hip1_ang", "hip1_spd(hid)", "knee1_ang", "knee1_spd(hid)",
                    "hip2_ang", "hip2_spd(hid)", "knee2_ang", "knee2_spd(hid)",
                    "leg1_gnd", "leg2_gnd"] + [f"lidar_{i}" for i in range(10)],
}

def _make_single_env(env_id, seed=0):
    po_wrappers = _get_po_wrappers()
    if env_id in po_wrappers:
        env = po_wrappers[env_id]()
    else:
        env = gym.make(env_id)
    env.reset(seed=seed)
    return env

def _is_continuous(env_id):
    return env_id in (C.PEND_FULL, C.PEND_PO, C.BIPED_FULL, C.BIPED_PO)


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

_CTM_OVERRIDES = {}  # set by parse_args ablation flags
_CTM_N_STEPS: int = 512      # overridable via --ctm-n-steps
_LSTM_N_EPOCHS: int = 1      # overridable via --lstm-n-epochs

def _ctm_kwargs():
    kw = dict(
        d_model=C.CTM.d_model, synapse_hidden=C.CTM.synapse_hidden,
        synapse_depth=C.CTM.synapse_depth, memory_length=C.CTM.memory_length,
        nlm_hidden=C.CTM.nlm_hidden, nlm_depth=C.CTM.nlm_depth,
        n_synch_out=C.CTM.n_synch_out, synch_window=C.CTM.synch_window,
        synch_decay=C.CTM.synch_decay, n_ticks=C.CTM.n_ticks,
        input_hidden=C.CTM.input_hidden,
    )
    kw.update(_CTM_OVERRIDES)
    return kw

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training orchestration
# ─────────────────────────────────────────────────────────────────────────────

def train_agent(agent_name: str, env_id: str, total_steps: int,
                seed: int, results_dir: str, eval_every: int = None,
                eval_episodes: int = None, verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)  # Small matrices: thread-spawn overhead > compute
    eval_every = eval_every or C.TRAIN.eval_every
    eval_episodes = eval_episodes or C.TRAIN.eval_episodes

    algo = C.AGENTS[agent_name]["algo"]
    continuous = _is_continuous(env_id)

    # ── Create environments ───────────────────────────────────────────────
    if algo == "ppo":
        train_env = make_vec_env(env_id, n_envs=C.TRAIN.n_envs, seed=seed)
        eval_env = _make_single_env(env_id, seed=9999)
        obs_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.n
    elif algo == "cppo":
        train_env = make_vec_env(env_id, n_envs=C.TRAIN.n_envs, seed=seed)
        eval_env = _make_single_env(env_id, seed=9999)
        obs_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.shape[0]
        max_action = float(eval_env.action_space.high[0])
    elif algo in ("td3", "sac"):
        train_env = _make_single_env(env_id, seed=seed)
        eval_env = _make_single_env(env_id, seed=9999)
        obs_dim = eval_env.observation_space.shape[0]
        if continuous:
            action_dim = eval_env.action_space.shape[0]
            max_action = float(eval_env.action_space.high[0])
        else:
            action_dim = eval_env.action_space.n

    print(f"\n{'─'*60}")
    print(f"  Agent : {agent_name.upper()}")
    print(f"  Env   : {env_id}")
    print(f"  Steps : {total_steps:,}")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")

    t0 = time.time()
    tcfg = C.TrainConfig(total_steps=total_steps, seed=seed,
                          n_envs=C.TRAIN.n_envs, eval_episodes=eval_episodes,
                          eval_every=eval_every, log_dir=results_dir)

    # ── Build trainer ─────────────────────────────────────────────────────
    if agent_name == "ppo_mlp":
        policy = MLPActorCritic(obs_dim, action_dim, hidden_sizes=C.MLP.hidden_sizes)
        print(f"  Params: {count_params(policy):,}")
        trainer = PPOTrainer(policy, train_env, C.PPO, tcfg,
                             is_recurrent=False, agent_name=agent_name)

    elif agent_name == "ppo_lstm":
        policy = LSTMActorCritic(obs_dim, action_dim,
                                  hidden_size=C.LSTM.hidden_size, n_layers=C.LSTM.n_layers)
        print(f"  Params: {count_params(policy):,}")
        lstm_ppo = C.PPOConfig(
            lr=C.PPO.lr, gamma=C.PPO.gamma, gae_lambda=C.PPO.gae_lambda,
            clip_eps=C.PPO.clip_eps, vf_coef=C.PPO.vf_coef, ent_coef=C.PPO.ent_coef,
            max_grad_norm=C.PPO.max_grad_norm, n_steps=C.PPO.n_steps,
            n_epochs=_LSTM_N_EPOCHS,
            batch_size=C.PPO.batch_size, recurrent_seq_len=C.PPO.recurrent_seq_len)
        trainer = PPOTrainer(policy, train_env, lstm_ppo, tcfg,
                             is_recurrent=True, agent_name=agent_name)

    elif agent_name == "ppo_ctm":
        policy = CTMActorCritic(obs_dim, action_dim, **_ctm_kwargs())
        print(f"  Params: {count_params(policy):,}")
        # n_steps=512 matches LSTM rollout length → same gradient estimate quality.
        # recurrent_seq_len=100 keeps BPTT tractable (rollout split into ~5 chunks).
        ctm_ppo = C.PPOConfig(
            lr=5e-4, gamma=C.PPO.gamma, gae_lambda=C.PPO.gae_lambda,
            clip_eps=0.1, vf_coef=0.25, ent_coef=0.1,
            max_grad_norm=C.CTM.max_grad_norm,
            n_steps=_CTM_N_STEPS, n_epochs=1,
            batch_size=C.PPO.batch_size, recurrent_seq_len=100,
            target_kl=0.01, ent_coef_final=0.005, ent_anneal_fraction=0.5)
        trainer = PPOTrainer(policy, train_env, ctm_ppo, tcfg,
                             is_recurrent=True, agent_name=agent_name)

    elif agent_name == "td3_mlp":
        actor = TD3MLPActor(obs_dim, action_dim, hidden_sizes=(256, 256),
                            max_action=max_action)
        critic = TD3Critic(obs_dim, action_dim, hidden_sizes=(256, 256))
        print(f"  Params: {count_params(actor):,} (actor) + {count_params(critic):,} (critic)")
        trainer = TD3Trainer(actor, critic, train_env, C.TD3, tcfg,
                             is_recurrent=False, agent_name=agent_name)

    elif agent_name == "td3_lstm":
        actor = TD3LSTMActor(obs_dim, action_dim, hidden_size=128,
                             max_action=max_action)
        critic = TD3Critic(obs_dim, action_dim, hidden_sizes=(256, 256))
        print(f"  Params: {count_params(actor):,} (actor) + {count_params(critic):,} (critic)")
        trainer = TD3Trainer(actor, critic, train_env, C.TD3, tcfg,
                             is_recurrent=True, agent_name=agent_name)

    elif agent_name == "td3_ctm":
        actor = TD3CTMActor(obs_dim, action_dim, max_action=max_action,
                            **_ctm_kwargs())
        critic = TD3Critic(obs_dim, action_dim, hidden_sizes=(256, 256))
        print(f"  Params: {count_params(actor):,} (actor) + {count_params(critic):,} (critic)")
        # CTM: small batch + less frequent updates to keep training tractable
        td3_cfg = C.TD3Config(lr_actor=1e-4, lr_critic=C.TD3.lr_critic,
                               gamma=C.TD3.gamma, tau=C.TD3.tau,
                               policy_noise=C.TD3.policy_noise,
                               noise_clip=C.TD3.noise_clip,
                               exploration_noise=C.TD3.exploration_noise,
                               policy_delay=C.TD3.policy_delay,
                               buffer_size=C.TD3.buffer_size,
                               batch_size=16, learning_starts=C.TD3.learning_starts,
                               train_freq=4)
        trainer = TD3Trainer(actor, critic, train_env, td3_cfg, tcfg,
                             is_recurrent=True, agent_name=agent_name)

    elif agent_name == "cppo_mlp":
        policy = ContinuousMLPActorCritic(obs_dim, action_dim, max_action=max_action)
        print(f"  Params: {count_params(policy):,}")
        trainer = PPOTrainer(policy, train_env, C.CPPO, tcfg,
                             is_recurrent=False, agent_name=agent_name)

    elif agent_name == "cppo_lstm":
        policy = ContinuousLSTMActorCritic(obs_dim, action_dim,
                                            hidden_size=C.LSTM.hidden_size,
                                            max_action=max_action)
        print(f"  Params: {count_params(policy):,}")
        lstm_cppo = C.CPPOConfig(lr=1e-3, n_epochs=1)  # 1 epoch: strictly on-policy for LSTM
        trainer = PPOTrainer(policy, train_env, lstm_cppo, tcfg,
                             is_recurrent=True, agent_name=agent_name)

    elif agent_name == "cppo_ctm":
        policy = ContinuousCTMActorCritic(obs_dim, action_dim, max_action=max_action,
                                           **_ctm_kwargs())
        print(f"  Params: {count_params(policy):,}")
        ctm_cppo = C.CPPOConfig(lr=3e-4, n_epochs=50, max_grad_norm=C.CTM.max_grad_norm)
        trainer = PPOTrainer(policy, train_env, ctm_cppo, tcfg,
                             is_recurrent=True, agent_name=agent_name)

    elif agent_name == "sac_mlp":
        actor = SACPolicy(obs_dim, action_dim, hidden_sizes=C.MLP.hidden_sizes)
        q1 = SACQNetwork(obs_dim, action_dim, hidden_sizes=C.MLP.hidden_sizes)
        q2 = SACQNetwork(obs_dim, action_dim, hidden_sizes=C.MLP.hidden_sizes)
        print(f"  Params: {count_params(actor):,} (actor)")
        trainer = SACTrainer(actor, q1, q2, train_env, C.SAC, tcfg,
                             agent_name=agent_name)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    steps, returns = trainer.train(eval_env=eval_env, verbose=verbose)

    elapsed = time.time() - t0
    final = f"{returns[-1]:.1f}" if returns else "N/A"
    print(f"  Done in {elapsed:.0f}s  |  final eval: {final}")

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, f"{agent_name}_{env_id.replace('-', '_')}.pt")
    trainer.save(ckpt_path)

    result = {"steps": steps, "returns": returns}
    with open(ckpt_path.replace(".pt", ".json"), "w") as f:
        json.dump(result, f)

    train_env.close()
    eval_env.close()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CTM interpretability analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_ctm_analysis(env_id: str, results_dir: str, n_steps_collect: int = 100):
    print("\n── CTM interpretability analysis ────────────────────────")

    env = _make_single_env(env_id)
    po_wrappers = _get_po_wrappers()
    # For PO envs: also record the unmasked full obs so we can check if sync tracks hidden dims
    is_po = env_id in po_wrappers
    full_obs_names = OBS_NAMES.get(env_id, [f"obs[{i}]" for i in range(env.observation_space.shape[0])])
    # full_env_id: base gym env that matches this PO env's observation space
    _PO_TO_FULL = {
        C.ENV_PO: C.ENV_FULL, C.ENV_PO2: C.ENV_FULL,
        C.ACROBOT_PO: C.ACROBOT_FULL,
        C.LUNAR_PO: C.LUNAR_FULL, C.PEND_PO: C.PEND_FULL, C.BIPED_PO: C.BIPED_FULL,
    }
    full_env_id = _PO_TO_FULL.get(env_id, env_id)
    full_obs_names_full = OBS_NAMES.get(full_env_id, full_obs_names)

    obs_names = full_obs_names
    obs_dim = env.observation_space.shape[0]
    continuous = _is_continuous(env_id)
    device = torch.device(C.TRAIN.device)

    # Build the right CTM variant
    if continuous:
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        policy = TD3CTMActor(obs_dim, action_dim, max_action=max_action, **_ctm_kwargs())
        ckpt_key = "actor_state"
        agent_prefix = "td3_ctm"
    else:
        action_dim = env.action_space.n
        policy = CTMActorCritic(obs_dim, action_dim, **_ctm_kwargs())
        ckpt_key = "policy_state"
        agent_prefix = "ppo_ctm"

    policy.to(device)

    ckpt_path = os.path.join(results_dir, f"{agent_prefix}_{env_id.replace('-', '_')}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt[ckpt_key])
        print(f"  Loaded weights from {ckpt_path}")
    else:
        print("  No checkpoint found — using random weights")

    policy.eval()
    obs, _ = env.reset(seed=0)
    hidden = policy.init_hidden(1, device)

    obs_history, full_obs_history, sync_repr_history, sync_matrices, dynamics_snapshots = [], [], [], [], []

    for step in range(n_steps_collect):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            if continuous:
                action, hidden = policy.get_action(obs_t, hidden)
                action_np = action.cpu().numpy()[0]
            else:
                action, _, _, _, hidden = policy.get_action(obs_t, hidden)
                action_np = action.item()

        obs_history.append(obs.copy())
        # Record full unmasked obs before stepping (wrappers store it in last_full_obs)
        if is_po and hasattr(env, 'last_full_obs'):
            full_obs_history.append(env.last_full_obs.copy())
        else:
            full_obs_history.append(obs.copy())

        sr = policy.get_sync_saliency()
        if sr is not None:
            sync_repr_history.append(sr.cpu().numpy()[0])
        nd = policy.get_neural_dynamics()
        if nd is not None and len(nd) > 0:
            dynamics_snapshots.append([a[0].cpu().numpy() for a in nd])
        if sr is not None:
            s = sr.cpu().numpy()[0]
            sync_matrices.append(np.outer(s, s))

        obs, reward, term, trunc, info = env.step(action_np)
        if term or trunc:
            obs, _ = env.reset()
            hidden = policy.init_hidden(1, device)

    env.close()

    obs_arr = np.array(obs_history)
    full_obs_arr = np.array(full_obs_history)
    sync_arr = np.array(sync_repr_history) if sync_repr_history else np.zeros((0, 0))

    if dynamics_snapshots:
        plot_neural_dynamics(
            post_act_seq=dynamics_snapshots[min(5, len(dynamics_snapshots)-1)],
            obs=obs_arr[min(5, len(obs_arr)-1)],
            episode_step=min(5, len(dynamics_snapshots)-1),
            save_path=os.path.join(results_dir, "ctm_neural_dynamics.png"),
            obs_names=obs_names,
        )
    if sync_matrices:
        snapshot_steps = [0, 4, 9, 19, 39]
        snapshots = [sync_matrices[i] for i in snapshot_steps if i < len(sync_matrices)]
        plot_sync_matrix(sync_matrices=snapshots,
                         save_path=os.path.join(results_dir, "ctm_sync_matrix.png"))
    if len(sync_arr) > 10:
        plot_obs_saliency(sync_reprs=sync_arr, obs_history=obs_arr,
                          obs_names=obs_names,
                          save_path=os.path.join(results_dir, "ctm_saliency.png"))
        # For PO envs: also plot saliency vs full obs to reveal hidden-state tracking
        if is_po and len(full_obs_arr) == len(obs_arr):
            plot_obs_saliency(sync_reprs=sync_arr, obs_history=full_obs_arr,
                              obs_names=full_obs_names_full,
                              save_path=os.path.join(results_dir, "ctm_saliency_fullobs.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_AGENTS = list(C.AGENTS.keys())
ALL_ENVS = list(ENV_MAP.keys())

def parse_args():
    p = argparse.ArgumentParser(description="CTM vs MLP/LSTM comparison")
    p.add_argument("--agents", nargs="+", default=["ppo_mlp", "ppo_lstm", "ppo_ctm"],
                   choices=ALL_AGENTS)
    p.add_argument("--envs", nargs="+", default=["full", "po"],
                   choices=ALL_ENVS)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip training if result JSON already exists")
    p.add_argument("--no-ctm-analysis", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--ctm-n-ticks", type=int, default=None,
                   help="Override CTM n_ticks (e.g. 1 for ablation)")
    p.add_argument("--ctm-memory-length", type=int, default=None,
                   help="Override CTM memory_length (e.g. 1 for ablation)")
    p.add_argument("--no-ctm-sync", action="store_true",
                   help="Ablation: disable CTM sync head (use raw post-act mean)")
    p.add_argument("--ctm-n-steps", type=int, default=None,
                   help="Override CTM PPO n_steps (e.g. 100 for CartPole, 512 for LunarLander)")
    p.add_argument("--lstm-n-epochs", type=int, default=None,
                   help="Override LSTM PPO n_epochs (e.g. 4 for CartPole, 1 for LunarLander)")
    return p.parse_args()


def load_existing(agent, env_id, results_dir):
    path = os.path.join(results_dir, f"{agent}_{env_id.replace('-','_')}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"steps": [], "returns": []}


def _plot_env_pair(full_id, po_id, all_results, agents, results_dir, label, suffix=""):
    empty = lambda: {a: {"steps": [], "returns": []} for a in agents}
    full_res = all_results.get(full_id, {}) or empty()
    po_res = all_results.get(po_id, {}) or empty()
    if not any(d.get("steps") for d in full_res.values()) and \
       not any(d.get("steps") for d in po_res.values()):
        return
    plot_training_curves_both_envs(
        full_results=full_res, po_results=po_res,
        save_path=os.path.join(results_dir, f"training_curves{suffix}.png"),
        env_label=label)
    plot_final_summary(
        results_full=full_res, results_po=po_res,
        save_path=os.path.join(results_dir, f"final_summary{suffix}.png"))


def main():
    args = parse_args()

    # Apply ablation overrides to CTM kwargs
    global _CTM_OVERRIDES, _CTM_N_STEPS, _LSTM_N_EPOCHS
    if args.ctm_n_ticks is not None:
        _CTM_OVERRIDES["n_ticks"] = args.ctm_n_ticks
    if args.ctm_memory_length is not None:
        _CTM_OVERRIDES["memory_length"] = args.ctm_memory_length
    if args.ctm_n_steps is not None:
        _CTM_N_STEPS = args.ctm_n_steps
    if args.lstm_n_epochs is not None:
        _LSTM_N_EPOCHS = args.lstm_n_epochs

    total_steps = args.steps or (5_000 if args.smoke_test else C.TRAIN.total_steps)
    eval_every = 1_000 if args.smoke_test else C.TRAIN.eval_every
    eval_episodes = 5 if args.smoke_test else C.TRAIN.eval_episodes
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    verbose = not args.quiet

    env_ids = [ENV_MAP[e] for e in args.envs]

    print("=" * 60)
    print("  CTM-RL Comparison")
    print("=" * 60)
    print(f"  Agents : {args.agents}")
    print(f"  Envs   : {env_ids}")
    print(f"  Steps  : {total_steps:,} per agent per env")
    if args.smoke_test:
        print("  ** SMOKE TEST **")

    # ── Training ──────────────────────────────────────────────────────────
    all_results = {eid: {} for eid in env_ids}

    for env_id in env_ids:
        continuous = _is_continuous(env_id)
        for agent in args.agents:
            algo = C.AGENTS[agent]["algo"]
            # Skip incompatible agent-env pairs
            if algo == "ppo" and continuous:
                continue
            if algo in ("td3", "cppo") and not continuous:
                continue
            if algo == "sac" and continuous:
                continue

            if args.plot_only or (args.skip_existing and load_existing(agent, env_id, results_dir)["steps"]):
                all_results[env_id][agent] = load_existing(agent, env_id, results_dir)
            else:
                result = train_agent(
                    agent_name=agent, env_id=env_id, total_steps=total_steps,
                    seed=args.seed, results_dir=results_dir,
                    eval_every=eval_every, eval_episodes=eval_episodes,
                    verbose=verbose)
                all_results[env_id][agent] = result

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n── Generating plots ────────────────────────────────────")
    _plot_env_pair(C.ENV_FULL, C.ENV_PO, all_results, args.agents,
                   results_dir, "CartPole", "_cartpole")
    _plot_env_pair(C.ENV_FULL, C.ENV_PO2, all_results, args.agents,
                   results_dir, "CartPole-PO2", "_cartpole_po2")
    _plot_env_pair(C.PEND_FULL, C.PEND_PO, all_results, args.agents,
                   results_dir, "Pendulum", "_pendulum")
    _plot_env_pair(C.BIPED_FULL, C.BIPED_PO, all_results, args.agents,
                   results_dir, "BipedalWalker", "_bipedal")
    _plot_env_pair(C.LUNAR_FULL, C.LUNAR_PO, all_results, args.agents,
                   results_dir, "LunarLander", "_lunar")

    # CTM analysis
    if not args.no_ctm_analysis:
        ctm_agents = [a for a in args.agents if "ctm" in a]
        if ctm_agents:
            # Pick best env for analysis
            for env_key in ["pend_po", "pend_full", "biped_po", "biped_full",
                            "lunar_po", "lunar_full", "po2", "po", "full"]:
                if env_key in args.envs:
                    run_ctm_analysis(ENV_MAP[env_key], results_dir,
                                     n_steps_collect=50 if args.smoke_test else 200)
                    break

    print("\n── Done! ───────────────────────────────────────────────")
    print(f"  Results: {results_dir}/")
    for f in sorted(os.listdir(results_dir)):
        size = os.path.getsize(os.path.join(results_dir, f))
        print(f"    {f:50s} {size//1024:>5} KB")


if __name__ == "__main__":
    main()
