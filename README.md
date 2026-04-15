# CTM-RL: Continuous Thought Machines for Interpretable Robot Control

Comparing CTM-based RL policies against standard baselines on partially observable control tasks.

| Agent | Algorithm | Memory | Notes |
|-------|-----------|--------|-------|
| PPO-MLP | PPO | None | Vanilla baseline |
| SAC-MLP | SAC | None | Off-policy baseline |
| PPO-LSTM | PPO | LSTM hidden state | Memory baseline |
| PPO-CTM | PPO | Synchronization matrix | Our architecture |

## Environments

- **CartPole-v1** — fully observable (pos, vel, angle, ang_vel). All agents should solve this.
- **CartPole-PO** — partially observable: velocity dimensions masked. Forces memory use.

## Project structure

```
ctm_robotics/
├── README.md
├── pyproject.toml
├── run_comparison.py           # Main entry point — trains all agents
├── plot_results.py             # Re-plot from saved checkpoints
├── ctm_robotics/
│   ├── config.py               # All hyperparameters in one place
│   ├── models/
│   │   ├── ctm.py              # CTM actor-critic (faithful to Sakana paper)
│   │   ├── lstm_policy.py      # LSTM actor-critic
│   │   └── mlp_policy.py       # MLP actor-critic + SAC components
│   ├── training/
│   │   ├── ppo.py              # PPO trainer (MLP, LSTM, CTM)
│   │   ├── sac.py              # Discrete SAC trainer
│   │   └── rollout_buffer.py   # Rollout buffer with recurrent support
│   ├── envs/
│   │   └── cartpole_po.py      # Partially observable CartPole wrapper
│   └── analysis/
│       └── visualize.py        # Training curves, sync matrix, saliency plots
└── results/                    # Generated outputs (gitignored)
```

## Setup

```bash
pip install -e .
```

## Usage

```bash
# Full run — all 4 agents x 2 envs
python run_comparison.py

# Quick smoke test
python run_comparison.py --smoke-test

# Single agent on one env
python run_comparison.py --agents ppo_ctm --envs po --steps 200000

# Re-plot from saved results
python plot_results.py

# Re-plot including CTM interpretability analysis
python plot_results.py --ctm-analysis
```

## CTM implementation

Faithful to `models/ctm_rl.py` in the [Sakana repository](https://github.com/SakanaAI/continuous-thought-machines):

- **Synapse model**: U-NET-style MLP connecting neurons at each internal tick
- **Neuron-level models (NLMs)**: private MLP per neuron processing pre-activation history
- **Synchronization**: sliding window of post-activations with exponential decay, diagonal dot-products
- **No cross-attention** in the RL variant (`heads=0` as in the paper)
- **Continuous history** Z^t maintained across environment steps (not reset between steps)

## Interpretability outputs

The CTM analysis pipeline produces three visualization types:

1. **Neural dynamics** — heatmap of neuron activations across internal ticks
2. **Synchronization matrix** — neuron-pair correlation snapshots over an episode
3. **Observation saliency** — correlation between sync representation and each observation variable
