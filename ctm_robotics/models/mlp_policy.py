"""
models/mlp_policy.py

Shared MLP building blocks and actor-critic for PPO-MLP and SAC.

Architecture:
    obs → [Linear → ReLU] × n_layers → [actor_head | critic_head]

For SAC (continuous actions) the actor outputs mean + log_std.
CartPole has discrete actions, so SAC is adapted with Gumbel-softmax
or simply treated as a discrete SAC variant.

Note: We use PPO for the MLP baseline too — pure SAC with discrete
actions requires a separate treatment. SAC-MLP here uses a continuous
relaxation of the discrete action via a tanh-squashed output mapped
to action probabilities (standard trick for discrete SAC).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def build_mlp(in_dim: int, hidden_sizes: tuple, out_dim: int,
              activation=nn.ReLU, output_activation=None) -> nn.Sequential:
    """Utility to build a fully-connected MLP."""
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class MLPActorCritic(nn.Module):
    """
    Actor-Critic for PPO with a shared MLP trunk.

    Output:
        actor:  logits over discrete actions (CartPole has 2)
        critic: scalar state value V(s)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Separate actor and critic networks (no shared trunk)
        # Separate nets are slightly less sample-efficient but more stable
        self.actor = build_mlp(obs_dim, hidden_sizes, action_dim)
        self.critic = build_mlp(obs_dim, hidden_sizes, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init — standard for PPO."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Output layer of actor: smaller gain for stability
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs):
        """
        Args:
            obs: (batch, obs_dim)
        Returns:
            logits: (batch, action_dim)
            values: (batch,)
        """
        logits = self.actor(obs)
        values = self.critic(obs).squeeze(-1)
        return logits, values

    def get_action(self, obs):
        """Sample action and return (action, log_prob, value, entropy)."""
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), values, dist.entropy()

    def evaluate_actions(self, obs, actions):
        """Compute log_prob, entropy and value for given (obs, action) pairs."""
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


# ─── SAC components ──────────────────────────────────────────────────────────

class SACQNetwork(nn.Module):
    """Q-network for discrete SAC. Takes obs, outputs Q(s,a) for all actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        return self.net(obs)  # (batch, action_dim)


class SACPolicy(nn.Module):
    """
    Discrete SAC Actor.
    Outputs action probabilities via softmax (not Gumbel-softmax — simpler).

    Reference: Christodoulou (2019) "Soft Actor-Critic for Discrete Action Spaces"
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs):
        """Returns action probabilities and log-probs (for entropy)."""
        logits = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def get_action(self, obs):
        probs, log_probs = self.forward(obs)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, log_probs, probs
