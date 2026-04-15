"""
envs/cartpole_po.py

Partially observable CartPole wrapper.
Masks velocity dimensions to force agents to integrate information over time.

CartPole obs: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
PO version:   [cart_pos, 0.0,      pole_angle, 0.0         ]  (vel dims zeroed)

This turns a trivially solvable MDP into a genuine POMDP:
- A memoryless agent (PPO-MLP) will struggle because velocity is needed
  to compute a good control signal.
- A memory agent (LSTM, CTM) can infer velocity by differencing positions
  over consecutive timesteps.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PartialObsCartPole(gym.Wrapper):
    """
    Wraps CartPole-v1 and zeros out the velocity observation dimensions.

    Args:
        masked_indices: list of observation indices to zero out.
                        Default: [1, 3] (cart_vel, pole_ang_vel)
        noise_scale:    optional Gaussian noise on remaining obs (default 0.0)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, masked_indices=(1, 3), noise_scale=0.0, **kwargs):
        env = gym.make("CartPole-v1", **kwargs)
        super().__init__(env)
        self.masked_indices = list(masked_indices)
        self.noise_scale = noise_scale

        # Observation space remains same shape — agents see zeros in masked dims
        # This is intentional: the agent knows there *are* hidden dimensions,
        # it just can't read them.
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            dtype=np.float32,
        )

    def _mask(self, obs):
        obs = obs.copy()
        for idx in self.masked_indices:
            obs[idx] = 0.0
        if self.noise_scale > 0.0:
            obs += np.random.randn(*obs.shape).astype(np.float32) * self.noise_scale
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._mask(obs), reward, terminated, truncated, info


def make_env(env_id: str, seed: int = 0, rank: int = 0):
    """
    Factory for creating environments, compatible with gymnasium vectorized envs.

    Args:
        env_id:  "CartPole-v1" or "CartPole-PO-v1"
        seed:    base random seed
        rank:    worker index (added to seed for diversity)
    """
    def _init():
        if env_id == "CartPole-PO-v1":
            env = PartialObsCartPole()
        else:
            env = gym.make("CartPole-v1")
        env.reset(seed=seed + rank)
        return env
    return _init


def make_vec_env(env_id: str, n_envs: int, seed: int = 0):
    """
    Creates a vectorized (synchronous) environment.
    Uses gymnasium's SyncVectorEnv for simplicity.
    """
    fns = [make_env(env_id, seed=seed, rank=i) for i in range(n_envs)]
    return gym.vector.SyncVectorEnv(fns)


# Register the PO variant so gym.make("CartPole-PO-v1") works
gym.register(
    id="CartPole-PO-v1",
    entry_point=PartialObsCartPole,
    max_episode_steps=500,
)
