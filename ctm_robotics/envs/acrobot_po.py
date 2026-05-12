"""
envs/acrobot_po.py

Partially observable Acrobot wrapper.
Masks angular velocity dimensions to force temporal integration.

Acrobot obs: [cos(th1), sin(th1), cos(th2), sin(th2), dtheta1, dtheta2]
PO version:  [cos(th1), sin(th1), cos(th2), sin(th2), 0.0,     0.0    ]

Without angular velocities, reactive MLP control fails: swing-up
requires knowing how fast the links are moving to apply torque correctly.
A memory agent can estimate velocity by differencing cos/sin over timesteps.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PartialObsAcrobot(gym.Wrapper):
    """Wraps Acrobot-v1 and zeros out angular velocity dimensions (indices 4, 5)."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, masked_indices=(4, 5), **kwargs):
        env = gym.make("Acrobot-v1", **kwargs)
        super().__init__(env)
        self.masked_indices = list(masked_indices)
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            dtype=np.float32,
        )

    def _mask(self, obs):
        obs = obs.copy()
        for idx in self.masked_indices:
            obs[idx] = 0.0
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_full_obs = obs.copy()
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_full_obs = obs.copy()
        return self._mask(obs), reward, terminated, truncated, info


gym.register(
    id="Acrobot-PO-v1",
    entry_point=PartialObsAcrobot,
    max_episode_steps=500,
)
