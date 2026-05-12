"""
envs/pendulum_po.py

Partially observable Pendulum wrapper.

Pendulum obs: [cos(theta), sin(theta), angular_velocity]
PO version:   [cos(theta), sin(theta), 0.0             ]
                                        ^^ masked index 2
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PartialObsPendulum(gym.Wrapper):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, masked_indices=(2,), **kwargs):
        env = gym.make("Pendulum-v1", **kwargs)
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
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._mask(obs), reward, terminated, truncated, info


gym.register(
    id="Pendulum-PO-v1",
    entry_point=PartialObsPendulum,
    max_episode_steps=200,
)
