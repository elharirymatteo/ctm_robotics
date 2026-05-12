"""
envs/lunarlander_po.py

Partially observable LunarLander wrapper.
Masks velocity dimensions to force agents to integrate information over time.

LunarLander obs: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
PO version:      [x, y, 0,  0,  angle, 0,           left_leg, right_leg]
                          ^^  ^^         ^^
                        masked indices [2, 3, 5]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PartialObsLunarLander(gym.Wrapper):
    """
    Wraps LunarLander-v3 and zeros out velocity observation dimensions.

    Args:
        masked_indices: list of observation indices to zero out.
                        Default: [2, 3, 5] (vx, vy, angular_vel)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, masked_indices=(2, 3, 5), **kwargs):
        env = gym.make("LunarLander-v3", **kwargs)
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
    id="LunarLander-PO-v3",
    entry_point=PartialObsLunarLander,
    max_episode_steps=1000,
)
