"""
envs/bipedal_po.py

Partially observable BipedalWalker wrapper.
Masks joint angular velocity dimensions.

BipedalWalker-v3 obs (24-dim):
  0: hull_angle           4: hip1_speed     8: knee1_angle   12: hip2_speed    16: knee2_angle   20: leg2_ground
  1: hull_angularVelocity 5: hip1_angle     9: knee1_speed   13: hip2_angle    17: knee2_speed   21: reserved
  2: vel_x                6: hip1_speed(?)  10: leg1_ground   14: hip2_speed(?) 18: leg2_ground   22: reserved
  3: vel_y                7: knee1_angle(?) 11: reserved      15: knee2_angle(?) 19: reserved     23: reserved

Actual layout per gymnasium docs (24 obs):
  [0]  hull_angle, [1] hull_angularVelocity, [2] vel_x, [3] vel_y,
  [4]  hip1_angle, [5] hip1_speed, [6] knee1_angle, [7] knee1_speed,
  [8]  hip2_angle, [9] hip2_speed, [10] knee2_angle, [11] knee2_speed,
  [12] leg1_ground_contact, [13] leg2_ground_contact,
  [14..23] 10 lidar rangefinder readings

PO: mask velocity indices [1, 2, 3, 5, 7, 9, 11]
  (hull_angvel, vel_x, vel_y, hip1_speed, knee1_speed, hip2_speed, knee2_speed)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PartialObsBipedalWalker(gym.Wrapper):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, masked_indices=(1, 2, 3, 5, 7, 9, 11), **kwargs):
        env = gym.make("BipedalWalker-v3", **kwargs)
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
    id="BipedalWalker-PO-v3",
    entry_point=PartialObsBipedalWalker,
    max_episode_steps=1600,
)
