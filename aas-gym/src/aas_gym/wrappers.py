"""
Reward wrappers for AAS Gym environments.
"""

import numpy as np
import gymnasium as gym


class WaypointRewardWrapper(gym.Wrapper):
    """
    Wrapper that adds reward for reaching a fixed waypoint.

    Reward structure:
    - Distance reward: -distance_to_target (encourages getting closer)
    - Arrival bonus: +100 when within arrival_threshold
    - Velocity penalty: -0.1 * speed (encourages smooth flight)

    Episode terminates when drone reaches the waypoint.
    """

    def __init__(self, env, target_position=None, arrival_threshold=2.0):
        """
        Args:
            env: The base environment
            target_position: [x, y, z] target in meters (ENU frame). Default [20, 0, 10]
            arrival_threshold: Distance in meters to consider "arrived"
        """
        super().__init__(env)

        if target_position is None:
            target_position = [20.0, 0.0, 10.0]  # 20m east, 10m altitude

        self.target_position = np.array(target_position, dtype=np.float64)
        self.arrival_threshold = arrival_threshold
        self.prev_distance = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_distance = self._get_distance(obs)
        info["target_position"] = self.target_position.tolist()
        info["distance_to_target"] = self.prev_distance
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Calculate distance to target
        distance = self._get_distance(obs)

        # Reward components
        # 1. Progress reward (positive if getting closer)
        progress_reward = (self.prev_distance - distance) * 10.0

        # 2. Distance penalty (small continuous penalty for being far)
        distance_penalty = -0.01 * distance

        # 3. Velocity penalty (discourage excessive speed)
        velocity = obs["velocity"]
        speed = np.linalg.norm(velocity)
        velocity_penalty = -0.05 * speed

        # 4. Arrival bonus
        arrival_bonus = 0.0
        if distance < self.arrival_threshold:
            arrival_bonus = 100.0
            terminated = True  # End episode on arrival

        # Total reward
        reward = progress_reward + distance_penalty + velocity_penalty + arrival_bonus

        # Update state
        self.prev_distance = distance

        # Add info
        info["target_position"] = self.target_position.tolist()
        info["distance_to_target"] = distance
        info["arrived"] = distance < self.arrival_threshold

        return obs, reward, terminated, truncated, info

    def _get_distance(self, obs):
        """Calculate Euclidean distance to target."""
        position = obs["position"]
        return np.linalg.norm(position - self.target_position)


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the Dict observation space into a single Box.
    Required for standard MLP policies in Stable Baselines3.

    Flattened order: [position(3), velocity(3), orientation(4), heading(1)] = 11 values
    """

    def __init__(self, env):
        super().__init__(env)

        # Calculate flattened size
        self.obs_size = 3 + 3 + 4 + 1  # position + velocity + orientation + heading

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float64
        )

    def observation(self, obs):
        """Flatten the dict observation into a 1D array."""
        return np.concatenate([
            obs["position"],      # 3
            obs["velocity"],      # 3
            obs["orientation"],   # 4
            obs["heading"],       # 1
        ])
