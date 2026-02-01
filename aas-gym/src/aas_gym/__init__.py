from aas_gym.aas_env import AASEnv
from aas_gym.wrappers import WaypointRewardWrapper, FlattenObservationWrapper


def make(render_mode=None, **kwargs):
    """Create an AASEnv instance."""
    return AASEnv(render_mode=render_mode, **kwargs)


def make_waypoint_env(target_position=None, arrival_threshold=2.0, flatten_obs=True, **kwargs):
    """
    Create an AASEnv wrapped with waypoint reward.

    Args:
        target_position: [x, y, z] target in meters. Default [20, 0, 10]
        arrival_threshold: Distance to consider arrived (meters)
        flatten_obs: If True, flatten Dict obs to Box (required for MLP policies)
        **kwargs: Passed to AASEnv (instance, gym_freq_hz, etc.)

    Returns:
        Wrapped environment ready for training
    """
    env = AASEnv(**kwargs)
    env = WaypointRewardWrapper(env, target_position=target_position, arrival_threshold=arrival_threshold)
    if flatten_obs:
        env = FlattenObservationWrapper(env)
    return env


__all__ = [
    "AASEnv",
    "WaypointRewardWrapper",
    "FlattenObservationWrapper",
    "make",
    "make_waypoint_env",
]
