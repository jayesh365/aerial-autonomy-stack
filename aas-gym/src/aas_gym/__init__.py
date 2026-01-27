"""
AAS Gym - Gymnasium environments for the Aerial Autonomy Stack.

Available Environments:
- AASEnv: Base environment with dummy action/observation spaces
- AASVelocityEnv: Velocity-controlled environment for RL training
- AASForwardFlightEnv: Simplified environment for learning forward flight
- AASSimpleCommandEnv: Simplified environment with discrete commands (hover, forward, left, right, backward)
"""

from aas_gym.aas_env import AASEnv, AASVelocityEnv, AASForwardFlightEnv, AASSimpleCommandEnv

__all__ = ['AASEnv', 'AASVelocityEnv', 'AASForwardFlightEnv', 'AASSimpleCommandEnv']
