"""
AAS Gym - Gymnasium environments for the Aerial Autonomy Stack.

Available Environments:
- AASEnv: Base environment with dummy action/observation spaces
- AASVelocityEnv: Velocity-controlled environment for RL training
- AASForwardFlightEnv: Simplified environment for learning forward flight
"""

from aas_gym.aas_env import AASEnv, AASVelocityEnv, AASForwardFlightEnv

__all__ = ['AASEnv', 'AASVelocityEnv', 'AASForwardFlightEnv']
