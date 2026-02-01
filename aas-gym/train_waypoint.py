#!/usr/bin/env python3
"""
Train a drone to fly to a fixed waypoint using PPO.

Usage:
    python train_waypoint.py

The training runs headless (no GUI). Use --render to see the simulation.
"""

import argparse
import os
from datetime import datetime

import aas_gym

# Check if stable-baselines3 is installed
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("stable-baselines3 not installed. Run:")
    print("  pip install stable-baselines3")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train drone waypoint navigation")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--target", type=float, nargs=3, default=[20.0, 0.0, 10.0],
                        help="Target position [x, y, z] in meters")
    parser.add_argument("--render", action="store_true", help="Enable GUI rendering")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000, help="Save checkpoint every N steps")
    parser.add_argument("--continue-from", type=str, default=None, help="Continue training from saved model")
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./training_logs/waypoint_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    # Create environment
    render_mode = "human" if args.render else None
    env = aas_gym.make_waypoint_env(
        target_position=args.target,
        arrival_threshold=2.0,
        flatten_obs=True,
        render_mode=render_mode,
    )
    env = Monitor(env, log_dir)

    print(f"Target waypoint: {args.target}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Create or load model
    if args.continue_from:
        print(f"Loading model from: {args.continue_from}")
        model = PPO.load(args.continue_from, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Entropy bonus for exploration
            tensorboard_log=log_dir,
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=log_dir,
        name_prefix="ppo_waypoint"
    )

    # Train
    print(f"\nStarting training for {args.timesteps} timesteps...")
    print("This will take a while. Each episode requires simulation reset (~80s).\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_path = os.path.join(log_dir, "ppo_waypoint_final")
    model.save(final_path)
    print(f"\nModel saved to: {final_path}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
