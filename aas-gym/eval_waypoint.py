#!/usr/bin/env python3
"""
Evaluate a trained waypoint navigation model.

Usage:
    python eval_waypoint.py --model training_logs/waypoint_xxx/ppo_waypoint_final.zip
    python eval_waypoint.py --model training_logs/waypoint_xxx/ppo_waypoint_final.zip --render
"""

import argparse
import numpy as np

import aas_gym

try:
    from stable_baselines3 import PPO
except ImportError:
    print("stable-baselines3 not installed. Run:")
    print("  pip install stable-baselines3")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained waypoint model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .zip")
    parser.add_argument("--target", type=float, nargs=3, default=[20.0, 0.0, 10.0],
                        help="Target position [x, y, z]")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Enable GUI rendering")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    args = parser.parse_args()

    # Create environment
    render_mode = "human" if args.render else "ansi"
    env = aas_gym.make_waypoint_env(
        target_position=args.target,
        arrival_threshold=2.0,
        flatten_obs=True,
        render_mode=render_mode,
    )

    # Load model
    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    print(f"Target: {args.target}")
    print(f"Running {args.episodes} episodes...\n")

    # Run episodes
    results = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        arrived = False

        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                arrived = info.get("arrived", False)
                break

        results.append({
            "episode": ep + 1,
            "steps": steps,
            "reward": total_reward,
            "arrived": arrived,
            "final_distance": info.get("distance_to_target", -1),
        })

        print(f"\nEpisode {ep + 1}:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Arrived: {arrived}")
        print(f"  Final Distance: {info.get('distance_to_target', -1):.2f}m")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    arrival_rate = sum(r["arrived"] for r in results) / len(results) * 100
    avg_reward = np.mean([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_distance = np.mean([r["final_distance"] for r in results])

    print(f"Arrival Rate: {arrival_rate:.1f}%")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Final Distance: {avg_distance:.2f}m")

    env.close()


if __name__ == "__main__":
    main()
