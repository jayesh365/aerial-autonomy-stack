import numpy as np
import gymnasium as gym
import argparse
import time
import itertools
import subprocess
import shutil

from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from aas_gym.aas_env import AASEnv, AASVelocityEnv, AASForwardFlightEnv, AASSimpleCommandEnv


# Register the environments so we can create them with gym.make()
gym.register(
    id="AASEnv-v0",
    entry_point=AASEnv,
)

gym.register(
    id="AASVelocityEnv-v0",
    entry_point=AASVelocityEnv,
)

gym.register(
    id="AASForwardFlightEnv-v0",
    entry_point=AASForwardFlightEnv,
)

gym.register(
    id="AASSimpleCommandEnv-v0",
    entry_point=AASSimpleCommandEnv,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="step", choices=[
        "step", "speedup", "vectorenv-speedup", "learn",
        "velocity-step", "velocity-train", "forward-flight",
        "simple-command"
    ])
    parser.add_argument("--repetitions", type=int, default=1),
    parser.add_argument("--autopilot", type=str, default="ardupilot", choices=["px4", "ardupilot"])
    parser.add_argument("--camera", action=argparse.BooleanOptionalAction, default=False, help="Enable/Disable Camera")
    parser.add_argument("--lidar", action=argparse.BooleanOptionalAction, default=False, help="Enable/Disable Lidar")
    parser.add_argument("--num_quads", type=int, default=1)
    parser.add_argument("--max_velocity", type=float, default=5.0, help="Maximum velocity in m/s")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps for training")
    args = parser.parse_args()

    if args.mode == "step":
        env = gym.make(
            "AASEnv-v0",
            gym_freq_hz=1,
            autopilot=args.autopilot,
            camera=args.camera,
            lidar=args.lidar,
            num_quads=args.num_quads,
            render_mode="human"
        )
        obs, info = env.reset()
        print(f"Reset result -- Obs: {obs}")
        for i in itertools.count():
            user_input = input("Press Enter to step, 'r' then Enter to reset, 'q' then Enter to exit...")
            stripped_input = user_input.strip().lower()
            if stripped_input and stripped_input in ('q', 'quit'):
                break
            if stripped_input and stripped_input in ('r', 'reset'):
                obs, info = env.reset()
                print(f"\nReset result -- Obs: {obs}")
            else:
                rnd_action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(rnd_action)
                print(f"\nStep {i} -- action: {rnd_action} result -- Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        print("\nClosing environment.")
        env.close()

    elif args.mode == "speedup":
        REPETITIONS = args.repetitions
        CTRL_FREQ_HZ = 50
        env = gym.make(
            "AASEnv-v0",
            instance=1,
            gym_freq_hz=CTRL_FREQ_HZ,
            autopilot=args.autopilot,
            camera=args.camera,
            lidar=args.lidar,
            num_quads=args.num_quads,
            render_mode="ansi" # "ansi" for progress bar, "human" for GUI
        )
        TIME_TO_SIMULATE_SEC = 250
        STEPS = TIME_TO_SIMULATE_SEC * CTRL_FREQ_HZ
        print(f"Starting speed test: {REPETITIONS} runs of {STEPS} steps each.")
        run_times = []
        for i in range(REPETITIONS):
            obs, info = env.reset()
            start_time = time.time()
            for _ in range(STEPS):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()
            duration = time.time() - start_time
            run_times.append(duration)
        avg_time = np.mean(run_times)
        std_time = np.std(run_times)
        all_speedups = [TIME_TO_SIMULATE_SEC / t for t in run_times]
        avg_speedup = np.mean(all_speedups)
        std_speedup = np.std(all_speedups)
        all_throughputs = [STEPS / t for t in run_times]
        avg_throughput = np.mean(all_throughputs)
        std_throughput = np.std(all_throughputs)
        print(f"\nAvg Duration:       {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"Avg Step Time:      {(avg_time / STEPS) * 1000:.3f} ms")
        print(f"Avg Speedup:        {avg_speedup:.2f}x ± {std_speedup:.2f}x wall-clock")
        print(f"Avg Throughput:     {avg_throughput:.2f} ± {std_throughput:.2f} steps/second")
        env.close()

    elif args.mode == "vectorenv-speedup":
        REPETITIONS = args.repetitions
        NUM_ENVS = 2 # Number of parallel environments (adjust based on CPU/RAM and GPU/VRAM usage, check with htop and nvidia-smi)
        CTRL_FREQ_HZ = 50
        TIME_TO_SIMULATE_SEC = 250
        STEPS_PER_ENV = TIME_TO_SIMULATE_SEC * CTRL_FREQ_HZ 
        print(f"Starting parallel speed test: {REPETITIONS} runs with {NUM_ENVS} envs, stepping each for {STEPS_PER_ENV} steps")
        def make_env(rank, freq_hz):
            def _init():
                return gym.make(
                    "AASEnv-v0",
                    instance=rank,
                    gym_freq_hz=freq_hz,
                    autopilot=args.autopilot,
                    camera=args.camera,
                    lidar=args.lidar,
                    num_quads=args.num_quads,
                    render_mode=None
                )
            return _init
        env_fns = [make_env(i, CTRL_FREQ_HZ) for i in range(NUM_ENVS)]
        envs = gym.vector.AsyncVectorEnv(env_fns)
        print(f"Running the test with render_mode=None")
        run_times = []
        for i in range(REPETITIONS):
            obs, info = envs.reset()
            start_time = time.time()
            for _ in range(STEPS_PER_ENV):
                actions = envs.action_space.sample() # Returns array of shape (NUM_ENVS, action_dim)
                obs, rewards, terminateds, truncateds, infos = envs.step(actions) # AsyncVectorEnv automatically resets individual envs when they terminate/truncate
            duration = time.time() - start_time
            run_times.append(duration)
        avg_time = np.mean(run_times)
        std_time = np.std(run_times)
        all_speedups = [(TIME_TO_SIMULATE_SEC * NUM_ENVS) / t for t in run_times]
        avg_speedup = np.mean(all_speedups)
        std_speedup = np.std(all_speedups)
        all_throughputs = [(STEPS_PER_ENV * NUM_ENVS) / t for t in run_times]
        avg_throughput = np.mean(all_throughputs)
        std_throughput = np.std(all_throughputs)
        print(f"\nAvg Duration:       {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"Avg Speedup:        {avg_speedup:.2f}x ± {std_speedup:.2f}x wall-clock (aggregate)")
        print(f"Avg Throughput:     {avg_throughput:.2f} ± {std_throughput:.2f} steps/second")
        envs.close()

    elif args.mode == "learn":
        print(f"TODO")
        # env = gym.make("AASEnv-v0")
        # try:
        #     # check_env(env) # Throws warning
        #     # check_env(env.unwrapped)
        #     sb3_check_env(env)
        #     print("\nEnvironment passes all checks!")
        # except Exception as e:
        #     print(f"\nEnvironment has issues: {e}")

        # env.reset()
        # env.step(env.action_space.sample())
        # env.reset()

        # # Instantiate the agent
        # model = PPO("MlpPolicy", env, verbose=1, device='cpu')

        # # Train the agent
        # print("Training agent...")
        # model.learn(total_timesteps=40000)
        # print("Training complete.")

        # # Save the agent
        # model_path = "ppo_agent.zip"
        # model.save(model_path)
        # print(f"Model saved to {model_path}")

        # # Load and test the trained agent
        # del model # remove to demonstrate loading
        # model = PPO.load(model_path, device='cpu')

        # print("\nTesting trained agent...")
        # obs, info = env.reset()
        # for _ in range(800): # Run for 800 steps
        #     action, _states = model.predict(obs, deterministic=True)
        #     obs, reward, terminated, truncated, info = env.step(action)

        #     if terminated or truncated:
        #         print("Episode finished. Resetting.")
        #         obs, info = env.reset()

        # env.close()

    elif args.mode == "velocity-step":
        """Manual stepping with velocity control - useful for testing."""
        print("=== Velocity Control Manual Stepping Mode ===")
        print(f"Autopilot: {args.autopilot}, Max Velocity: {args.max_velocity} m/s")

        env = gym.make(
            "AASVelocityEnv-v0",
            gym_freq_hz=10,  # Lower frequency for manual control
            autopilot=args.autopilot,
            camera=args.camera,
            lidar=args.lidar,
            num_quads=args.num_quads,
            max_velocity=args.max_velocity,
            render_mode="human"
        )

        obs, info = env.reset()
        print(f"\nReset complete!")
        print(f"Initial position: {info['position']}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        print("\n--- Controls ---")
        print("Enter velocity commands as: vx vy vz yaw_rate (normalized -1 to 1)")
        print("Examples:")
        print("  '0.5 0 0 0' - Move forward at 50% max velocity")
        print("  '0 0.3 0 0' - Move right at 30% max velocity")
        print("  '0 0 0.2 0' - Move up at 20% max velocity")
        print("  'r' - Reset environment")
        print("  'q' - Quit")
        print("  '' (empty) - Random action")

        for i in itertools.count():
            user_input = input("\nAction [vx vy vz yaw] or r/q: ").strip().lower()

            if user_input in ('q', 'quit'):
                break
            elif user_input in ('r', 'reset'):
                obs, info = env.reset()
                print(f"Reset! Position: {info['position']}")
                continue
            elif user_input == '':
                action = env.action_space.sample()
            else:
                try:
                    parts = user_input.split()
                    if len(parts) == 4:
                        action = np.array([float(p) for p in parts], dtype=np.float32)
                        action = np.clip(action, -1.0, 1.0)
                    else:
                        print("Invalid input. Enter 4 values or press Enter for random.")
                        continue
                except ValueError:
                    print("Invalid input. Enter numeric values.")
                    continue

            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {i}:")
            print(f"  Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, {action[3]:.2f}]")
            print(f"  Position: [{info['position'][0]:.1f}, {info['position'][1]:.1f}, {info['position'][2]:.1f}]")
            print(f"  Velocity: [{info['velocity'][0]:.2f}, {info['velocity'][1]:.2f}, {info['velocity'][2]:.2f}]")
            print(f"  Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}")

            if terminated or truncated:
                print("\nEpisode ended! Press Enter to reset or 'q' to quit.")
                if input().strip().lower() == 'q':
                    break
                obs, info = env.reset()
                print(f"Reset! Position: {info['position']}")

        print("\nClosing environment.")
        env.close()

    elif args.mode == "velocity-train":
        """Train a PPO agent with velocity control."""
        print("=== Velocity Control PPO Training Mode ===")
        print(f"Autopilot: {args.autopilot}, Max Velocity: {args.max_velocity} m/s")
        print(f"Total timesteps: {args.timesteps}")

        env = gym.make(
            "AASVelocityEnv-v0",
            gym_freq_hz=50,
            autopilot=args.autopilot,
            camera=args.camera,
            lidar=args.lidar,
            num_quads=args.num_quads,
            max_velocity=args.max_velocity,
            render_mode=None
        )

        # Validate environment
        print("\nValidating environment...")
        try:
            sb3_check_env(env)
            print("Environment passes all checks!")
        except Exception as e:
            print(f"Warning: Environment has issues: {e}")

        # Test reset and step
        print("\nTesting reset and step...")
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Initial position: {info['position']}")

        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        print(f"Step successful. Reward: {reward:.3f}")

        # Create PPO agent
        print("\nCreating PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device='cpu',
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

        # Train
        print(f"\nTraining for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps)
        print("Training complete!")

        # Save model
        model_path = "ppo_velocity_agent.zip"
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # Test trained agent
        print("\nTesting trained agent...")
        obs, info = env.reset()
        total_reward = 0
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 50 == 0:
                print(f"Step {step}: pos={info['position']}, reward={reward:.2f}")

            if terminated or truncated:
                print(f"Episode ended at step {step}. Total reward: {total_reward:.2f}")
                break

        env.close()
        print("Done!")

    elif args.mode == "forward-flight":
        """Train forward flight behavior."""
        print("=== Forward Flight Training Mode ===")
        print(f"Autopilot: {args.autopilot}")
        print(f"Total timesteps: {args.timesteps}")

        env = gym.make(
            "AASForwardFlightEnv-v0",
            gym_freq_hz=50,
            autopilot=args.autopilot,
            camera=False,
            lidar=False,
            num_quads=1,
            max_velocity=args.max_velocity,
            render_mode=None
        )

        # Validate environment
        print("\nValidating environment...")
        try:
            sb3_check_env(env)
            print("Environment passes all checks!")
        except Exception as e:
            print(f"Warning: Environment has issues: {e}")

        # Create PPO agent
        print("\nCreating PPO agent for forward flight...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device='cpu',
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

        # Train
        print(f"\nTraining forward flight for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps)
        print("Training complete!")

        # Save model
        model_path = "ppo_forward_flight.zip"
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # Test trained agent
        print("\nTesting trained forward flight agent...")
        obs, info = env.reset()
        total_reward = 0
        initial_x = info['position'][0]

        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 50 == 0:
                forward_dist = info['position'][0] - initial_x
                print(f"Step {step}: forward={forward_dist:.1f}m, alt={info['position'][2]:.1f}m, reward={reward:.2f}")

            if terminated or truncated:
                forward_dist = info['position'][0] - initial_x
                print(f"Episode ended at step {step}.")
                print(f"Total forward distance: {forward_dist:.1f}m")
                print(f"Total reward: {total_reward:.2f}")
                break

        env.close()
        print("Done!")

    elif args.mode == "simple-command":
        """Interactive drone control with simple discrete commands."""
        print("=" * 60)
        print("=== SIMPLE COMMAND DRONE CONTROL ===")
        print("=" * 60)
        print(f"Autopilot: {args.autopilot}")
        print(f"Movement velocity: {args.max_velocity} m/s")
        print("\nThe drone will automatically take off and hover.")
        print("Then you can control it with simple commands.")
        print("=" * 60)

        env = gym.make(
            "AASSimpleCommandEnv-v0",
            gym_freq_hz=10,  # Lower frequency for interactive control
            autopilot=args.autopilot,
            camera=args.camera,
            lidar=args.lidar,
            num_quads=args.num_quads,
            move_velocity=args.max_velocity,
            takeoff_altitude=40.0,
            render_mode="human"  # Show Gazebo GUI
        )

        obs, info = env.reset()

        print("\n" + "=" * 60)
        print("CONTROLS:")
        print("  w / f - Move FORWARD")
        print("  a / l - Move LEFT")
        print("  d / r - Move RIGHT")
        print("  s / b - Move BACKWARD")
        print("  h / (space/empty) - HOVER (stop)")
        print("  reset - Reset environment (restart episode)")
        print("  q / quit - Quit and close")
        print("=" * 60)
        print("\nEnter a command and press Enter:")

        step_count = 0
        running = True

        while running:
            try:
                user_input = input("\nCommand> ").strip().lower()

                # Parse command
                if user_input in ('q', 'quit', 'exit'):
                    print("\nQuitting...")
                    running = False
                    break

                elif user_input in ('reset', 'restart'):
                    print("\nResetting environment...")
                    obs, info = env.reset()
                    step_count = 0
                    print("Reset complete! Ready for commands.")
                    continue

                elif user_input in ('w', 'f', 'forward'):
                    action = env.unwrapped.ACTION_FORWARD
                elif user_input in ('a', 'l', 'left'):
                    action = env.unwrapped.ACTION_LEFT
                elif user_input in ('d', 'r', 'right'):
                    action = env.unwrapped.ACTION_RIGHT
                elif user_input in ('s', 'b', 'back', 'backward'):
                    action = env.unwrapped.ACTION_BACKWARD
                elif user_input in ('h', 'hover', 'stop', ''):
                    action = env.unwrapped.ACTION_HOVER
                else:
                    # Try to parse as a number
                    try:
                        action = int(user_input)
                        if action < 0 or action > 4:
                            print("Invalid action number. Use 0-4.")
                            continue
                    except ValueError:
                        print(f"Unknown command: '{user_input}'")
                        print("Use: w/f=forward, a/l=left, d/r=right, s/b=backward, h=hover, reset, q=quit")
                        continue

                # Execute the action multiple times to make movement visible
                num_steps = 5  # Execute 5 steps per command for noticeable movement
                total_reward = 0

                for _ in range(num_steps):
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    step_count += 1

                    if terminated or truncated:
                        break

                # Print status
                pos = info['position']
                vel = info['velocity']
                print(f"[{info['action_name']:8s}] "
                      f"Pos: ({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:6.1f}) | "
                      f"Vel: ({vel[0]:5.1f}, {vel[1]:5.1f}, {vel[2]:5.1f}) | "
                      f"Reward: {total_reward:.2f}")

                if terminated:
                    print("\n*** Episode TERMINATED ***")
                    print("Enter 'reset' to restart or 'q' to quit.")
                elif truncated:
                    print("\n*** Episode TRUNCATED (time limit) ***")
                    print("Enter 'reset' to restart or 'q' to quit.")

            except KeyboardInterrupt:
                print("\n\nInterrupted! Closing...")
                running = False
                break

        print("\nClosing environment...")
        env.close()
        print("Done!")

    else:
        print(f"Unknown mode: {args.mode}")

def configure_host_x11():
    if not shutil.which("xhost"):
        print("Error: 'xhost' command not found. GUI rendering may fail.")
        return
    try: # Check if already configured
        output = subprocess.check_output(["xhost"], text=True)
        if "LOCAL:" in output or "local:docker" in output:
            return # Already configured, return silently
    except subprocess.CalledProcessError:
        pass
    print("Granting X Server access to Docker containers...")
    try:
        subprocess.run(["xhost", "+local:docker"], check=True)
        print("X Server access granted.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not configure xhost: {e}")

if __name__ == '__main__':
    configure_host_x11()
    main()
