import numpy as np
import gymnasium as gym
import docker
import zmq
import time
import struct
import os
import subprocess
import shutil
import concurrent.futures

from docker.types import NetworkingConfig, EndpointConfig, DeviceRequest


class AASEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
            instance: int=0,
            gym_freq_hz: int=50,
            autopilot: str="px4",
            camera: bool=True,
            lidar: bool=True,
            num_quads: int=1,
            render_mode=None
        ):
        super().__init__()

        self.GYM_FREQ_HZ = gym_freq_hz
        self.GYM_INIT_DURATION = 80.0  # Seconds to run unpaused during reset (seconds)
        self.MAX_EPISODE_LENGTH_SEC = 300.0  # Max episode length in seconds (excluding init duration)
        
        # [DUMMY] Action Space: [/action] between -1.0 and 1.0
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,), dtype=np.float32
        )
        # [DUMMY] Observation Space is the Gazebo Sim /clocl [seconds, nanoseconds]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float64),
            high=np.array([np.inf, 1e9], dtype=np.float64),
            dtype=np.float64
        )
        # Initialize storage for the clock
        self.sim_sec = 0.0
        self.sim_nanosec = 0.0

        self.max_steps = int(self.MAX_EPISODE_LENGTH_SEC*self.GYM_FREQ_HZ)  # Max steps per episode
        self.step_count = 0
        
        # Rendering
        self.render_mode = render_mode

        # AAS Setup
        self.HEADLESS = False if self.render_mode == "human" else True # Only display GUIs if render_mode is "human"
        self.AUTOPILOT = autopilot
        self.CAMERA = camera
        self.LIDAR = lidar
        self.NUM_QUADS = num_quads
        self.NUM_VTOLS = 0
        self.WORLD = "impalpable_greyness"
        #
        self.SIM_SUBNET = "10.42"
        self.AIR_SUBNET = "10.22"
        self.SIM_ID = "100"
        # self.GROUND_ID = "101" # Unused
        #
        self.GND_CONTAINER = False # Do NOT use the ground-image to run Zenoh (nor QGC)
        self.RTF = 15.0 # Note: RTFs > 10 can destabilize PX4/ArduPilot SITL
        self.START_AS_PAUSED = True # Start the simulation paused and manually step with gz-sim WorldControl
        self.INSTANCE = instance
        #
        sim_parts = self.SIM_SUBNET.split('.')
        self.SIM_SUBNET = f"{sim_parts[0]}.{int(sim_parts[1]) + self.INSTANCE}"
        # air_parts = self.AIR_SUBNET.split('.') # Unused
        # self.AIR_SUBNET = f"{air_parts[0]}.{int(air_parts[1]) + self.INSTANCE}" # Unused
        #
        self.SIM_NET_NAME = f"aas-sim-network-inst{self.INSTANCE}"
        # self.AIR_NET_NAME = f"aas-air-network-inst{self.INSTANCE}" # Unused
        self.SIM_CONT_NAME = f"simulation-container-inst{self.INSTANCE}"
        # self.GND_CONT_NAME = f"ground-container-inst{self.INSTANCE}" # Unused

        # X Server access (this is redundancy for configure_host_x11() in gym_run.py)
        if shutil.which("xhost"):
            try:
                current_acls = subprocess.check_output(["xhost"], text=True)
                if "LOCAL:" not in current_acls and "local:docker" not in current_acls:
                    print("Granting X Server access to Docker containers...")
                    subprocess.run(["xhost", "+local:docker"], check=True)
                    print("X Server access granted.")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not configure xhost: {e}")

        # Docker setup
        try:
            self.client = docker.from_env()
        except Exception as e:
            raise RuntimeError("Could not connect to the Docker daemon. Ensure Docker is running.") from e
        #
        def force_container_cleanup(name): # Needed to use AsyncVectorEnv
            try:
                old_container = self.client.containers.get(name)
                print(f"Found existing container '{name}'. Removing it...")
                old_container.remove(force=True)
                time.sleep(1.0)
            except docker.errors.NotFound:
                pass
            except Exception as e:
                print(f"Warning during cleanup of {name}: {e}")
        #
        networks_config = [
            {"name": self.SIM_NET_NAME, "subnet_base": self.SIM_SUBNET},
            # {"name": self.AIR_NET_NAME, "subnet_base": self.AIR_SUBNET} # Unused
        ]
        self.networks = {}
        for net_config in networks_config:
            net_name = net_config["name"]
            base_ip = net_config["subnet_base"]
            print(f"Setting up Docker Network: {net_name}...")
            try:
                existing_network = self.client.networks.get(net_name)
                existing_network.remove()
                print(f"Existing network '{net_name}' removed.")
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as e:
                print(f"Warning: Could not remove {net_name}: {e}")
            ipam_pool = docker.types.IPAMPool(
                subnet=f"{base_ip}.0.0/16",
                gateway=f"{base_ip}.0.1"
            )
            ipam_config = docker.types.IPAMConfig(
                pool_configs=[ipam_pool]
            )
            new_network = self.client.networks.create(
                net_name,
                driver="bridge",
                ipam=ipam_config
            )
            self.networks[net_name] = new_network
            print(f"Network '{net_name}' created on subnet {base_ip}.0.0/16")
        #
        env_display = os.environ.get('DISPLAY', '')
        env_xdg = os.environ.get('XDG_RUNTIME_DIR', '')
        gpu_requests = [
            DeviceRequest(count=-1, capabilities=[['gpu']]) # Replaces "--gpus all"
        ]
        self.ZMQ_IPC_SOCKET_DIR = '/tmp/aas_zmq_sockets'
        os.makedirs(self.ZMQ_IPC_SOCKET_DIR, exist_ok=True)
        volume_binds = {
            '/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'rw'}, # Replaces "--volume /tmp/.X11-unix:/tmp/.X11-unix:rw"
            self.ZMQ_IPC_SOCKET_DIR: {'bind': self.ZMQ_IPC_SOCKET_DIR, 'mode': 'rw'} # For ZMQ IPC sockets
        }
        device_binds = ['/dev/dri:/dev/dri:rwm'] # Replaces "--device /dev/dri"
        #
        force_container_cleanup(self.SIM_CONT_NAME)
        print(f"Creating Simulation Container ({self.SIM_CONT_NAME})...")
        self.simulation_container = self.client.containers.create(
            "simulation-image:latest",
            name=self.SIM_CONT_NAME,
            tty=True, # Replaces -it
            detach=True,
            auto_remove=False,
            privileged=True, # Replaces --privileged
            volumes=volume_binds,
            devices=device_binds,
            device_requests=gpu_requests,
            environment={
                "DISPLAY": env_display,
                "QT_X11_NO_MITSHM": "1",
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "XDG_RUNTIME_DIR": env_xdg,
                "GST_DEBUG": "3",
                "AUTOPILOT": self.AUTOPILOT,
                "HEADLESS": str(self.HEADLESS).lower(),
                "CAMERA": str(self.CAMERA).lower(),
                "LIDAR": str(self.LIDAR).lower(),
                "NUM_QUADS": str(self.NUM_QUADS),
                "NUM_VTOLS": str(self.NUM_VTOLS),
                "WORLD": self.WORLD,
                "SIMULATED_TIME": "true",
                "RTF": str(self.RTF),
                "START_AS_PAUSED": str(self.START_AS_PAUSED).lower(),
                "SIM_SUBNET": self.SIM_SUBNET,
                # "GROUND_ID": self.GROUND_ID,
                "GND_CONTAINER": str(self.GND_CONTAINER).lower(),
                "ROS_DOMAIN_ID": self.SIM_ID,
                "GYMNASIUM" : "true",
                "GYM_FREQ_HZ" : str(self.GYM_FREQ_HZ),
                "GYM_INIT_DURATION" : str(self.GYM_INIT_DURATION),
                "INSTANCE": str(self.INSTANCE),
            }
        )
        print(f"Connecting {self.SIM_CONT_NAME} to {self.SIM_NET_NAME}...")
        self.networks[self.SIM_NET_NAME].connect(
            self.simulation_container,
            ipv4_address=f"{self.SIM_SUBNET}.90.{self.SIM_ID}"
        )
        # print(f"Connecting {self.SIM_CONT_NAME} to {self.AIR_NET_NAME}...")
        # self.networks[self.AIR_NET_NAME].connect(
        #     self.simulation_container,
        #     ipv4_address=f"{self.AIR_SUBNET}.90.{self.SIM_ID}"
        # )
        # self.simulation_container.start()
        #
        self.aircraft_containers = []
        for i in range(1, self.NUM_QUADS + self.NUM_VTOLS + 1):            
            air_cont_name = f"aircraft-container-inst{self.INSTANCE}_{i}"
            force_container_cleanup(air_cont_name)
            print(f"Creating Aircraft Container {air_cont_name}...")
            drone_type = "quad" if i <= self.NUM_QUADS else "vtol"
            air_cont = self.client.containers.create(
                "aircraft-image:latest",
                name=air_cont_name,
                tty=True, # Replaces -it
                detach=True,
                auto_remove=False,
                privileged=True, # Replaces --privileged
                volumes=volume_binds,
                devices=device_binds,
                device_requests=gpu_requests,
                environment={
                    "DISPLAY": env_display,
                    "QT_X11_NO_MITSHM": "1",
                    "NVIDIA_DRIVER_CAPABILITIES": "all",
                    "XDG_RUNTIME_DIR": env_xdg,
                    "GST_DEBUG": "3",
                    "AUTOPILOT": self.AUTOPILOT,
                    "HEADLESS": str(self.HEADLESS).lower(),
                    "CAMERA": str(self.CAMERA).lower(),
                    "LIDAR": str(self.LIDAR).lower(),
                    "DRONE_TYPE": drone_type,
                    "DRONE_ID": str(i),
                    "SIMULATED_TIME": "true",
                    "SIM_SUBNET": self.SIM_SUBNET,
                    # "AIR_SUBNET": self.AIR_SUBNET,
                    "SIM_ID": self.SIM_ID,
                    # "GROUND_ID": self.GROUND_ID,
                    "GND_CONTAINER": str(self.GND_CONTAINER).lower(),
                    "ROS_DOMAIN_ID": str(i),
                    "GYMNASIUM" : "true",
                }
            )
            print(f"Connecting {air_cont_name} to {self.SIM_NET_NAME}...")
            self.networks[self.SIM_NET_NAME].connect(
                air_cont,
                ipv4_address=f"{self.SIM_SUBNET}.90.{i}"
            )
            # print(f"Connecting {air_cont_name} to {self.AIR_NET_NAME}...")
            # self.networks[self.AIR_NET_NAME].connect(
            #     air_cont,
            #     ipv4_address=f"{self.AIR_SUBNET}.90.{i}"
            # )
            # air_cont.start()
            self.aircraft_containers.append(air_cont)
        print("Docker setup complete. All containers are running and connected.")

        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.ZMQ_TRANSPORT = "tcp" # "tcp" or "ipc", also uncomment the corresponding block in zero_bridge.cpp
        if self.ZMQ_TRANSPORT == "tcp":
            self.ZMQ_PORT = 5555
            self.ZMQ_IP = f"{self.SIM_SUBNET}.90.{self.SIM_ID}"

    def _get_obs(self):
        return np.array([self.sim_sec, self.sim_nanosec], dtype=np.float64)

    def _get_info(self):
        return {"sim_time_sec": self.sim_sec, "sim_time_nanosec": self.sim_nanosec}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Handle seeding

        # Close existing ZMQ connection if any
        if self.socket:
            self.socket.close()
        # Restart Docker containers
        try:
            print("Restarting all containers in parallel...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.simulation_container.restart)]
                for air_cont in self.aircraft_containers:
                    futures.append(executor.submit(air_cont.restart))
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        except Exception as e:
            print(f"Error restarting containers: {e}")
            raise e
        # Establish ZeroMQ connection
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 60 * 1000) # 1000 ms = 1 seconds, only a placeholder, will be changed during reset
        if self.ZMQ_TRANSPORT == "tcp":
            self.socket.connect(f"tcp://{self.ZMQ_IP}:{self.ZMQ_PORT}")
            print(f"ZeroMQ socket connected to {self.ZMQ_IP}:{self.ZMQ_PORT}")
        elif self.ZMQ_TRANSPORT == "ipc":
            ipc_file = f"{self.ZMQ_IPC_SOCKET_DIR}/bridge_inst{self.INSTANCE}.ipc"
            self.socket.connect(f"ipc://{ipc_file}")
            print(f"ZeroMQ socket connected via IPC: {ipc_file}")
        else:
            raise ValueError(f"Invalid ZMQ_TRANSPORT: {self.ZMQ_TRANSPORT}")
        ###########################################################################################
        # ZeroMQ REQ/REP to the ROS2 sim ##########################################################
        ###########################################################################################
        try:
            self.socket.setsockopt(zmq.RCVTIMEO, 300 * 1000) # Temporarily increase timeout to 300s to reset the simulation
            reset = 9999.0 # A special action to reset the environment
            action_payload = struct.pack('d', reset) # Serialize the action 
            self.socket.send(action_payload) # Send the REQ
            reply_bytes = self.socket.recv() # Wait for the REP (synchronous block) this call will block until a reply is received or it times out
            self.socket.setsockopt(zmq.RCVTIMEO, 60 * 1000) # Restore standard timeout (60s) for stepping
            unpacked = struct.unpack('iI', reply_bytes) # Deserialize: i = int32 (sec), I = uint32 (nanosec)
            self.sim_sec, self.sim_nanosec = unpacked
            self.start_sim_sec = float(self.sim_sec) + (float(self.sim_nanosec) * 1e-9)
        except zmq.error.Again:
            print("ZMQ Error: Reply from container timed out.")
        except ValueError:
            print("ZMQ Error: Reply format error. Received garbage state.")
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################
        self.step_count = 0
        
        if self.render_mode == "ansi":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        force = action[0]
        ###########################################################################################
        # ZeroMQ REQ/REP to the ROS2 sim ##########################################################
        ###########################################################################################
        try:
            action_payload = struct.pack('d', force) # Serialize the action
            self.socket.send(action_payload) # Send the REQ
            reply_bytes = self.socket.recv() # Wait for the REP (synchronous block) this call will block until a reply is received or it times out
            unpacked = struct.unpack('iI', reply_bytes) # Deserialize: i = int32 (sec), I = uint32 (nanosec)
            sec, nanosec = unpacked
            self.sim_sec, self.sim_nanosec = unpacked
        except zmq.error.Again:
            print("ZMQ Error: Reply from container timed out.")
        except ValueError:
            print("ZMQ Error: Reply format error. Received garbage state.")
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################
        self.step_count += 1
        # Calculate reward
        reward = float(-1.0)
        # Check for termination
        terminated = False  # This is a continuing task, never "terminates"
        # Check for truncation (episode ends due to time limit)
        truncated = self.step_count >= self.max_steps
        # Get obs and info
        obs = self._get_obs()
        info = self._get_info()
        
        # Handle rendering
        if self.render_mode == "ansi":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            self._render_frame()

    def _render_frame(self):
        bar_width = 40        
        current_abs_time = self.sim_sec + (self.sim_nanosec * 1e-9)
        start_time = getattr(self, 'start_sim_sec', 0.0)
        episode_time = current_abs_time - start_time
        progress = min(max(episode_time / self.MAX_EPISODE_LENGTH_SEC, 0.0), 1.0)
        filled_len = int(bar_width * progress)
        bar = '=' * filled_len + '-' * (bar_width - filled_len)
        print(f"\r[{bar}] {episode_time:6.2f}s / {self.MAX_EPISODE_LENGTH_SEC:.0f}s", end="")

    def close(self):
        if self.render_mode == "ansi":
            print() # Add a newline after the final render
        
        try:
            self.simulation_container.stop()
            self.simulation_container.remove(force=True)
            print(f"Simulation container '{self.simulation_container.name}' stopped.")
        except Exception:
            pass
        for container in self.aircraft_containers:
            try:
                container.stop()
                container.remove(force=True)
                print(f"Aircraft container '{container.name}' stopped.")
            except Exception:
                pass
        for net_name, network_obj in self.networks.items():
            try:
                network_obj.remove()
                print(f"Network {net_name} removed.")
            except Exception as e:
                print(f"Warning: Could not remove {net_name}: {e}")

        # Close ZMQ resources
        if self.socket:
            self.socket.close(linger=0)
        if self.zmq_context:
            self.zmq_context.term()


class AASVelocityEnv(AASEnv):
    """
    Aerial Autonomy Stack Gym Environment with Velocity Control.

    This environment extends AASEnv to provide:
    - Action Space: Velocity commands [vx, vy, vz, yaw_rate] in m/s and rad/s
    - Observation Space: Drone state [x, y, z, vx, vy, vz, qw, qx, qy, qz]

    The environment communicates with:
    1. Simulation container (via ZMQ) for world stepping
    2. Aircraft container (via ZMQ) for velocity control and state feedback

    Coordinate Frame:
    - ArduPilot: ENU (East-North-Up) - vx=East, vy=North, vz=Up
    - PX4: NED (North-East-Down) - vx=North, vy=East, vz=Down (handled internally)
    """

    # ZMQ message format constants for aircraft control
    ACTION_FORMAT = '4d'  # 4 doubles: vx, vy, vz, yaw_rate
    ACTION_SIZE = struct.calcsize(ACTION_FORMAT)
    STATE_FORMAT = '10d'  # 10 doubles: x, y, z, vx, vy, vz, qw, qx, qy, qz
    STATE_SIZE = struct.calcsize(STATE_FORMAT)

    def __init__(self,
            instance: int = 0,
            gym_freq_hz: int = 50,
            autopilot: str = "ardupilot",  # Default to ArduPilot for velocity control
            camera: bool = False,  # Disable camera by default for faster training
            lidar: bool = False,   # Disable lidar by default for faster training
            num_quads: int = 1,
            render_mode = None,
            max_velocity: float = 10.0,  # Maximum velocity in m/s
            max_yaw_rate: float = 1.0,   # Maximum yaw rate in rad/s
            target_position: np.ndarray = None,  # Optional target for reward calculation
        ):
        """
        Initialize the velocity-controlled drone environment.

        Args:
            instance: Environment instance ID (for parallel environments)
            gym_freq_hz: Control frequency in Hz
            autopilot: Autopilot type ("ardupilot" or "px4")
            camera: Enable camera sensor
            lidar: Enable lidar sensor
            num_quads: Number of quadcopter drones
            render_mode: Rendering mode ("human", "ansi", or None)
            max_velocity: Maximum velocity command magnitude in m/s
            max_yaw_rate: Maximum yaw rate command in rad/s
            target_position: Optional target position [x, y, z] for reward calculation
        """
        # Initialize parent class (handles Docker, networks, simulation ZMQ)
        super().__init__(
            instance=instance,
            gym_freq_hz=gym_freq_hz,
            autopilot=autopilot,
            camera=camera,
            lidar=lidar,
            num_quads=num_quads,
            render_mode=render_mode
        )

        self.max_velocity = max_velocity
        self.max_yaw_rate = max_yaw_rate
        self.target_position = target_position if target_position is not None else np.array([100.0, 0.0, -50.0])

        # Override Action Space: [vx, vy, vz, yaw_rate] normalized to [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Override Observation Space: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        # Position bounds are large to accommodate flight area
        # Velocity bounds match max_velocity
        # Quaternion components are bounded [-1, 1]
        obs_low = np.array([
            -1000.0, -1000.0, -1000.0,  # Position (m)
            -max_velocity, -max_velocity, -max_velocity,  # Velocity (m/s)
            -1.0, -1.0, -1.0, -1.0  # Quaternion
        ], dtype=np.float64)
        obs_high = np.array([
            1000.0, 1000.0, 1000.0,  # Position (m)
            max_velocity, max_velocity, max_velocity,  # Velocity (m/s)
            1.0, 1.0, 1.0, 1.0  # Quaternion
        ], dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float64
        )

        # Drone state storage
        self.drone_position = np.zeros(3)
        self.drone_velocity = np.zeros(3)
        self.drone_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        # Aircraft ZMQ setup (separate from simulation ZMQ)
        self.aircraft_zmq_port = 5556  # Port for gym_control_node.py
        self.aircraft_socket = None

    def _get_obs(self):
        """Get current observation (drone state)."""
        return np.concatenate([
            self.drone_position,
            self.drone_velocity,
            self.drone_orientation
        ]).astype(np.float64)

    def _get_info(self):
        """Get additional info dict."""
        return {
            "sim_time_sec": self.sim_sec,
            "sim_time_nanosec": self.sim_nanosec,
            "position": self.drone_position.copy(),
            "velocity": self.drone_velocity.copy(),
            "orientation": self.drone_orientation.copy(),
            "distance_to_target": np.linalg.norm(self.drone_position - self.target_position)
        }

    def _connect_aircraft_zmq(self):
        """Connect to aircraft container ZMQ socket."""
        if self.aircraft_socket is not None:
            try:
                self.aircraft_socket.close()
            except Exception:
                pass

        self.aircraft_socket = self.zmq_context.socket(zmq.REQ)
        self.aircraft_socket.setsockopt(zmq.RCVTIMEO, 10 * 1000)  # 10 second timeout
        self.aircraft_socket.setsockopt(zmq.SNDTIMEO, 10 * 1000)

        # Connect to first aircraft container
        # Aircraft IP is at SIM_SUBNET.90.1 (drone ID 1)
        aircraft_ip = f"{self.SIM_SUBNET}.90.1"
        self.aircraft_socket.connect(f"tcp://{aircraft_ip}:{self.aircraft_zmq_port}")
        print(f"Aircraft ZMQ socket connected to {aircraft_ip}:{self.aircraft_zmq_port}")

    def _send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float) -> bool:
        """
        Send velocity command to aircraft and receive state.

        Args:
            vx, vy, vz: Velocity commands in m/s
            yaw_rate: Yaw rate command in rad/s

        Returns:
            True if successful, False otherwise
        """
        try:
            # Pack and send action
            action_payload = struct.pack(self.ACTION_FORMAT, vx, vy, vz, yaw_rate)
            self.aircraft_socket.send(action_payload)

            # Receive state
            reply_bytes = self.aircraft_socket.recv()

            if len(reply_bytes) == self.STATE_SIZE:
                state = struct.unpack(self.STATE_FORMAT, reply_bytes)
                self.drone_position = np.array(state[0:3])
                self.drone_velocity = np.array(state[3:6])
                self.drone_orientation = np.array(state[6:10])
                return True
            else:
                print(f"Warning: Invalid state size received: {len(reply_bytes)}")
                return False

        except zmq.error.Again:
            print("Aircraft ZMQ Error: Timeout waiting for state - reconnecting...")
            self._reconnect_aircraft_zmq()
            return False
        except zmq.error.ZMQError as e:
            print(f"Aircraft ZMQ Error: {e} - reconnecting...")
            self._reconnect_aircraft_zmq()
            return False
        except struct.error as e:
            print(f"Aircraft ZMQ Error: Struct error: {e}")
            return False
        except Exception as e:
            print(f"Aircraft ZMQ Error: {e}")
            return False

    def _reconnect_aircraft_zmq(self):
        """Reconnect the aircraft ZMQ socket after an error."""
        try:
            if self.aircraft_socket is not None:
                self.aircraft_socket.close(linger=0)
        except Exception:
            pass

        # Small delay before reconnecting
        time.sleep(0.5)

        # Recreate socket
        self.aircraft_socket = self.zmq_context.socket(zmq.REQ)
        self.aircraft_socket.setsockopt(zmq.RCVTIMEO, 10 * 1000)
        self.aircraft_socket.setsockopt(zmq.SNDTIMEO, 10 * 1000)
        self.aircraft_socket.setsockopt(zmq.LINGER, 0)

        aircraft_ip = f"{self.SIM_SUBNET}.90.1"
        self.aircraft_socket.connect(f"tcp://{aircraft_ip}:{self.aircraft_zmq_port}")
        print(f"Aircraft ZMQ socket reconnected to {aircraft_ip}:{self.aircraft_zmq_port}")

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # Call parent reset (handles simulation reset and container restart)
        obs, info = super().reset(seed=seed, options=options)

        # Wait a bit for aircraft container to fully initialize
        time.sleep(2.0)

        # Connect to aircraft ZMQ
        self._connect_aircraft_zmq()

        # Wait for gym_control_node to be ready and get initial state
        max_retries = 30
        for i in range(max_retries):
            try:
                # Send a "get state" request (zero velocity)
                if self._send_velocity_command(0.0, 0.0, 0.0, 0.0):
                    print(f"Aircraft ZMQ connected. Initial position: {self.drone_position}")
                    break
            except Exception as e:
                if i < max_retries - 1:
                    print(f"Waiting for aircraft ZMQ... ({i+1}/{max_retries})")
                    time.sleep(1.0)
                else:
                    print(f"Warning: Could not connect to aircraft ZMQ after {max_retries} retries")

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: Normalized velocity command [vx, vy, vz, yaw_rate] in [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action from [-1, 1] to actual velocity
        vx = float(action[0]) * self.max_velocity
        vy = float(action[1]) * self.max_velocity
        vz = float(action[2]) * self.max_velocity
        yaw_rate = float(action[3]) * self.max_yaw_rate

        # Step simulation (parent class handles this)
        # Send dummy action to simulation to advance time
        try:
            action_payload = struct.pack('d', 0.0)  # Dummy action for simulation stepping
            self.socket.send(action_payload)
            reply_bytes = self.socket.recv()
            unpacked = struct.unpack('iI', reply_bytes)
            self.sim_sec, self.sim_nanosec = unpacked
        except zmq.error.Again:
            print("Simulation ZMQ Error: Reply from container timed out.")
        except Exception as e:
            print(f"Simulation ZMQ Error: {e}")

        # Send velocity command to aircraft and get state
        self._send_velocity_command(vx, vy, vz, yaw_rate)

        self.step_count += 1

        # Calculate reward (customize this for your task)
        reward = self._calculate_reward(action)

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps

        # Get observation and info
        obs = self._get_obs()
        info = self._get_info()

        # Handle rendering
        if self.render_mode == "ansi":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action) -> float:
        """
        Calculate reward for the current step.

        Default implementation: negative distance to target + velocity bonus toward target.
        Override this method to implement custom reward functions.

        Args:
            action: The action taken

        Returns:
            Reward value
        """
        # Distance to target
        distance = np.linalg.norm(self.drone_position - self.target_position)

        # Direction to target
        direction_to_target = self.target_position - self.drone_position
        direction_norm = np.linalg.norm(direction_to_target)
        if direction_norm > 0.1:
            direction_to_target = direction_to_target / direction_norm
        else:
            direction_to_target = np.zeros(3)

        # Velocity toward target (dot product)
        velocity_toward_target = np.dot(self.drone_velocity, direction_to_target)

        # Reward components
        distance_reward = -0.01 * distance  # Penalize distance
        velocity_reward = 0.1 * velocity_toward_target  # Reward moving toward target

        # Action smoothness penalty (penalize large actions)
        action_penalty = -0.01 * np.sum(np.square(action))

        # Goal bonus
        goal_bonus = 10.0 if distance < 5.0 else 0.0

        reward = distance_reward + velocity_reward + action_penalty + goal_bonus

        return float(reward)

    def _check_terminated(self) -> bool:
        """
        Check if episode should terminate.

        Default: terminate if drone goes out of bounds or crashes.
        Override this method to implement custom termination conditions.

        Returns:
            True if episode should terminate
        """
        # Check position bounds
        if np.any(np.abs(self.drone_position) > 500.0):
            print("Episode terminated: Out of bounds")
            return True

        # Check if crashed (z too low for NED, too high for ENU)
        # This depends on coordinate frame - adjust as needed
        if self.AUTOPILOT == "ardupilot":
            # ENU: z is up, crash if z < 0 (below ground)
            if self.drone_position[2] < 0.5:
                print("Episode terminated: Crashed (z < 0.5)")
                return True
        else:
            # NED: z is down, crash if z > -0.5 (too close to ground)
            if self.drone_position[2] > -0.5:
                print("Episode terminated: Crashed (z > -0.5)")
                return True

        # Check if reached target
        distance = np.linalg.norm(self.drone_position - self.target_position)
        if distance < 2.0:
            print("Episode terminated: Reached target!")
            return True

        return False

    def close(self):
        """Clean up resources."""
        # Close aircraft ZMQ
        if self.aircraft_socket is not None:
            try:
                self.aircraft_socket.close(linger=0)
            except Exception:
                pass

        # Call parent close
        super().close()


class AASForwardFlightEnv(AASVelocityEnv):
    """
    Simplified environment for learning forward flight.

    This environment rewards the drone for moving forward (positive x direction)
    while maintaining altitude and orientation.

    Action Space: [forward_velocity, lateral_velocity, vertical_velocity, yaw_rate]
    Observation Space: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
    """

    def __init__(self, **kwargs):
        # Set defaults for forward flight training
        kwargs.setdefault('autopilot', 'ardupilot')
        kwargs.setdefault('max_velocity', 5.0)  # Lower max velocity for stability
        kwargs.setdefault('max_yaw_rate', 0.5)
        kwargs.setdefault('camera', False)
        kwargs.setdefault('lidar', False)

        super().__init__(**kwargs)

        # Target altitude (ENU frame, z is up)
        self.target_altitude = 40.0  # meters
        self.initial_position = None

    def reset(self, seed=None, options=None):
        """Reset and record initial position."""
        obs, info = super().reset(seed=seed, options=options)
        self.initial_position = self.drone_position.copy()
        return obs, info

    def _calculate_reward(self, action) -> float:
        """
        Reward function for forward flight.

        Rewards:
        - Forward velocity (positive x)
        - Maintaining target altitude
        - Staying close to initial y position (no drift)
        - Smooth actions
        """
        # Forward progress reward
        forward_velocity = self.drone_velocity[0]  # vx in ENU
        forward_reward = 0.5 * forward_velocity  # Reward forward motion

        # Altitude maintenance reward (ENU: z is up)
        altitude_error = abs(self.drone_position[2] - self.target_altitude)
        altitude_reward = -0.1 * altitude_error

        # Lateral drift penalty
        if self.initial_position is not None:
            lateral_drift = abs(self.drone_position[1] - self.initial_position[1])
            drift_penalty = -0.05 * lateral_drift
        else:
            drift_penalty = 0.0

        # Action smoothness penalty
        action_penalty = -0.01 * np.sum(np.square(action))

        # Survival bonus (small positive reward for staying alive)
        survival_bonus = 0.1

        reward = forward_reward + altitude_reward + drift_penalty + action_penalty + survival_bonus

        return float(reward)

    def _check_terminated(self) -> bool:
        """Check termination for forward flight."""
        # Check altitude bounds (ENU)
        if self.drone_position[2] < 5.0:  # Too low
            print("Episode terminated: Too low")
            return True
        if self.drone_position[2] > 100.0:  # Too high
            print("Episode terminated: Too high")
            return True

        # Check lateral drift
        if self.initial_position is not None:
            lateral_drift = abs(self.drone_position[1] - self.initial_position[1])
            if lateral_drift > 50.0:
                print("Episode terminated: Too much lateral drift")
                return True

        # Check if drone has traveled far enough (success condition)
        if self.initial_position is not None:
            forward_distance = self.drone_position[0] - self.initial_position[0]
            if forward_distance > 200.0:
                print("Episode terminated: Reached forward distance goal!")
                return True

        return False


class AASSimpleCommandEnv(AASVelocityEnv):
    """
    Simplified environment with discrete movement commands.

    The drone automatically takes off to a set altitude and hovers.
    Then it accepts simple discrete commands to move at a constant velocity.

    Action Space (Discrete):
        0: HOVER - Stay in place (zero velocity)
        1: FORWARD - Move forward (positive X in ENU)
        2: LEFT - Move left (positive Y in ENU)
        3: RIGHT - Move right (negative Y in ENU)
        4: BACKWARD - Move backward (negative X in ENU)

    The drone maintains altitude automatically while moving.

    Observation Space: [x, y, z, vx, vy, vz, qw, qx, qy, qz]

    Use reset() to restart the episode (drone returns to start and takes off again).
    Use close() to quit and clean up resources.
    """

    # Action constants
    ACTION_HOVER = 0
    ACTION_FORWARD = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_BACKWARD = 4

    ACTION_NAMES = {
        0: "HOVER",
        1: "FORWARD",
        2: "LEFT",
        3: "RIGHT",
        4: "BACKWARD"
    }

    def __init__(self,
            instance: int = 0,
            gym_freq_hz: int = 10,  # Lower frequency for manual control
            autopilot: str = "ardupilot",
            camera: bool = False,
            lidar: bool = False,
            num_quads: int = 1,
            render_mode = "ansi",  # Default to ANSI rendering for status display
            move_velocity: float = 2.0,  # Constant velocity in m/s
            takeoff_altitude: float = 40.0,  # Altitude to take off and hover at
            altitude_tolerance: float = 3.0,  # Tolerance for altitude check
        ):
        """
        Initialize the simple command drone environment.

        Args:
            instance: Environment instance ID
            gym_freq_hz: Control frequency (lower for manual control)
            autopilot: Autopilot type (default: ardupilot)
            camera: Enable camera (default: False for faster startup)
            lidar: Enable lidar (default: False for faster startup)
            num_quads: Number of drones (default: 1)
            render_mode: Rendering mode ("ansi" shows status, "human" for GUI)
            move_velocity: Constant velocity for movement commands in m/s
            takeoff_altitude: Target altitude for takeoff and hover
            altitude_tolerance: Tolerance for altitude maintenance
        """
        # Initialize parent with velocity control capabilities
        super().__init__(
            instance=instance,
            gym_freq_hz=gym_freq_hz,
            autopilot=autopilot,
            camera=camera,
            lidar=lidar,
            num_quads=num_quads,
            render_mode=render_mode,
            max_velocity=move_velocity,  # Use move_velocity as max
            max_yaw_rate=0.5,  # Fixed yaw rate
        )

        self.move_velocity = move_velocity
        self.takeoff_altitude = takeoff_altitude
        self.altitude_tolerance = altitude_tolerance

        # Override action space to be discrete
        self.action_space = gym.spaces.Discrete(5)

        # State tracking
        self.is_flying = False
        self.initial_position = None
        self.current_action = self.ACTION_HOVER
        self.last_action_name = "HOVER"

        # Reduce max episode length for interactive use
        self.MAX_EPISODE_LENGTH_SEC = 600.0  # 10 minutes
        self.max_steps = int(self.MAX_EPISODE_LENGTH_SEC * self.GYM_FREQ_HZ)

    def _wait_for_takeoff_complete(self, timeout: float = 120.0) -> bool:
        """
        Wait for the drone to reach takeoff altitude.
        The gymnasium_setup.py handles the actual takeoff command.
        This method monitors until the drone reaches altitude.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if takeoff completed, False if timeout
        """
        print(f"\nWaiting for drone to reach altitude {self.takeoff_altitude}m...")
        print("(The gymnasium_setup node handles takeoff automatically)")

        start_time = time.time()
        last_print_time = 0

        while (time.time() - start_time) < timeout:
            # Get current state
            try:
                self._send_velocity_command(0.0, 0.0, 0.0, 0.0)
            except Exception:
                time.sleep(0.5)
                continue

            current_altitude = self.drone_position[2]  # ENU: z is up

            # Print progress every 5 seconds
            elapsed = int(time.time() - start_time)
            if elapsed > last_print_time and elapsed % 5 == 0:
                print(f"  Current altitude: {current_altitude:.1f}m / {self.takeoff_altitude}m (elapsed: {elapsed}s)")
                last_print_time = elapsed

            # Check if we've reached altitude
            if current_altitude >= (self.takeoff_altitude - self.altitude_tolerance):
                print(f"\nTakeoff complete! Altitude: {current_altitude:.1f}m")
                self.is_flying = True
                return True

            time.sleep(0.5)

        print(f"\nWarning: Takeoff timeout. Current altitude: {self.drone_position[2]:.1f}m")
        return False

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        This will:
        1. Restart containers (handled by parent)
        2. Wait for automatic takeoff to complete
        3. Return initial observation
        """
        print("\n" + "="*60)
        print("RESETTING ENVIRONMENT")
        print("="*60)

        # Call parent reset (restarts containers, connects ZMQ)
        obs, info = super().reset(seed=seed, options=options)

        # Wait for takeoff to complete
        self._wait_for_takeoff_complete(timeout=120.0)

        # Record initial position
        self.initial_position = self.drone_position.copy()
        self.current_action = self.ACTION_HOVER
        self.last_action_name = "HOVER"

        print("\n" + "="*60)
        print("ENVIRONMENT READY")
        print(f"Position: x={self.drone_position[0]:.1f}, y={self.drone_position[1]:.1f}, z={self.drone_position[2]:.1f}")
        print(f"Commands: HOVER(0), FORWARD(1), LEFT(2), RIGHT(3), BACKWARD(4)")
        print("="*60 + "\n")

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """
        Execute one step with a discrete action.

        Args:
            action: Discrete action (0=HOVER, 1=FORWARD, 2=LEFT, 3=RIGHT, 4=BACKWARD)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert discrete action to velocity command
        self.current_action = action
        self.last_action_name = self.ACTION_NAMES.get(action, "UNKNOWN")

        # Map discrete action to velocity
        if action == self.ACTION_HOVER:
            vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
        elif action == self.ACTION_FORWARD:
            vx, vy, vz, yaw_rate = self.move_velocity, 0.0, 0.0, 0.0
        elif action == self.ACTION_LEFT:
            vx, vy, vz, yaw_rate = 0.0, self.move_velocity, 0.0, 0.0
        elif action == self.ACTION_RIGHT:
            vx, vy, vz, yaw_rate = 0.0, -self.move_velocity, 0.0, 0.0
        elif action == self.ACTION_BACKWARD:
            vx, vy, vz, yaw_rate = -self.move_velocity, 0.0, 0.0, 0.0
        else:
            print(f"Warning: Unknown action {action}, defaulting to HOVER")
            vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0

        # Add altitude hold - maintain takeoff altitude
        altitude_error = self.takeoff_altitude - self.drone_position[2]
        # Simple P controller for altitude
        vz = np.clip(altitude_error * 0.5, -2.0, 2.0)

        # Step simulation (parent class handles this)
        try:
            action_payload = struct.pack('d', 0.0)
            self.socket.send(action_payload)
            reply_bytes = self.socket.recv()
            unpacked = struct.unpack('iI', reply_bytes)
            self.sim_sec, self.sim_nanosec = unpacked
        except zmq.error.Again:
            print("Simulation ZMQ Error: Timeout")
        except Exception as e:
            print(f"Simulation ZMQ Error: {e}")

        # Send velocity command to aircraft
        self._send_velocity_command(vx, vy, vz, yaw_rate)

        self.step_count += 1

        # Simple reward: distance traveled from start
        reward = self._calculate_reward(action)

        # Check termination
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info['action_name'] = self.last_action_name

        if self.render_mode == "ansi":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action) -> float:
        """Simple reward: just staying alive and maintaining altitude."""
        # Altitude maintenance reward
        altitude_error = abs(self.drone_position[2] - self.takeoff_altitude)
        altitude_reward = -0.1 * altitude_error

        # Survival bonus
        survival_bonus = 0.1

        return float(altitude_reward + survival_bonus)

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Check altitude bounds (ENU: z is up)
        if self.drone_position[2] < 5.0:
            print("\nEpisode terminated: Altitude too low (crashed)")
            return True
        if self.drone_position[2] > 100.0:
            print("\nEpisode terminated: Altitude too high")
            return True

        # Check if too far from start
        if self.initial_position is not None:
            distance_from_start = np.linalg.norm(
                self.drone_position[:2] - self.initial_position[:2]
            )
            if distance_from_start > 500.0:
                print("\nEpisode terminated: Too far from start position")
                return True

        return False

    def _render_frame(self):
        """Render current state to terminal."""
        # Clear line and print status
        pos = self.drone_position
        vel = self.drone_velocity

        status = (
            f"\r[{self.last_action_name:8s}] "
            f"Pos: ({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:6.1f}) | "
            f"Vel: ({vel[0]:5.1f}, {vel[1]:5.1f}, {vel[2]:5.1f}) | "
            f"Step: {self.step_count}"
        )
        print(status, end="", flush=True)

    def _get_info(self):
        """Get info dict with additional simple command info."""
        info = super()._get_info()
        info['action'] = self.current_action
        info['action_name'] = self.last_action_name
        info['is_flying'] = self.is_flying
        info['takeoff_altitude'] = self.takeoff_altitude
        return info
