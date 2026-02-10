#!/usr/bin/env python3
"""
Gym Control Node - A Python ROS2 node for RL-based drone velocity control.

This node runs in the aircraft container and provides:
- ZMQ REP socket for receiving velocity commands from the gym environment
- Subscribes to drone state topics (position, velocity, orientation)
- Sets GUIDED mode and publishes velocity commands to MAVROS (ArduPilot)
- Returns drone state observations to the gym environment

Communication Protocol:
- Receives: struct with [vx, vy, vz, yaw_rate] as 4 doubles (32 bytes)
- Sends: struct with [x, y, z, vx, vy, vz, qw, qx, qy, qz] as 10 doubles (80 bytes)
"""

import argparse
import struct
import threading
import time

import cv2
import numpy as np
import zmq

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Message types for ArduPilot (MAVROS)
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
from cv_bridge import CvBridge

# MAVROS services and messages
from mavros_msgs.srv import SetMode, CommandBool
from mavros_msgs.msg import State

# Message types for PX4
try:
    from px4_msgs.msg import VehicleLocalPosition, VehicleOdometry, TrajectorySetpoint, OffboardControlMode
    PX4_AVAILABLE = True
except ImportError:
    PX4_AVAILABLE = False


class GymControlNode(Node):
    """ROS2 Node that bridges gym environment with drone control."""

    # ZMQ message format constants
    ACTION_FORMAT = '4d'  # 4 doubles: vx, vy, vz, yaw_rate
    ACTION_SIZE = struct.calcsize(ACTION_FORMAT)
    STATE_FORMAT = '10d'  # 10 doubles: x, y, z, vx, vy, vz, qw, qx, qy, qz
    STATE_SIZE = struct.calcsize(STATE_FORMAT)

    # Special action value for reset/init
    RESET_SIGNAL = 9999.0

    def __init__(self, drone_id: str, autopilot: str, zmq_port: int = 5556):
        super().__init__('gym_control_node')

        self.drone_id = drone_id
        self.autopilot = autopilot.lower()
        self.zmq_port = zmq_port

        # State storage (thread-safe with lock)
        self.state_lock = threading.Lock()
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z (NED for PX4, ENU for ArduPilot)
        self.velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion w, x, y, z
        self.state_valid = False

        # Frame storage (JPEG-compressed, thread-safe)
        self.frame_lock = threading.Lock()
        self.latest_frame_jpeg = b''  # Empty bytes = no frame available
        self.bridge = CvBridge()

        # Drone state tracking
        self.drone_armed = False
        self.drone_mode = ""
        self.guided_mode_set = False
        self.mavros_connected = False

        # Check simulation time
        if self.get_parameter('use_sim_time').as_bool():
            self.get_logger().info("Simulation time is enabled.")
        else:
            self.get_logger().warn("Simulation time is disabled.")

        # QoS profile for subscribers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Callback group for parallel execution
        self.callback_group = ReentrantCallbackGroup()

        # Setup based on autopilot type
        if self.autopilot == "ardupilot":
            self._setup_ardupilot(qos_profile)
        elif self.autopilot == "px4":
            self._setup_px4(qos_profile)
        else:
            raise ValueError(f"Unsupported autopilot: {self.autopilot}")

        # Subscribe to raw YOLO frames (published without bounding boxes)
        self.frame_sub = self.create_subscription(
            Image,
            'yolo_frame',
            self._frame_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            ),
            callback_group=self.callback_group
        )

        # Create timer for periodic velocity publishing (needed to maintain GUIDED mode)
        self.last_velocity_cmd = (0.0, 0.0, 0.0, 0.0)
        self.velocity_publish_timer = self.create_timer(
            0.1,  # 10 Hz
            self._velocity_publish_callback,
            callback_group=self.callback_group
        )

        # Wait for MAVROS to be ready before starting ZMQ
        self.get_logger().info("Waiting for MAVROS state data...")
        self._wait_for_mavros_ready()

        # ZMQ setup (runs in separate thread)
        self.zmq_running = True
        self.zmq_thread = threading.Thread(target=self._zmq_listener, daemon=True)
        self.zmq_thread.start()

        self.get_logger().info(f"GymControlNode initialized for {autopilot} drone {drone_id}")
        self.get_logger().info(f"ZMQ REP socket listening on port {zmq_port}")

    def _setup_ardupilot(self, qos_profile):
        """Setup publishers and subscribers for ArduPilot (MAVROS)."""
        self.get_logger().info("Setting up ArduPilot (MAVROS) interface...")

        # Publisher for velocity commands
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            10
        )

        # Subscribers for state
        self.odom_sub = self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self._ardupilot_odom_callback,
            qos_profile,
            callback_group=self.callback_group
        )

        # MAVROS state subscriber
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self._mavros_state_callback,
            qos_profile,
            callback_group=self.callback_group
        )

        # Global position for reference
        self.global_pos_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self._ardupilot_global_callback,
            qos_profile,
            callback_group=self.callback_group
        )

        # Service clients for mode and arming
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')

    def _setup_px4(self, qos_profile):
        """Setup publishers and subscribers for PX4."""
        if not PX4_AVAILABLE:
            raise ImportError("px4_msgs not available. Install px4_msgs package.")

        self.get_logger().info("Setting up PX4 interface...")

        # Publishers
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            'fmu/in/trajectory_setpoint',
            10
        )
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            'fmu/in/offboard_control_mode',
            10
        )

        # Subscribers
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self._px4_local_position_callback,
            qos_profile,
            callback_group=self.callback_group
        )
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            'fmu/out/vehicle_odometry',
            self._px4_odometry_callback,
            qos_profile,
            callback_group=self.callback_group
        )

    def _mavros_state_callback(self, msg: State):
        """Handle MAVROS state updates."""
        self.drone_armed = msg.armed
        self.drone_mode = msg.mode
        self.mavros_connected = msg.connected

    def _wait_for_mavros_ready(self):
        """Wait for MAVROS to be connected and receiving data."""
        self.mavros_connected = False
        max_wait = 120  # seconds
        start_time = time.time()

        while (time.time() - start_time) < max_wait:
            # Spin to process callbacks
            rclpy.spin_once(self, timeout_sec=0.5)

            # Check if we have valid state data
            with self.state_lock:
                has_position = self.state_valid and not np.allclose(self.position, [0, 0, 0])

            if self.mavros_connected and has_position:
                self.get_logger().info(f"MAVROS ready! Armed: {self.drone_armed}, Mode: {self.drone_mode}")
                self.get_logger().info(f"Position: {self.position}")
                return True

            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0 and elapsed > 0:
                self.get_logger().info(f"Still waiting for MAVROS... ({elapsed}s) connected={self.mavros_connected}, valid={self.state_valid}")

        self.get_logger().warn("Timeout waiting for MAVROS - proceeding anyway")
        return False

    def _ardupilot_odom_callback(self, msg: Odometry):
        """Handle ArduPilot odometry (ENU frame)."""
        with self.state_lock:
            # Position (ENU)
            self.position[0] = msg.pose.pose.position.x
            self.position[1] = msg.pose.pose.position.y
            self.position[2] = msg.pose.pose.position.z

            # Velocity (body frame from MAVROS, we'll use it as-is)
            self.velocity[0] = msg.twist.twist.linear.x
            self.velocity[1] = msg.twist.twist.linear.y
            self.velocity[2] = msg.twist.twist.linear.z

            # Orientation quaternion
            self.orientation[0] = msg.pose.pose.orientation.w
            self.orientation[1] = msg.pose.pose.orientation.x
            self.orientation[2] = msg.pose.pose.orientation.y
            self.orientation[3] = msg.pose.pose.orientation.z

            self.state_valid = True

    def _ardupilot_global_callback(self, msg: NavSatFix):
        """Handle ArduPilot global position (for reference only)."""
        pass  # Can be used for lat/lon if needed

    def _frame_callback(self, msg: Image):
        """Handle raw YOLO frame (no bounding boxes). JPEG-compress and store."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.frame_lock:
                self.latest_frame_jpeg = jpeg_data.tobytes()
        except Exception as e:
            self.get_logger().warn(f"Frame conversion error: {e}")

    def _px4_local_position_callback(self, msg):
        """Handle PX4 local position (NED frame)."""
        with self.state_lock:
            # Position (NED)
            self.position[0] = msg.x
            self.position[1] = msg.y
            self.position[2] = msg.z

            # Velocity (NED)
            self.velocity[0] = msg.vx
            self.velocity[1] = msg.vy
            self.velocity[2] = msg.vz

            self.state_valid = True

    def _px4_odometry_callback(self, msg):
        """Handle PX4 odometry for orientation."""
        with self.state_lock:
            # Orientation quaternion
            self.orientation[0] = msg.q[0]  # w
            self.orientation[1] = msg.q[1]  # x
            self.orientation[2] = msg.q[2]  # y
            self.orientation[3] = msg.q[3]  # z

    def _set_guided_mode(self) -> bool:
        """Set ArduPilot to GUIDED mode for velocity control."""
        if self.autopilot != "ardupilot":
            return True

        if self.drone_mode == "GUIDED":
            self.get_logger().info("Already in GUIDED mode")
            return True

        if not self.set_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Set mode service not available")
            return False

        request = SetMode.Request()
        request.custom_mode = "GUIDED"

        future = self.set_mode_client.call_async(request)
        # Wait for result with timeout
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < 5.0:
            time.sleep(0.1)

        if future.done():
            result = future.result()
            if result.mode_sent:
                self.get_logger().info("GUIDED mode set successfully")
                self.guided_mode_set = True
                return True
            else:
                self.get_logger().error("Failed to set GUIDED mode")
                return False
        else:
            self.get_logger().error("Set mode service call timed out")
            return False

    def _velocity_publish_callback(self):
        """Periodic callback to publish velocity commands (maintains GUIDED mode)."""
        if self.autopilot == "ardupilot" and self.guided_mode_set:
            vx, vy, vz, yaw_rate = self.last_velocity_cmd
            self._publish_velocity_ardupilot(vx, vy, vz, yaw_rate)

    def _publish_velocity_ardupilot(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """Publish velocity command for ArduPilot."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # World frame

        # Linear velocity (ENU: x=East, y=North, z=Up)
        msg.twist.linear.x = float(vx)   # East
        msg.twist.linear.y = float(vy)   # North
        msg.twist.linear.z = float(vz)   # Up

        # Angular velocity (yaw rate)
        msg.twist.angular.z = float(yaw_rate)

        self.vel_pub.publish(msg)

    def _publish_velocity_px4(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """Publish velocity command for PX4."""
        # Offboard control mode (must be published regularly)
        mode_msg = OffboardControlMode()
        mode_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        mode_msg.velocity = True
        self.offboard_mode_pub.publish(mode_msg)

        # Trajectory setpoint with velocity
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Velocity (NED: x=North, y=East, z=Down)
        # Convert from user-friendly (forward, right, up) to NED
        traj_msg.velocity[0] = float(vx)   # North
        traj_msg.velocity[1] = float(vy)   # East
        traj_msg.velocity[2] = float(-vz)  # Down (negative of up)

        traj_msg.yawspeed = float(yaw_rate)

        self.trajectory_pub.publish(traj_msg)

    def publish_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """Publish velocity command to the appropriate autopilot."""
        # Store for periodic publishing
        self.last_velocity_cmd = (vx, vy, vz, yaw_rate)

        if self.autopilot == "ardupilot":
            # Ensure we're in GUIDED mode first
            if not self.guided_mode_set:
                self._set_guided_mode()
            self._publish_velocity_ardupilot(vx, vy, vz, yaw_rate)
        elif self.autopilot == "px4":
            self._publish_velocity_px4(vx, vy, vz, yaw_rate)

    def get_state(self) -> tuple:
        """Get current drone state (thread-safe)."""
        with self.state_lock:
            return (
                self.position.copy(),
                self.velocity.copy(),
                self.orientation.copy(),
                self.state_valid
            )

    def _zmq_listener(self):
        """ZMQ listener thread - receives actions, sends state."""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.zmq_port}")

        self.get_logger().info(f"ZMQ listener started on port {self.zmq_port}")

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        while self.zmq_running and rclpy.ok():
            # Poll with timeout to allow checking zmq_running flag
            socks = dict(poller.poll(100))  # 100ms timeout

            if socket in socks and socks[socket] == zmq.POLLIN:
                try:
                    # Receive action
                    request = socket.recv()

                    if len(request) == self.ACTION_SIZE:
                        # Unpack velocity command
                        vx, vy, vz, yaw_rate = struct.unpack(self.ACTION_FORMAT, request)

                        # Check for reset signal
                        if abs(vx - self.RESET_SIGNAL) < 0.001:
                            self.get_logger().info("Received reset signal")
                            # On reset, just return current state without publishing
                            self.guided_mode_set = False  # Reset guided mode flag
                        else:
                            # Publish velocity command
                            self.publish_velocity(vx, vy, vz, yaw_rate)
                    else:
                        self.get_logger().warn(f"Invalid action size: {len(request)}, expected {self.ACTION_SIZE}")

                    # Get current state
                    pos, vel, quat, valid = self.get_state()

                    # Get latest frame (JPEG bytes)
                    with self.frame_lock:
                        frame_jpeg = self.latest_frame_jpeg

                    # Pack and send state + frame
                    # Format: [10 doubles (state)] + [uint32 frame_len] + [frame JPEG bytes]
                    state_data = struct.pack(
                        self.STATE_FORMAT,
                        pos[0], pos[1], pos[2],
                        vel[0], vel[1], vel[2],
                        quat[0], quat[1], quat[2], quat[3]
                    )
                    frame_len = len(frame_jpeg)
                    reply = state_data + struct.pack('I', frame_len) + frame_jpeg
                    socket.send(reply)

                except zmq.ZMQError as e:
                    self.get_logger().error(f"ZMQ error: {e}")
                except struct.error as e:
                    self.get_logger().error(f"Struct error: {e}")

        socket.close()
        context.term()
        self.get_logger().info("ZMQ listener stopped")

    def destroy_node(self):
        """Clean shutdown."""
        self.zmq_running = False
        if self.zmq_thread.is_alive():
            self.zmq_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description='Gym Control Node for RL-based drone control')
    parser.add_argument('--drone_id', type=str, required=True, help='Drone ID (e.g., 1, 2)')
    parser.add_argument('--autopilot', type=str, default='ardupilot',
                        choices=['ardupilot', 'px4'], help='Autopilot type')
    parser.add_argument('--zmq_port', type=int, default=5556, help='ZMQ port for gym communication')

    # Parse known args to separate ROS args
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = GymControlNode(
        drone_id=parsed_args.drone_id,
        autopilot=parsed_args.autopilot,
        zmq_port=parsed_args.zmq_port
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
