"""
Gym Control Node

Bridges gym commands to the autopilot_interface.
- Subscribes to /gym_command (geometry_msgs/Vector3) for position deltas [dx, dy, dz]
- Tracks current position from /mavros/local_position/odom
- Calls /DroneN/set_reposition service with absolute position
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

import os
import threading

from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from autopilot_interface_msgs.srv import SetReposition


class GymControlNode(Node):
    def __init__(self):
        super().__init__('gym_control_node')

        self.drone_id = None
        drone_id_str = os.environ.get('DRONE_ID')
        if drone_id_str is None:
            self.get_logger().error("DRONE_ID environment variable not set.")
        else:
            try:
                self.drone_id = int(drone_id_str)
            except ValueError:
                self.get_logger().error(f"Could not parse DRONE_ID='{drone_id_str}' as an integer.")

        self.data_lock = threading.Lock()
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.position_received = False

        # Callback groups
        self.subscriber_callback_group = ReentrantCallbackGroup()
        self.service_callback_group = ReentrantCallbackGroup()

        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Subscribe to local position (ENU frame)
        self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self.odom_callback,
            qos_profile,
            callback_group=self.subscriber_callback_group
        )

        # Subscribe to gym commands
        self.create_subscription(
            Vector3,
            '/gym_command',
            self.gym_command_callback,
            10,
            callback_group=self.subscriber_callback_group
        )

        # Service client for set_reposition
        if self.drone_id is not None:
            self._reposition_client = self.create_client(
                SetReposition,
                f'/Drone{self.drone_id}/set_reposition',
                callback_group=self.service_callback_group
            )
            self.get_logger().info(f"Gym control node initialized for Drone{self.drone_id}")
        else:
            self._reposition_client = None
            self.get_logger().error("Service client not created - DRONE_ID not set")

    def odom_callback(self, msg):
        """Update current position from odometry."""
        with self.data_lock:
            self.current_x = msg.pose.pose.position.x  # East
            self.current_y = msg.pose.pose.position.y  # North
            self.current_z = msg.pose.pose.position.z  # Up
            self.position_received = True

    def gym_command_callback(self, msg):
        """
        Receive delta position command and call set_reposition.
        msg.x = dx (east delta)
        msg.y = dy (north delta)
        msg.z = dz (altitude delta)
        """
        if self._reposition_client is None:
            self.get_logger().error("Cannot process command - service client not available")
            return

        with self.data_lock:
            if not self.position_received:
                self.get_logger().warn("No position data yet, ignoring command")
                return

            # Compute absolute position from current + delta
            target_east = self.current_x + msg.x
            target_north = self.current_y + msg.y
            target_alt = self.current_z + msg.z

        self.get_logger().info(
            f"Gym command: delta=[{msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}] -> "
            f"target=[{target_east:.2f}, {target_north:.2f}, {target_alt:.2f}]"
        )

        # Call set_reposition service
        if not self._reposition_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("set_reposition service not available")
            return

        request = SetReposition.Request()
        request.east = target_east
        request.north = target_north
        request.altitude = target_alt

        future = self._reposition_client.call_async(request)
        future.add_done_callback(self.reposition_response_callback)

    def reposition_response_callback(self, future):
        """Handle response from set_reposition service."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Reposition command accepted")
            else:
                self.get_logger().warn(f"Reposition command rejected: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GymControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
