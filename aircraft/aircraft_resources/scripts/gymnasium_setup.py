#!/usr/bin/env python3
"""
Gymnasium Setup Node - Initializes drone for RL training.

This node handles the initial setup sequence:
1. Wait for autopilot interface to be ready
2. Send takeoff command
3. Wait for takeoff to complete and drone to stabilize
4. Keep running to maintain ROS connections

Based on the mission sequence pattern from test_mission.yaml
"""

import time
import argparse

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from autopilot_interface_msgs.action import Takeoff, Offboard
from nav_msgs.msg import Odometry
from std_msgs.msg import String

# Try to import MAVROS state for ArduPilot
try:
    from mavros_msgs.msg import State as MavrosState
    MAVROS_AVAILABLE = True
except ImportError:
    MAVROS_AVAILABLE = False


class GymnasiumSetup(Node):
    def __init__(self, drone_id, takeoff_altitude=40.0):
        super().__init__('gymnasium_setup_node')
        self.drone_id = drone_id
        self.takeoff_altitude = takeoff_altitude

        # State tracking
        self.current_altitude = 0.0
        self.drone_armed = False
        self.drone_mode = ""
        self.mavros_connected = False
        self.position_valid = False
        self.system_status = 0  # MAV_STATE: 0=uninit, 3=standby (ready to arm), 4=active

        # Action clients
        self.takeoff_client = ActionClient(self, Takeoff, f'/Drone{drone_id}/takeoff_action')
        self.offboard_client = ActionClient(self, Offboard, f'/Drone{drone_id}/offboard_action')

        # QoS profile for subscribers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to odometry to track altitude
        self.odom_sub = self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self._odom_callback,
            qos_profile
        )

        # Subscribe to MAVROS state if available
        if MAVROS_AVAILABLE:
            self.state_sub = self.create_subscription(
                MavrosState,
                '/mavros/state',
                self._mavros_state_callback,
                qos_profile
            )

        self.get_logger().info(f'GymnasiumSetup initialized for Drone{drone_id}')
        self.get_logger().info(f'Target takeoff altitude: {takeoff_altitude}m')

    def _odom_callback(self, msg: Odometry):
        """Track current altitude from odometry."""
        self.current_altitude = msg.pose.pose.position.z  # ENU: z is up
        self.position_valid = True

    def _mavros_state_callback(self, msg):
        """Track MAVROS connection and arm state."""
        self.mavros_connected = msg.connected
        self.drone_armed = msg.armed
        self.drone_mode = msg.mode
        self.system_status = msg.system_status  # MAV_STATE: 3=STANDBY (ready to arm)

    def wait_for_server(self, client, name, timeout=60.0):
        """Wait for action server with timeout."""
        self.get_logger().info(f'Waiting for {name} action server...')
        start_time = time.time()
        while not client.wait_for_server(timeout_sec=2.0):
            if time.time() - start_time > timeout:
                self.get_logger().error(f'{name} server not available after {timeout}s')
                return False
            self.get_logger().info(f'{name} not available yet. Retrying...')
        self.get_logger().info(f'{name} server is ready.')
        return True

    def wait_for_mavros(self, timeout=120.0):
        """Wait for MAVROS to be connected and publishing data."""
        self.get_logger().info('Waiting for MAVROS connection...')
        start_time = time.time()

        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.5)

            if self.position_valid:
                self.get_logger().info(f'MAVROS connected! Altitude: {self.current_altitude:.1f}m')
                if MAVROS_AVAILABLE:
                    self.get_logger().info(f'Armed: {self.drone_armed}, Mode: {self.drone_mode}')
                return True

            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0 and elapsed > 0:
                self.get_logger().info(f'Still waiting for MAVROS... ({elapsed}s)')

        self.get_logger().warn('Timeout waiting for MAVROS')
        return False

    def wait_for_ekf_ready(self, timeout=60.0):
        """
        Wait for ArduPilot EKF to be ready for arming.

        ArduPilot SITL needs ~40 seconds for pre-arm checks (GPS lock, EKF convergence).
        We check system_status == 3 (MAV_STATE_STANDBY) which indicates ready to arm.
        """
        self.get_logger().info('Waiting for ArduPilot EKF/pre-arm checks...')
        self.get_logger().info('(This takes ~40s for ArduPilot SITL GPS/EKF convergence)')
        start_time = time.time()

        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.5)

            # MAV_STATE_STANDBY (3) = Ready to arm
            # MAV_STATE_ACTIVE (4) = Already armed
            if self.system_status >= 3:
                self.get_logger().info(f'ArduPilot ready! System status: {self.system_status} (3=STANDBY, 4=ACTIVE)')
                return True

            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0 and elapsed > 0:
                self.get_logger().info(
                    f'Waiting for EKF... ({elapsed}s) '
                    f'status={self.system_status} (need >=3), mode={self.drone_mode}'
                )

        self.get_logger().warn(f'Timeout waiting for EKF. Current status: {self.system_status}')
        return False

    def send_takeoff(self):
        """Send takeoff command and wait for completion."""
        if not self.wait_for_server(self.takeoff_client, 'Takeoff'):
            return False

        goal_msg = Takeoff.Goal()
        goal_msg.takeoff_altitude = self.takeoff_altitude
        # VTOL parameters (used if aircraft is VTOL)
        goal_msg.vtol_transition_heading = 300.0
        goal_msg.vtol_loiter_nord = 100.0
        goal_msg.vtol_loiter_east = 100.0
        goal_msg.vtol_loiter_alt = self.takeoff_altitude + 20.0

        self.get_logger().info(f'Sending Takeoff Goal (altitude: {self.takeoff_altitude}m)...')

        send_goal_future = self.takeoff_client.send_goal_async(
            goal_msg,
            feedback_callback=self._takeoff_feedback_callback
        )
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Takeoff Goal Rejected!')
            return False

        self.get_logger().info('Takeoff Goal Accepted. Waiting for result...')

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result()
        self.get_logger().info(f'Takeoff completed! Result: {result}')

        return True

    def _takeoff_feedback_callback(self, feedback_msg):
        """Handle takeoff feedback."""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Takeoff feedback: {feedback}')

    def wait_for_altitude(self, target_altitude, tolerance=2.0, timeout=60.0):
        """Wait for drone to reach target altitude."""
        self.get_logger().info(f'Waiting for altitude {target_altitude}m (tolerance: {tolerance}m)...')
        start_time = time.time()

        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)

            if self.position_valid:
                if abs(self.current_altitude - target_altitude) < tolerance:
                    self.get_logger().info(f'Reached target altitude: {self.current_altitude:.1f}m')
                    return True

                elapsed = int(time.time() - start_time)
                if elapsed % 5 == 0 and elapsed > 0:
                    self.get_logger().info(f'Current altitude: {self.current_altitude:.1f}m, target: {target_altitude}m')

        self.get_logger().warn(f'Timeout waiting for altitude. Current: {self.current_altitude:.1f}m')
        return False

    def spin_wait(self, seconds):
        """Wait for specified duration while processing callbacks."""
        self.get_logger().info(f'Waiting {seconds} seconds for stabilization...')
        target_time = self.get_clock().now() + Duration(seconds=seconds)
        while self.get_clock().now() < target_time:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('Wait complete.')

    def run_setup_sequence(self):
        """Run the complete setup sequence."""
        self.get_logger().info('='*50)
        self.get_logger().info('Starting Gymnasium Setup Sequence')
        self.get_logger().info('='*50)

        # Step 1: Wait for MAVROS to be ready
        self.get_logger().info('Step 1: Waiting for MAVROS connection...')
        if not self.wait_for_mavros(timeout=120.0):
            self.get_logger().error('Failed to connect to MAVROS')
            return False

        # Step 2: Wait for ArduPilot SITL pre-arm checks
        # CRITICAL: ArduPilot SITL needs ~40 seconds of WALL CLOCK TIME for:
        #   - GPS lock and home position
        #   - EKF convergence
        #   - Gyro calibration
        #   - AHRS initialization
        # The simulation's GYM_INIT_DURATION is in SIM time (with RTF=15, 80s sim = 5s real)
        # So we MUST wait here for real wall-clock time.
        self.get_logger().info('Step 2: Waiting for ArduPilot SITL initialization...')
        self.get_logger().info('(ArduPilot needs ~40s wall-clock for GPS/EKF/gyro calibration)')

        # Wait for system_status to reach STANDBY, with minimum 40s wall-clock wait
        init_start = time.time()
        min_wait_seconds = 40.0  # Minimum wall-clock time to wait

        while True:
            rclpy.spin_once(self, timeout_sec=1.0)
            elapsed = time.time() - init_start

            # Check if we have valid status AND enough time has passed
            if self.system_status >= 3 and elapsed >= min_wait_seconds:
                self.get_logger().info(f'ArduPilot ready after {elapsed:.1f}s! Status: {self.system_status}')
                break

            # Timeout after 90 seconds
            if elapsed > 90.0:
                self.get_logger().warn(f'Timeout waiting for ArduPilot. Status: {self.system_status}')
                break

            # Progress logging every 10 seconds
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                remaining = max(0, min_wait_seconds - elapsed)
                self.get_logger().info(
                    f'Waiting... ({elapsed:.0f}s elapsed, {remaining:.0f}s min remaining) '
                    f'status={self.system_status}, mode={self.drone_mode}, armed={self.drone_armed}'
                )

        # Step 3: Send takeoff command (with extended retries for arming)
        self.get_logger().info('Step 3: Sending takeoff command...')
        takeoff_success = False
        max_retries = 5  # Increased retries
        retry_delay = 10.0  # Longer delay between retries for EKF to converge
        for attempt in range(max_retries):
            if self.send_takeoff():
                takeoff_success = True
                break
            self.get_logger().warn(f'Takeoff attempt {attempt+1}/{max_retries} failed')
            if attempt < max_retries - 1:
                self.get_logger().info(f'Waiting {retry_delay}s before retry (ArduPilot may need more time)...')
                self.spin_wait(retry_delay)

        if not takeoff_success:
            self.get_logger().error('Failed to takeoff after multiple attempts')
            return False

        # Step 4: Wait for drone to reach altitude
        self.get_logger().info('Step 4: Waiting for target altitude...')
        self.wait_for_altitude(self.takeoff_altitude, tolerance=3.0, timeout=60.0)

        # Step 5: Stabilization wait (like "wait: 5.0" in mission)
        self.get_logger().info('Step 5: Stabilization wait...')
        self.spin_wait(5.0)

        self.get_logger().info('='*50)
        self.get_logger().info('Setup sequence complete!')
        self.get_logger().info(f'Drone is at altitude: {self.current_altitude:.1f}m')
        self.get_logger().info(f'Armed: {self.drone_armed}, Mode: {self.drone_mode}')
        self.get_logger().info('Ready for gym control commands.')
        self.get_logger().info('='*50)

        return True


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description='Gymnasium Setup Node')
    parser.add_argument('--drone_id', type=str, required=True, help='The ID of the drone')
    parser.add_argument('--takeoff_altitude', type=float, default=40.0, help='Takeoff altitude in meters')
    parsed_args, _ = parser.parse_known_args()

    node = GymnasiumSetup(
        drone_id=parsed_args.drone_id,
        takeoff_altitude=parsed_args.takeoff_altitude
    )

    # Run setup sequence
    success = node.run_setup_sequence()

    if success:
        # Keep the node running to maintain ROS connections
        # and continue reporting status
        node.get_logger().info('Keeping node alive for status monitoring...')
        try:
            rate = node.create_rate(0.2)  # 0.2 Hz = every 5 seconds
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=5.0)
                node.get_logger().info(
                    f'Status: alt={node.current_altitude:.1f}m, '
                    f'armed={node.drone_armed}, mode={node.drone_mode}'
                )
        except KeyboardInterrupt:
            pass
    else:
        node.get_logger().error('Setup sequence failed!')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
