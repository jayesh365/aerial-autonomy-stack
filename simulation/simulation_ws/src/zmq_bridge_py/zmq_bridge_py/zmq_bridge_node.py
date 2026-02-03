#!/usr/bin/env python3
"""
Python ZMQ Bridge - Replaces the C++ zeromq_bridge.cpp

Handles:
1. ZMQ REQ/REP communication with host gym
2. Gazebo simulation control (pause/step) via gz.transport
3. ROS2 subscriptions for state (odom, clock)
4. Camera images via GStreamer
"""

import os
import struct
import threading
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

import zmq

# Gazebo transport
import gz.transport13
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean as GzBoolean

# Command types
CMD_RESET = 0
CMD_STEP = 1


class ZMQBridgePython(Node):
    def __init__(self):
        super().__init__('zmq_bridge_py')

        # Parameters
        self.declare_parameter('step_size', 250)
        self.declare_parameter('physics_dt', 0.004)
        self.declare_parameter('init_duration', 80.0)
        self.declare_parameter('camera_port', 5600)
        self.declare_parameter('enable_camera', True)

        self.step_size = self.get_parameter('step_size').value
        self.physics_dt = self.get_parameter('physics_dt').value
        self.init_duration = self.get_parameter('init_duration').value
        self.camera_port = self.get_parameter('camera_port').value
        self.enable_camera = self.get_parameter('enable_camera').value

        # State
        self.current_sim_time = 0.0
        self.state_lock = threading.Lock()
        self.clock_lock = threading.Lock()
        self.clock_event = threading.Event()

        # State payload: [x, y, z, vx, vy, vz, qx, qy, qz, qw, heading] = 11 doubles
        self.current_state = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0,
            'heading': 0.0
        }

        # Camera frame
        self.camera_lock = threading.Lock()
        self.current_frame = None
        self.frame_shape = (240, 320, 3)  # H, W, C - matches sensor_camera model.sdf

        # ROS2 subscriptions
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_sub = self.create_subscription(
            Odometry, '/mavros/local_position/odom',
            self.odom_callback, qos
        )
        self.clock_sub = self.create_subscription(
            Clock, '/clock',
            self.clock_callback, 10
        )

        # Gazebo transport
        self.gz_node = gz.transport13.Node()
        world_name = os.environ.get('WORLD', 'default')
        self.gz_control_topic = f"/world/{world_name}/control"
        self.get_logger().info(f"Gazebo control topic: {self.gz_control_topic}")

        # ZMQ setup
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind("tcp://*:5555")
        self.get_logger().info("ZMQ REP socket bound to port 5555")

        # Camera setup (GStreamer)
        if self.enable_camera:
            self.start_camera_capture()

        # Start ZMQ listener thread
        self.running = True
        self.zmq_thread = threading.Thread(target=self.zmq_listener, daemon=True)
        self.zmq_thread.start()

        self.get_logger().info("Python ZMQ Bridge initialized")

    def start_camera_capture(self):
        """Start GStreamer camera capture in background thread."""
        gst_pipeline = (
            f"udpsrc port={self.camera_port} ! "
            "application/x-rtp, media=(string)video, encoding-name=(string)H264 ! "
            "rtph264depay ! avdec_h264 ! videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
        )

        def camera_loop():
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                self.get_logger().warn(f"Failed to open camera on port {self.camera_port}")
                return

            self.get_logger().info(f"Camera capture started on port {self.camera_port}")
            while self.running:
                ret, frame = cap.read()
                if ret:
                    with self.camera_lock:
                        self.current_frame = frame
                else:
                    time.sleep(0.01)
            cap.release()

        self.camera_thread = threading.Thread(target=camera_loop, daemon=True)
        self.camera_thread.start()

    def odom_callback(self, msg):
        """Update state from odometry."""
        with self.state_lock:
            self.current_state['x'] = msg.pose.pose.position.x
            self.current_state['y'] = msg.pose.pose.position.y
            self.current_state['z'] = msg.pose.pose.position.z
            self.current_state['vx'] = msg.twist.twist.linear.x
            self.current_state['vy'] = msg.twist.twist.linear.y
            self.current_state['vz'] = msg.twist.twist.linear.z
            self.current_state['qx'] = msg.pose.pose.orientation.x
            self.current_state['qy'] = msg.pose.pose.orientation.y
            self.current_state['qz'] = msg.pose.pose.orientation.z
            self.current_state['qw'] = msg.pose.pose.orientation.w

            # Compute heading from quaternion
            qw, qx, qy, qz = (
                self.current_state['qw'], self.current_state['qx'],
                self.current_state['qy'], self.current_state['qz']
            )
            siny = 2.0 * (qw * qz + qx * qy)
            cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
            heading = np.arctan2(siny, cosy) * 180.0 / np.pi
            if heading < 0:
                heading += 360.0
            self.current_state['heading'] = heading

    def clock_callback(self, msg):
        """Update simulation time from clock."""
        with self.clock_lock:
            self.current_sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.clock_event.set()

    def set_gazebo_pause(self, pause: bool):
        """Pause or unpause Gazebo simulation."""
        req = WorldControl()
        req.pause = pause
        result, _ = self.gz_node.request(
            self.gz_control_topic, req, WorldControl, GzBoolean, 1000
        )
        return result

    def step_gazebo(self):
        """Step Gazebo simulation by step_size physics steps."""
        req = WorldControl()
        req.pause = True
        req.multi_step = self.step_size
        result, _ = self.gz_node.request(
            self.gz_control_topic, req, WorldControl, GzBoolean, 1000
        )
        return result

    def pack_state_reply(self, include_image=True):
        """Pack state (and optionally image) into bytes for ZMQ reply."""
        with self.state_lock:
            state_bytes = struct.pack(
                '=11d',
                self.current_state['x'], self.current_state['y'], self.current_state['z'],
                self.current_state['vx'], self.current_state['vy'], self.current_state['vz'],
                self.current_state['qx'], self.current_state['qy'], self.current_state['qz'],
                self.current_state['qw'], self.current_state['heading']
            )

        if include_image and self.enable_camera:
            with self.camera_lock:
                if self.current_frame is not None:
                    # Encode image as JPEG for compression
                    _, img_encoded = cv2.imencode('.jpg', self.current_frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
                    img_bytes = img_encoded.tobytes()
                else:
                    img_bytes = b''

            # Pack: [state (88 bytes)] [img_size (4 bytes)] [img_data (variable)]
            img_size = len(img_bytes)
            return state_bytes + struct.pack('=I', img_size) + img_bytes
        else:
            # Just state, no image
            return state_bytes

    def zmq_listener(self):
        """Main ZMQ request handler loop."""
        self.get_logger().info("ZMQ listener started")

        poller = zmq.Poller()
        poller.register(self.zmq_socket, zmq.POLLIN)

        while self.running and rclpy.ok():
            events = dict(poller.poll(timeout=100))

            if self.zmq_socket in events:
                try:
                    request = self.zmq_socket.recv()

                    # Parse command: [cmd_type (1 byte), dx, dy, dz (3 doubles)]
                    cmd_type = struct.unpack('=B', request[:1])[0]
                    dx, dy, dz = struct.unpack('=3d', request[1:25])

                    if cmd_type == CMD_RESET:
                        self.get_logger().info("RESET received")

                        # Unpause simulation
                        self.set_gazebo_pause(False)

                        # Wait for init_duration
                        while self.running and rclpy.ok():
                            self.clock_event.wait(timeout=1.0)
                            self.clock_event.clear()
                            with self.clock_lock:
                                if self.current_sim_time >= self.init_duration:
                                    break

                        # Pause simulation
                        self.set_gazebo_pause(True)
                        self.get_logger().info(f"Init complete at t={self.current_sim_time:.1f}s, paused")

                    elif cmd_type == CMD_STEP:
                        # Publish gym command (handled by gym_control_node)
                        # For now we just step - the command publishing is separate

                        # Record target time
                        with self.clock_lock:
                            target_time = self.current_sim_time + (self.step_size * self.physics_dt) - 0.0001

                        # Step simulation
                        self.step_gazebo()

                        # Wait for step to complete
                        timeout = 30.0
                        start = time.time()
                        while self.running and rclpy.ok() and (time.time() - start) < timeout:
                            self.clock_event.wait(timeout=1.0)
                            self.clock_event.clear()
                            with self.clock_lock:
                                if self.current_sim_time >= target_time:
                                    break

                    # Send reply
                    reply = self.pack_state_reply(include_image=True)
                    self.zmq_socket.send(reply)

                except Exception as e:
                    self.get_logger().error(f"ZMQ error: {e}")
                    # Send empty reply to unblock client
                    self.zmq_socket.send(b'\x00' * 88)

    def destroy_node(self):
        self.running = False
        self.zmq_socket.close()
        self.zmq_context.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ZMQBridgePython()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
