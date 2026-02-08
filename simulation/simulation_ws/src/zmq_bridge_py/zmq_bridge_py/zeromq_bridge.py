"""
Python ZMQ Bridge for Gazebo and ROS 2.

Replaces zmq_bridge_cpp/src/zeromq_bridge.cpp with identical behavior:
- ZMQ REP socket on port 5555 receives actions from the gym environment
- Publishes actions to /action ROS 2 topic
- Controls Gazebo simulation stepping via gz-transport
- Returns simulation clock time to the gym environment

Wire protocol (unchanged from C++ version):
  Request:  8 bytes  - struct.pack('d', action)      double
  Reply:    8 bytes  - struct.pack('iI', sec, nanosec) int32 + uint32
  Reset signal: action == 9999.0
"""

import os
import struct
import threading

import zmq

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock

# Gazebo transport (same imports as gz_step.py)
import gz.transport13
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean as GzBoolean


class ZMQBridge(Node):

    # Wire format constants — must match aas_env.py's struct.pack('d') / struct.unpack('iI')
    ACTION_FORMAT = 'd'       # double (8 bytes)
    CLOCK_FORMAT = 'iI'       # int32 sec + uint32 nanosec (8 bytes)
    RESET_SIGNAL = 9999.0

    def __init__(self):
        super().__init__('zmq_bridge_node')

        # --- ROS 2 Parameters (same as C++ version) ---
        # step_size: how many physics substeps per gym step
        #   C++: this->declare_parameter("step_size", 250);
        self.declare_parameter('step_size', 250)
        self.step_size = self.get_parameter('step_size').value

        # physics_dt: duration of one physics substep in seconds
        #   C++: this->declare_parameter("physics_dt", 0.004);
        self.declare_parameter('physics_dt', 0.004)
        self.physics_dt = self.get_parameter('physics_dt').value

        # init_duration: sim-seconds to run unpaused during reset (EKF convergence)
        #   C++: this->declare_parameter("init_duration", 80.0);
        self.declare_parameter('init_duration', 80.0)
        self.init_duration = self.get_parameter('init_duration').value

        self.get_logger().info(
            f'Config: step_size={self.step_size}, physics_dt={self.physics_dt}, '
            f'init_duration={self.init_duration}'
        )

        # --- ZMQ Setup ---
        # C++: context_(1), socket_(context_, zmq::socket_type::rep)
        #      socket_.bind("tcp://*:5555");
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind('tcp://*:5555')
        self.get_logger().info('ZMQ REP socket bound to port 5555')

        # --- ROS 2 Pub/Sub ---
        # C++: publisher_ = this->create_publisher<Float64>("/action", 10);
        self.callback_group = ReentrantCallbackGroup()
        self.action_pub = self.create_publisher(Float64, '/action', 10)

        # C++: subscription_ = this->create_subscription<Clock>("/clock", 10, ...)
        self.clock_sub = self.create_subscription(
            Clock, '/clock', self._clock_callback, 10,
            callback_group=self.callback_group
        )

        # --- Shared state (protected by lock + condition variable) ---
        # C++: std::mutex clock_mutex_;
        #      std::condition_variable clock_cv_;
        #      ClockPayload current_clock_;
        #      double current_sim_time_ = 0.0;
        self.clock_lock = threading.Lock()
        self.clock_cv = threading.Condition(self.clock_lock)
        self.current_sec = 0        # int32
        self.current_nanosec = 0    # uint32
        self.current_sim_time = 0.0 # double

        # --- Gazebo transport ---
        # C++: gz::transport::Node gz_node_;
        #      service_topic_ = "/world/" + world_name + "/control";
        self.gz_node = gz.transport13.Node()
        world_name = os.environ.get('WORLD', 'default')
        self.service_topic = f'/world/{world_name}/control'
        self.get_logger().info(f'Gazebo service topic: {self.service_topic}')

        # --- Start ZMQ listener thread ---
        # C++: running_ = true;
        #      zmq_thread_ = std::thread(&ZMQBridge::zmq_listener, this);
        self.running = True
        self.zmq_thread = threading.Thread(target=self._zmq_listener, daemon=True)
        self.zmq_thread.start()

    # ---- Clock callback (runs in ROS 2 executor thread) ----
    # C++: void clock_callback(const Clock::SharedPtr msg)
    def _clock_callback(self, msg):
        with self.clock_cv:                          # lock_guard<mutex> lock(clock_mutex_)
            self.current_sec = msg.clock.sec         # current_clock_.sec = msg->clock.sec
            self.current_nanosec = msg.clock.nanosec # current_clock_.nanosec = msg->clock.nanosec
            self.current_sim_time = (               # current_sim_time_ = sec + nanosec * 1e-9
                msg.clock.sec + msg.clock.nanosec * 1e-9
            )
            self.clock_cv.notify_all()               # clock_cv_.notify_all()

    # ---- Gazebo control (called from ZMQ thread) ----
    # C++: void set_gazebo_pause(bool pause)
    def _set_gazebo_pause(self, pause):
        req = WorldControl()
        req.pause = pause
        result, _response = self.gz_node.request(
            self.service_topic, req, WorldControl, GzBoolean, 1000
        )
        if not result:
            self.get_logger().error('Gazebo pause service call failed')

    # C++: void step_gazebo()
    def _step_gazebo(self):
        req = WorldControl()
        req.pause = True
        req.multi_step = self.step_size
        result, _response = self.gz_node.request(
            self.service_topic, req, WorldControl, GzBoolean, 1000
        )
        if not result:
            self.get_logger().error('Gazebo step service call failed')

    # ---- ZMQ listener (runs in its own thread) ----
    # C++: void zmq_listener()
    def _zmq_listener(self):
        self.get_logger().info('ZMQ listener thread started.')

        # C++: zmq::pollitem_t items[] = {{ socket_, 0, ZMQ_POLLIN, 0 }};
        poller = zmq.Poller()
        poller.register(self.zmq_socket, zmq.POLLIN)

        while rclpy.ok() and self.running:
            # C++: zmq::poll(&items[0], 1, 100);
            socks = dict(poller.poll(100))  # 100ms timeout

            if self.zmq_socket not in socks:
                continue

            try:
                # 1. Receive action
                # C++: zmq::message_t request;
                #      socket_.recv(request);
                #      double action = *static_cast<double*>(request.data());
                request = self.zmq_socket.recv()
                action = struct.unpack(self.ACTION_FORMAT, request)[0]

                received = True

                # 2. Logic dispatch
                if abs(action - self.RESET_SIGNAL) < 0.001:
                    # --- RESET MODE ---
                    # C++: set_gazebo_pause(false);
                    self.get_logger().info('Received reset action (9999.0). Running unpaused...')
                    self._set_gazebo_pause(False)

                    # C++: while (rclcpp::ok() && running_) {
                    #          clock_cv_.wait(lock);
                    #          if (current_sim_time_ >= init_duration_) break;
                    #      }
                    with self.clock_cv:
                        while rclpy.ok() and self.running:
                            self.clock_cv.wait(timeout=1.0)
                            if self.current_sim_time >= self.init_duration:
                                break

                    # C++: set_gazebo_pause(true);
                    self._set_gazebo_pause(True)
                    self.get_logger().info('Initialization complete. Paused.')

                else:
                    # --- STEP MODE ---
                    # A. Publish action
                    # C++: auto ros_msg = Float64(); ros_msg.data = action;
                    #      publisher_->publish(ros_msg);
                    ros_msg = Float64()
                    ros_msg.data = action
                    self.action_pub.publish(ros_msg)

                    # B. Calculate target time
                    # C++: target_time = current_sim_time_ + (step_size_ * physics_dt_) - 0.0001;
                    #      clock_ready_ = false;
                    with self.clock_cv:
                        target_time = (
                            self.current_sim_time
                            + (self.step_size * self.physics_dt)
                            - 0.0001
                        )

                    # C. Trigger step
                    # C++: step_gazebo();
                    self._step_gazebo()

                    # D. Wait for clock to reach target
                    # C++: clock_cv_.wait_for(lock, 30000ms, [this, target_time]{
                    #          return current_sim_time_ >= target_time;
                    #      });
                    with self.clock_cv:
                        received = self.clock_cv.wait_for(
                            lambda: self.current_sim_time >= target_time,
                            timeout=30.0
                        )

                # 3. Send reply
                # C++: zmq::message_t reply(sizeof(ClockPayload));
                #      memcpy(reply.data(), &current_clock_, sizeof(ClockPayload));
                if received:
                    reply = struct.pack(
                        self.CLOCK_FORMAT,
                        self.current_sec,
                        self.current_nanosec
                    )
                else:
                    self.get_logger().warn('Update timeout!')
                    reply = struct.pack(self.CLOCK_FORMAT, 0, 0)

                self.zmq_socket.send(reply)

            except zmq.ZMQError as e:
                self.get_logger().error(f'ZMQ Error: {e}')
            except struct.error as e:
                self.get_logger().error(f'Struct Error: {e}')

    # ---- Cleanup ----
    # C++: ~ZMQBridge()
    def destroy_node(self):
        self.running = False
        if self.zmq_thread.is_alive():
            self.zmq_thread.join(timeout=2.0)
        self.zmq_socket.close()
        self.zmq_context.term()
        super().destroy_node()


# ---- Entry point ----
# C++: int main(int argc, char* argv[]) {
#          rclcpp::init(argc, argv);
#          rclcpp::spin(std::make_shared<ZMQBridge>());
#          rclcpp::shutdown();
#      }
def main(args=None):
    rclpy.init(args=args)
    node = ZMQBridge()

    # MultiThreadedExecutor so clock callbacks run while the ZMQ thread is blocking
    # (same pattern as gym_control_node.py)
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
