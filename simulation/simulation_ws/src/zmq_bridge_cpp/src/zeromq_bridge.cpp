#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstdlib>
#include <string>
#include <cstring>
#include <sys/stat.h>
// ROS 2 Includes
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/vector3.hpp"
// Gazebo Includes (libgz-transport13-*, libgz-msgs10-dev)
#include <gz/transport/Node.hh>
#include <gz/msgs/world_control.pb.h>
#include <gz/msgs/boolean.pb.h>
// ZeroMQ Includes (requires cppzmq installed with libzmq3-dev)
#include <zmq.hpp>

using namespace std::chrono_literals;

// Command types from gym
constexpr uint8_t CMD_RESET = 0;
constexpr uint8_t CMD_STEP = 1;

// Input payload: [cmd_type (1 byte), dx, dy, dz (3 doubles)]
#pragma pack(push, 1)
struct GymCommand {
    uint8_t cmd_type;
    double dx;
    double dy;
    double dz;
};
#pragma pack(pop)

// Output payload: [x, y, z, vx, vy, vz, qx, qy, qz, qw, heading] (11 doubles)
#pragma pack(push, 1)
struct StatePayload {
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
    double qx;
    double qy;
    double qz;
    double qw;
    double heading;
};
#pragma pack(pop)

class ZMQBridge : public rclcpp::Node {
public:
    ZMQBridge() : Node("zmq_bridge_node"), context_(1), socket_(context_, zmq::socket_type::rep) {

        this->declare_parameter("step_size", 250); // How many multiples of the timestep in the world SDF (250Hz/4ms for PX4, 500Hz/2ms for ArduPilot)
        step_size_ = this->get_parameter("step_size").as_int();
        this->declare_parameter("physics_dt", 0.004); // Size of the physics timestep in seconds (4ms for PX4, 2ms for ArduPilot)
        physics_dt_ = this->get_parameter("physics_dt").as_double();
        RCLCPP_INFO(this->get_logger(), "Config: step size = %d, physics dt = %.4f", step_size_, physics_dt_);
        this->declare_parameter("init_duration", 80.0); // Duration to run unpaused during reset (seconds)
        init_duration_ = this->get_parameter("init_duration").as_double();

        // 1.a ZMQ TCP Setup (see self.ZMQ_TRANSPORT in aas_env.py)
        socket_.bind("tcp://*:5555"); // '*' binds to all interfaces (equivalent to 0.0.0.0)
        RCLCPP_INFO(this->get_logger(), "ZMQ REP socket bound to port 5555");

        // 2. ROS 2 Setup
        // Publisher for gym commands (position deltas)
        gym_cmd_publisher_ = this->create_publisher<geometry_msgs::msg::Vector3>("/gym_command", 10);

        // Subscribe to clock
        clock_subscription_ = this->create_subscription<rosgraph_msgs::msg::Clock>(
            "/clock", 10,
            std::bind(&ZMQBridge::clock_callback, this, std::placeholders::_1));

        // Subscribe to odometry for state
        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/mavros/local_position/odom", 10,
            std::bind(&ZMQBridge::odom_callback, this, std::placeholders::_1));

        const char* env_world = std::getenv("WORLD");
        if (env_world == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Environment variable 'WORLD' was not set.");
        }
        std::string world_name = env_world;
        service_topic_ = "/world/" + world_name + "/control";

        // 3. Start ZMQ Thread
        running_ = true;
        zmq_thread_ = std::thread(&ZMQBridge::zmq_listener, this);
    }

    ~ZMQBridge() {
        running_ = false;
        // Clean shutdown for ZMQ
        context_.close();
        if (zmq_thread_.joinable()) {
            zmq_thread_.join();
        }
    }

private:
    zmq::context_t context_;
    zmq::socket_t socket_;

    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr gym_cmd_publisher_;
    rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr clock_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

    gz::transport::Node gz_node_;
    std::string service_topic_;

    std::thread zmq_thread_;
    std::atomic<bool> running_;

    // Clock state
    std::mutex clock_mutex_;
    std::condition_variable clock_cv_;
    double current_sim_time_ = 0.0;

    // Odometry state
    std::mutex odom_mutex_;
    StatePayload current_state_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    bool odom_received_ = false;

    int step_size_;
    double physics_dt_;
    double init_duration_;

    void clock_callback(const rosgraph_msgs::msg::Clock::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(clock_mutex_);
        current_sim_time_ = msg->clock.sec + (msg->clock.nanosec * 1e-9);
        clock_cv_.notify_all();
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(odom_mutex_);

        // Position (ENU frame)
        current_state_.x = msg->pose.pose.position.x;
        current_state_.y = msg->pose.pose.position.y;
        current_state_.z = msg->pose.pose.position.z;

        // Velocity
        current_state_.vx = msg->twist.twist.linear.x;
        current_state_.vy = msg->twist.twist.linear.y;
        current_state_.vz = msg->twist.twist.linear.z;

        // Orientation (quaternion)
        current_state_.qx = msg->pose.pose.orientation.x;
        current_state_.qy = msg->pose.pose.orientation.y;
        current_state_.qz = msg->pose.pose.orientation.z;
        current_state_.qw = msg->pose.pose.orientation.w;

        // Compute heading from quaternion (yaw in degrees)
        // yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        double siny_cosp = 2.0 * (current_state_.qw * current_state_.qz + current_state_.qx * current_state_.qy);
        double cosy_cosp = 1.0 - 2.0 * (current_state_.qy * current_state_.qy + current_state_.qz * current_state_.qz);
        double yaw_rad = std::atan2(siny_cosp, cosy_cosp);
        current_state_.heading = yaw_rad * 180.0 / M_PI;
        if (current_state_.heading < 0) {
            current_state_.heading += 360.0;
        }

        odom_received_ = true;
    }

    void set_gazebo_pause(bool pause) {
        gz::msgs::WorldControl req;
        req.set_pause(pause);
        gz::msgs::Boolean rep;
        bool result;
        gz_node_.Request(service_topic_, req, 1000, rep, result);
    }

    void step_gazebo() {
        gz::msgs::WorldControl req;
        req.set_pause(true);
        req.set_multi_step(static_cast<unsigned int>(step_size_));
        gz::msgs::Boolean rep;
        bool result;
        gz_node_.Request(service_topic_, req, 1000, rep, result);
    }

    void zmq_listener() {
        RCLCPP_INFO(this->get_logger(), "ZMQ listener thread started.");

        zmq::pollitem_t items[] = {
            { static_cast<void*>(socket_), 0, ZMQ_POLLIN, 0 }
        };

        while (rclcpp::ok() && running_) {
            zmq::poll(&items[0], 1, 100); // Poll with 100ms timeout to allow checking running_ flag

            if (items[0].revents & ZMQ_POLLIN) {
                try {
                    // 1. Receive Command
                    zmq::message_t request;
                    auto res = socket_.recv(request, zmq::recv_flags::none);
                    if (!res) continue;

                    // Parse command
                    if (request.size() < sizeof(GymCommand)) {
                        RCLCPP_ERROR(this->get_logger(), "Received message too small: %zu bytes", request.size());
                        // Send empty state
                        zmq::message_t reply(sizeof(StatePayload));
                        StatePayload empty_state = {0};
                        std::memcpy(reply.data(), &empty_state, sizeof(StatePayload));
                        socket_.send(reply, zmq::send_flags::none);
                        continue;
                    }

                    GymCommand cmd;
                    std::memcpy(&cmd, request.data(), sizeof(GymCommand));

                    // 2. Logic Dispatch
                    if (cmd.cmd_type == CMD_RESET) {
                        RCLCPP_INFO(this->get_logger(), "Received RESET command. Running unpaused for %.1f seconds...", init_duration_);

                        // A. Unpause simulation
                        set_gazebo_pause(false);

                        // B. Wait loop until init_duration reached
                        while (rclcpp::ok() && running_) {
                            std::unique_lock<std::mutex> lock(clock_mutex_);
                            clock_cv_.wait(lock);
                            if (current_sim_time_ >= init_duration_) {
                                break;
                            }
                        }

                        // C. Pause simulation
                        set_gazebo_pause(true);
                        RCLCPP_INFO(this->get_logger(), "Initialization Complete. Paused at sim time %.2f", current_sim_time_);

                    } else if (cmd.cmd_type == CMD_STEP) {
                        // A. Publish gym command (position delta)
                        auto ros_msg = geometry_msgs::msg::Vector3();
                        ros_msg.x = cmd.dx;
                        ros_msg.y = cmd.dy;
                        ros_msg.z = cmd.dz;
                        gym_cmd_publisher_->publish(ros_msg);

                        // B. Calculate target time
                        double target_time;
                        {
                            std::lock_guard<std::mutex> lock(clock_mutex_);
                            target_time = current_sim_time_ + (step_size_ * physics_dt_) - 0.0001;
                        }

                        // C. Trigger step
                        step_gazebo();

                        // D. Wait for step to complete
                        std::unique_lock<std::mutex> lock(clock_mutex_);
                        bool received = clock_cv_.wait_for(lock, 30000ms, [this, target_time]{
                            return current_sim_time_ >= target_time;
                        });

                        if (!received) {
                            RCLCPP_WARN(this->get_logger(), "Step timeout!");
                        }
                    } else {
                        RCLCPP_WARN(this->get_logger(), "Unknown command type: %d", cmd.cmd_type);
                    }

                    // 3. Send state reply
                    zmq::message_t reply(sizeof(StatePayload));
                    {
                        std::lock_guard<std::mutex> lock(odom_mutex_);
                        std::memcpy(reply.data(), &current_state_, sizeof(StatePayload));
                    }
                    socket_.send(reply, zmq::send_flags::none);

                } catch (const std::exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "ZMQ Error: %s", e.what());
                }
            }
        }
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZMQBridge>());
    rclcpp::shutdown();
    return 0;
}
