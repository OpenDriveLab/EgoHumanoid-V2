#ifndef STARK_NODE_HPP
#define STARK_NODE_HPP

#include <chrono>
#include <sys/stat.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/u_int16.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/int8.hpp>
#include "ros2_stark_interfaces/msg/motor_status.hpp"
// #include "ros2_stark_interfaces/msg/touch_status_item.hpp"
#include "ros2_stark_interfaces/msg/touch_status.hpp"
#include "ros2_stark_controller/stark-sdk.h"

class StarkNode : public rclcpp::Node {
public:
    StarkNode();
    ~StarkNode();

private:
    // ROS 2 Publishers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Publisher<ros2_stark_interfaces::msg::MotorStatus>::SharedPtr motor_status_pub_;
    rclcpp::Publisher<ros2_stark_interfaces::msg::MotorStatus>::SharedPtr motor_status_pub_r;
    rclcpp::Publisher<ros2_stark_interfaces::msg::TouchStatus>::SharedPtr touch_status_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr turbo_mode_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt16>::SharedPtr voltage_pub_;

    // ROS 2 Subscribers
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_sub_left;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_sub_right;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr force_level_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr turbo_mode_sub_;

    // ROS 2 Timer
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr info_timer_;

    // Stark SDK handle
    ModbusHandle* handle_ = nullptr;
    ModbusHandle* handle_l = nullptr;
    ModbusHandle* handle_r = nullptr;
    uint8_t slave_id_;
    uint8_t slave_id_l;
    uint8_t slave_id_r;

    struct SerialLink {
        struct stat device{};
        std::chrono::steady_clock::time_point retry_after{};
        unsigned failures = 0;
        bool feedback_ok = false;
    } link_l_, link_r_;

    // Last commanded positions per finger (uint16, 0~1000 SDK internal scale).
    // Used by publish_motor_status to unwrap the uint8 wraparound in SDK reading.
    uint16_t last_cmd_pos_l[6] = {0};
    uint16_t last_cmd_pos_r[6] = {0};
    bool last_cmd_valid_l_ = false;
    bool last_cmd_valid_r_ = false;
    static uint16_t unwrap_pos(uint8_t raw, uint16_t anchor);

    // Configuration parameters
    std::string port_;
    int protect_current_;
    int max_current_;
    int hold_period_ms_;
    bool turbo_always_on_;
    using TurboModeFn = void (*)(void*, uint8_t, bool);
    TurboModeFn turbo_mode_fn_ = nullptr;
    std::string port_l;
    std::string port_r;
    uint32_t baudrate_;
    StarkFirmwareType fw_type_;
    StarkProtocolType protocol_type_;
    LogLevel log_level_;
    bool log_screen_;

    // Callback functions
    void timer_callback();
    void info_timer_callback();
    void joint_cmd_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void joint_cmd_callback_left(const sensor_msgs::msg::JointState::SharedPtr msg);
    void joint_cmd_callback_right(const sensor_msgs::msg::JointState::SharedPtr msg);
    void force_level_callback(const std_msgs::msg::Int8::SharedPtr msg);
    void turbo_mode_callback(const std_msgs::msg::Bool::SharedPtr msg);
    bool set_turbo_mode(ModbusHandle* handle, uint8_t slave_id, bool enabled);

    // Helper functions
    bool initialize_modbus();
    void reconnect_hand(bool left);
    void disconnect_hand(bool left);
    bool hand_current(bool left);
    void record_feedback(bool left, bool ok);
    void configure_hand(bool left);
    void publish_joint_state();
    void publish_device_status();
    void publish_motor_status();
    void publish_motor_status_r();
    void publish_touch_status();
};
#endif // STARK_NODE_HPP
