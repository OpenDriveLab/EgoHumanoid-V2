#include <cstdlib>
#include <stdexcept>
#include "ros2_stark_controller/stark_node.hpp"

// Optional SDK 2.x symbol. The asm alias avoids a declaration collision when
// building with a newer header, and ELF weak linkage avoids a libdl dependency.
extern "C" void brainco_set_turbo_mode_optional(void*, uint8_t, bool)
    __asm__("stark_set_turbo_mode_enabled") __attribute__((weak));


// Recover the 0~1000 position estimate from the wrapped SDK uint8 reading.
// SDK exposes MotorStatusData.positions[6] as uint8_t but the underlying scale
// is 0~1000 — values > 255 wrap around. We pick the candidate {raw, raw+256,
// raw+512, raw+768} closest to the last commanded position (anchor).
uint16_t StarkNode::unwrap_pos(uint8_t raw, uint16_t anchor) {
    uint16_t best = raw;
    int best_diff = std::abs(static_cast<int>(anchor) - static_cast<int>(raw));
    for (int k = 1; k <= 4; ++k) {
        uint16_t cand = static_cast<uint16_t>(raw) + static_cast<uint16_t>(k * 256);
        if (cand > 1023) break;
        int d = std::abs(static_cast<int>(anchor) - static_cast<int>(cand));
        if (d < best_diff) {
            best_diff = d;
            best = cand;
        }
    }
    return best;
}

StarkNode::StarkNode() : Node("stark_node") {
    // Declare and Get parameters
    port_l = this->declare_parameter<std::string>("port_l", "/dev/ttyUSB0");
    port_r = this->declare_parameter<std::string>("port_r", "/dev/ttyUSB1");
    baudrate_ = this->declare_parameter<int>("baudrate", 115200);
    slave_id_l = this->declare_parameter<int>("slave_id_l", 1);
    slave_id_r = this->declare_parameter<int>("slave_id_r", 1);
    fw_type_ = static_cast<StarkFirmwareType>(this->declare_parameter<int>("firmware_type", 2));
    protocol_type_ = static_cast<StarkProtocolType>(this->declare_parameter<int>("protocol_type", 1));
    log_level_ = static_cast<LogLevel>(this->declare_parameter<int>("log_level", 2));
    log_screen_ = static_cast<bool>(this->declare_parameter<bool>("log_screen", true));
    protect_current_ = this->declare_parameter<int>("protect_current", 500);  // mA, default 500, range 100~1500
    max_current_ = this->declare_parameter<int>("max_current", 1000);  // mA, default 1000
    hold_period_ms_ = this->declare_parameter<int>("hold_period_ms", 10);
    turbo_always_on_ = this->declare_parameter<bool>("turbo_always_on", true);
    if (hold_period_ms_ < 5 || hold_period_ms_ > 1000) {
        throw std::invalid_argument("hold_period_ms must be in [5, 1000]");
    }

    // SDK 2.x exports this API, while older SDK versions may not. Resolve
    // it at runtime so the driver remains binary-compatible with both.
    turbo_mode_fn_ = brainco_set_turbo_mode_optional;
    if (turbo_always_on_ && !turbo_mode_fn_) {
        RCLCPP_WARN(this->get_logger(),
                    "Turbo API unavailable in installed Stark SDK; preserving device Turbo setting");
    }

    RCLCPP_INFO(this->get_logger(), "Left  Hand: Port: %s, Baudrate: %d, Slave_id: %d",
                port_l.c_str(), baudrate_, slave_id_l);
    RCLCPP_INFO(this->get_logger(), "Right Hand: Port: %s, Baudrate: %d, Slave_id: %d",
                port_r.c_str(), baudrate_, slave_id_r);
    RCLCPP_INFO(this->get_logger(), "Firmware_type: %d, Protocol_type: %d, Log_level: %d\n",
                fw_type_, protocol_type_, log_level_);

    // Initialize Stark SDK
    init_cfg(fw_type_, protocol_type_, log_level_);
    // list_available_ports();
    if (!initialize_modbus()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize Modbus");
        throw std::runtime_error("Modbus initialization failed");
    }
    else {
        RCLCPP_INFO(this->get_logger(), "Waiting for joint cmd ...\n");
        RCLCPP_INFO(this->get_logger(), "protect_current=%d mA, max_current=%d mA per finger (defaults 500/1000)", protect_current_, max_current_);
    }

    // Initialize publishers
    motor_status_pub_ = create_publisher<ros2_stark_interfaces::msg::MotorStatus>("motor_status", 10);
    motor_status_pub_r = create_publisher<ros2_stark_interfaces::msg::MotorStatus>("motor_status_r", 10);
    touch_status_pub_ = create_publisher<ros2_stark_interfaces::msg::TouchStatus>("touch_status", 10);
    turbo_mode_pub_ = create_publisher<std_msgs::msg::Bool>("turbo_mode", 10);
    voltage_pub_ = create_publisher<std_msgs::msg::UInt16>("voltage", 10);


    joint_cmd_sub_left = create_subscription<sensor_msgs::msg::JointState>(
        "joint_commands_left", 10, std::bind(&StarkNode::joint_cmd_callback_left, this, std::placeholders::_1));
    joint_cmd_sub_right = create_subscription<sensor_msgs::msg::JointState>(
        "joint_commands_right", 10, std::bind(&StarkNode::joint_cmd_callback_right, this, std::placeholders::_1));
    // Initialize timer
    timer_ = create_wall_timer(
        std::chrono::milliseconds(hold_period_ms_),
        std::bind(&StarkNode::timer_callback, this));

    // 设备电量等信息，不需要频繁更新
    info_timer_ = create_wall_timer(
        std::chrono::seconds(30),
        std::bind(&StarkNode::info_timer_callback, this));
}

StarkNode::~StarkNode() {
    if (handle_l) {
        set_turbo_mode(handle_l, slave_id_l, false);
        modbus_close(handle_l);
    }
    if (handle_r) {
        set_turbo_mode(handle_r, slave_id_r, false);
        modbus_close(handle_r);
    }
}

bool StarkNode::set_turbo_mode(ModbusHandle* handle, uint8_t slave_id, bool enabled) {
    if (!handle || !turbo_mode_fn_) return false;
    turbo_mode_fn_(static_cast<void*>(handle), slave_id, enabled);
    return true;
}

bool StarkNode::initialize_modbus() {
    if (port_l.empty() && port_r.empty()) return false;
    // Missing USB hardware is recoverable; an empty port intentionally stays off.
    reconnect_hand(true);
    reconnect_hand(false);
    return true;
}

void StarkNode::configure_hand(bool left) {
    auto handle = left ? handle_l : handle_r;
    const auto slave = left ? slave_id_l : slave_id_r;
    for (int fid = 1; fid <= 6; ++fid) {
        const auto finger = static_cast<StarkFingerId>(fid);
        modbus_set_finger_protect_current(handle, slave, finger, static_cast<uint16_t>(protect_current_));
        modbus_set_finger_max_current(handle, slave, finger, static_cast<uint16_t>(max_current_));
    }
    if (turbo_always_on_) set_turbo_mode(handle, slave, true);
}

void StarkNode::disconnect_hand(bool left) {
    auto& handle = left ? handle_l : handle_r;
    auto& link = left ? link_l_ : link_r_;
    (left ? last_cmd_valid_l_ : last_cmd_valid_r_) = false;
    link.feedback_ok = false;
    link.failures = 0;
    link.retry_after = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    if (handle) {
        // No device writes (including Turbo changes) through a stale USB handle.
        modbus_close(handle);
        handle = nullptr;
        RCLCPP_WARN(this->get_logger(), "%s hand disconnected; cached target dropped, retry in 1 s",
                    left ? "Left" : "Right");
    }
}

bool StarkNode::hand_current(bool left) {
    const auto handle = left ? handle_l : handle_r;
    if (!handle) return false;
    const auto& port = left ? port_l : port_r;
    const auto& link = left ? link_l_ : link_r_;
    struct stat current{};
    // A by-id name can stay the same while its ttyUSB node is replaced. Compare
    // inode as well as device number, including replacement at the same tty name.
    if (stat(port.c_str(), &current) == 0 &&
        current.st_dev == link.device.st_dev && current.st_ino == link.device.st_ino &&
        current.st_rdev == link.device.st_rdev) return true;
    disconnect_hand(left);
    return false;
}

void StarkNode::reconnect_hand(bool left) {
    const auto& port = left ? port_l : port_r;
    if (port.empty() || hand_current(left)) return;
    auto& link = left ? link_l_ : link_r_;
    const auto now = std::chrono::steady_clock::now();
    if (now < link.retry_after) return;
    link.retry_after = now + std::chrono::seconds(1);
    struct stat device{};
    if (stat(port.c_str(), &device) != 0 || !S_ISCHR(device.st_mode)) return;
    auto& handle = left ? handle_l : handle_r;
    handle = modbus_open(port.c_str(), baudrate_, left ? slave_id_l : slave_id_r);
    if (!handle) {
        RCLCPP_WARN(this->get_logger(), "Cannot reopen %s; retry in 1 s", port.c_str());
        return;
    }
    link.device = device;
    link.feedback_ok = false;
    link.failures = 0;
    (left ? last_cmd_valid_l_ : last_cmd_valid_r_) = false;
    if (!hand_current(left)) return;  // USB changed during open.
    configure_hand(left);
    RCLCPP_INFO(this->get_logger(), "%s hand opened %s; waiting for feedback and a fresh command",
                left ? "Left" : "Right", port.c_str());
}

void StarkNode::record_feedback(bool left, bool ok) {
    auto& link = left ? link_l_ : link_r_;
    link.feedback_ok = ok;
    if (ok) {
        link.failures = 0;
    } else {
        (left ? last_cmd_valid_l_ : last_cmd_valid_r_) = false;
        if (++link.failures >= 3) disconnect_hand(left);
    }
}

void StarkNode::timer_callback() {
    reconnect_hand(true);
    reconnect_hand(false);
    publish_motor_status();
    publish_motor_status_r();
    // publish_touch_status();

    // Re-send latest target each tick (~20Hz) so that if a finger stalls on
    // a temporary obstacle (protect_current triggered → motor stopped), the
    // motion automatically resumes as soon as the obstacle is removed. Without
    // this, the hand keeps the last "stopped" state forever until a new ROS2
    // joint_commands_{left,right} message arrives.
    if (hand_current(true) && link_l_.feedback_ok && last_cmd_valid_l_) {
        uint16_t pos[6];
        for (int i = 0; i < 6; ++i) pos[i] = last_cmd_pos_l[i] / 10;  // 0~1000 → 0~100
        modbus_set_finger_positions(handle_l, slave_id_l, pos, 6);
    }
    if (hand_current(false) && link_r_.feedback_ok && last_cmd_valid_r_) {
        uint16_t pos[6];
        for (int i = 0; i < 6; ++i) pos[i] = last_cmd_pos_r[i] / 10;
        modbus_set_finger_positions(handle_r, slave_id_r, pos, 6);
    }
}

void StarkNode::info_timer_callback() {
    publish_device_status();
}

void StarkNode::publish_joint_state() {
    if (!hand_current(true)) return;
    auto motor_status = modbus_get_motor_status(handle_l, slave_id_l);
    if (!motor_status) {
        if (log_screen_) {RCLCPP_WARN(this->get_logger(), "Failed to get LEFT motor status");}
        return;
    }

    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = this->now();
    msg.name = {"thumb", "thumb_aux", "index", "middle", "ring", "pinky"};
    for (int i = 0; i < 6; i++) {
        msg.position.push_back(motor_status->positions[i] / 100.0);  // Normalize to 0-1
        msg.velocity.push_back(motor_status->speeds[i] / 100.0);     // Normalize to -1-1
        msg.effort.push_back(motor_status->currents[i]);
    }

    joint_state_pub_->publish(msg);
    free_motor_status_data(motor_status);
}

void StarkNode::publish_motor_status() {
    if (!hand_current(true)) return;
    auto motor_status = modbus_get_motor_status(handle_l, slave_id_l);
    record_feedback(true, motor_status != nullptr);
    if (!motor_status) {
        if (log_screen_) {RCLCPP_WARN(this->get_logger(), "Failed to get LEFT motor status");}
        return;
    }

    auto msg = ros2_stark_interfaces::msg::MotorStatus();
    // 使用 std::copy 填充 std::array
    for (int i = 0; i < 6; ++i) {
        msg.positions[i] = unwrap_pos(motor_status->positions[i], last_cmd_pos_l[i]);
    }
    std::copy(motor_status->speeds, motor_status->speeds + 6, msg.speeds.begin());
    std::copy(motor_status->currents, motor_status->currents + 6, msg.currents.begin());
    std::copy(motor_status->states, motor_status->states + 6, msg.states.begin());

    motor_status_pub_->publish(msg);
    free_motor_status_data(motor_status);
}

void StarkNode::publish_motor_status_r() {
    if (!hand_current(false)) return;
    auto motor_status_r = modbus_get_motor_status(handle_r, slave_id_r);
    record_feedback(false, motor_status_r != nullptr);
    if (!motor_status_r) {
        if (log_screen_) {RCLCPP_WARN(this->get_logger(), "Failed to get RIGHT motor status");}
        return;
    }

    auto msg = ros2_stark_interfaces::msg::MotorStatus();
    // 使用 std::copy 填充 std::array
    for (int i = 0; i < 6; ++i) {
        msg.positions[i] = unwrap_pos(motor_status_r->positions[i], last_cmd_pos_r[i]);
    }
    std::copy(motor_status_r->speeds, motor_status_r->speeds + 6, msg.speeds.begin());
    std::copy(motor_status_r->currents, motor_status_r->currents + 6, msg.currents.begin());
    std::copy(motor_status_r->states, motor_status_r->states + 6, msg.states.begin());

    motor_status_pub_r->publish(msg);
    free_motor_status_data(motor_status_r);
}

void StarkNode::publish_touch_status() {
    auto touch_status = modbus_get_touch_status(handle_l, slave_id_l);
    // auto touch_status_r = modbus_get_touch_status(handle_r, slave_id_r);
    // if (!touch_status) {
    //     RCLCPP_WARN(this->get_logger(), "Failed to get LEFT touch status");
    //     return;
    // }
    // if (!touch_status_r) {
    //     RCLCPP_WARN(this->get_logger(), "Failed to get RIGHT touch status");
    //     return;
    // }

    auto msg = ros2_stark_interfaces::msg::TouchStatus();
    // 填充固定 5 个元素的数组, 代表 5个手指上对应的传感器信息
    for (int i = 0; i < 5; i++) {
        msg.data[i].normal_force1 = touch_status->data[i].normal_force1;
        msg.data[i].normal_force2 = touch_status->data[i].normal_force2;
        msg.data[i].normal_force3 = touch_status->data[i].normal_force3;
        msg.data[i].tangential_force1 = touch_status->data[i].tangential_force1;
        msg.data[i].tangential_force2 = touch_status->data[i].tangential_force2;
        msg.data[i].tangential_force3 = touch_status->data[i].tangential_force3;
        msg.data[i].tangential_direction1 = touch_status->data[i].tangential_direction1;
        msg.data[i].tangential_direction2 = touch_status->data[i].tangential_direction2;
        msg.data[i].tangential_direction3 = touch_status->data[i].tangential_direction3;
        msg.data[i].self_proximity1 = touch_status->data[i].self_proximity1;
        msg.data[i].self_proximity2 = touch_status->data[i].self_proximity2;
        msg.data[i].mutual_proximity = touch_status->data[i].mutual_proximity;
        msg.data[i].status = touch_status->data[i].status;
    }

    touch_status_pub_->publish(msg);
    free_touch_status_data(touch_status);
}

void StarkNode::publish_device_status() {
    if (!hand_current(true) || !link_l_.feedback_ok) return;
    // 电量
    auto voltage_l = modbus_get_voltage(handle_l, slave_id_l);
    auto voltage_msg = std_msgs::msg::UInt16();
    voltage_msg.data = voltage_l;
    voltage_pub_->publish(voltage_msg);

}



void StarkNode::joint_cmd_callback_left(const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (!hand_current(true) || !link_l_.feedback_ok) return;
    if (log_screen_) {RCLCPP_INFO(this->get_logger(), "Received joint commands left");}
    if (msg->position.size() >= 6) {
        uint16_t positions[6];
        for (int i = 0; i < 6; i++) {
            positions[i] = static_cast<uint16_t>(msg->position[i] * 100);  // Scale to 0-100
            last_cmd_pos_l[i] = positions[i] * 10;  // SDK 0~1000 internal scale (for unwrap)
        }
        if (log_screen_) {
            RCLCPP_INFO(this->get_logger(), "Left Joint positions: %d, %d, %d, %d, %d, %d",
                    positions[0], positions[1], positions[2], positions[3], positions[4], positions[5]);}
        last_cmd_valid_l_ = true;
        modbus_set_finger_positions(handle_l, slave_id_l, positions, 6);
    }
}

void StarkNode::joint_cmd_callback_right(const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (!hand_current(false) || !link_r_.feedback_ok) return;
    if (log_screen_) {RCLCPP_INFO(this->get_logger(), "Received joint commands right");}
    if (msg->position.size() >= 6) {
        uint16_t positions[6];
        for (int i = 0; i < 6; i++) {
            positions[i] = static_cast<uint16_t>(msg->position[i] * 100);  // Scale to 0-100
            last_cmd_pos_r[i] = positions[i] * 10;  // SDK 0~1000 internal scale (for unwrap)
        }
        if (log_screen_) {
            RCLCPP_INFO(this->get_logger(), "Right Joint positions: %d, %d, %d, %d, %d, %d",
                    positions[0], positions[1], positions[2], positions[3], positions[4], positions[5]);}
        last_cmd_valid_r_ = true;
        modbus_set_finger_positions(handle_r, slave_id_r, positions, 6);
    }
}


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StarkNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
