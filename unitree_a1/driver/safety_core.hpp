// Safety layer between the 50 Hz Python policy and the 500 Hz motor loop.
//
// Header-only and free of the Unitree SDK so it builds and is unit-tested off the robot. Everything here
// runs inside the motor loop, so a hung or crashed Python process still ends in damping.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>

namespace talon_a1 {

constexpr int kNumJoints = 12;  // unitree_legged_sdk order: FR, FL, RR, RL x (hip, thigh, calf)
using Joints = std::array<double, kNumJoints>;

// Latched until clear_faults(). Any fault forces damping.
enum Fault : uint32_t {
  kFaultStaleTarget = 1u << 0,    // active and no set_target() for longer than target_timeout_s
  kFaultStaleState = 1u << 1,     // active and LowState.tick stopped changing
  kFaultTilt = 1u << 2,           // active and |roll| or |pitch| beyond tilt_limit_rad
  kFaultEstopButton = 1u << 3,    // remote L2 + B, honoured in every mode
  kFaultEstopSoftware = 1u << 4,  // estop() called
  kFaultBadTarget = 1u << 5,      // non-finite target, or gains outside [0, max]
  kFaultOverspeed = 1u << 6,      // active and |dq| beyond dq_limit
  kFaultOverpower = 1u << 7,      // active and sum |tau_est * dq| beyond power_limit_w for power_hold_steps in a row
  kFaultOvertorque = 1u << 8,     // active and |tau_est| beyond tau_limit
};

enum class Mode : int { kDamping = 0, kActive = 1 };

struct SafetyConfig {
  // talon-rl D2 stale-data contract: <=20 ms normal, 20-40 ms hold the previous target, >40 ms safe stop + latch.
  double target_timeout_s = 0.040;
  double state_timeout_s = 0.050;
  double tilt_limit_rad = 0.5;  // past the training terminations (roll 0.4, pitch 0.2) but well before a fall lands
  double dq_limit = 30.0;  // above the 21 rad/s DCMotor limit used in training, so it only trips on faults
  double tau_limit = 30.0;  // N*m; sport mode peaked at 29 walking on the robot, the motor saturates near 33.5
  double power_limit_w = 300.0;  // about twice sport mode's 140 W peak walking on the robot; not yet set from a policy run
  int power_hold_steps = 10;  // 20 ms at 500 Hz, so one noisy tau_est sample does not stop the robot
  double kp_max = 60.0;  // top of the Kp range the deployment RMA policy trained with
  double kd_max = 5.0;
  double damping_kd = 1.0;
  double q_margin = 0.05;  // targets are clamped this far inside the joint limits
  // a1_const.h joint limits per joint type (hip, thigh, calf)
  std::array<double, 3> q_min = {{-0.802, -1.05, -2.7}};
  std::array<double, 3> q_max = {{0.802, 4.19, -0.916}};
};

struct Sensors {
  Joints q{};
  Joints dq{};
  Joints tau{};  // MotorState.tauEst
  double roll = 0.0;
  double pitch = 0.0;
  uint32_t tick = 0;  // LowState.tick, changes with every fresh packet
  bool estop_button = false;
};

struct MotorCommand {
  Joints q{};
  Joints dq{};
  Joints kp{};
  Joints kd{};
  Joints tau{};
};

class SafetyCore {
 public:
  explicit SafetyCore(const SafetyConfig& cfg = SafetyConfig()) : cfg_(cfg) {}

  // -- policy side --------------------------------------------------------------------------------------
  bool set_target(const Joints& q, double kp, double kd, double now) {
    std::lock_guard<std::mutex> lock(mu_);
    bool ok = std::isfinite(kp) && std::isfinite(kd) && kp >= 0.0 && kd >= 0.0 && kp <= cfg_.kp_max &&
              kd <= cfg_.kd_max;
    for (double v : q) ok = ok && std::isfinite(v);
    if (!ok) {
      latch(kFaultBadTarget);
      return false;
    }
    for (int i = 0; i < kNumJoints; ++i) target_[i] = clamp_joint(i, q[i]);
    kp_ = kp;
    kd_ = kd;
    target_time_ = now;
    return true;
  }

  // Active needs: no fault, a fresh target, and live robot state.
  bool arm(double now) {
    std::lock_guard<std::mutex> lock(mu_);
    if (faults_ != 0 || target_time_ < 0.0 || now - target_time_ > cfg_.target_timeout_s) return false;
    if (!state_seen_ || now - last_tick_time_ > cfg_.state_timeout_s) return false;
    mode_ = Mode::kActive;
    return true;
  }

  void disarm() {
    std::lock_guard<std::mutex> lock(mu_);
    mode_ = Mode::kDamping;
  }

  void estop() {
    std::lock_guard<std::mutex> lock(mu_);
    latch(kFaultEstopSoftware);
  }

  // Clears every latch. The old target is dropped, so arming again needs a new set_target().
  void clear_faults() {
    std::lock_guard<std::mutex> lock(mu_);
    faults_ = 0;
    mode_ = Mode::kDamping;
    target_time_ = -1.0;
  }

  Mode mode() const {
    std::lock_guard<std::mutex> lock(mu_);
    return mode_;
  }

  uint32_t faults() const {
    std::lock_guard<std::mutex> lock(mu_);
    return faults_;
  }

  double target_age(double now) const {
    std::lock_guard<std::mutex> lock(mu_);
    return target_time_ < 0.0 ? -1.0 : now - target_time_;
  }

  // -- motor loop ---------------------------------------------------------------------------------------
  MotorCommand step(const Sensors& s, double now) {
    std::lock_guard<std::mutex> lock(mu_);
    if (s.tick != last_tick_) {
      if (last_tick_time_ >= 0.0) state_seen_ = true;  // seen it change at least once: the robot is talking
      last_tick_ = s.tick;
      last_tick_time_ = now;
    }
    if (s.estop_button) latch(kFaultEstopButton);
    if (mode_ == Mode::kActive) {
      if (now - target_time_ > cfg_.target_timeout_s) latch(kFaultStaleTarget);
      if (now - last_tick_time_ > cfg_.state_timeout_s) latch(kFaultStaleState);
      if (std::fabs(s.roll) > cfg_.tilt_limit_rad || std::fabs(s.pitch) > cfg_.tilt_limit_rad) latch(kFaultTilt);
      for (double v : s.dq) {
        if (!(std::fabs(v) <= cfg_.dq_limit)) {
          latch(kFaultOverspeed);
          break;
        }
      }
      for (double v : s.tau) {
        if (!(std::fabs(v) <= cfg_.tau_limit)) {
          latch(kFaultOvertorque);
          break;
        }
      }
      double power = 0.0;
      for (int i = 0; i < kNumJoints; ++i) power += std::fabs(s.tau[i] * s.dq[i]);
      power_over_ = power <= cfg_.power_limit_w ? 0 : power_over_ + 1;
      if (power_over_ >= cfg_.power_hold_steps) latch(kFaultOverpower);
    } else {
      power_over_ = 0;
    }

    MotorCommand c;
    for (int i = 0; i < kNumJoints; ++i) {
      if (mode_ == Mode::kActive) {
        c.q[i] = target_[i];
        c.kp[i] = kp_;
        c.kd[i] = kd_;
      } else {
        c.q[i] = s.q[i];
        c.kp[i] = 0.0;
        c.kd[i] = cfg_.damping_kd;
      }
    }
    return c;
  }

 private:
  void latch(uint32_t fault) {
    faults_ |= fault;
    mode_ = Mode::kDamping;
  }

  double clamp_joint(int i, double v) const {
    const int type = i % 3;
    const double lo = cfg_.q_min[type] + cfg_.q_margin;
    const double hi = cfg_.q_max[type] - cfg_.q_margin;
    return v < lo ? lo : (v > hi ? hi : v);
  }

  mutable std::mutex mu_;
  SafetyConfig cfg_;
  Mode mode_ = Mode::kDamping;
  uint32_t faults_ = 0;
  Joints target_{};
  double kp_ = 0.0;
  double kd_ = 0.0;
  double target_time_ = -1.0;
  uint32_t last_tick_ = 0;
  double last_tick_time_ = -1.0;
  bool state_seen_ = false;
  int power_over_ = 0;
};

}  // namespace talon_a1
