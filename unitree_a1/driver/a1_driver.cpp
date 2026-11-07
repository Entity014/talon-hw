// pybind11 module `_a1_driver`: owns the 500 Hz motor loop and the SafetyCore.
//
// Built with TALON_A1_WITH_SDK it talks to the A1 motor controller through unitree_legged_sdk v3.3.x in
// LOWLEVEL mode. Without it the transport is a loopback plant (first-order joints, level IMU) so the loop,
// the safety core and the whole Python stack can run on a desk. Joint order is always the SDK's.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#include "safety_core.hpp"

#ifdef TALON_A1_WITH_SDK
// Not unitree_legged_sdk.h: it pulls in lcm.h (LCM) and loop.h (Boost), neither of which we use.
#include "unitree_legged_sdk/comm.h"
#include "unitree_legged_sdk/quadruped.h"
#include "unitree_legged_sdk/udp.h"
#endif

namespace py = pybind11;

namespace talon_a1 {

// Wireless remote bytes in LowState.wirelessRemote. Layout of xRockerBtnDataStruct in unitree_joystick.h
// (SDK v3.3.1; the header is gone from the 3.3.4 tree but the bytes are unchanged).
#pragma pack(push, 1)
struct RemoteRaw {
  uint8_t head[2];
  uint16_t keys;  // xKeySwitchUnion
  float lx;
  float rx;
  float ry;
  float L2;
  float ly;
  uint8_t idle[16];
};
#pragma pack(pop)
static_assert(sizeof(RemoteRaw) == 40, "wirelessRemote is 40 bytes");

// xKeySwitchUnion bit positions
constexpr uint16_t kKeyR1 = 1u << 0, kKeyL1 = 1u << 1, kKeyStart = 1u << 2, kKeySelect = 1u << 3,
                   kKeyR2 = 1u << 4, kKeyL2 = 1u << 5, kKeyF1 = 1u << 6, kKeyF2 = 1u << 7, kKeyA = 1u << 8,
                   kKeyB = 1u << 9, kKeyX = 1u << 10, kKeyY = 1u << 11, kKeyUp = 1u << 12,
                   kKeyRight = 1u << 13, kKeyDown = 1u << 14, kKeyLeft = 1u << 15;

// Snapshot handed to Python (SDK joint order).
struct RobotState {
  Joints q{};
  Joints dq{};
  Joints tau_est{};
  std::array<double, 4> quat{{1.0, 0.0, 0.0, 0.0}};  // w, x, y, z
  std::array<double, 3> gyro{};                      // body frame, rad/s
  std::array<double, 3> acc{};
  std::array<double, 3> rpy{};
  std::array<int, 4> foot_force{};
  uint32_t tick = 0;
  uint16_t keys = 0;
  double lx = 0.0, ly = 0.0, rx = 0.0, ry = 0.0;
  // filled from the safety core and the loop
  int mode = 0;
  uint32_t faults = 0;
  double target_age = -1.0;
  uint64_t loops = 0;
  uint64_t overruns = 0;
  double max_period = 0.0;
};

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

#ifdef TALON_A1_WITH_SDK

class Transport {
 public:
  Transport() : udp_(UNITREE_LEGGED_SDK::LOWLEVEL) {
    udp_.InitCmdData(cmd_);
  }

  void receive(RobotState& out) {
    udp_.Recv();  // non-blocking socket: returns at once when no packet is waiting
    udp_.GetRecv(state_);
    for (int i = 0; i < kNumJoints; ++i) {
      out.q[i] = state_.motorState[i].q;
      out.dq[i] = state_.motorState[i].dq;
      out.tau_est[i] = state_.motorState[i].tauEst;
    }
    for (int i = 0; i < 4; ++i) out.quat[i] = state_.imu.quaternion[i];
    for (int i = 0; i < 3; ++i) {
      out.gyro[i] = state_.imu.gyroscope[i];
      out.acc[i] = state_.imu.accelerometer[i];
      out.rpy[i] = state_.imu.rpy[i];
    }
    for (int i = 0; i < 4; ++i) out.foot_force[i] = state_.footForce[i];
    out.tick = state_.tick;
    RemoteRaw r;
    std::memcpy(&r, state_.wirelessRemote, sizeof(r));
    out.keys = r.keys;
    out.lx = r.lx;
    out.ly = r.ly;
    out.rx = r.rx;
    out.ry = r.ry;
  }

  // No SDK Safety::PowerProtect: when it trips it calls exit(), which skips damping and loses the run log.
  void send(const MotorCommand& c) {
    for (int i = 0; i < kNumJoints; ++i) {
      UNITREE_LEGGED_SDK::MotorCmd& m = cmd_.motorCmd[i];
      m.mode = 0x0A;  // servo mode
      m.q = static_cast<float>(c.q[i]);
      m.dq = static_cast<float>(c.dq[i]);
      m.Kp = static_cast<float>(c.kp[i]);
      m.Kd = static_cast<float>(c.kd[i]);
      m.tau = static_cast<float>(c.tau[i]);
    }
    udp_.SetSend(cmd_);
    udp_.Send();
  }

 private:
  UNITREE_LEGGED_SDK::UDP udp_;
  UNITREE_LEGGED_SDK::LowCmd cmd_ = {0};
  UNITREE_LEGGED_SDK::LowState state_ = {0};
};

#else

// Desk plant: each joint is an independent damped inertia driven by the commanded PD law, with a level IMU.
// It only exists to exercise the loop, the safety core and the Python stack; it is not a robot model.
class Transport {
 public:
  Transport() {
    for (int leg = 0; leg < 4; ++leg) {  // lying down, roughly
      q_[3 * leg + 0] = 0.0;
      q_[3 * leg + 1] = 1.1;
      q_[3 * leg + 2] = -2.6;
    }
  }

  void receive(RobotState& out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!freeze_) ++tick_;
    out.q = q_;
    out.dq = dq_;
    out.tick = tick_;
    out.rpy = {{roll_, pitch_, 0.0}};
    // quaternion from roll/pitch (yaw 0)
    const double cr = std::cos(roll_ / 2), sr = std::sin(roll_ / 2), cp = std::cos(pitch_ / 2),
                 sp = std::sin(pitch_ / 2);
    out.quat = {{cr * cp, sr * cp, cr * sp, -sr * sp}};
    out.tau_est = tau_est_;
    out.keys = keys_;
    out.lx = lx_;
    out.ly = ly_;
    out.rx = rx_;
    out.ry = ry_;
  }

  void send(const MotorCommand& c) {
    std::lock_guard<std::mutex> lock(mu_);
    const double dt = 0.002, inertia = 0.05, friction = 0.1;
    for (int i = 0; i < kNumJoints; ++i) {
      const double tau = c.kp[i] * (c.q[i] - q_[i]) + c.kd[i] * (c.dq[i] - dq_[i]) + c.tau[i];
      dq_[i] += dt * (tau - friction * dq_[i]) / inertia;
      q_[i] += dt * dq_[i];
    }
  }

  void set_remote(uint16_t keys, double lx, double ly, double rx, double ry) {
    std::lock_guard<std::mutex> lock(mu_);
    keys_ = keys;
    lx_ = lx;
    ly_ = ly;
    rx_ = rx;
    ry_ = ry;
  }

  void set_tau_est(int joint, double tau) {
    std::lock_guard<std::mutex> lock(mu_);
    tau_est_[joint] = tau;
  }

  void set_tilt(double roll, double pitch) {
    std::lock_guard<std::mutex> lock(mu_);
    roll_ = roll;
    pitch_ = pitch;
  }

  void freeze_state(bool freeze) {
    std::lock_guard<std::mutex> lock(mu_);
    freeze_ = freeze;
  }

 private:
  std::mutex mu_;
  Joints q_{};
  Joints dq_{};
  uint32_t tick_ = 0;
  bool freeze_ = false;
  double roll_ = 0.0, pitch_ = 0.0;
  Joints tau_est_{};
  uint16_t keys_ = 0;
  double lx_ = 0.0, ly_ = 0.0, rx_ = 0.0, ry_ = 0.0;
};

#endif

class Driver {
 public:
  Driver(const SafetyConfig& cfg, double dt) : core_(cfg), dt_(dt) {}
  ~Driver() { stop(); }

  void start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread(&Driver::loop, this);
    // SCHED_FIFO needs root or CAP_SYS_NICE; without it the loop still runs, just with more jitter.
    sched_param p;
    p.sched_priority = 90;
    realtime_ = pthread_setschedparam(thread_.native_handle(), SCHED_FIFO, &p) == 0;
  }

  void stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
  }

  RobotState state() const {
    RobotState s;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      s = state_;
    }
    s.mode = static_cast<int>(core_.mode());
    s.faults = core_.faults();
    s.target_age = core_.target_age(now_s());
    return s;
  }

  bool set_target(const Joints& q, double kp, double kd) { return core_.set_target(q, kp, kd, now_s()); }
  bool arm() { return core_.arm(now_s()); }
  void disarm() { core_.disarm(); }
  void estop() { core_.estop(); }
  void clear_faults() { core_.clear_faults(); }
  bool realtime() const { return realtime_; }
  bool running() const { return running_; }

#ifndef TALON_A1_WITH_SDK
  Transport& plant() { return transport_; }
#endif

 private:
  void loop() {
    using clock = std::chrono::steady_clock;
    const auto period = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(dt_));
    auto next = clock::now();
    double last = now_s();
    RobotState s;
    while (running_) {
      transport_.receive(s);
      Sensors sen;
      sen.q = s.q;
      sen.dq = s.dq;
      sen.tau = s.tau_est;
      sen.roll = s.rpy[0];
      sen.pitch = s.rpy[1];
      sen.tick = s.tick;
      sen.estop_button = (s.keys & kKeyL2) && (s.keys & kKeyB);
      const double t = now_s();
      transport_.send(core_.step(sen, t));

      s.loops += 1;
      s.max_period = std::max(s.max_period, t - last);
      last = t;
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        state_ = s;
      }
      next += period;
      const auto now = clock::now();
      if (now > next) {
        s.overruns += 1;
        next = now;
      } else {
        std::this_thread::sleep_until(next);
      }
    }
    // Leave the motors damped: the last frames the robot sees are Kp = 0.
    core_.disarm();
    for (int k = 0; k < 50; ++k) {
      transport_.receive(s);
      Sensors sen;
      sen.q = s.q;
      sen.tick = s.tick;
      transport_.send(core_.step(sen, now_s()));
      std::this_thread::sleep_for(period);
    }
  }

  SafetyCore core_;
  Transport transport_;
  double dt_;
  std::atomic<bool> running_{false};
  bool realtime_ = false;
  std::thread thread_;
  mutable std::mutex state_mu_;
  RobotState state_;
};

}  // namespace talon_a1

PYBIND11_MODULE(_a1_driver, m) {
  using namespace talon_a1;
  m.doc() = "A1 low-level motor loop with a latched safety core (SDK joint order)";
#ifdef TALON_A1_WITH_SDK
  m.attr("WITH_SDK") = true;
#else
  m.attr("WITH_SDK") = false;
#endif

  py::dict faults;
  faults["stale_target"] = static_cast<uint32_t>(kFaultStaleTarget);
  faults["stale_state"] = static_cast<uint32_t>(kFaultStaleState);
  faults["tilt"] = static_cast<uint32_t>(kFaultTilt);
  faults["estop_button"] = static_cast<uint32_t>(kFaultEstopButton);
  faults["estop_software"] = static_cast<uint32_t>(kFaultEstopSoftware);
  faults["bad_target"] = static_cast<uint32_t>(kFaultBadTarget);
  faults["overspeed"] = static_cast<uint32_t>(kFaultOverspeed);
  faults["overpower"] = static_cast<uint32_t>(kFaultOverpower);
  faults["overtorque"] = static_cast<uint32_t>(kFaultOvertorque);
  m.attr("FAULTS") = faults;

  py::dict keys;
  const std::pair<const char*, uint16_t> key_bits[] = {
      {"R1", kKeyR1}, {"L1", kKeyL1}, {"START", kKeyStart}, {"SELECT", kKeySelect}, {"R2", kKeyR2},
      {"L2", kKeyL2}, {"F1", kKeyF1}, {"F2", kKeyF2},       {"A", kKeyA},           {"B", kKeyB},
      {"X", kKeyX},   {"Y", kKeyY},   {"UP", kKeyUp},       {"RIGHT", kKeyRight},   {"DOWN", kKeyDown},
      {"LEFT", kKeyLeft}};
  for (const auto& kv : key_bits) keys[kv.first] = kv.second;
  m.attr("KEYS") = keys;

  py::class_<SafetyConfig>(m, "SafetyConfig")
      .def(py::init<>())
      .def_readwrite("target_timeout_s", &SafetyConfig::target_timeout_s)
      .def_readwrite("state_timeout_s", &SafetyConfig::state_timeout_s)
      .def_readwrite("tilt_limit_rad", &SafetyConfig::tilt_limit_rad)
      .def_readwrite("dq_limit", &SafetyConfig::dq_limit)
      .def_readwrite("tau_limit", &SafetyConfig::tau_limit)
      .def_readwrite("power_limit_w", &SafetyConfig::power_limit_w)
      .def_readwrite("kp_max", &SafetyConfig::kp_max)
      .def_readwrite("kd_max", &SafetyConfig::kd_max)
      .def_readwrite("damping_kd", &SafetyConfig::damping_kd)
      .def_readwrite("q_margin", &SafetyConfig::q_margin)
      .def_readwrite("q_min", &SafetyConfig::q_min)
      .def_readwrite("q_max", &SafetyConfig::q_max);

  py::class_<RobotState>(m, "RobotState")
      .def_readonly("q", &RobotState::q)
      .def_readonly("dq", &RobotState::dq)
      .def_readonly("tau_est", &RobotState::tau_est)
      .def_readonly("quat", &RobotState::quat)
      .def_readonly("gyro", &RobotState::gyro)
      .def_readonly("acc", &RobotState::acc)
      .def_readonly("rpy", &RobotState::rpy)
      .def_readonly("foot_force", &RobotState::foot_force)
      .def_readonly("tick", &RobotState::tick)
      .def_readonly("keys", &RobotState::keys)
      .def_readonly("lx", &RobotState::lx)
      .def_readonly("ly", &RobotState::ly)
      .def_readonly("rx", &RobotState::rx)
      .def_readonly("ry", &RobotState::ry)
      .def_readonly("mode", &RobotState::mode)
      .def_readonly("faults", &RobotState::faults)
      .def_readonly("target_age", &RobotState::target_age)
      .def_readonly("loops", &RobotState::loops)
      .def_readonly("overruns", &RobotState::overruns)
      .def_readonly("max_period", &RobotState::max_period);

  py::class_<Driver> drv(m, "Driver");
  drv.def(py::init<const SafetyConfig&, double>(), py::arg("cfg"), py::arg("dt") = 0.002)
      .def("start", &Driver::start)
      .def("stop", &Driver::stop, py::call_guard<py::gil_scoped_release>())
      .def("state", &Driver::state)
      .def("set_target", &Driver::set_target, py::arg("q"), py::arg("kp"), py::arg("kd"))
      .def("arm", &Driver::arm)
      .def("disarm", &Driver::disarm)
      .def("estop", &Driver::estop)
      .def("clear_faults", &Driver::clear_faults)
      .def_property_readonly("realtime", &Driver::realtime)
      .def_property_readonly("running", &Driver::running);
#ifndef TALON_A1_WITH_SDK
  drv.def("sim_set_remote",
          [](Driver& d, uint16_t keys, double lx, double ly, double rx, double ry) {
            d.plant().set_remote(keys, lx, ly, rx, ry);
          },
          py::arg("keys"), py::arg("lx") = 0.0, py::arg("ly") = 0.0, py::arg("rx") = 0.0, py::arg("ry") = 0.0)
      .def("sim_set_tilt", [](Driver& d, double roll, double pitch) { d.plant().set_tilt(roll, pitch); })
      .def("sim_set_tau_est", [](Driver& d, int joint, double tau) { d.plant().set_tau_est(joint, tau); })
      .def("sim_freeze_state", [](Driver& d, bool freeze) { d.plant().freeze_state(freeze); });
#endif
}
