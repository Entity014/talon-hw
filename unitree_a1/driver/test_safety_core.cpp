// Unit tests for SafetyCore. No framework: each CHECK prints the failing line and the run exits non-zero.
#include <cstdio>
#include <limits>

#include "safety_core.hpp"

using namespace talon_a1;

static int failures = 0;
#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

static Joints stand() {
  Joints q{};
  for (int leg = 0; leg < 4; ++leg) {
    q[3 * leg + 0] = 0.0;
    q[3 * leg + 1] = 0.9;
    q[3 * leg + 2] = -1.5;
  }
  return q;
}

// Feeds two ticks so the core has seen the robot talk.
static void live(SafetyCore& core, Sensors& s, double& t) {
  s.tick = 1;
  core.step(s, t);
  t += 0.002;
  s.tick = 2;
  core.step(s, t);
}

static void test_arm_preconditions() {
  SafetyCore core;
  Sensors s;
  double t = 0.0;
  CHECK(core.mode() == Mode::kDamping);
  CHECK(!core.arm(t));  // no target, no state
  CHECK(core.set_target(stand(), 25.0, 0.5, t));
  CHECK(!core.arm(t));  // no state yet
  live(core, s, t);
  CHECK(core.set_target(stand(), 25.0, 0.5, t));
  CHECK(core.arm(t));
  MotorCommand c = core.step(s, t);
  CHECK(c.kp[0] == 25.0 && c.kd[0] == 0.5 && c.q[1] == 0.9);
}

static void test_damping_output() {
  SafetyCore core;
  Sensors s;
  s.q[4] = 1.23;
  MotorCommand c = core.step(s, 0.0);
  CHECK(c.kp[4] == 0.0 && c.kd[4] == SafetyConfig().damping_kd && c.q[4] == 1.23);
}

static void test_stale_target_hold_then_latch() {
  SafetyCore core;
  Sensors s;
  double t = 0.0;
  live(core, s, t);
  core.set_target(stand(), 25.0, 0.5, t);
  CHECK(core.arm(t));
  for (int k = 0; k < 15; ++k) {  // 30 ms with fresh state: still holding the old target
    t += 0.002;
    s.tick += 1;
    core.step(s, t);
  }
  CHECK(core.mode() == Mode::kActive && core.faults() == 0);
  for (int k = 0; k < 6; ++k) {  // past 40 ms
    t += 0.002;
    s.tick += 1;
    core.step(s, t);
  }
  CHECK(core.mode() == Mode::kDamping && (core.faults() & kFaultStaleTarget));
  core.clear_faults();
  CHECK(core.faults() == 0);
  CHECK(!core.arm(t));  // the old target was dropped
  core.set_target(stand(), 25.0, 0.5, t);
  CHECK(core.arm(t));
}

static void test_stale_state() {
  SafetyCore core;
  Sensors s;
  double t = 0.0;
  live(core, s, t);
  core.set_target(stand(), 25.0, 0.5, t);
  CHECK(core.arm(t));
  for (int k = 0; k < 30; ++k) {  // tick frozen for 60 ms, target kept fresh
    t += 0.002;
    core.set_target(stand(), 25.0, 0.5, t);
    core.step(s, t);
  }
  CHECK(core.faults() & kFaultStaleState);
}

static void test_estop_button_any_mode() {
  SafetyCore core;
  Sensors s;
  s.estop_button = true;
  core.step(s, 0.0);
  CHECK(core.faults() & kFaultEstopButton);
}

static void test_bad_target() {
  SafetyCore core;
  Joints q = stand();
  q[3] = std::numeric_limits<double>::quiet_NaN();
  CHECK(!core.set_target(q, 25.0, 0.5, 0.0));
  CHECK(core.faults() & kFaultBadTarget);
  core.clear_faults();
  CHECK(!core.set_target(stand(), 100.0, 0.5, 0.0));  // kp above kp_max
  CHECK(core.faults() & kFaultBadTarget);
}

static void test_tilt_and_overspeed_only_when_active() {
  SafetyCore core;
  Sensors s;
  double t = 0.0;
  s.roll = 1.2;
  live(core, s, t);
  CHECK(core.faults() == 0);  // lifting a damped robot is fine
  s.roll = 0.0;
  core.set_target(stand(), 25.0, 0.5, t);
  CHECK(core.arm(t));
  s.pitch = -0.9;
  s.tick += 1;
  core.step(s, t);
  CHECK(core.faults() & kFaultTilt);

  SafetyCore core2;
  Sensors s2;
  t = 0.0;
  live(core2, s2, t);
  core2.set_target(stand(), 25.0, 0.5, t);
  CHECK(core2.arm(t));
  s2.dq[7] = 40.0;
  s2.tick += 1;
  core2.step(s2, t);
  CHECK(core2.faults() & kFaultOverspeed);
}

static void test_default_tilt_limit() {
  SafetyCore core;
  Sensors s;
  double t = 0.0;
  live(core, s, t);
  core.set_target(stand(), 25.0, 0.5, t);
  CHECK(core.arm(t));
  s.roll = 0.45;
  s.tick += 1;
  core.step(s, t);
  CHECK(core.faults() == 0);
  s.roll = 0.55;
  s.tick += 1;
  core.step(s, t);
  CHECK(core.faults() & kFaultTilt);
}

static void test_overtorque_only_when_active() {
  SafetyCore core;
  Sensors s;
  double t = 0.0;
  s.tau[5] = 31.0;
  live(core, s, t);
  CHECK(core.faults() == 0);  // the robot is damped, so nothing here is commanding that torque
  s.tau[5] = 29.0;
  core.set_target(stand(), 25.0, 0.5, t);
  CHECK(core.arm(t));
  s.tick += 1;
  core.step(s, t);
  CHECK(core.faults() == 0 && core.mode() == Mode::kActive);
  s.tau[5] = -31.0;
  s.tick += 1;
  MotorCommand c = core.step(s, t);
  CHECK(core.faults() & kFaultOvertorque);
  CHECK(core.mode() == Mode::kDamping && c.kp[5] == 0.0);

  SafetyCore core2;
  Sensors s2;
  t = 0.0;
  live(core2, s2, t);
  core2.set_target(stand(), 25.0, 0.5, t);
  CHECK(core2.arm(t));
  s2.tau[0] = std::numeric_limits<double>::quiet_NaN();
  s2.tick += 1;
  core2.step(s2, t);
  CHECK(core2.faults() & kFaultOvertorque);
}

static void test_joint_clamp() {
  SafetyCore core;
  Sensors s;
  double t = 0.0;
  live(core, s, t);
  Joints q = stand();
  q[0] = 2.0;    // hip above 0.802
  q[2] = -0.5;   // calf above -0.916
  core.set_target(q, 25.0, 0.5, t);
  CHECK(core.arm(t));
  MotorCommand c = core.step(s, t);
  CHECK(std::fabs(c.q[0] - (0.802 - 0.05)) < 1e-12);
  CHECK(std::fabs(c.q[2] - (-0.916 - 0.05)) < 1e-12);
}

int main() {
  test_arm_preconditions();
  test_damping_output();
  test_stale_target_hold_then_latch();
  test_stale_state();
  test_estop_button_any_mode();
  test_bad_target();
  test_tilt_and_overspeed_only_when_active();
  test_default_tilt_limit();
  test_overtorque_only_when_active();
  test_joint_clamp();
  if (failures == 0) std::printf("test_safety_core: all passed\n");
  return failures == 0 ? 0 : 1;
}
