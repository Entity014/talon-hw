"""Run the deployment RMA policy on the A1 (or on the loopback desk plant when the driver was built without the SDK).

    python3 scripts/run.py --policy policies/rma_a1.npz

Hang the robot on a gantry for the first runs. Remote combos are listed in talon_a1/controller.py.
Ctrl-C disarms, leaves the motors damped and writes the run log.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from talon_a1 import _a1_driver, contract, sport  # noqa: E402
from talon_a1.controller import Controller, RunLog, obs_noise_amplitude, provenance  # noqa: E402
from talon_a1.policy import OnnxPolicy, RmaPolicy, load_policy  # noqa: E402
from talon_a1.remote import MacroRemote, combo  # noqa: E402


def desk_demo(drv):
    """Loopback only: press the remote combos a person would, so the full stack runs unattended."""
    for at, keys, ly in ((0.5, combo("L1", "A"), 0.0), (0.7, 0, 0.0), (4.0, combo("L1", "X"), 0.0),
                         (4.2, 0, 0.6), (7.0, combo("L1", "Y"), 0.0), (7.2, 0, 0.0), (9.0, combo("L1", "B"), 0.0)):
        time.sleep(max(0.0, at - (time.monotonic() - desk_demo.t0)))
        drv.sim_set_remote(keys, 0.0, ly, 0.0, 0.0)


def wait_for_robot(drv, timeout=1.0):
    first = drv.state().tick
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if drv.state().tick != first:
            return True
        time.sleep(0.01)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--action-clip", type=float, default=None,
                   help="bound on |action|. Default: 1 for the RMA policy (tanh-bounded), 10 for PROMO npz/onnx policies, "
                        "whose output is linear and needs |a| > 1 to step (clipping at 1 made oursA lean and never walk)")
    p.add_argument("--kp", type=float, default=None,
                   help="bring-up, restrained only: policy Kp, below 50 is outside training (default from policy)")
    p.add_argument("--action-scale", type=float, default=None,
                   help="rad per unit action; below default is outside training (default from policy)")
    p.add_argument("--w", nargs=3, type=float, default=[0.333, 0.333, 0.334],
                   help="preference vector w = (track, stab, eff), for PROMO models (default: balanced 0.333 0.333 0.334)")
    p.add_argument("--soft-start", type=float, default=1.0,
                   help="seconds to ease the action and gains in after L1+X, 0 = jump at once")
    p.add_argument("--stand-only", action="store_true", help="L1+X does nothing: the policy can never start")
    p.add_argument("--vx-max", type=float, default=0.5)
    p.add_argument("--vy-max", type=float, default=0.3)
    p.add_argument("--wz-max", type=float, default=0.5)
    p.add_argument("--macro-vx", type=float, default=0.6,
                   help="forward speed vx (m/s) for the R1+X joystick walk macro (default: 0.6)")
    p.add_argument("--macro-dist", type=float, default=3.0,
                   help="target distance (m) for the R1+X joystick walk macro (default: 3.0)")
    p.add_argument("--macro-ramp", type=float, default=0.5,
                   help="ramp up/down time (s) for the R1+X joystick walk macro (default: 0.5)")
    p.add_argument("--stand-time", type=float, default=3.0)
    p.add_argument("--power-limit", type=float, default=300.0, help="W of sum |tau_est * dq|, latched fault after 20 ms")
    p.add_argument("--tilt-limit", type=float, default=0.5, help="rad, latched fault while armed")
    p.add_argument("--tau-limit", type=float, default=30.0, help="N*m on tau_est, latched fault while armed")
    p.add_argument("--damping-kd", type=float, default=1.0, help="Kd of the damping mode every fault ends in")
    p.add_argument("--flip-imu-x", action="store_true", default=False,
                   help="rotate IMU frame 180 deg about X (fixes inverted IMU on certain A1 units)")
    p.add_argument("--obs-noise-scale", type=float, default=0.0,
                   help="PROMO policies only: add the TRAINING observation noise (gyro 0.2, gravity 0.05, joint_pos 0.01, joint_vel 1.5, uniform +-) times this scale to the observation the policy sees. 0 = none (default). The real dq is much cleaner than the training noise: this is the A/B test of that mismatch")
    p.add_argument("--dq-noise", type=float, default=None, help="PROMO policies only: uniform +- noise on joint_vel in rad/s (overrides the joint_vel channel of --obs-noise-scale); --dq-noise 1.5 alone is the dq-only A/B test")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--action-smooth", type=float, default=0.0,
                   help="EMA action smoothing factor alpha in [0, 1): a_applied = (1 - alpha) * a_new + alpha * a_prev. 0 = off, recommended 0.2-0.4 to reduce hardware shaking/chatter")
    p.add_argument("--pitch-bias", type=float, default=0.0,
                   help="rad: flex rear calves (RL/RR) in WALK to pitch nose up and shift weight off front legs (try 0.05-0.12)")
    p.add_argument("--front-lift-bias", type=float, default=0.0,
                   help="rad: flex front thigh/calf (FL/FR) proportional to forward cmd vx > 0 to increase front foot clearance (try 0.05-0.12)")
    p.add_argument("--imu-roll-bias", type=float, default=0.0,
                   help="rad: subtract static IMU roll mounting offset from projected_gravity (e.g. 0.07)")
    p.add_argument("--imu-pitch-bias", type=float, default=0.0,
                   help="rad: subtract static IMU pitch mounting offset from projected_gravity")
    p.add_argument("--imu-zero", action="store_true",
                   help="auto-calibrate imu-roll-bias and imu-pitch-bias from STAND pose right before L1+X")
    p.add_argument("--lead-ms", type=float, default=0.0, help="PROMO policies only: lead compensation, joint_pos seen by the policy += joint_vel x lead (ms). 0 = off. In simulation it did not help (bounce 0.52 -> 0.49-0.51 at 15 ms delay); kept as an A/B option for the robot, where the delay structure may differ")
    p.add_argument("--duration", type=float, default=None, help="seconds; default runs until Ctrl-C")
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    p.add_argument("--desk-demo", action="store_true", help="loopback build only: scripted remote presses")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()
    policy = load_policy(args.policy, w=args.w)
    base_action_scale = getattr(policy, "action_scale", contract.ACTION_SCALE)
    base_kp = getattr(policy, "kp", contract.KP)

    action_scale = args.action_scale if args.action_scale is not None else base_action_scale
    policy_kp = args.kp if args.kp is not None else base_kp

    if args.action_clip is None:
        args.action_clip = 10.0 if hasattr(policy, "hist_len") else 1.0
    if not 0.0 < args.action_clip <= 10.0:
        p.error("--action-clip must be in (0, 10]")
    if not 0.0 <= args.action_smooth < 1.0:
        p.error("--action-smooth must be in [0, 1)")
    if not 0.0 < action_scale <= base_action_scale:
        p.error("--action-scale must be in (0, %g]" % base_action_scale)
    if args.soft_start < 0.0:
        p.error("--soft-start must not be negative")
    if not 0.0 < policy_kp <= base_kp:
        p.error("--kp must be in (0, %g]" % base_kp)

    cfg = _a1_driver.SafetyConfig()
    cfg.tilt_limit_rad = args.tilt_limit
    cfg.tau_limit = args.tau_limit
    cfg.power_limit_w = args.power_limit
    cfg.flip_imu_x = args.flip_imu_x
    if not 0.0 < args.damping_kd <= cfg.kd_max:
        p.error("--damping-kd must be in (0, %g]" % cfg.kd_max)
    cfg.damping_kd = args.damping_kd

    if _a1_driver.WITH_SDK:
        sport.require_stopped()
        print("LOW-LEVEL control of the real A1. The robot must hang on the gantry, remote in hand (L2+B = e-stop).")
        if not args.yes:
            input("Press Enter to start the motor loop...")
    else:
        print("Loopback desk plant: no robot is connected.")
    if args.desk_demo and _a1_driver.WITH_SDK:
        sys.exit("--desk-demo is for the loopback build only")

    drv = _a1_driver.Driver(cfg, 0.002)
    drv.start()
    print("motor loop running, SCHED_FIFO: %s" % drv.realtime)
    if not wait_for_robot(drv):
        print("warning: LowState.tick is not changing; arming will be refused until the robot answers")

    obs_dim = getattr(policy, "obs_dim", contract.OBS_DIM)
    log = RunLog(obs_dim=obs_dim)
    remote = MacroRemote(args.vx_max, args.vy_max, args.wz_max,
                         macro_vx=args.macro_vx, macro_dist=args.macro_dist, macro_ramp=args.macro_ramp,
                         stand_time=args.stand_time, soft_start=args.soft_start)
    print("Joystick macro ready: R1+X = stand -> walk %.1f m @ vx=%.2f m/s -> stop -> sit (R1+Y or stick = cancel)"
          % (args.macro_dist, args.macro_vx))
    ctl = Controller(drv, policy, remote, dict(_a1_driver.FAULTS),
                     stand_time=args.stand_time, action_clip=args.action_clip, log=log, policy_kp=policy_kp,
                     allow_walk=not args.stand_only, soft_start=args.soft_start,
                     action_scale=action_scale, w=args.w,
                     obs_noise=obs_noise_amplitude(args.obs_noise_scale, args.dq_noise) if hasattr(policy, "hist_len") else None, noise_seed=args.noise_seed, lead_ms=args.lead_ms if hasattr(policy, "hist_len") else 0.0,
                     action_smooth=args.action_smooth, pitch_bias=args.pitch_bias, front_lift_bias=args.front_lift_bias,
                     imu_roll_bias=args.imu_roll_bias, imu_pitch_bias=args.imu_pitch_bias, imu_zero=args.imu_zero)
    duration = args.duration
    if args.desk_demo:
        desk_demo.t0 = time.monotonic()
        threading.Thread(target=desk_demo, args=(drv,), daemon=True).start()
        duration = duration or 10.0
    try:
        ctl.run(duration)
    except KeyboardInterrupt:
        pass
    finally:
        drv.disarm()
        drv.stop()
        # Compressing the log takes seconds on the Atom; a second Ctrl-C here used to lose the whole run.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("saving the log, please wait...")
        final = drv.state()
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
        path = os.path.join(args.log_dir, time.strftime("run_%Y%m%d_%H%M%S.npz"))
        meta = {"args": vars(args), "policy": policy.meta, "with_sdk": bool(_a1_driver.WITH_SDK),
                "realtime": drv.realtime, "policy_overruns": ctl.overruns, "motor_overruns": final.overruns,
                "motor_max_period_s": final.max_period}
        meta.update(provenance(args.policy, policy))
        log.save(path, meta)
        print(json.dumps({k: meta[k] for k in ("policy_overruns", "motor_overruns", "motor_max_period_s")}))
        print("log: %s" % path)


if __name__ == "__main__":
    main()
