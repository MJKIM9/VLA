"""Run gravity compensation on GELLO without connecting to a robot."""

import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tyro


@dataclass
class Args:
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NPRF-if00-port0"
    """GELLO serial port."""

    baudrate: int = 4000000
    """Dynamixel baudrate."""

    joint_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    """Dynamixel motor IDs."""

    joint_signs: Tuple[float, ...] = (1.0, -1.0, -1.0, -1.0, -1.0, 1.0)
    """Joint sign corrections."""

    joint_offsets: Tuple[float, ...] = (4.1356, 3.1431, 0.1273, 4.6434, 7.8724, 4.7017)
    """Joint offsets from ur10_gello.yaml calibration."""

    xml_path: Optional[str] = None
    """Path to MuJoCo XML. Defaults to yam.xml."""

    torque_to_pwm: Optional[Tuple[float, ...]] = None
    """PWM/Nm per joint (6 values). Defaults based on motor max torque."""

    hz: float = 50.0
    """Gravity compensation loop frequency."""


def main():
    args = tyro.cli(Args)

    xml_path = args.xml_path or str(
        Path(__file__).parents[1]
        / "third_party/mujoco_menagerie/i2rt_yam/yam.xml"
    )

    from gello.dynamixel.driver import DynamixelDriver
    from gello.robots.gravity_comp import GravityCompensator, GravityCompThread

    print(f"Connecting to GELLO on {args.port} ...")
    driver = DynamixelDriver(
        ids=args.joint_ids,
        port=args.port,
        baudrate=args.baudrate,
        use_fake_fallback=False,
    )
    print("Connected.")

    torque_to_pwm = list(args.torque_to_pwm) if args.torque_to_pwm else None

    compensator = GravityCompensator(
        xml_path=xml_path,
        joint_signs=args.joint_signs,
        joint_offsets=args.joint_offsets,
        torque_to_pwm=torque_to_pwm,
    )

    gc = GravityCompThread(
        driver=driver,
        compensator=compensator,
        get_joints_fn=driver.get_joints,
        n_joints=len(args.joint_ids),
        hz=args.hz,
    )

    def shutdown(sig, frame):
        print("\nStopping gravity compensation...")
        gc.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    gc.start()
    print("Gravity compensation running. Press Ctrl+C to stop.")

    signal.pause()


if __name__ == "__main__":
    main()
