import threading
from typing import Dict, Optional

import numpy as np

from gello.robots.robot import Robot


def _rotvec_to_rpy(rotvec: np.ndarray) -> np.ndarray:
    """UR rotation vector (axis-angle) → RPY (roll, pitch, yaw) 변환."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float32)
    axis = rotvec / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-10:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw  = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
        self,
        robot_ip: str = "192.168.1.10",
        no_gripper: bool = False,
        gripper_port: Optional[str] = None,
    ):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)

        self._datc_gripper = None
        if gripper_port is not None:
            from gello.robots.datc_gripper import DATCGripper
            self._datc_gripper = DATCGripper(port=gripper_port)
        elif not no_gripper:
            from gello.robots.robotiq_gripper import RobotiqGripper
            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=robot_ip, port=63352)
            print("gripper connected")

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = (not no_gripper) or (gripper_port is not None)
        self._prev_tcp_pos: Optional[np.ndarray] = None

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        import time

        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._datc_gripper is not None:
            pos = np.append(robot_joints, 0.0)  # DATC gripper has no position feedback
        elif self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray, current_joints=None) -> None:
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 30       # 제어 주기
        lookahead_time = 0.1
        gain = 300
        robot_joints = joint_state[:6]

        # 현재 위치에서 최대 변화량 제한 (rad)
        if current_joints is None:
            current_joints = np.array(self.r_inter.getActualQ())
        max_delta = np.array([0.030, 0.030, 0.030, 0.030, 0.060, 0.060])
        delta = robot_joints - current_joints
        delta = np.clip(delta, -max_delta, max_delta)
        robot_joints = current_joints + delta

        self.robot.servoJ(robot_joints, velocity, acceleration, dt, lookahead_time, gain)
        if self._datc_gripper is not None and len(joint_state) > 6:
            t = np.clip(joint_state[6], 0, 1)
            gripper_pos = int(990 - t * (990 - 1))
            threading.Thread(target=self._datc_gripper.set_position, args=(gripper_pos,), daemon=True).start()
    def stop(self, deceleration: float = 2.0) -> None:
        """Stop all joints immediately with given deceleration (rad/s²)."""
        try:
            self.robot.stopJ(deceleration)
        except Exception as e:
            print(f"[URRobot] stop failed: {e}")

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self, full: bool = True) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        joint_vels = np.array(self.r_inter.getActualQd())
        gripper_pos = np.array([joints[-1] if self._use_gripper else 0.0])

        tcp_pose = np.array(self.r_inter.getActualTCPPose())
        tcp_xyz = tcp_pose[:3]
        tcp_rpy = _rotvec_to_rpy(tcp_pose[3:])
        if self._prev_tcp_pos is None:
            tcp_xyz_delta = np.zeros(3, dtype=np.float32)
        else:
            tcp_xyz_delta = (tcp_xyz - self._prev_tcp_pos).astype(np.float32)
        self._prev_tcp_pos = tcp_xyz.copy()

        obs = {
            "joint_positions":  joints,
            "joint_velocities": joint_vels,
            "gripper_position": gripper_pos,
            "tcp_xyz_delta":    tcp_xyz_delta,
            "tcp_rpy":          tcp_rpy,
        }
        if full:
            obs["ee_pos_quat"] = tcp_pose
            obs["wrench"]      = np.array(self.r_inter.getActualTCPForce())
        return obs


def main():
    robot_ip = "192.168.1.11"
    ur = URRobot(robot_ip, no_gripper=True)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
