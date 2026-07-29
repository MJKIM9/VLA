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


def _unwrap_rotvec(rotvec_new: np.ndarray, rotvec_prev: np.ndarray) -> np.ndarray:
    """회전각이 ±π 근처일 때 축각 표현이 부호를 뒤집는 불연속(±2π 점프)을 보정.
    (axis, θ)와 (-axis, 2π-θ)는 동일한 회전이므로, 이전 프레임에 더 가까운 쪽을 선택한다."""
    theta = np.linalg.norm(rotvec_new)
    if theta < 1e-8:
        return rotvec_new
    rotvec_alt = rotvec_new - 2 * np.pi * (rotvec_new / theta)
    if np.linalg.norm(rotvec_new - rotvec_prev) <= np.linalg.norm(rotvec_alt - rotvec_prev):
        return rotvec_new
    return rotvec_alt


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
        self,
        robot_ip: str = "192.168.1.10",
        no_gripper: bool = False,
        gripper_port: Optional[str] = None,
        unwrap_rotvec: bool = True,
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
        self._prev_tcp_rotvec: Optional[np.ndarray] = None
        # v19처럼 unwrap 적용 전(rotvec ±π 불연속이 낀 채로) 학습된 체크포인트를 쓸 때는
        # False로 꺼서 학습 당시와 같은(불연속 있는) state 분포를 재현해야 한다.
        self._unwrap_rotvec_enabled = unwrap_rotvec

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

    def state_orientation_key(self) -> str:
        """state 조립 시 사용할 orientation 필드명.
        unwrap_rotvec=True(v20용)면 'tcp_rotvec', False(v19용)면 'tcp_rpy'."""
        return "tcp_rotvec" if self._unwrap_rotvec_enabled else "tcp_rpy"

    def uses_rotvec_action_delta(self) -> bool:
        """액션의 회전 델타 계산 방식.
        True(v20)면 회전행렬 합성 기반 rotvec 델타, False(v19)면 RPY 성분별 차."""
        return self._unwrap_rotvec_enabled

    def reset_delta_tracking(self):
        """Δxyz/rotvec 연속성 추적 상태 초기화.

        VLA 추론처럼 새 제어 세션을 시작할 때 호출하지 않으면, 직전(하이브리드
        사전 동작 등)의 큰 이동이 첫 tcp_xyz_delta에 그대로 섞여 들어가
        모델이 비정상적으로 큰 액션을 예측하는 원인이 된다."""
        self._prev_tcp_pos = None
        self._prev_tcp_rotvec = None

    def set_unwrap_rotvec(self, enabled: bool) -> None:
        """v19/v20 orientation 모드를 실행 중에 전환. 체크포인트 전환 시(UI 토글)
        반드시 델타 추적도 같이 리셋해야 이전 모드의 값이 섞이지 않는다."""
        self._unwrap_rotvec_enabled = enabled
        self.reset_delta_tracking()

    def get_observations(self, full: bool = True) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        joint_vels = np.array(self.r_inter.getActualQd())
        gripper_pos = np.array([joints[-1] if self._use_gripper else 0.0])

        tcp_pose = np.array(self.r_inter.getActualTCPPose())
        tcp_xyz = tcp_pose[:3]
        tcp_rotvec = tcp_pose[3:].astype(np.float32)  # 축각(axis-angle), 변환 없이 그대로 사용
        if self._unwrap_rotvec_enabled and self._prev_tcp_rotvec is not None:
            tcp_rotvec = _unwrap_rotvec(tcp_rotvec, self._prev_tcp_rotvec).astype(np.float32)
        self._prev_tcp_rotvec = tcp_rotvec.copy()
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
            "tcp_rotvec":       tcp_rotvec,
            # v19는 rotvec이 아니라 RPY로 state를 학습했으므로 항상 같이 제공한다
            # (raw rotvec 기준 변환 — unwrap 여부와 무관하게 원본 그대로).
            "tcp_rpy":          _rotvec_to_rpy(tcp_pose[3:]),
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
