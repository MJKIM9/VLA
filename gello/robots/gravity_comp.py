"""MuJoCo-based gravity compensation for GELLO using PWM control mode."""

import threading
import time
from typing import Callable, Optional, Sequence

import mujoco
import numpy as np

# Torque-to-PWM scale per joint: PWM = gravity_torque(Nm) * scale
# scale = PWM_MAX(885) / max_motor_torque(Nm)
# XM430-W350(ID1): 3.8Nm, XC430-W240(ID2): 1.9Nm, XC430-W150(ID3): 1.2Nm
# XL330-M077(ID4-6): 0.093Nm (very small, likely saturates for any real torque)
DEFAULT_TORQUE_TO_PWM = np.array([
    885 / 3.8,   # ID1 XM430-W350
    885 / 1.9,   # ID2 XC430-W240
    885 / 1.2,   # ID3 XC430-W150
    885 / 0.5,   # ID4 XL330-M077
    885 / 0.5,   # ID5 XL330-M077
    885 / 0.5,   # ID6 XL330-M077
]) / 20.0  # Start conservative — increase if still sagging


class GravityCompensator:
    """Computes per-joint gravity compensation PWM values via MuJoCo forward kinematics."""

    def __init__(
        self,
        xml_path: str,
        joint_signs: Sequence[float],
        joint_offsets: Sequence[float],
        torque_to_pwm: Optional[np.ndarray] = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.joint_signs = np.array(joint_signs[:6])
        self.joint_offsets = np.array(joint_offsets[:6])
        self.torque_to_pwm = (
            np.array(torque_to_pwm)
            if torque_to_pwm is not None
            else DEFAULT_TORQUE_TO_PWM.copy()
        )

    def compute_pwm(self, q_raw: np.ndarray) -> np.ndarray:
        """Return gravity compensation PWM values for 6 arm joints.

        Args:
            q_raw: Raw encoder angles from driver.get_joints() (rad), length >= 6.

        Returns:
            PWM values (-885 ~ 885) per joint.
        """
        # Raw encoder → GELLO output frame → MuJoCo joint frame
        q_gello = (q_raw[:6] - self.joint_offsets) * self.joint_signs
        self.data.qpos[:6] = q_gello
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Gravity compensation torques in MuJoCo joint frame (Nm)
        tau_mj = self.data.qfrc_gravcomp[:6].copy()

        # Convert to motor encoder frame (apply joint signs)
        tau_motor = tau_mj * self.joint_signs

        # Convert torque → PWM
        pwm = tau_motor * self.torque_to_pwm
        return np.clip(pwm, -885, 885)


class GravityCompThread:
    """Background thread that applies gravity compensation via PWM control mode."""

    def __init__(
        self,
        driver,
        compensator: GravityCompensator,
        get_joints_fn: Callable[[], np.ndarray],
        n_joints: int,
        hz: float = 50.0,
    ):
        self._driver = driver
        self._compensator = compensator
        self._get_joints = get_joints_fn
        self._n_joints = n_joints
        self._hz = hz
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        from gello.dynamixel.driver import PWM_CONTROL_MODE

        self._driver.set_torque_mode(False)
        time.sleep(0.2)
        self._driver.set_operating_mode(PWM_CONTROL_MODE)
        time.sleep(0.1)
        self._driver.set_torque_mode(True)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("Gravity compensation started (PWM mode).")

    def stop(self):
        from gello.dynamixel.driver import POSITION_CONTROL_MODE

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        import time
        self._driver.set_torque_mode(False)
        time.sleep(0.3)

        # 모드 전환 재시도 (XL330 타임아웃 대응)
        for attempt in range(3):
            try:
                self._driver.set_operating_mode(POSITION_CONTROL_MODE)
                break
            except Exception as e:
                print(f"[GravComp] 모드 전환 재시도 {attempt+1}/3: {e}")
                time.sleep(0.3)

        # 현재 실제 위치를 Goal Position에 써서 스냅백 방지
        time.sleep(0.1)
        try:
            current_pos = self._get_joints()
            self._driver.set_torque_mode(True)
            self._driver.set_joints(current_pos.tolist())
        except Exception as e:
            print(f"[GravComp] 위치 초기화 실패: {e}")

        print("Gravity compensation stopped.")

    def _loop(self):
        dt = 1.0 / self._hz
        while not self._stop.is_set():
            try:
                q_raw = self._get_joints()
                pwm = self._compensator.compute_pwm(q_raw)

                if self._n_joints > 6:
                    pwm = np.append(pwm, 0.0)  # gripper: no compensation

                self._driver.set_pwm(pwm.tolist())
                print(f"\r[GravComp] PWM: {np.round(pwm[:6], 1)}", end="", flush=True)
            except Exception as e:
                print(f"[GravComp] {e}")
            time.sleep(dt)
