import atexit
import faulthandler
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

faulthandler.enable()  # segfault 시 C 스택 트레이스 출력

import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict


def _detect_color_centroid(img_bgr, lower_hsv, upper_hsv):
    """HSV 범위로 색상 검출 후 픽셀 중심 (u, v) 반환. 검출 실패 시 None."""
    import cv2 as _cv2
    import numpy as _np2
    hsv = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2HSV)
    mask = _cv2.inRange(hsv, _np2.array(lower_hsv), _np2.array(upper_hsv))
    mask = _cv2.erode(mask, None, iterations=2)
    mask = _cv2.dilate(mask, None, iterations=2)
    contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=_cv2.contourArea)
    M = _cv2.moments(c)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def compute_alignment_xyz(img_bgr, wrist_cam):
    """손목 카메라에서 케이블(검정)-노란 물체의 xyz 오정렬 벡터(미터) 반환.

    반환값: np.array([dx, dy, dz]) = cable_xyz - yellow_xyz (카메라 프레임)
    두 물체 중 하나라도 검출 실패 시 [0, 0, 0] 반환.
    """
    import numpy as _np2
    yellow = _detect_color_centroid(img_bgr, [20, 80, 80], [35, 255, 255])
    cable  = _detect_color_centroid(img_bgr, [0, 0, 0], [180, 50, 50])
    if yellow is None or cable is None:
        return _np2.zeros(3, dtype=_np2.float32)
    yellow_3d = wrist_cam.pixel_to_3d(*yellow)
    cable_3d  = wrist_cam.pixel_to_3d(*cable)
    if _np2.all(yellow_3d == 0) or _np2.all(cable_3d == 0):
        return _np2.zeros(3, dtype=_np2.float32)
    return (cable_3d - yellow_3d)  # x, y, z 모두 반환

# Global variables for cleanup
active_threads: List[threading.Thread] = []
active_servers: List[Any] = []
cleanup_in_progress = False


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)

    print("Cleanup completed.")


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)


def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    right_cfg = None
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    # Read launch parameters from YAML
    gripper_port   = left_cfg.get("gripper_port")
    gc_cfg         = left_cfg.get("gravity_comp", {})
    dataset_cfg    = left_cfg.get("dataset", {})
    vla_cfg        = left_cfg.get("vla", {})

    # Create one shared DATCGripper instance and inject into robot
    gripper = None
    if gripper_port:
        from gello.robots.datc_gripper import DATCGripper
        gripper = DATCGripper(port=gripper_port)
        if hasattr(left_robot, "_datc_gripper"):
            left_robot._datc_gripper = gripper
            left_robot._use_gripper = True

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)  # type: ignore[union-attr]
        server_host = cfg["robot"].get("host", "127.0.0.1")  # type: ignore[union-attr]

        # Start server in background (non-daemon for proper cleanup)
        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(robot)

        # Wait for server to be ready
        print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        print("Server ready!")

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Get server configuration (use a different default port for hardware)
        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = "127.0.0.1"

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(server)

        # Wait for server to be ready
        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, hardware_host)
        print("Hardware server ready!")

        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=hardware_port, host=hardware_host)

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30))

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    # ── 2π 오프셋 자동 보정 (Dynamixel 초기 읽기 안정화 후 실행) ──────
    import numpy as _np2
    import time as _time2
    _gello_robot = getattr(agent, "_robot", None)
    if _gello_robot is not None:
        try:
            _time2.sleep(0.5)  # Dynamixel background thread 안정화 대기
            _ur_joints  = _np2.array(env.get_obs()["joint_positions"][:6])
            _gello_joints = _gello_robot.get_joint_state()[:6]
            # 비정상적으로 큰 값이면 AutoCorrect 건너뜀 (garbage 읽기 방지)
            if _np2.any(_np2.abs(_gello_joints) > 30.0):
                print(f"[AutoCorrect] GELLO 값이 비정상 (max={_np2.max(_np2.abs(_gello_joints)):.1f} rad) → 건너뜀")
            else:
                _changed = False
                for _i in range(6):
                    _diff = _gello_joints[_i] - _ur_joints[_i]
                    _n = round(_diff / (2 * _np2.pi))
                    if abs(_n) >= 1 and abs(_diff - _n * 2 * _np2.pi) < 0.3:
                        _gello_robot._joint_offsets[_i] -= _n * 2 * _np2.pi
                        print(f"[AutoCorrect] Joint {_i+1}: {_n*360:+.0f}° 보정됨")
                        _changed = True
                if _changed:
                    _new_offsets = _gello_robot._joint_offsets[:6].tolist()
                    import re
                    with open(args.left_config_path, "r") as _f:
                        _yaml_str = _f.read()
                    _fmt = "[" + ", ".join(f"{v:.4f}" for v in _new_offsets) + "]"
                    _yaml_str = re.sub(r"joint_offsets:\s*\[.*?\]", f"joint_offsets: {_fmt}", _yaml_str)
                    with open(args.left_config_path, "w") as _f:
                        _f.write(_yaml_str)
                    print(f"[AutoCorrect] YAML 저장 완료: {_fmt}")
                else:
                    print("[AutoCorrect] 2π 오프셋 이상 없음.")
        except Exception as _e:
            print(f"[AutoCorrect] 건너뜀: {_e}")

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    from gello.ui.control_panel import ControlPanel

    # LeRobot recorder (카메라 없이도 state/action 저장 가능)
    recorder = None
    cameras = {}
    if dataset_cfg.get("dir"):
        from gello.data_utils.lerobot_recorder import LeRobotRecorder, make_lerobot_dataset
        from gello.cameras.realsense_camera import get_device_ids, RealSenseCamera

        device_ids = get_device_ids()
        if len(device_ids) >= 2:
            cameras = {
                "exterior": RealSenseCamera(device_id=device_ids[0]),
                "wrist":    RealSenseCamera(device_id=device_ids[1]),
            }
        elif len(device_ids) == 1:
            cameras = {"wrist": RealSenseCamera(device_id=device_ids[0])}

        # state = tcp_xyz_delta(3) + tcp_rpy(3) + gripper(1) + align_xyz(3) = 10
        STATE_DIM  = 3 + 3 + 1 + 3
        ACTION_DIM = left_robot.num_dofs()
        dataset = make_lerobot_dataset(
            repo_id=dataset_cfg.get("repo_id", "koras/ur10_task"),
            root=str(Path(dataset_cfg["dir"]).expanduser()),
            fps=int(cfg.get("hz", 30)),
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            camera_keys=list(cameras.keys()),
        )
        recorder = LeRobotRecorder(
            dataset=dataset,
            task=dataset_cfg.get("task_name", "teleoperation"),
        )

    # VLA 정책은 버튼 클릭 시 lazy-load (CUDA init 전에 fork하면 segfault 발생)
    vla_demo_fn = None
    _vla_policy = [None]  # lazy-load용 컨테이너

    # Teleoperation toggle event (set = active); starts OFF
    teleop_event = threading.Event()

    # Control loop records frames when recorder is active
    import queue as _queue
    import numpy as _np
    _record_queue: _queue.Queue = _queue.Queue(maxsize=300)
    _home_target: list = []  # 비어있으면 go_home 비활성

    _record_frame_count = [0]

    def record_worker():
        """add_frame을 별도 스레드에서 처리. 카메라는 제어 루프에서 미리 읽어서 전달."""
        while True:
            item = _record_queue.get()
            if item is None:
                _record_queue.task_done()
                break
            obs_snap, action_snap, imgs = item
            try:
                if not recorder.is_recording:
                    continue
                wrist_img = imgs.get("wrist")
                wrist_cam = cameras.get("wrist")
                if wrist_img is not None and wrist_cam is not None:
                    import cv2 as _cv2
                    wrist_bgr = _cv2.cvtColor(wrist_img, _cv2.COLOR_RGB2BGR)
                    align_xyz = compute_alignment_xyz(wrist_bgr, wrist_cam)
                else:
                    align_xyz = _np.zeros(3, dtype=_np.float32)
                state = _np.concatenate([
                    obs_snap["tcp_xyz_delta"],
                    obs_snap["tcp_rpy"],
                    obs_snap["gripper_position"],
                    align_xyz,
                ])
                # delta action = GELLO_t - UR_t
                delta_action = action_snap - obs_snap["joint_positions"]
                recorder.add_frame(state=state, action=delta_action, images=imgs)
                _record_frame_count[0] += 1
            except KeyError as e:
                import traceback as _tb
                print(f"[RecordWorker] KeyError: {e} — episode_buffer 손상, 초기화합니다")
                _tb.print_exc()
                # episode_buffer에 size 키가 없는 경우 복구
                try:
                    eb = getattr(recorder.dataset, 'episode_buffer', None)
                    if eb is not None and 'size' not in eb:
                        recorder.dataset.episode_buffer = recorder.dataset.create_episode_buffer()
                except Exception:
                    pass
            except Exception as e:
                import traceback as _tb
                print(f"[RecordWorker] Error: {e}")
                _tb.print_exc()
            finally:
                _record_queue.task_done()
        print(f"[RecordWorker] Done. Total frames added: {_record_frame_count[0]}")

    record_thread = threading.Thread(target=record_worker, daemon=True)
    record_thread.start()

    def control_loop_with_record():
        import traceback
        try:
            obs = env.get_obs()
        except Exception as e:
            print(f"[ControlLoop] Failed to get initial obs: {e}")
            return
        print("[ControlLoop] Started. Waiting for Teleop ON...")
        while True:
            try:
                if not teleop_event.is_set():
                    if _home_target:
                        # servoJ로 홈 위치까지 천천히 이동
                        # obs가 stale할 수 있으므로 항상 현재 위치를 새로 읽음
                        try:
                            obs = env.get_obs()
                        except Exception:
                            pass
                        target = _np.array(_home_target)
                        current = _np.array(obs["joint_positions"][:6])
                        diff = target - current
                        if _np.max(_np.abs(_np.rad2deg(diff))) < 1.0:
                            _home_target.clear()
                            print("[GoHome] Done.")
                        else:
                            step = _np.clip(diff, -0.01164, 0.01164)  # 20 deg/s @ 30Hz
                            obs = env.step(_np.append(current + step, [1.0]))
                        _cached_ur_joints[0] = obs["joint_positions"][:6]
                    else:
                        # idle 상태에서도 GELLO/UR 관절값을 갱신해야
                        # 텔레오프 재시작 시 안전 체크에 stale 값이 쓰이지 않음
                        # VLA 실행 중(_vla_stop이 clear)에는 env를 건드리지 않음
                        # (VLA와 같은 ZMQ 소켓을 동시에 쓰면 ZMQError 발생)
                        vla_running = vla_cfg.get("checkpoint") and not _vla_stop.is_set()
                        if not vla_running:
                            try:
                                obs = env.get_obs()
                                _cached_ur_joints[0] = obs["joint_positions"][:6]
                            except Exception:
                                pass
                        try:
                            idle_action = agent.act(obs)
                            _cached_gello_joints[0] = idle_action[:6].copy()
                        except Exception:
                            pass
                        time.sleep(1.0 / cfg.get("hz", 30))
                    continue
                action = agent.act(obs)
                _cached_gello_joints[0] = action[:6].copy()  # agent.act()가 gello 관절값 반환
                # obs_T + action_T를 먼저 기록한 후 env.step() 실행
                # (step 이후의 obs를 기록하면 state[T+1]과 action[T]가 짝지어지는 버그)
                # 카메라도 제어 루프에서 직접 읽어서 robot state와 동기화
                if recorder and recorder.is_recording:
                    try:
                        imgs = {}
                        for _cam_key, _cam in cameras.items():
                            _img, _ = _cam.read()
                            imgs[_cam_key] = _img
                        _record_queue.put_nowait((dict(obs), action.copy(), imgs))
                    except _queue.Full:
                        pass  # 큐가 꽉 찬 경우 프레임 드롭
                obs = env.step(action)
                _cached_ur_joints[0] = obs["joint_positions"][:6]
            except Exception as e:
                print(f"\n[ControlLoop] Error: {e}")
                time.sleep(0.5)

    # 제어 루프에서 캐싱된 관절값 — UI 스레드와 Dynamixel/ZMQ 동시 접근 방지
    _cached_ur_joints   = [None]
    _cached_gello_joints = [None]

    control_thread = threading.Thread(target=control_loop_with_record, daemon=True)
    control_thread.start()

    # Parse gravity comp torque_to_pwm from YAML list
    gc_torque_to_pwm = None
    if gc_cfg.get("torque_to_pwm"):
        import numpy as np
        gc_torque_to_pwm = np.array(gc_cfg["torque_to_pwm"], dtype=float)

    # Get DynamixelRobot from GelloAgent for gravity compensation
    gello_robot = getattr(agent, "_robot", None)

    # Joint position getters — 캐시 반환 (UI 스레드에서 Dynamixel/ZMQ 직접 접근 금지)
    def get_gello_joints():
        return _cached_gello_joints[0]

    def get_ur_joints():
        return _cached_ur_joints[0]

    # VLA Demo function — 첫 실행 시 lazy-load (fork-after-CUDA segfault 방지)
    if vla_cfg.get("checkpoint"):
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from run_vla import load_policy, run_inference
        import torch as _torch
        _vla_device = "cuda" if _torch.cuda.is_available() else "cpu"
        _vla_checkpoint = str(Path(vla_cfg["checkpoint"]).expanduser())
        _vla_stop = threading.Event()

        _vla_stats = [None]

        def vla_demo_fn():
            if _vla_policy[0] is None:
                print(f"[VLA] 정책 로딩 중... (device={_vla_device})")
                _vla_policy[0], _vla_stats[0] = load_policy(_vla_checkpoint, _vla_device)
                print("[VLA] 정책 로딩 완료.")
            _vla_stop.clear()
            teleop_event.clear()
            time.sleep(0.3)  # 제어루프 마지막 servoJ 전송 완료 대기
            # servoJ 스크립트를 완전히 종료하고 RTDE 스크립트 재시작
            # (reuploadScript 없이 moveL 호출 시 "another thread controlling" 에러 발생)
            try:
                left_robot.robot.servoStop(10.0)
            except Exception:
                pass
            time.sleep(0.1)
            try:
                left_robot.robot.reuploadScript()
                print("[Hybrid] RTDE 스크립트 재시작 완료")
            except Exception as e:
                print(f"[Hybrid] reuploadScript 실패: {e}")
            time.sleep(0.3)
            try:
                cur = left_robot.r_inter.getActualTCPPose()
                print(f"[Hybrid] 현재 TCP: {[round(v,4) for v in cur]}")
            except Exception as e:
                print(f"[Hybrid] TCP 읽기 실패: {e}")

            def _euler_zyx_deg_to_rot_vec(rx_deg, ry_deg, rz_deg):
                """ZYX Euler 각도(degree) → UR rotation vector(radian) 변환."""
                rx = _np.deg2rad(rx_deg)
                ry = _np.deg2rad(ry_deg)
                rz = _np.deg2rad(rz_deg)
                Rx = _np.array([[1, 0, 0], [0, _np.cos(rx), -_np.sin(rx)], [0, _np.sin(rx), _np.cos(rx)]])
                Ry = _np.array([[_np.cos(ry), 0, _np.sin(ry)], [0, 1, 0], [-_np.sin(ry), 0, _np.cos(ry)]])
                Rz = _np.array([[_np.cos(rz), -_np.sin(rz), 0], [_np.sin(rz), _np.cos(rz), 0], [0, 0, 1]])
                R = Rz @ Ry @ Rx
                angle = _np.arccos(_np.clip((_np.trace(R) - 1) / 2, -1.0, 1.0))
                if abs(angle) < 1e-10:
                    return [0.0, 0.0, 0.0]
                axis = _np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2 * _np.sin(angle))
                return (axis * angle).tolist()

            def move_cs(cs_pose, label="", speed=0.1, accel=0.5):
                """카르테시안 직선 이동. cs_pose = [x, y, z, rx_deg, ry_deg, rz_deg]"""
                if _vla_stop.is_set():
                    return False
                print(f"[Hybrid] {label}")
                rot_vec = _euler_zyx_deg_to_rot_vec(cs_pose[3], cs_pose[4], cs_pose[5])
                pose_rad = list(cs_pose[:3]) + rot_vec
                print(f"[Hybrid]   moveL 목표: {[round(v,4) for v in pose_rad]}")

                # 매 이동 전 RTDE 스크립트 완전 재시작
                # (이전 moveL/servoJ 잔류 상태가 다음 moveL을 막는 것 방지)
                try:
                    left_robot.robot.reuploadScript()
                except Exception as e:
                    print(f"[Hybrid] reuploadScript 실패: {e}")
                for _ in range(30):  # 스크립트 실행 확인 (최대 3초)
                    try:
                        if left_robot.robot.isProgramRunning():
                            break
                    except Exception:
                        pass
                    time.sleep(0.1)
                time.sleep(0.2)  # 스크립트 초기화 완료 대기

                # STOP watchdog: 20ms 간격으로 감시, 감지 시 즉시 stopL
                _wd_stop = threading.Event()
                def _watchdog():
                    while not _wd_stop.wait(timeout=0.02):
                        if _vla_stop.is_set():
                            try:
                                left_robot.robot.stopL(2.0)
                            except Exception:
                                pass
                            return
                threading.Thread(target=_watchdog, daemon=True).start()

                try:
                    left_robot.robot.moveL(pose_rad, speed, accel)  # sync
                    return not _vla_stop.is_set()
                except Exception as e:
                    print(f"[Hybrid] moveL 실패: {e}")
                    return False
                finally:
                    _wd_stop.set()

            def set_gripper(value, label=""):
                """그리퍼 직접 제어. value = raw position (0=닫힘, 1000=열림)."""
                print(f"[Hybrid] {label}")
                if gripper is None:
                    print("[Hybrid] gripper 없음 — 건너뜀")
                    return
                try:
                    gripper.set_position(int(value))
                except Exception as e:
                    print(f"[Hybrid] gripper 제어 실패: {e}")
                time.sleep(0.5)

            def move_home(label="6. home 자세"):
                """홈 자세로 관절 공간 이동."""
                if _vla_stop.is_set():
                    return False
                print(f"[Hybrid] {label}")
                _home_target.clear()
                _home_target.extend(_HOME_RAD)
                while _home_target:
                    if _vla_stop.is_set():
                        return False
                    time.sleep(0.05)
                time.sleep(0.3)
                return True

            # ── 1단계: 사전 티칭 동작 ───────────────────────────────────────
            set_gripper(_GRIPPER_OPEN,      "1. gripper open")
            time.sleep(0.5)
            if not move_cs(_PRE_GRASP_CS,  "2. 케이블 파지 전 자세"): return
            if not move_cs(_GRASP_CS,       "3. 케이블 파지 자세"):   return
            set_gripper(_GRIPPER_CLOSE,     "4. gripper close")
            time.sleep(0.5)
            if not move_cs(_POST_GRASP_CS,  "5. 케이블 파지 후 자세"): return
            if not move_home():                                          return

            # ── 2단계: VLA 추론 ─────────────────────────────────────────────
            print("[Hybrid] 사전 동작 완료. VLA 추론 시작.")
            time.sleep(0.5)
            run_inference(
                env=env,
                cameras=cameras,
                policy=_vla_policy[0],
                stats=_vla_stats[0] or {},
                device=_vla_device,
                fps=cfg.get("hz", 30),
                chunk_size=vla_cfg.get("chunk_size", 20),
                stop_event=_vla_stop,
                speed_scale=vla_cfg.get("speed_scale", 1.0),
                direct_robot=left_robot,  # ZMQ 우회: 로봇 직접 제어
            )

    estop_fn = getattr(left_robot, "stop", None)

    def stop_fn():
        """VLA 및 텔레오퍼레이션을 부드럽게 정지 (하드웨어 E-STOP 아님)."""
        teleop_event.clear()
        _home_target.clear()
        if vla_cfg.get("checkpoint"):
            try:
                _vla_stop.set()
            except NameError:
                pass
        print("[STOP] VLA/Teleop stopped.")

    import numpy as _np
    _HOME_DEG = [-89.797, -80.051, -109.583, -80.419, 89.491, 0.047]
    _HOME_RAD = _np.deg2rad(_HOME_DEG).tolist()

    # ── 하이브리드 데모용 사전 티칭 자세 (CS: x, y, z [m] / rx, ry, rz [deg]) ──
    _PRE_GRASP_CS  = [-0.46883, -0.54319, 0.41521,  179.989,  0.002, -179.997]  # 1. 케이블 파지 전 자세
    _GRASP_CS      = [-0.46882, -0.54321, 0.26036,  179.988,  0.002, -179.995]  # 3. 케이블 파지 자세
    _POST_GRASP_CS = [-0.46883, -0.54319, 0.41521,  179.989,  0.002, -179.997]  # 5. 케이블 파지 후 자세

    _GRIPPER_OPEN  = 500   # gripper open (0=닫힘, 1000=열림)
    _GRIPPER_CLOSE = 0     # gripper close

    def go_home_fn():
        print("[GoHome] Moving to home position...")
        _home_target.clear()
        _home_target.extend(_HOME_RAD)

    panel = ControlPanel(
        teleop_event=teleop_event,
        gripper=gripper,
        recorder=recorder,
        vla_demo_fn=vla_demo_fn,
        gello_robot=gello_robot,
        gc_xml_path=gc_cfg.get("xml_path"),
        gc_torque_to_pwm=gc_torque_to_pwm,
        get_gello_joints_fn=get_gello_joints,
        get_ur_joints_fn=get_ur_joints,
        estop_fn=estop_fn,
        stop_fn=stop_fn,
        go_home_fn=go_home_fn,
        cameras=cameras,
        record_queue=_record_queue,
    )
    if recorder is not None:
        import atexit as _atexit
        _atexit.register(recorder.close)

    try:
        panel.run()
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
