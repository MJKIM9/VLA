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


_CABLE_ROI_TOP    = 0.51
_CABLE_ROI_BOTTOM = 0.65
_CABLE_ROI_LEFT   = 0.448
_CABLE_ROI_RIGHT  = 0.593
_CABLE_TIP_STRIP  = 20
_CABLE_TIP_Y_OFFSET = 0.05  # 케이블 tip y 좌표 위로 이동 (이미지 높이 비율)
_MIN_AREA_COLOR   = 5    # 노란색 검출 최소 픽셀
_MIN_AREA_CABLE   = 50   # 케이블(검정) 검출 최소 픽셀
_COLOR_ROI_LEFT   = 0.15
_COLOR_ROI_RIGHT  = 0.85


def _detect_color_centroid(img_bgr, lower_hsv, upper_hsv):
    """HSV 범위로 검출된 모든 픽셀의 무게중심 (u, v) 반환. 검출 실패 시 None.
    노란 커넥터처럼 케이블에 가려 두 덩어리로 쪼개지는 경우에도 중심을 올바르게 추정한다."""
    import cv2 as _cv2
    import numpy as _np2
    hsv = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2HSV)
    mask = _cv2.inRange(hsv, _np2.array(lower_hsv), _np2.array(upper_hsv))
    mask = _cv2.erode(mask, None, iterations=2)
    mask = _cv2.dilate(mask, None, iterations=2)
    _, _w = mask.shape
    mask[:, :int(_w * _COLOR_ROI_LEFT)]  = 0
    mask[:, int(_w * _COLOR_ROI_RIGHT):] = 0
    pts = _np2.argwhere(mask > 0)  # (row, col)
    if len(pts) < _MIN_AREA_COLOR:
        return None
    cy = int(pts[:, 0].mean())
    cx = int(pts[:, 1].mean())
    return cx, cy


def _detect_color_blobs(img_bgr, lower_hsv, upper_hsv):
    """HSV 범위로 검출된 픽셀을 개별 blob(connected component) 단위로 분리해
    각 blob의 중심 좌표와 면적을 반환. x좌표(왼쪽→오른쪽) 순으로 정렬된 리스트."""
    import cv2 as _cv2
    import numpy as _np2
    hsv = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2HSV)
    mask = _cv2.inRange(hsv, _np2.array(lower_hsv), _np2.array(upper_hsv))
    mask = _cv2.erode(mask, None, iterations=2)
    mask = _cv2.dilate(mask, None, iterations=2)
    _, _w = mask.shape
    mask[:, :int(_w * _COLOR_ROI_LEFT)]  = 0
    mask[:, int(_w * _COLOR_ROI_RIGHT):] = 0
    n_labels, _labels, stats, centroids = _cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n_labels):  # 0번 라벨은 배경
        area = stats[i, _cv2.CC_STAT_AREA]
        if area < _MIN_AREA_COLOR:
            continue
        cx, cy = centroids[i]
        blobs.append({"cx": int(cx), "cy": int(cy), "area": int(area)})
    blobs.sort(key=lambda b: b["cx"])
    return blobs


def _detect_cable_tip(img_bgr):
    """케이블(검정) ROI 적용 후 하단 tip 픽셀 좌표 반환. 검출 실패 시 None."""
    import cv2 as _cv2
    import numpy as _np2
    hsv = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2HSV)
    mask = _cv2.inRange(hsv, _np2.array([0, 0, 0]), _np2.array([180, 80, 80]))
    mask = _cv2.erode(mask, None, iterations=2)
    mask = _cv2.dilate(mask, None, iterations=2)
    h, w = mask.shape
    top_y   = int(h * _CABLE_ROI_TOP)
    bot_y   = int(h * _CABLE_ROI_BOTTOM)
    left_x  = int(w * _CABLE_ROI_LEFT)
    right_x = int(w * _CABLE_ROI_RIGHT)
    mask[:top_y] = 0
    mask[bot_y:] = 0
    mask[:, :left_x] = 0
    mask[:, right_x:] = 0
    contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=_cv2.contourArea)
    if _cv2.contourArea(c) < _MIN_AREA_CABLE:
        return None
    strip_top = max(0, bot_y - _CABLE_TIP_STRIP)
    strip = mask[strip_top:bot_y, :]
    cols = _np2.where(strip.any(axis=0))[0]
    if len(cols) == 0:
        return None
    h_full = mask.shape[0]
    return int(cols.mean()), bot_y - _CABLE_TIP_STRIP // 2 - int(h_full * _CABLE_TIP_Y_OFFSET)


def compute_alignment_xyz(img_bgr, wrist_cam):
    """손목 카메라에서 케이블(검정) tip - 노란 물체의 xy 오정렬 벡터(미터) 반환.
    depth 센서 대신 고정 작업거리로 픽셀 → 미터 변환 (z=0).
    두 물체 중 하나라도 검출 실패 시 [0, 0, 0] 반환.
    """
    import numpy as _np2
    _FIXED_DEPTH = 0.30
    yellow = _detect_color_centroid(img_bgr, [22, 150, 120], [32, 255, 255])
    cable  = _detect_cable_tip(img_bgr)
    if yellow is None or cable is None:
        return _np2.zeros(3, dtype=_np2.float32)
    dx = (cable[0] - yellow[0]) * _FIXED_DEPTH / wrist_cam.fx
    dy = (cable[1] - yellow[1]) * _FIXED_DEPTH / wrist_cam.fy
    return _np2.array([dx, dy, 0.0], dtype=_np2.float32)

# Global variables for cleanup
active_threads: List[threading.Thread] = []
active_servers: List[Any] = []
active_recorder: List[Any] = [None]  # signal_handler에서 recorder를 안전하게 닫기 위한 참조
cleanup_in_progress = False


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    # recorder를 먼저 닫아 meta parquet writer의 footer를 반드시 기록한다.
    # (Ctrl+C 등으로 os._exit() 호출 시 atexit/finally가 스킵되어 이게 없으면
    #  녹화 중이던 에피소드의 메타데이터 파일이 미완성 상태로 남아 다음 로드 시 손상됨)
    if active_recorder[0] is not None:
        try:
            if active_recorder[0].is_recording:
                active_recorder[0].end_episode(save=True)
            active_recorder[0].close()
            print("Recorder closed.")
        except Exception as e:
            print(f"Error closing recorder: {e}")

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
    datasets_root_dir    = left_cfg.get("datasets_root_dir", "~/datasets")
    checkpoints_root_dir = left_cfg.get("checkpoints_root_dir", "~/checkpoints")

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
    set_dataset_fn = None
    _current_dataset_name = None
    _cartesian_action = dataset_cfg.get("cartesian_action", False)
    if dataset_cfg.get("dir"):
        from gello.data_utils.lerobot_recorder import LeRobotRecorder, make_lerobot_dataset
        from gello.cameras.realsense_camera import get_device_ids, RealSenseCamera

        _use_exterior_camera = dataset_cfg.get("use_exterior_camera", True)
        device_ids = get_device_ids()
        if len(device_ids) >= 2 and _use_exterior_camera:
            cameras = {
                "exterior": RealSenseCamera(device_id=device_ids[0]),
                "wrist":    RealSenseCamera(device_id=device_ids[1]),
            }
        elif len(device_ids) >= 1:
            # exterior 미사용 또는 카메라 1대만 연결된 경우: wrist만 사용
            wrist_device_id = device_ids[1] if len(device_ids) >= 2 else device_ids[0]
            cameras = {"wrist": RealSenseCamera(device_id=wrist_device_id)}

        # state = tcp_xyz_delta(3) + tcp_rotvec(3) + gripper(1) + align_xy(2) = 9
        # action = tcp_xyz_delta(3) + tcp_rotvec_delta(3) + gripper_delta(1) = 7  (cartesian_action=True)
        #        = joint_delta(7)                                                  (cartesian_action=False)
        STATE_DIM  = 3 + 3 + 1 + 2
        ACTION_DIM = 7 if _cartesian_action else left_robot.num_dofs()
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
        active_recorder[0] = recorder
        _current_dataset_name = Path(dataset_cfg["dir"]).expanduser().name

        def set_dataset_fn(dataset_name: str):
            """UI에서 데이터셋 선택 시 호출. recorder를 새 데이터셋으로 교체하고 yaml에도 반영."""
            nonlocal recorder
            if recorder is not None and recorder.is_recording:
                print("[Dataset] 녹화 중에는 데이터셋을 변경할 수 없습니다.")
                return
            try:
                new_root = Path(datasets_root_dir).expanduser() / dataset_name
                new_repo_id = f"koras/{dataset_name}"
                new_dataset = make_lerobot_dataset(
                    repo_id=new_repo_id,
                    root=str(new_root),
                    fps=int(cfg.get("hz", 30)),
                    state_dim=STATE_DIM,
                    action_dim=ACTION_DIM,
                    camera_keys=list(cameras.keys()),
                )
                if recorder is not None:
                    try:
                        recorder.close()
                    except Exception:
                        pass
                recorder = LeRobotRecorder(
                    dataset=new_dataset,
                    task=dataset_cfg.get("task_name", "teleoperation"),
                )
                active_recorder[0] = recorder
                panel.set_recorder(recorder)
                dataset_cfg["dir"] = str(new_root)
                dataset_cfg["repo_id"] = new_repo_id
                try:
                    _update_yaml_scalar(args.left_config_path, "dir", f"~/datasets/{dataset_name}")
                    _update_yaml_scalar(args.left_config_path, "repo_id", new_repo_id)
                except Exception as e:
                    print(f"[Dataset] yaml 저장 실패: {e}")
                print(f"[Dataset] 활성 데이터셋 변경: {dataset_name}")
            except Exception as e:
                print(f"[Dataset] 변경 실패: {e}")

    # VLA 정책은 버튼 클릭 시 lazy-load (CUDA init 전에 fork하면 segfault 발생)
    vla_demo_fn = None
    set_checkpoint_fn = None
    _vla_policy = [None]  # lazy-load용 컨테이너

    def _update_yaml_scalar(yaml_path: str, key: str, value) -> bool:
        """yaml 파일에서 key: 값을 주석/포맷 보존하며 갱신. 찾으면 True."""
        import re
        with open(yaml_path, "r") as f:
            text = f.read()
        if isinstance(value, bool):
            val_str = "true" if value else "false"
        elif isinstance(value, float):
            val_str = repr(value)
        else:
            val_str = str(value)
        pattern = re.compile(rf"^(\s*){re.escape(key)}:\s*[^\n#]*(#.*)?$", re.MULTILINE)

        def _repl(m):
            comment = m.group(2) or ""
            sep = "  " if comment else ""
            return f"{m.group(1)}{key}: {val_str}{sep}{comment}"

        new_text, n = pattern.subn(_repl, text, count=1)
        if n == 0:
            return False
        with open(yaml_path, "w") as f:
            f.write(new_text)
        return True

    def set_vla_params_fn(new_values: dict):
        """UI에서 파라미터 적용 시 호출. vla_cfg를 갱신하고 yaml 파일에도 반영해 재시작 후에도 유지되게 한다."""
        vla_cfg.update(new_values)
        for key, value in new_values.items():
            try:
                if not _update_yaml_scalar(args.left_config_path, key, value):
                    print(f"[VLA] yaml에서 '{key}' 키를 찾지 못해 파일에는 반영 못함 (메모리에는 반영됨)")
            except Exception as e:
                print(f"[VLA] yaml 저장 실패 ({key}): {e}")
        print(f"[VLA] 파라미터 업데이트: {new_values}")

    _train_scripts_dir = Path(__file__).resolve().parent

    def start_act_training_fn(params: dict):
        """UI에서 'Start ACT Training' 클릭 시 호출. 별도 프로세스로 학습을 백그라운드 시작."""
        import subprocess, sys
        dataset_name = params["dataset_name"]
        cmd = [
            sys.executable, str(_train_scripts_dir / "train_act.py"),
            "--repo_id", f"koras/{dataset_name}",
            "--dataset_dir", str(Path(datasets_root_dir).expanduser() / dataset_name),
            "--output_dir", str(Path(checkpoints_root_dir).expanduser() / params["output_name"]),
            "--batch_size", str(params["batch_size"]),
            "--num_workers", str(params["num_workers"]),
            "--steps", str(params["steps"]),
            "--save_freq", str(params["save_freq"]),
            "--lr", str(params["lr"]),
            "--chunk_size", str(params["chunk_size"]),
        ]
        print(f"[Train:ACT] 실행: {' '.join(cmd)}")
        subprocess.Popen(cmd)
        print("[Train:ACT] 백그라운드에서 학습을 시작했습니다. 진행 상황은 터미널 로그를 확인하세요.")

    def start_align_training_fn(params: dict):
        """UI에서 'Start Align Training' 클릭 시 호출. 별도 프로세스로 학습을 백그라운드 시작."""
        import subprocess, sys
        dataset_name = params["dataset_name"]
        cmd = [
            sys.executable, str(_train_scripts_dir / "train_align.py"),
            "--dataset_dir", str(Path(datasets_root_dir).expanduser() / dataset_name),
            "--output_path", str(Path(checkpoints_root_dir).expanduser() / params["output_name"]),
            "--fine_threshold", str(params["fine_threshold"]),
            "--epochs", str(params["epochs"]),
            "--batch_size", str(params["batch_size"]),
            "--lr", str(params["lr"]),
        ]
        print(f"[Train:Align] 실행: {' '.join(cmd)}")
        subprocess.Popen(cmd)
        print("[Train:Align] 백그라운드에서 학습을 시작했습니다. 진행 상황은 터미널 로그를 확인하세요.")

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
                wrist_img = imgs.get("wrist")
                wrist_cam = cameras.get("wrist")
                if wrist_img is not None and wrist_cam is not None:
                    import cv2 as _cv2
                    wrist_bgr = _cv2.cvtColor(wrist_img, _cv2.COLOR_RGB2BGR)
                    align_xyz = compute_alignment_xyz(wrist_bgr, wrist_cam)
                else:
                    align_xyz = _np.zeros(3, dtype=_np.float32)
                # state: 9-dim (align_xy[:2]만 포함, 추론 obs_to_tensor와 일치)
                state = _np.concatenate([
                    obs_snap["tcp_xyz_delta"],
                    obs_snap[left_robot.state_orientation_key()],
                    obs_snap["gripper_position"],
                    align_xyz[:2],
                ])
                if _cartesian_action:
                    # Cartesian action: FK(GELLO) - FK(UR)의 xyz 델타 + 회전 델타 + 그리퍼 델타
                    # 회전 델타 방식은 left_robot.uses_rotvec_action_delta()로 결정
                    # (v20: 회전행렬 합성 기반 rotvec 델타 / v19: RPY 성분별 차)
                    try:
                        gello_fk = left_robot.robot.getForwardKinematics(action_snap[:6].tolist())
                        ur_fk = left_robot.robot.getForwardKinematics(obs_snap["joint_positions"][:6].tolist())
                        xyz_delta_act = _np.array(gello_fk[:3]) - _np.array(ur_fk[:3])
                        if left_robot.uses_rotvec_action_delta():
                            from scipy.spatial.transform import Rotation as _RotRW
                            R_ur = _RotRW.from_rotvec(ur_fk[3:6]).as_matrix()
                            R_gello = _RotRW.from_rotvec(gello_fk[3:6]).as_matrix()
                            rot_delta_act = _RotRW.from_matrix(R_gello @ R_ur.T).as_rotvec()
                        else:
                            from gello.robots.ur import _rotvec_to_rpy
                            gello_rpy = _rotvec_to_rpy(_np.array(gello_fk[3:6]))
                            ur_rpy = _rotvec_to_rpy(_np.array(ur_fk[3:6]))
                            rot_delta_act = gello_rpy - ur_rpy
                        gripper_delta_act = _np.array([action_snap[-1] - obs_snap["joint_positions"][-1]])
                        delta_action = _np.concatenate([xyz_delta_act, rot_delta_act, gripper_delta_act])
                    except Exception as _fk_e:
                        print(f"[RecordWorker] FK 실패: {_fk_e}, zero action 사용")
                        delta_action = _np.zeros(7, dtype=_np.float32)
                else:
                    # joint space action (v17 방식)
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
    _vla_stop = threading.Event()  # 제어루프보다 먼저 정의 (스코프 오류 방지)
    _vla_stop.set()                # 초기 상태: VLA 미실행

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
        _vla_checkpoint = [str(Path(vla_cfg["checkpoint"]).expanduser())]  # UI에서 교체 가능한 컨테이너
        _vla_stats = [None]

        def set_checkpoint_fn(pretrained_model_path: str):
            """UI에서 체크포인트 선택 시 호출. 다음 VLA Demo 실행 시 새로 로드되며 yaml에도 반영."""
            if not _vla_stop.is_set():
                print("[VLA] 추론 실행 중에는 체크포인트를 변경할 수 없습니다.")
                return
            _vla_checkpoint[0] = pretrained_model_path
            _vla_policy[0] = None
            _vla_stats[0] = None
            vla_cfg["checkpoint"] = pretrained_model_path
            try:
                _update_yaml_scalar(args.left_config_path, "checkpoint", pretrained_model_path)
            except Exception as e:
                print(f"[VLA] yaml 저장 실패: {e}")
            print(f"[VLA] 체크포인트 선택됨: {pretrained_model_path} (다음 실행 시 로드)")

        def vla_demo_fn():
            if _vla_policy[0] is None:
                print(f"[VLA] 정책 로딩 중... (device={_vla_device})")
                print(f"[VLA] 체크포인트 경로: {_vla_checkpoint[0]}")
                _vla_policy[0], _vla_stats[0] = load_policy(_vla_checkpoint[0], _vla_device)
                print("[VLA] 정책 로딩 완료.")
            else:
                print(f"[VLA] 이미 로드된 정책 재사용: {_vla_checkpoint[0]}")
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

            def move_cs(cs_pose, label="", speed=0.2, accel=1.0):
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
                time.sleep(1.0)

            def move_z_fixed_rp(z_target, label="", speed=0.025, accel=0.15, tcp_frame=False):
                """z를 이동. tcp_frame=True이면 TCP z축 방향으로 이동 (base_z가 z_target에 도달하도록)."""
                if _vla_stop.is_set():
                    return False
                try:
                    cur = left_robot.r_inter.getActualTCPPose()
                except Exception as e:
                    print(f"[Hybrid] TCP 읽기 실패: {e}")
                    return False
                # 현재 회전행렬 계산
                rv = _np.array(cur[3:6])
                angle = _np.linalg.norm(rv)
                if angle < 1e-10:
                    R = _np.eye(3)
                else:
                    ax = rv / angle
                    K = _np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                    R = _np.eye(3) + _np.sin(angle) * K + (1 - _np.cos(angle)) * (K @ K)

                if tcp_frame:
                    # TCP z축 방향으로 base_z가 z_target에 도달하도록 이동
                    tcp_z = R[:, 2]  # TCP z축 (base 프레임)
                    dz_base = z_target - cur[2]
                    if abs(tcp_z[2]) < 0.1:
                        print("[Hybrid] TCP z축이 수평에 가까워 tcp_frame 이동 불가 — base frame으로 대체")
                        tcp_frame = False
                    else:
                        scale = dz_base / tcp_z[2]
                        new_xyz = _np.array(cur[:3]) + tcp_z * scale
                        target_pose = list(new_xyz) + list(cur[3:6])
                if not tcp_frame:
                    yaw = _np.arctan2(R[1, 0], R[0, 0])
                    roll_r  = _np.deg2rad(-179.99)
                    pitch_r = _np.deg2rad(0.0)
                    cr, sr = _np.cos(roll_r),  _np.sin(roll_r)
                    cp, sp = _np.cos(pitch_r), _np.sin(pitch_r)
                    cy, sy = _np.cos(yaw),     _np.sin(yaw)
                    Rx2 = _np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
                    Ry2 = _np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                    Rz2 = _np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                    R2 = Rz2 @ Ry2 @ Rx2
                    ang2 = _np.arccos(_np.clip((_np.trace(R2) - 1) / 2, -1.0, 1.0))
                    if abs(ang2) < 1e-10:
                        rv2 = [0.0, 0.0, 0.0]
                    else:
                        rv2 = list((ang2 / (2 * _np.sin(ang2))) * _np.array([
                            R2[2,1] - R2[1,2], R2[0,2] - R2[2,0], R2[1,0] - R2[0,1]
                        ]))
                    target_pose = list(cur[:2]) + [z_target] + rv2
                print(f"[Hybrid] {label} → TCP: {[round(v,4) for v in target_pose]}")
                try:
                    left_robot.robot.reuploadScript()
                except Exception as e:
                    print(f"[Hybrid] reuploadScript 실패: {e}")
                for _ in range(30):
                    try:
                        if left_robot.robot.isProgramRunning():
                            break
                    except Exception:
                        pass
                    time.sleep(0.1)
                time.sleep(0.2)
                _wd_stop2 = threading.Event()
                def _watchdog2():
                    while not _wd_stop2.wait(timeout=0.02):
                        if _vla_stop.is_set():
                            try:
                                left_robot.robot.stopL(2.0)
                            except Exception:
                                pass
                            return
                threading.Thread(target=_watchdog2, daemon=True).start()
                try:
                    left_robot.robot.moveL(target_pose, speed, accel)
                    return not _vla_stop.is_set()
                except Exception as e:
                    print(f"[Hybrid] moveL 실패: {e}")
                    return False
                finally:
                    _wd_stop2.set()

            def move_home(label="6. home 자세", speed=0.5, accel=0.5):
                """홈 자세로 관절 공간 이동 (moveJ)."""
                if _vla_stop.is_set():
                    return False
                print(f"[Hybrid] {label}")
                try:
                    left_robot.robot.moveJ(_HOME_RAD, speed, accel)
                    return True
                except Exception as e:
                    print(f"[Hybrid] moveJ 실패: {e}")
                    return False

            # ── 1단계: 사전 티칭 동작 ───────────────────────────────────────
            if not move_home("1. home 경유"):                            return

            # home 도착 직후: 노란색 ROI 내 blob 개수 검출 및 번호가 매겨진 스냅샷 저장
            _wrist_cam_snap = cameras.get("wrist")
            if _wrist_cam_snap is not None:
                try:
                    import cv2 as _cv2_snap
                    import os as _os_snap
                    _img_rgb_snap, _ = _wrist_cam_snap.read()
                    _img_bgr_snap = _cv2_snap.cvtColor(_img_rgb_snap, _cv2_snap.COLOR_RGB2BGR)
                    _yellow_blobs = _detect_color_blobs(_img_bgr_snap, [22, 150, 120], [32, 255, 255])
                    print(f"[Hybrid] home 도착 직후 노란색 blob 검출: {len(_yellow_blobs)}개")
                    for _i, _b in enumerate(_yellow_blobs, start=1):
                        _cv2_snap.circle(_img_bgr_snap, (_b["cx"], _b["cy"]), 8, (0, 255, 255), -1)
                        _cv2_snap.putText(_img_bgr_snap, f"yellow {_i}", (_b["cx"] + 10, _b["cy"]),
                                           _cv2_snap.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    _snap_dir = _os_snap.path.expanduser("~/vla_home_snapshots")
                    _os_snap.makedirs(_snap_dir, exist_ok=True)
                    _snap_path = _os_snap.path.join(_snap_dir, f"home_yellow_{int(time.time())}.png")
                    _cv2_snap.imwrite(_snap_path, _img_bgr_snap)
                    print(f"[Hybrid] 스냅샷 저장: {_snap_path}")
                except Exception as _e:
                    print(f"[Hybrid] 노란색 blob 스냅샷 실패: {_e}")
            set_gripper(_GRIPPER_OPEN,      "2. gripper open")
            time.sleep(0.5)
            if not move_cs(_PRE_GRASP_CS,  "3. 케이블 파지 전 자세"): return
            if not move_cs(_GRASP_CS,       "4. 케이블 파지 자세"):   return
            set_gripper(_GRIPPER_CLOSE,     "5. gripper close")
            time.sleep(0.5)
            if not move_cs(_POST_GRASP_CS,  "6. 케이블 파지 후 자세"): return
            if not move_home("7. home"):                                 return

            # ── 2단계: VLA 추론 ─────────────────────────────────────────────
            print("[Hybrid] 사전 동작 완료. VLA 추론 시작.")
            time.sleep(0.5)
            # VLA 추론 중: 소프트웨어 어드미턴스만 사용 (hardware forceMode 없음)
            # UI ON 상태면 yaml의 admittance_gain 적용, OFF면 0.0
            _eff_gain = vla_cfg.get("admittance_gain", 0.0) if _admittance_active[0] else 0.0
            if _admittance_active[0]:
                print(f"[Admittance] VLA 추론 중 소프트웨어 어드미턴스 적용 (gain={_eff_gain})")
            _z_approach = vla_cfg.get("z_approach")
            _z_floor    = vla_cfg.get("z_floor")
            _z_insert   = vla_cfg.get("z_insert", 0.24)
            _align_thr  = vla_cfg.get("align_insert_threshold", 0.002)
            insertion_ready = run_inference(
                env=env,
                cameras=cameras,
                policy=_vla_policy[0],
                stats=_vla_stats[0] or {},
                device=_vla_device,
                fps=cfg.get("hz", 30),
                chunk_size=vla_cfg.get("chunk_size", 20),
                stop_event=_vla_stop,
                speed_scale=vla_cfg.get("speed_scale", 1.0),
                delta_scale=vla_cfg.get("delta_scale", 1.0),
                use_servoing=vla_cfg.get("use_servoing", False),
                fix_orientation=vla_cfg.get("fix_orientation", False),
                direct_robot=left_robot,
                z_approach_threshold=_z_approach,
                z_floor=_z_floor,
                use_z_freeze=vla_cfg.get("use_z_freeze", False),
                align_insert_threshold=_align_thr,
                cartesian_action=_cartesian_action,
                control_gripper=vla_cfg.get("control_gripper", True),
                admittance_gain=_eff_gain,
                admittance_deadband=vla_cfg.get("admittance_deadband", 3.0),
                admittance_spring_k=vla_cfg.get("admittance_spring_k", 0.0),
                admittance_damping_b=vla_cfg.get("admittance_damping_b", 0.0),
            )

            # ── 3단계: 삽입 시퀀스 (z+align 조건 충족 시) ─────────────────
            if insertion_ready:
                print("[VLA] 삽입 시퀀스 시작")
                _vla_stop.clear()
                try:
                    left_robot.robot.servoStop(10.0)
                except Exception:
                    pass
                time.sleep(0.1)
                if move_z_fixed_rp(_z_insert, f"삽입 z={_z_insert}", tcp_frame=True):
                    set_gripper(_GRIPPER_OPEN, "그리퍼 open")
                    move_z_fixed_rp(0.34, "삽입 후 복귀 z=0.34", tcp_frame=True)

            # ── 4단계: home 복귀 ─────────────────────────────────────────────
            if insertion_ready:
                set_gripper(_GRIPPER_CLOSE, "그리퍼 close")
                print("[VLA] home 복귀")
                move_home("home 복귀")


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
        try:
            left_robot.robot.stopL(2.0)
        except Exception:
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

    _admittance_active = [False]

    _adm_bg_stop = [threading.Event()]
    _adm_bg_stop[0].set()

    def _start_admittance_bg():
        import numpy as _np_adm
        from scipy.spatial.transform import Rotation as _RotAdm

        _ADM_GAIN      = 0.003   # (m/s) / N : 힘 → 속도 게인 (작은 힘에도 부드럽게 반응)
        _ADM_DEADBAND  = 1.0     # N
        _ADM_SPRING_K  = 1.5     # 1/s : 원위치 복원력 계수 (클수록 빨리 복귀)
        _ADM_DAMPING_B = 0.3     # 무차원 : 실제 속도에 반대로 작용해 진동 억제
        _ADM_FILTER_ALPHA = 0.2  # 낮을수록 힘 신호를 더 부드럽게(떨림 억제)

        stop_ev = threading.Event()
        _adm_bg_stop[0] = stop_ev
        _origin_xyz = _np_adm.array(left_robot.r_inter.getActualTCPPose()[:3])
        _ft_filtered = _np_adm.zeros(3)

        def _loop():
            nonlocal _ft_filtered
            while not stop_ev.is_set():
                # VLA 실행 중엔 건너뜀 (_vla_stop이 clear = VLA 실행 중)
                if not _vla_stop.is_set():
                    time.sleep(0.02)
                    continue
                try:
                    ft_tcp = _np_adm.array(left_robot.r_inter.getActualTCPForce()[:3])
                    tcp    = left_robot.r_inter.getActualTCPPose()
                    pos    = _np_adm.array(tcp[:3])
                    R      = _RotAdm.from_rotvec(tcp[3:6]).as_matrix()
                    ft_base = R @ ft_tcp
                    ft_base[0] = -ft_base[0]  # x축 반대 방향 보정
                    ft_base[2] = -ft_base[2]  # z축 반대 방향 보정
                    _ft_filtered = _ADM_FILTER_ALPHA * ft_base + (1 - _ADM_FILTER_ALPHA) * _ft_filtered
                    ft_base = _ft_filtered
                    ft_clipped = _np_adm.where(
                        _np_adm.abs(ft_base) > _ADM_DEADBAND,
                        (ft_base - _np_adm.sign(ft_base) * _ADM_DEADBAND) * _ADM_GAIN,
                        0.0,
                    )
                    v_actual = _np_adm.array(left_robot.r_inter.getActualTCPSpeed()[:3])
                    spring_term = -_ADM_SPRING_K * (pos - _origin_xyz)
                    damping_term = -_ADM_DAMPING_B * v_actual
                    vel = ft_clipped + spring_term + damping_term
                    left_robot.robot.speedL(list(vel) + [0.0, 0.0, 0.0], 0.5, 0.04)
                except Exception:
                    pass
                time.sleep(0.02)

        threading.Thread(target=_loop, daemon=True).start()

    def go_home_fn():
        print("[GoHome] Moving to home position...")
        teleop_event.clear()
        time.sleep(0.3)  # 제어루프 마지막 servoJ 전송 완료 대기
        # servoJ 스크립트를 완전히 종료하고 RTDE 스크립트 재시작
        # (reuploadScript 없이 moveJ 호출 시 "another thread controlling" 에러 발생)
        try:
            left_robot.robot.servoStop(10.0)
        except Exception:
            pass
        time.sleep(0.1)
        try:
            left_robot.robot.reuploadScript()
        except Exception as e:
            print(f"[GoHome] reuploadScript 실패: {e}")
        # 스크립트가 로봇 컨트롤러에서 실제로 실행 중인지 확인 후 이동
        # (확인 없이 바로 moveJ 호출 시 에러 없이 조용히 무시될 수 있음)
        for _ in range(30):  # 최대 3초
            try:
                if left_robot.robot.isProgramRunning():
                    break
            except Exception:
                pass
            time.sleep(0.1)
        time.sleep(0.2)  # 스크립트 초기화 완료 대기
        print("[GoHome] RTDE 스크립트 재시작 완료")
        try:
            left_robot.robot.moveJ(_HOME_RAD, 0.5, 0.5)
            print("[GoHome] Done.")
        except Exception as e:
            print(f"[GoHome] moveJ 실패: {e}")

    def admittance_on_fn():
        _adm_bg_stop[0].set()
        time.sleep(0.05)
        _admittance_active[0] = True
        _start_admittance_bg()
        print("[Admittance] ON")

    def admittance_off_fn():
        _admittance_active[0] = False
        _adm_bg_stop[0].set()
        try:
            left_robot.robot.stopL(2.0)
        except Exception:
            pass
        print("[Admittance] OFF")

    _current_checkpoint_label = None
    if vla_cfg.get("checkpoint"):
        _ckpt_path = Path(vla_cfg["checkpoint"]).expanduser()
        # 기대 구조: <version>/checkpoints/<step>/pretrained_model
        if _ckpt_path.name == "pretrained_model" and _ckpt_path.parent.parent.name == "checkpoints":
            _current_checkpoint_label = f"{_ckpt_path.parent.parent.parent.name}/{_ckpt_path.parent.name}"

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
        admittance_on_fn=admittance_on_fn,
        admittance_off_fn=admittance_off_fn,
        cameras=cameras,
        record_queue=_record_queue,
        checkpoints_dir=checkpoints_root_dir,
        datasets_dir=datasets_root_dir,
        current_checkpoint=_current_checkpoint_label,
        current_dataset=_current_dataset_name,
        set_checkpoint_fn=set_checkpoint_fn,
        set_dataset_fn=set_dataset_fn,
        vla_params=vla_cfg,
        set_vla_params_fn=set_vla_params_fn,
        start_act_training_fn=start_act_training_fn,
        start_align_training_fn=start_align_training_fn,
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
