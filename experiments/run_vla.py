"""ACT policy inference on UR10 robot.

Usage:
    python experiments/run_vla.py \
        --left-config-path configs/ur10_gello.yaml \
        --checkpoint-path ~/checkpoints/ur10_act/final \
        --gripper-port /dev/ttyUSB1
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict


@dataclass
class Args:
    left_config_path: str
    """Robot config YAML (same as used for data collection)."""

    checkpoint_path: str
    """Path to trained ACT checkpoint directory."""

    gripper_port: Optional[str] = None
    """DATC gripper serial port."""

    chunk_size: int = 20
    """Number of actions to execute per inference call."""

    fps: int = 30
    """Control frequency."""

    camera_serials: List[str] = field(default_factory=list)
    """RealSense serial numbers. Empty = auto-detect."""

    img_height: int = 480
    img_width: int = 640

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_policy(checkpoint_path: str, device: str):
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.config.n_action_steps = 15
    policy.to(device)
    policy.eval()

    # Load normalizer/unnormalizer stats from checkpoint
    import os
    from safetensors import safe_open
    stats = {}
    pre_file = os.path.join(
        checkpoint_path, "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
    post_file = os.path.join(
        checkpoint_path, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    for path in [pre_file, post_file]:
        if os.path.exists(path):
            with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    if k not in stats:
                        stats[k] = f.get_tensor(k)
    if not stats:
        print("[VLA] WARNING: No normalizer stats found — policy output will NOT be unnormalized!")
    else:
        print(f"[VLA] Loaded normalizer stats ({len(stats)} entries)")
    return policy, stats


def _detect_color_centroid(img_bgr, lower_hsv, upper_hsv):
    """HSV 범위로 색상 검출 후 픽셀 중심 (u, v) 반환. 검출 실패 시 None."""
    import cv2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def compute_alignment_xyz(img_bgr, wrist_cam) -> np.ndarray:
    """손목 카메라에서 케이블(검정)-노란 물체의 xyz 오정렬 벡터(미터) 반환."""
    yellow = _detect_color_centroid(img_bgr, [20, 80, 80], [35, 255, 255])
    cable  = _detect_color_centroid(img_bgr, [0, 0, 0], [180, 50, 50])
    if yellow is None or cable is None:
        return np.zeros(3, dtype=np.float32)
    yellow_3d = wrist_cam.pixel_to_3d(*yellow)
    cable_3d  = wrist_cam.pixel_to_3d(*cable)
    if np.all(yellow_3d == 0) or np.all(cable_3d == 0):
        return np.zeros(3, dtype=np.float32)
    return (cable_3d - yellow_3d)


def obs_to_tensor(obs: dict, camera_keys: list, device: str,
                  img_height: int, img_width: int, stats: dict,
                  wrist_cam=None):
    """Convert robot obs dict to ACTPolicy batch dict with normalization."""
    import torch
    import cv2

    # 손목 카메라로 빨강-노랑 xyz 오정렬 계산
    align_xyz = np.zeros(3, dtype=np.float32)
    if wrist_cam is not None and "observation.images.wrist" in obs:
        wrist_img = obs["observation.images.wrist"]   # RGB
        wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
        align_xyz = compute_alignment_xyz(wrist_bgr, wrist_cam)

    state = np.concatenate([
        obs["tcp_xyz_delta"],      # 3
        obs["tcp_rpy"],            # 3
        obs["gripper_position"],   # 1
        align_xyz,                 # 3
    ])  # 총 10차원
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    # Normalize observation.state: z = (x - mean) / (std + eps)
    if "observation.state.mean" in stats:
        mean = stats["observation.state.mean"].to(device)
        std = stats["observation.state.std"].to(device)
        state_t = (state_t - mean) / (std + 1e-8)

    batch = {"observation.state": state_t}

    for key, img in obs.items():
        if key.startswith("observation.images."):
            t = torch.from_numpy(img.copy()).float() / 255.0   # (H,W,3) → [0,1]
            t = t.permute(2, 0, 1).unsqueeze(0).to(device)    # (1,3,H,W)
            cam_name = key[len("observation.images."):]
            mean_key = f"observation.images.{cam_name}.mean"
            std_key = f"observation.images.{cam_name}.std"
            if mean_key in stats:
                img_mean = stats[mean_key].to(device)
                img_std = stats[std_key].to(device)
                t = (t - img_mean) / (img_std + 1e-8)
            batch[key] = t

    return batch


def unnormalize_action(action: "torch.Tensor", stats: dict, device: str) -> "torch.Tensor":
    """Unnormalize model output: x = z * std + mean."""
    if "action.mean" not in stats:
        return action
    mean = stats["action.mean"].to(device)
    std = stats["action.std"].to(device)
    return action * std + mean


class _CameraBuffer:
    """카메라를 백그라운드 스레드에서 계속 읽고 최신 프레임을 즉시 반환."""
    def __init__(self, cam, img_size):
        self._cam = cam
        self._img_size = img_size
        self._frame = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            img, _ = self._cam.read(img_size=self._img_size)
            with self._lock:
                self._frame = img

    def get(self):
        with self._lock:
            return self._frame


def run_inference(
    env,
    cameras: dict,
    policy,
    stats: dict,
    device: str,
    fps: int,
    chunk_size: int,
    stop_event: threading.Event,
    img_height: int = 480,
    img_width: int = 640,
    speed_scale: float = 1.0,
    direct_robot=None,
):
    """Main inference loop. Runs until stop_event is set.

    direct_robot: URRobot 인스턴스를 직접 전달하면 ZMQ 없이 로봇 직접 제어.
                  None이면 기존 env(ZMQ) 경로 사용.
    """
    dt = speed_scale / fps
    camera_keys = [f"observation.images.{k}" for k in cameras]

    effective_hz = fps / speed_scale
    mode = "직접(ZMQ 우회)" if direct_robot is not None else "ZMQ"
    print(f"[VLA] Starting inference at {fps}Hz × 1/{speed_scale:.1f} = {effective_hz:.1f}Hz effective, chunk_size={chunk_size}, mode={mode}")
    policy.reset()

    obs = direct_robot.get_observations() if direct_robot is not None else env.get_obs()

    # 카메라 백그라운드 버퍼 시작 (블로킹 방지)
    cam_buffers = {key: _CameraBuffer(cam, (img_width, img_height)) for key, cam in cameras.items()}
    # 모든 카메라 버퍼에 첫 프레임이 들어올 때까지 대기
    print("[VLA] 카메라 버퍼 준비 대기...")
    while not stop_event.is_set():
        if all(cam_buffers[k].get() is not None for k in cameras):
            break
        time.sleep(0.01)
    print("[VLA] 카메라 버퍼 준비 완료")

    # 비동기 추론: chunk 경계에서 다음 chunk를 백그라운드로 미리 계산
    import concurrent.futures
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _next_future = None   # 다음 chunk 추론 결과 (Future)
    _step_in_chunk = 0
    _cached_actions = []  # 현재 chunk 액션 리스트

    def _infer_chunk(batch_snapshot):
        """백그라운드에서 chunk_size개 액션을 한 번에 계산."""
        actions = []
        with torch.no_grad():
            for _ in range(chunk_size):
                a = policy.select_action(batch_snapshot)
                a = unnormalize_action(a, stats, device)
                actions.append(a.squeeze(0).cpu().numpy())
        return actions

    # 첫 chunk 동기 실행 (워밍업)
    for key in cameras:
        img = cam_buffers[key].get()
        if img is not None:
            obs[f"observation.images.{key}"] = img
    _batch0 = obs_to_tensor(obs, camera_keys, device, img_height, img_width, stats,
                            wrist_cam=cameras.get("wrist"))
    policy.reset()
    _cached_actions = _infer_chunk(_batch0)
    _step_in_chunk = 0

    while not stop_event.is_set():
        t0 = time.time()

        # 카메라 버퍼에서 즉시 읽기 (블로킹 없음)
        for key in cameras:
            img = cam_buffers[key].get()
            if img is not None:
                obs[f"observation.images.{key}"] = img

        # chunk 첫 step에 다음 chunk 비동기 추론 시작 (15 step × 33ms = 500ms 여유)
        if _step_in_chunk == 0 and _next_future is None:
            batch_snap = obs_to_tensor(obs, camera_keys, device, img_height, img_width, stats,
                                       wrist_cam=cameras.get("wrist"))
            _next_future = _executor.submit(_infer_chunk, batch_snap)

        # 현재 chunk에서 액션 꺼내기
        delta_np = _cached_actions[_step_in_chunk]
        _step_in_chunk += 1

        # chunk 소진 시 다음 chunk로 교체
        if _step_in_chunk >= chunk_size:
            t_wait = time.time()
            _cached_actions = _next_future.result()
            wait_ms = (time.time() - t_wait) * 1000
            if wait_ms > 2.0:
                print(f"[VLA] 다음 chunk 대기: {wait_ms:.1f}ms")
            _next_future = None
            _step_in_chunk = 0

        q_current = np.array(obs["joint_positions"])
        action_np = q_current + delta_np

        if direct_robot is not None:
            direct_robot.command_joint_state(action_np, current_joints=q_current[:6])
            obs = direct_robot.get_observations(full=False)
        else:
            obs = env.step(action_np)

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
        else:
            print(f"[VLA] 주기 초과: {elapsed*1000:.1f}ms (목표 {dt*1000:.1f}ms, 초과 {-remaining*1000:.1f}ms)")

    _executor.shutdown(wait=False)
    print("[VLA] Inference stopped.")


def main():
    args = tyro.cli(Args)
    cfg = OmegaConf.to_container(OmegaConf.load(args.left_config_path), resolve=True)

    # --- Camera setup ---
    from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
    device_ids = args.camera_serials if args.camera_serials else get_device_ids()
    if len(device_ids) == 0:
        print("Warning: No cameras found. Running state-only inference.")
        cameras = {}
    elif len(device_ids) == 1:
        cameras = {"wrist": RealSenseCamera(device_id=device_ids[0])}
    else:
        cameras = {
            "wrist":    RealSenseCamera(device_id=device_ids[0]),
            "exterior": RealSenseCamera(device_id=device_ids[1]),
        }

    # --- Robot setup ---
    import zmq.error
    from gello.env import RobotEnv
    from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

    robot = instantiate_from_dict(cfg["robot"])

    if args.gripper_port:
        from gello.robots.datc_gripper import DATCGripper
        gripper = DATCGripper(port=args.gripper_port)
        if hasattr(robot, "_datc_gripper"):
            robot._datc_gripper = gripper
            robot._use_gripper = True

    hardware_port = cfg.get("hardware_server_port", 6001)
    server = ZMQServerRobot(robot, port=hardware_port)
    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()
    time.sleep(1.0)

    robot_client = ZMQClientRobot(port=hardware_port)
    env = RobotEnv(robot_client, control_rate_hz=args.fps)

    # --- Policy ---
    print(f"Loading policy from {args.checkpoint_path} ...")
    policy, stats = load_policy(
        str(Path(args.checkpoint_path).expanduser()),
        args.device,
    )
    print(f"Policy loaded. Device: {args.device}")

    # --- Run ---
    stop_event = threading.Event()
    try:
        run_inference(
            env=env,
            cameras=cameras,
            policy=policy,
            stats=stats,
            device=args.device,
            fps=args.fps,
            chunk_size=args.chunk_size,
            stop_event=stop_event,
            img_height=args.img_height,
            img_width=args.img_width,
        )
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[VLA] Interrupted.")


if __name__ == "__main__":
    main()
