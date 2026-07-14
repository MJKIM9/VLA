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


def detect_yellow_centroid(img_bgr):
    """손목 카메라 이미지에서 노란색 물체 픽셀 중심 (u, v) 반환. 검출 실패 시 None."""
    import cv2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([35, 255, 255]))
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


def compute_yellow_3d(img_bgr, wrist_cam) -> np.ndarray:
    """손목 카메라 이미지에서 노란색 물체의 카메라 프레임 3D 좌표(미터) 반환."""
    centroid = detect_yellow_centroid(img_bgr)
    if centroid is None:
        return np.zeros(3, dtype=np.float32)
    u, v = centroid
    return wrist_cam.pixel_to_3d(u, v)


def obs_to_tensor(obs: dict, camera_keys: list, device: str,
                  img_height: int, img_width: int, stats: dict,
                  wrist_cam=None):
    """Convert robot obs dict to ACTPolicy batch dict with normalization."""
    import torch
    import cv2

    # 손목 카메라로 노란 물체 3D 위치 계산
    yellow_xyz = np.zeros(3, dtype=np.float32)
    if wrist_cam is not None and f"observation.images.wrist" in obs:
        wrist_img = obs["observation.images.wrist"]   # RGB
        wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
        yellow_xyz = compute_yellow_3d(wrist_bgr, wrist_cam)

    state = np.concatenate([
        obs["joint_positions"],   # 7
        obs["joint_velocities"],  # 6
        obs["gripper_position"],  # 1
        yellow_xyz,               # 3
    ])
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
):
    """Main inference loop. Runs until stop_event is set.

    speed_scale: slow-motion factor. 1.0 = normal, 3.0 = 1/3 speed.
    The loop sleeps speed_scale times longer between steps.
    """
    dt = speed_scale / fps
    camera_keys = [f"observation.images.{k}" for k in cameras]

    effective_hz = fps / speed_scale
    print(f"[VLA] Starting inference at {fps}Hz × 1/{speed_scale:.1f} = {effective_hz:.1f}Hz effective, chunk_size={chunk_size}")
    policy.reset()

    obs = env.get_obs()

    while not stop_event.is_set():
        t0 = time.time()

        # Read cameras into obs dict
        for key, cam in cameras.items():
            img, _ = cam.read(img_size=(img_width, img_height))
            obs[f"observation.images.{key}"] = img

        batch = obs_to_tensor(obs, camera_keys, device, img_height, img_width, stats,
                              wrist_cam=cameras.get("wrist"))

        with torch.no_grad():
            action = policy.select_action(batch)   # (1, action_dim) — normalized space
            action = unnormalize_action(action, stats, device)

        action_np = action.squeeze(0).cpu().numpy()
        obs = env.step(action_np)

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

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
