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


# 케이블 검출 ROI (비율)
_CABLE_ROI_TOP    = 0.51
_CABLE_ROI_BOTTOM = 0.65
_CABLE_ROI_LEFT   = 0.448
_CABLE_ROI_RIGHT  = 0.593
_CABLE_TIP_STRIP  = 20   # ROI 하단선 위 슬라이스 두께 (px)
_CABLE_TIP_Y_OFFSET = 0.05  # 케이블 tip y 좌표 위로 이동 (이미지 높이 비율)
_MIN_AREA_COLOR   = 5    # 노란색 검출 최소 픽셀
_MIN_AREA_CABLE   = 50   # 케이블(검정) 검출 최소 픽셀
_COLOR_ROI_LEFT   = 0.15
_COLOR_ROI_RIGHT  = 0.85

# Visual servoing 파라미터
_SERVOING_THRESHOLD = 0.007  # ACT xy_mag 이 값 미만이면 servoing 모드 진입
_SERVOING_K        = 0.05    # 비례 게인 (실험으로 튜닝) — 낮출수록 정렬 오차에 덜 민감하게(부드럽게) 반응

# ACT 예측 z 델타에 곱하는 게인 — z_freeze 진입 전(순수 ACT 구간)의 z 민감도 조절
_Z_GAIN = 1.5

# TCP orientation 고정 (roll/pitch)
_TARGET_ROLL_RAD  = np.deg2rad(-179.99)
_TARGET_PITCH_RAD = np.deg2rad(0.0)


def _rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
    """RPY (roll, pitch, yaw) → UR rotation vector (axis-angle)."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if abs(ang) < 1e-10:
        return np.zeros(3)
    return (ang / (2 * np.sin(ang))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])


def _fix_tcp_orientation(tcp_pose_6d):
    """rotation vector 자세에서 roll/pitch를 목표값으로 고정하고 yaw는 유지."""
    rv = np.array(tcp_pose_6d[3:6], dtype=np.float64)
    angle = np.linalg.norm(rv)
    if angle < 1e-10:
        R = np.eye(3)
    else:
        ax = rv / angle
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    yaw = np.arctan2(R[1, 0], R[0, 0])

    cr, sr = np.cos(_TARGET_ROLL_RAD),  np.sin(_TARGET_ROLL_RAD)
    cp, sp = np.cos(_TARGET_PITCH_RAD), np.sin(_TARGET_PITCH_RAD)
    cy, sy = np.cos(yaw),               np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R2 = Rz @ Ry @ Rx

    ang2 = np.arccos(np.clip((np.trace(R2) - 1) / 2, -1.0, 1.0))
    if abs(ang2) < 1e-10:
        rv2 = [0.0, 0.0, 0.0]
    else:
        rv2 = (ang2 / (2 * np.sin(ang2))) * np.array([
            R2[2, 1] - R2[1, 2], R2[0, 2] - R2[2, 0], R2[1, 0] - R2[0, 1]
        ])

    return list(tcp_pose_6d[:3]) + list(rv2)


def _detect_color_centroid(img_bgr, lower_hsv, upper_hsv):
    """HSV 범위로 검출된 모든 픽셀의 무게중심 (u, v) 반환. 검출 실패 시 None.
    노란 커넥터처럼 케이블에 가려 두 덩어리로 쪼개지는 경우에도 중심을 올바르게 추정한다."""
    import cv2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    h, w = mask.shape
    mask[:, :int(w * _COLOR_ROI_LEFT)]  = 0
    mask[:, int(w * _COLOR_ROI_RIGHT):] = 0
    pts = np.argwhere(mask > 0)  # (row, col)
    if len(pts) < _MIN_AREA_COLOR:
        return None
    cy = int(pts[:, 0].mean())
    cx = int(pts[:, 1].mean())
    return cx, cy


def _detect_cable_tip(img_bgr):
    """케이블(검정) ROI 적용 후 하단 tip 픽셀 좌표 반환. 검출 실패 시 None."""
    import cv2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 80, 80]))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    h, w = mask.shape
    top_y   = int(h * _CABLE_ROI_TOP)
    bot_y   = int(h * _CABLE_ROI_BOTTOM)
    left_x  = int(w * _CABLE_ROI_LEFT)
    right_x = int(w * _CABLE_ROI_RIGHT)
    mask[:top_y] = 0
    mask[bot_y:] = 0
    mask[:, :left_x] = 0
    mask[:, right_x:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < _MIN_AREA_CABLE:
        return None
    strip_top = max(0, bot_y - _CABLE_TIP_STRIP)
    strip = mask[strip_top:bot_y, :]
    cols = np.where(strip.any(axis=0))[0]
    if len(cols) == 0:
        return None
    h_full = mask.shape[0]
    return int(cols.mean()), bot_y - _CABLE_TIP_STRIP // 2 - int(h_full * _CABLE_TIP_Y_OFFSET)


_ALIGN_FIXED_DEPTH = 0.30  # depth 센서 대신 사용할 고정 작업거리 (m)


class _AlignBuffer:
    """백그라운드 스레드에서 정렬 계산을 지속 실행하고 최신 결과를 캐싱."""
    def __init__(self, cam_buffer, wrist_cam):
        self._cam_buffer = cam_buffer
        self._wrist_cam = wrist_cam
        self._result = np.zeros(2, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import cv2
        while not self._stop.is_set():
            img_rgb = self._cam_buffer.get()
            if img_rgb is None:
                time.sleep(0.005)
                continue
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            xy = compute_alignment_xyz(img_bgr, self._wrist_cam)[:2]
            with self._lock:
                self._result = xy

    def get(self) -> np.ndarray:
        with self._lock:
            return self._result.copy()

    def stop(self):
        self._stop.set()


def compute_alignment_xyz(img_bgr, wrist_cam) -> np.ndarray:
    """손목 카메라에서 케이블(검정) tip - 노란 물체의 xy 오정렬 벡터(미터) 반환.
    depth 센서 대신 고정 작업거리로 픽셀 → 미터 변환 (z=0)."""
    yellow = _detect_color_centroid(img_bgr, [22, 150, 120], [32, 255, 255])
    cable  = _detect_cable_tip(img_bgr)
    if yellow is None or cable is None:
        return np.zeros(3, dtype=np.float32)
    d = _ALIGN_FIXED_DEPTH
    dx = (cable[0] - yellow[0]) * d / wrist_cam.fx
    dy = (cable[1] - yellow[1]) * d / wrist_cam.fy
    return np.array([dx, dy, 0.0], dtype=np.float32)


def obs_to_tensor(obs: dict, camera_keys: list, device: str,
                  img_height: int, img_width: int, stats: dict,
                  wrist_cam=None, fix_orientation: bool = True,
                  orientation_key: str = "tcp_rotvec"):
    """Convert robot obs dict to ACTPolicy batch dict with normalization.

    orientation_key: 'tcp_rotvec'(v20, unwrap 적용) 또는 'tcp_rpy'(v19, unwrap 이전 학습)."""
    import torch
    import cv2

    # 손목 카메라로 빨강-노랑 xyz 오정렬 계산
    align_xyz = np.zeros(3, dtype=np.float32)
    if wrist_cam is not None and "observation.images.wrist" in obs:
        wrist_img = obs["observation.images.wrist"]   # RGB
        wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
        align_xyz = compute_alignment_xyz(wrist_bgr, wrist_cam)

    if orientation_key == "tcp_rpy":
        orientation = obs["tcp_rpy"].copy()
        if fix_orientation:
            orientation[0] = np.deg2rad(-179.99)
            orientation[1] = np.deg2rad(0.0)
    else:
        orientation = obs["tcp_rotvec"].copy()
        if fix_orientation:
            # roll/pitch 고정, yaw는 실제값 유지 — RPY로 임시 변환해 고정 후 축각으로 복원
            from gello.robots.ur import _rotvec_to_rpy
            rpy = _rotvec_to_rpy(orientation)
            rpy[0] = np.deg2rad(-179.99)
            rpy[1] = np.deg2rad(0.0)
            orientation = _rpy_to_rotvec(rpy)
    state = np.concatenate([
        obs["tcp_xyz_delta"],      # 3
        orientation,                # 3 (orientation_key에 따라 축각 또는 RPY)
        obs["gripper_position"],   # 1
        align_xyz[:2],             # 2 (align_x, align_y)
    ])  # 총 9차원
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
        self._depth = None  # RGB와 동일 타이밍의 depth (원본 해상도, uint16 mm)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            img, _ = self._cam.read(img_size=self._img_size)
            depth = getattr(self._cam, '_depth_raw', None)
            if depth is not None:
                depth = depth.copy()
            with self._lock:
                self._frame = img
                self._depth = depth

    def get(self):
        with self._lock:
            return self._frame

    def get_with_depth(self):
        with self._lock:
            return self._frame, self._depth


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
    delta_scale: float = 1.0,
    direct_robot=None,
    use_servoing: bool = True,
    fix_orientation: bool = False,
    z_approach_threshold: float = None,
    z_floor: float = None,
    use_z_freeze: bool = False,
    align_insert_threshold: float = 0.002,
    cartesian_action: bool = False,
    control_gripper: bool = True,
    admittance_gain: float = 0.0,
    admittance_deadband: float = 3.0,
    admittance_spring_k: float = 0.0,
    admittance_damping_b: float = 0.0,
) -> bool:
    """Main inference loop. Runs until stop_event is set.

    direct_robot: URRobot 인스턴스를 직접 전달하면 ZMQ 없이 로봇 직접 제어.
                  None이면 기존 env(ZMQ) 경로 사용.
    반환값: True = z+align 조건으로 삽입 준비 완료, False = 외부 stop 또는 정상 종료.
    """
    dt = speed_scale / fps
    camera_keys = [f"observation.images.{k}" for k in cameras]
    _orientation_key = direct_robot.state_orientation_key() if direct_robot is not None else "tcp_rotvec"
    print(f"[VLA] state orientation 표현: {_orientation_key}")

    effective_hz = fps / speed_scale
    mode = "직접(ZMQ 우회)" if direct_robot is not None else "ZMQ"
    print(f"[VLA] Starting inference at {fps}Hz × 1/{speed_scale:.1f} = {effective_hz:.1f}Hz effective, chunk_size={chunk_size}, mode={mode}")
    policy.reset()

    # 직전(하이브리드 사전 동작 등)의 큰 이동이 첫 tcp_xyz_delta에 섞여 들어가지 않도록
    # Δxyz/rotvec 추적 상태를 리셋한 뒤 obs를 다시 읽는다.
    if direct_robot is not None:
        direct_robot.reset_delta_tracking()
    obs = direct_robot.get_observations() if direct_robot is not None else env.get_obs()
    print(f"[VLA][reset] Δxyz/rotvec 리셋 직후 tcp_xyz_delta={obs['tcp_xyz_delta'].tolist()}")

    # 안전장치: 학습 데이터 통계와 무관한 절대 선속도 상한 (m/s).
    # speed_scale/delta_scale이 적용된 뒤 실제로 servoL에 들어가는 물리적 이동거리 기준.
    _MAX_LINEAR_SPEED = 0.1  # m/s
    _max_raw_xyz_norm = (_MAX_LINEAR_SPEED * dt) / max(speed_scale * delta_scale, 1e-9)

    # 어드미턴스 스프링 기준점: VLA 추론 시작 시점의 TCP 위치로 복귀
    _adm_origin_xyz = None
    if direct_robot is not None:
        try:
            _adm_origin_xyz = np.array(direct_robot.r_inter.getActualTCPPose()[:3])
        except Exception:
            _adm_origin_xyz = None
    _adm_ft_filtered = np.zeros(3)
    _ADM_FILTER_ALPHA = 0.2  # 낮을수록 힘 신호를 더 부드럽게(떨림 억제)

    # 카메라 백그라운드 버퍼 시작 (블로킹 방지)
    cam_buffers = {key: _CameraBuffer(cam, (img_width, img_height)) for key, cam in cameras.items()}
    # 모든 카메라 버퍼에 첫 프레임이 들어올 때까지 대기
    print("[VLA] 카메라 버퍼 준비 대기...")
    while not stop_event.is_set():
        if all(cam_buffers[k].get() is not None for k in cameras):
            break
        time.sleep(0.01)
    print("[VLA] 카메라 버퍼 준비 완료")

    # 정렬 계산 백그라운드 버퍼 시작
    _wrist_cam = cameras.get("wrist")
    _align_buf = None
    if _wrist_cam is not None and "wrist" in cam_buffers:
        _align_buf = _AlignBuffer(cam_buffers["wrist"], _wrist_cam)

    # 서보잉 목표: 두 점을 정확히 일치(align_y=0)시키는 게 아니라,
    # 노란점이 파란(케이블)점보다 이미지 높이의 약 3%만큼 위에 오도록 정렬한다.
    # align_y는 "양수 = cable이 yellow보다 아래(yellow가 위)"이므로 목표값은 양수.
    _SERVOING_TARGET_Y_PCT = 0.03  # 이미지 높이 대비 목표 오프셋 비율
    if _wrist_cam is not None:
        _servoing_target_y = _SERVOING_TARGET_Y_PCT * img_height * _ALIGN_FIXED_DEPTH / _wrist_cam.fy
    else:
        _servoing_target_y = 0.0

    # 비동기 추론: chunk 경계에서 다음 chunk를 백그라운드로 미리 계산
    import concurrent.futures
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _next_future = None   # 다음 chunk 추론 결과 (Future)
    _step_in_chunk = 0
    _cached_actions = []  # 현재 chunk 액션 리스트
    _servoing_mode = False  # 한번 전환되면 에피소드 끝까지 유지
    _insertion_ready = False  # z+align 조건으로 삽입 준비 완료 시 True
    _z_freeze_mode = False

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
                            wrist_cam=cameras.get("wrist"), fix_orientation=fix_orientation,
                            orientation_key=_orientation_key)
    policy.reset()
    _cached_actions = _infer_chunk(_batch0)
    _step_in_chunk = 0

    # 디버그용 손목 카메라 영상 저장 (VLA Demo 실행 중 전체를 mp4로)
    import os as _os_dbg
    _debug_dir = _os_dbg.path.expanduser(f"~/vla_debug_frames/{int(time.time())}")
    _os_dbg.makedirs(_debug_dir, exist_ok=True)
    _debug_video_path = f"{_debug_dir}/wrist.mp4"
    print(f"[VLA] 손목 카메라 디버그 영상 저장 경로: {_debug_video_path}")
    _debug_video_writer = None

    while not stop_event.is_set():
        t0 = time.time()

        # 카메라 버퍼에서 즉시 읽기 (블로킹 없음)
        for key in cameras:
            img = cam_buffers[key].get()
            if img is not None:
                obs[f"observation.images.{key}"] = img

        wrist_img_now = obs.get("observation.images.wrist")
        if wrist_img_now is not None:
            import cv2 as _cv2_dbg
            _bgr_dbg = _cv2_dbg.cvtColor(wrist_img_now, _cv2_dbg.COLOR_RGB2BGR)

            # 케이블(검정) ROI 시각화
            _h_dbg0, _w_dbg0 = _bgr_dbg.shape[:2]
            _cv2_dbg.rectangle(
                _bgr_dbg,
                (int(_w_dbg0 * _CABLE_ROI_LEFT), int(_h_dbg0 * _CABLE_ROI_TOP)),
                (int(_w_dbg0 * _CABLE_ROI_RIGHT), int(_h_dbg0 * _CABLE_ROI_BOTTOM)),
                (255, 255, 0), 1,
            )

            # 노란 물체 / 케이블 tip 검출 결과 오버레이
            _yellow_px = _detect_color_centroid(_bgr_dbg, [22, 150, 120], [32, 255, 255])
            _cable_px = _detect_cable_tip(_bgr_dbg)
            if _yellow_px is not None:
                _cv2_dbg.circle(_bgr_dbg, _yellow_px, 6, (0, 255, 255), -1)
                _cv2_dbg.putText(_bgr_dbg, "yellow", (_yellow_px[0] + 8, _yellow_px[1]),
                                  _cv2_dbg.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            if _cable_px is not None:
                _cv2_dbg.circle(_bgr_dbg, _cable_px, 6, (255, 0, 0), -1)
                _cv2_dbg.putText(_bgr_dbg, "cable", (_cable_px[0] + 8, _cable_px[1]),
                                  _cv2_dbg.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            if _yellow_px is not None and _cable_px is not None:
                _cv2_dbg.line(_bgr_dbg, _yellow_px, _cable_px, (0, 255, 0), 1)

            # 현재 정렬(align_x, align_y) 값 텍스트 표시
            if _align_buf is not None:
                _align_xy_dbg = _align_buf.get()
                _cv2_dbg.putText(
                    _bgr_dbg,
                    f"align x={_align_xy_dbg[0]*1000:.1f}mm y={_align_xy_dbg[1]*1000:.1f}mm",
                    (10, _h_dbg0 - 10), _cv2_dbg.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                )

            if _debug_video_writer is None:
                _h_dbg, _w_dbg = _bgr_dbg.shape[:2]
                _fourcc_dbg = _cv2_dbg.VideoWriter_fourcc(*"mp4v")
                _debug_video_writer = _cv2_dbg.VideoWriter(
                    _debug_video_path, _fourcc_dbg, effective_hz, (_w_dbg, _h_dbg))
            _debug_video_writer.write(_bgr_dbg)

        # chunk 첫 step에 다음 chunk 비동기 추론 시작 (15 step × 33ms = 500ms 여유)
        if _step_in_chunk == 0 and _next_future is None:
            batch_snap = obs_to_tensor(obs, camera_keys, device, img_height, img_width, stats,
                                       wrist_cam=cameras.get("wrist"), fix_orientation=fix_orientation,
                            orientation_key=_orientation_key)
            _next_future = _executor.submit(_infer_chunk, batch_snap)

        # 현재 chunk에서 액션 꺼내기
        delta_np = _cached_actions[_step_in_chunk].copy()
        _step_in_chunk += 1
        delta_np[2] = delta_np[2] * _Z_GAIN  # z_freeze 진입 전 ACT z 민감도 조절 (진입 후엔 0으로 덮어써짐)

        # align_xy: 백그라운드 스레드에서 계산된 최신 값 읽기
        _align_xy = _align_buf.get() if _align_buf is not None else np.zeros(2, dtype=np.float32)
        align_x_mag = abs(float(_align_xy[0]))   # x 오정렬 절댓값
        align_y_val = float(_align_xy[1])         # y: 양수 = cable이 yellow보다 아래 (yellow가 위)


        # Visual servoing: ACT와 동일하게 시작/종료, xy delta를 align 비례제어로 대체
        if use_servoing and np.linalg.norm(_align_xy) > 1e-4:
            _y_err = _align_xy[1] - _servoing_target_y
            delta_np[0] = _SERVOING_K * _align_xy[0]
            delta_np[1] = -_SERVOING_K * _y_err  # y축 부호 반전 + 목표 오프셋(노란점이 목표% 위)
            print(f"[VLA][servoing] align_y={_align_xy[1]:.4f} target={_servoing_target_y:.4f} "
                  f"err={_y_err:.4f} dy={delta_np[1]:.5f}")

        # z 제어 및 삽입 조건 체크
        if direct_robot is not None:
            try:
                tcp_z = direct_robot.r_inter.getActualTCPPose()[2]
                # z_approach 도달 전, ACT의 z 예측이 정체/수렴해 중간에 멈추는 것을 방지하기 위한
                # 최소 하강 속도 보장. 모델이 이미 이보다 빠르게 내려가고 있으면 그대로 두고,
                # 느리거나 멈추거나 올라가려 하면 강제로 이 속도로 내려가게 한다.
                _MIN_DESCENT_RATE = -0.0002  # m/step
                if (z_approach_threshold is not None and not _z_freeze_mode
                        and tcp_z > z_approach_threshold
                        and delta_np[2] > _MIN_DESCENT_RATE):
                    delta_np[2] = _MIN_DESCENT_RATE
                if use_z_freeze:
                    if z_approach_threshold is not None and not _z_freeze_mode and tcp_z <= z_approach_threshold:
                        _z_freeze_mode = True
                        print(f"[VLA] Z-freeze 진입 (z={tcp_z:.4f}m, 모드={use_z_freeze})")
                    if _z_freeze_mode:
                        delta_np[2] = 0.0
                        if use_z_freeze == "servoing":
                            # ACT 출력 무시, 서보잉으로 xy 제어
                            _y_err = _align_xy[1] - _servoing_target_y
                            delta_np[0] = _SERVOING_K * _align_xy[0]
                            delta_np[1] = -_SERVOING_K * _y_err  # y축 부호 반전 + 목표 오프셋(노란점이 목표% 위)
                            print(f"[VLA][servoing] align_y={_align_xy[1]:.4f} target={_servoing_target_y:.4f} "
                                  f"err={_y_err:.4f} dy={delta_np[1]:.5f}")
                else:
                    # z_floor 모드: z_floor 이하 하강 차단
                    if z_floor is not None and tcp_z <= z_floor:
                        delta_np[2] = max(delta_np[2], 0.0)
                # 삽입 조건: 모든 모드 공통
                if z_approach_threshold is not None and tcp_z <= z_approach_threshold:
                    _x_ok = 0 < align_x_mag < align_insert_threshold
                    _y_ok = 0.005 < align_y_val < 0.05
                    if _x_ok and _y_ok:
                        print(f"[VLA] 삽입 준비 완료 (x={align_x_mag*1000:.1f}mm, y={align_y_val*1000:.1f}mm)")
                        _insertion_ready = True
                        stop_event.set()
            except Exception:
                pass

        # 절대 속도 상한: 서보잉/z-freeze로 xy가 덮어써진 뒤의 최종 delta_np에 적용해야
        # 서보잉 출력도 빠짐없이 안전 범위 안으로 제한된다.
        _xyz_norm = float(np.linalg.norm(delta_np[:3]))
        if _xyz_norm > _max_raw_xyz_norm:
            _scale = _max_raw_xyz_norm / (_xyz_norm + 1e-12)
            print(f"[VLA][safety] 절대 속도 상한({_MAX_LINEAR_SPEED} m/s) 초과, 축소 적용: "
                  f"{np.round(delta_np[:3], 5).tolist()} -> {np.round(delta_np[:3] * _scale, 5).tolist()}")
            delta_np[:3] = delta_np[:3] * _scale

        # chunk 소진 시 다음 chunk로 교체
        if _step_in_chunk >= chunk_size:
            t_wait = time.time()
            try:
                _cached_actions = _next_future.result()
            except Exception as _fe:
                print(f"[VLA] 추론 오류: {_fe}")
                break
            wait_ms = (time.time() - t_wait) * 1000
            if wait_ms > 2.0:
                print(f"[VLA] 다음 chunk 대기: {wait_ms:.1f}ms")
            _next_future = None
            _step_in_chunk = 0

        q_current = np.array(obs["joint_positions"])

        if cartesian_action and direct_robot is not None:
            # delta_np = [dx, dy, dz, d_rx, d_ry, d_rz(축각 델타), d_gripper]
            try:
                _cur_tcp = list(direct_robot.r_inter.getActualTCPPose())  # [x,y,z,rx,ry,rz]
                # xyz: 직접 delta 적용
                new_xyz = [_cur_tcp[i] + delta_np[i] * speed_scale * delta_scale for i in range(3)]
                # orientation: 회전 델타 적용 방식은 기록 시와 대칭
                # (v20: 회전행렬 합성 기반 rotvec 델타 / v19: RPY 성분별 차)
                if direct_robot.uses_rotvec_action_delta():
                    from scipy.spatial.transform import Rotation as _RotApply
                    R_cur = _RotApply.from_rotvec(np.array(_cur_tcp[3:6], dtype=np.float64))
                    R_delta = _RotApply.from_rotvec(np.array(delta_np[3:6], dtype=np.float64) * speed_scale * delta_scale)
                    new_rv = (R_delta * R_cur).as_rotvec()
                else:
                    from gello.robots.ur import _rotvec_to_rpy
                    cur_rpy = _rotvec_to_rpy(np.array(_cur_tcp[3:6], dtype=np.float64)).astype(np.float64)
                    new_rpy = cur_rpy + np.array(delta_np[3:6], dtype=np.float64) * speed_scale * delta_scale
                    new_rv = _rpy_to_rotvec(new_rpy)
                # 소프트웨어 어드미턴스: F/T 센서 기반 위치 보정
                if admittance_gain > 0.0:
                    try:
                        ft_tcp = np.array(direct_robot.r_inter.getActualTCPForce()[:3])
                        from scipy.spatial.transform import Rotation as _Rot
                        R = _Rot.from_rotvec(_cur_tcp[3:6]).as_matrix()
                        ft_base = R @ ft_tcp
                        ft_base[0] = -ft_base[0]  # x축 반대 방향 보정
                        ft_base[2] = -ft_base[2]  # z축 반대 방향 보정
                        # 저역통과 필터: 힘 신호 노이즈로 인한 떨림 억제
                        _adm_ft_filtered = _ADM_FILTER_ALPHA * ft_base + (1 - _ADM_FILTER_ALPHA) * _adm_ft_filtered
                        ft_base = _adm_ft_filtered
                        print(f"[Adm] ft_base={[round(float(v),2) for v in ft_base]}")
                        # deadband: 노이즈 무시
                        ft_clipped = np.where(
                            np.abs(ft_base) > admittance_deadband,
                            ft_base - np.sign(ft_base) * admittance_deadband,
                            0.0,
                        )
                        adm_offset = ft_clipped * admittance_gain
                        # 스프링: 추론 시작 시점 위치로 되돌리는 복원력
                        if admittance_spring_k > 0.0 and _adm_origin_xyz is not None:
                            adm_offset = adm_offset - admittance_spring_k * (np.array(_cur_tcp[:3]) - _adm_origin_xyz)
                        # 댐핑: 실제 TCP 속도에 반대로 작용해 진동 억제
                        if admittance_damping_b > 0.0:
                            v_actual = np.array(direct_robot.r_inter.getActualTCPSpeed()[:3])
                            adm_offset = adm_offset - admittance_damping_b * v_actual
                        new_xyz = [new_xyz[i] + float(adm_offset[i]) for i in range(3)]
                    except Exception:
                        pass
                tcp_pose = new_xyz + list(new_rv)
                direct_robot.robot.servoL(tcp_pose, 0.5, 0.5, dt, 0.1, 300)
                if control_gripper and direct_robot._datc_gripper is not None and len(delta_np) > 6:
                    cur_gripper = q_current[6] if len(q_current) > 6 else 0.0
                    t = np.clip(cur_gripper + delta_np[6], 0, 1)
                    import threading as _threading
                    _threading.Thread(
                        target=direct_robot._datc_gripper.set_position,
                        args=(int(990 - t * (990 - 1)),), daemon=True
                    ).start()
            except Exception as _e:
                print(f"[VLA] servoL 오류: {_e}")
            obs = direct_robot.get_observations(full=False)
        else:
            action_np = q_current + delta_np

            # roll/pitch 고정: FK → orientation 보정 → IK (fix_orientation=True 일 때만)
            if fix_orientation and direct_robot is not None:
                try:
                    q_tgt = action_np[:6].tolist()
                    fk = direct_robot.robot.getForwardKinematics(q_tgt)
                    fk_fixed = _fix_tcp_orientation(fk)
                    q_fixed = direct_robot.robot.getInverseKinematics(fk_fixed, q_tgt)
                    if q_fixed is not None and len(q_fixed) == 6:
                        action_np[:6] = np.array(q_fixed)
                except Exception:
                    pass

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
    if _align_buf is not None:
        _align_buf.stop()
    if _debug_video_writer is not None:
        _debug_video_writer.release()
        print(f"[VLA] 손목 카메라 디버그 영상 저장 완료: {_debug_video_path}")
    print("[VLA] Inference stopped.")
    return _insertion_ready


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
