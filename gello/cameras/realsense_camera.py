import os
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from gello.cameras.camera import CameraDriver


def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(self, device_id: Optional[str] = None, flip: bool = False):
        import pyrealsense2 as rs

        self._device_id = device_id

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self._pipeline.start(config)
        self._flip = flip
        self._lock = threading.Lock()

        # 카메라 내부 파라미터 저장 (3D 변환에 사용)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # raw → meters
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        self.cx, self.cy = intr.ppx, intr.ppy

        # 단일 배경 스레드가 프레임을 지속 캐시 — 여러 스레드가 read()를 동시 호출해도 안전
        self._color_cache: Optional[np.ndarray] = None
        self._depth_cache: Optional[np.ndarray] = None
        self._cache_lock = threading.Lock()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        # 첫 프레임 대기
        for _ in range(300):
            with self._cache_lock:
                if self._color_cache is not None:
                    break
            time.sleep(0.01)

    def _capture_loop(self):
        """단일 스레드에서 RealSense 프레임을 지속 수신해 캐시에 저장."""
        while True:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=3000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).copy()
                with self._cache_lock:
                    self._color_cache = color
                    self._depth_cache = depth
                    self._depth_raw = depth
            except Exception:
                time.sleep(0.01)

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """캐시된 최신 프레임을 반환. 여러 스레드에서 동시 호출 안전."""
        import cv2

        while True:
            with self._cache_lock:
                if self._color_cache is not None:
                    color_image = self._color_cache.copy()
                    depth_image = self._depth_cache.copy()
                    break
            time.sleep(0.005)

        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)

        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]

        return image, depth

    def pixel_to_3d(self, u: int, v: int) -> np.ndarray:
        """픽셀 좌표 (u, v)를 카메라 프레임의 3D 좌표(미터)로 변환."""
        src = getattr(self, '_depth_raw', None)
        if src is None:
            return np.zeros(3, dtype=np.float32)
        h, w = src.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        d = float(src[v, u]) * self.depth_scale  # meters
        if d <= 0.01 or d > 2.0:
            return np.zeros(3, dtype=np.float32)
        x = (u - self.cx) * d / self.fx
        y = (v - self.cy) * d / self.fy
        return np.array([x, y, d], dtype=np.float32)


def _debug_read(camera, save_datastream=False):
    import cv2

    cv2.namedWindow("image")
    cv2.namedWindow("depth")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    while True:
        time.sleep(0.1)
        image, depth = camera.read()
        depth = np.concatenate([depth, depth, depth], axis=-1)
        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        cv2.imshow("depth", depth)
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"images/depth_{counter}.png", depth)
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)
