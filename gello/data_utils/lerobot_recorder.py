"""LeRobot v0.4 native dataset recorder for GELLO teleoperation."""

from typing import Dict, List, Optional

import numpy as np


def make_lerobot_dataset(
    repo_id: str,
    root: str,
    fps: int,
    state_dim: int,
    action_dim: int,
    camera_keys: List[str],
    img_height: int = 480,
    img_width: int = 640,
    robot_type: str = "ur10",
):
    """Create a new LeRobotDataset for recording.

    Args:
        repo_id:    HuggingFace-style id, e.g. "koras/ur10_task"
        root:       Local directory to save data
        fps:        Recording frame rate
        state_dim:  Dimension of robot state vector
        action_dim: Dimension of action vector
        camera_keys: List of camera names, e.g. ["wrist", "exterior"]
        robot_type: Robot type string
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"joint_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"joint_{i}" for i in range(action_dim)],
        },
    }

    for key in camera_keys:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (img_height, img_width, 3),
            "names": ["height", "width", "channel"],
        }

    import os
    import shutil
    import json
    meta_file = os.path.join(root, "meta", "tasks.parquet")
    info_file = os.path.join(root, "meta", "info.json")
    # libsvtav1은 segfault 유발 → 기존 데이터셋이어도 재생성
    bad_codec = False
    if os.path.exists(info_file):
        try:
            with open(info_file) as f:
                info = json.load(f)
            for feat in info.get("features", {}).values():
                if isinstance(feat, dict) and feat.get("info", {}).get("video.codec") == "libsvtav1":
                    bad_codec = True
                    break
        except Exception:
            bad_codec = True
    if os.path.exists(meta_file) and not bad_codec:
        try:
            dataset = LeRobotDataset(repo_id=repo_id, root=root, vcodec="h264")
            return dataset
        except Exception as e:
            print(f"[Dataset] 기존 데이터셋 로드 실패 ({e}), 기존 데이터 보존 후 새로 생성합니다.")
            import datetime
            backup = root.rstrip("/") + f"_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil as _shutil
            _shutil.move(root, backup)
            print(f"[Dataset] 기존 데이터 백업: {backup}")
    elif os.path.exists(root) and not bad_codec:
        # 디렉토리는 있지만 meta/tasks.parquet 없음 → 빈/불완전 디렉토리, 삭제 후 재생성
        print(f"[Dataset] 불완전한 데이터셋 디렉토리 감지 → 삭제 후 재생성합니다.")
        shutil.rmtree(root)
    elif bad_codec:
        print("[Dataset] libsvtav1 codec 감지 → h264로 재생성합니다.")
        if os.path.exists(root):
            shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=robot_type,
        use_videos=True,
        image_writer_threads=4,
        vcodec="h264",
    )
    return dataset


class LeRobotRecorder:
    """Thin wrapper around LeRobotDataset for frame-by-frame recording."""

    def __init__(self, dataset, task: str = "teleoperation"):
        self.dataset = dataset
        self.task = task
        self._recording = False
        try:
            self._episode_count = dataset.num_episodes
        except Exception:
            self._episode_count = 0

    @property
    def is_recording(self):
        return self._recording

    def start_episode(self):
        self._recording = True
        print(f"Recording episode {self._episode_count}...")

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        images: Dict[str, np.ndarray],
    ):
        """Add one frame to the current episode."""
        frame = {
            "task": self.task,
            "observation.state": state.astype(np.float32),
            "action": action.astype(np.float32),
        }
        for key, img in images.items():
            frame[f"observation.images.{key}"] = img  # (H, W, 3) uint8

        self.dataset.add_frame(frame)

    def end_episode(self, save: bool = True, record_queue=None):
        """Finalize the current episode."""
        if not self._recording:
            return

        self._recording = False  # 새 프레임이 큐에 추가되는 것을 막음

        # record_queue가 다 빠질 때까지 대기 (최대 30초)
        # _recording=False 후에도 큐에 남은 프레임은 add_frame으로 정상 처리됨
        if record_queue is not None:
            import threading as _threading
            _done = _threading.Event()
            _threading.Thread(target=lambda: (record_queue.join(), _done.set()), daemon=True).start()
            if not _done.wait(timeout=30):
                print("[Recorder] record_queue join timeout, proceeding anyway")

        if save:
            try:
                print("[Recorder] 비디오 인코딩 중... (30~60초 소요될 수 있음)")
                self.dataset.save_episode()
                # 에피소드 저장 직후 footer를 즉시 기록해 파일을 완성한다.
                # _writer_closed_for_reading = True 로 설정하면 다음 save_episode()가
                # 새 파일(file-001.parquet 등)을 열어 기존 파일을 덮어쓰지 않는다.
                try:
                    self.dataset._close_writer()
                    self.dataset._writer_closed_for_reading = True
                except Exception as e:
                    print(f"[Recorder] data writer close warning: {e}")
                try:
                    self.dataset.meta._flush_metadata_buffer()
                except Exception as e:
                    print(f"[Recorder] meta flush warning: {e}")
                self._episode_count += 1
                print(f"Saved. Total episodes: {self._episode_count}")
            except Exception as e:
                import traceback
                print(f"[Recorder] save_episode 실패: {e}")
                traceback.print_exc()
        else:
            self.dataset.clear_episode_buffer()
            print("Episode discarded.")

    def close(self):
        """Flush and close parquet writers. Call once when recording session is complete."""
        try:
            self.dataset._close_writer()
        except Exception as e:
            print(f"[Recorder] data writer close warning: {e}")
        try:
            self.dataset.meta._close_writer()
        except Exception as e:
            print(f"[Recorder] meta writer close warning: {e}")
