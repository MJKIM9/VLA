"""여러 LeRobotDataset 폴더에 흩어진 에피소드를 하나의 대상 데이터셋으로 병합.

Ctrl+C 등으로 recorder가 제대로 안 닫혀 매번 새 데이터셋이 만들어지면서
같은 태스크의 에피소드가 여러 폴더(_backup_* 등)에 흩어졌을 때 사용.

Usage:
    python experiments/merge_episodes.py \
        --dst_repo_id koras/ur10_task_v20 \
        --dst_root ~/datasets/ur10_task_v20 \
        --src_roots ~/datasets/ur10_task_v20_backup_20260724_101518 \
                    ~/datasets/ur10_task_v20_backup_20260724_102050 \
                    ~/datasets/ur10_task_v20_backup_20260724_102259
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import tyro

from gello.data_utils.lerobot_recorder import LeRobotRecorder


@dataclass
class Args:
    dst_repo_id: str
    """병합 대상(이미 존재하는) 데이터셋의 repo_id."""

    dst_root: str
    """병합 대상 데이터셋의 로컬 디렉터리."""

    src_roots: List[str] = field(default_factory=list)
    """에피소드를 가져올 원본 데이터셋 디렉터리들 (순서대로 병합)."""


def _frame_to_add_frame_args(item: dict, camera_keys: list):
    state = item["observation.state"].numpy().astype(np.float32)
    action = item["action"].numpy().astype(np.float32)
    images = {}
    for key in camera_keys:
        img_t = item[f"observation.images.{key}"]
        img = (img_t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        images[key] = img
    return state, action, images


def main():
    args = tyro.cli(Args)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dst_root = str(Path(args.dst_root).expanduser())
    dst_dataset = LeRobotDataset(repo_id=args.dst_repo_id, root=dst_root, vcodec="h264")
    camera_keys = list(dst_dataset.meta.camera_keys)
    recorder = LeRobotRecorder(dataset=dst_dataset, task="teleoperation")

    print(f"[Merge] 병합 대상: {dst_root} (기존 {recorder._episode_count}개 에피소드)")

    total_added = 0
    for src_root_str in args.src_roots:
        src_root = str(Path(src_root_str).expanduser())
        print(f"\n[Merge] 원본 로드: {src_root}")
        src_dataset = LeRobotDataset(repo_id=Path(src_root).name, root=src_root, vcodec="h264")
        print(f"[Merge]   에피소드 {src_dataset.num_episodes}개, 프레임 {src_dataset.num_frames}개")

        # 프레임을 순서대로 훑으며 episode_index가 바뀌는 지점마다 에피소드 경계로 처리
        current_ep = None
        for i in range(len(src_dataset)):
            item = src_dataset[i]
            ep_idx = int(item["episode_index"].item())
            if current_ep is None:
                recorder.start_episode()
                current_ep = ep_idx
            elif ep_idx != current_ep:
                recorder.end_episode(save=True)
                print(f"[Merge]   에피소드 저장 완료 (누적 {recorder._episode_count}개)")
                recorder.start_episode()
                current_ep = ep_idx

            state, action, images = _frame_to_add_frame_args(item, camera_keys)
            recorder.add_frame(state=state, action=action, images=images)

        if current_ep is not None:
            recorder.end_episode(save=True)
            print(f"[Merge]   에피소드 저장 완료 (누적 {recorder._episode_count}개)")
        total_added += src_dataset.num_episodes

    recorder.close()
    print(f"\n[Merge] 완료. 원본에서 총 {total_added}개 에피소드를 가져왔습니다. "
          f"최종 {dst_root}의 에피소드 수: {recorder._episode_count}")


if __name__ == "__main__":
    main()
