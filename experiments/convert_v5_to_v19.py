"""v5 데이터셋 (관절 공간 action 7-dim) → v19 (Cartesian action 7-dim) 변환 스크립트.

변환 규칙:
  state:  10-dim → 9-dim (align_z 제거, [:9])
  action: joint_delta[7] → cartesian_delta[7]
    [0:3] = state[t+1][0:3]       (다음 프레임 tcp_xyz_delta = 실제 TCP 이동)
    [3:6] = state[t+1][3:6] - state[t][3:6]  (rpy 변화량)
    [6]   = joint_action[t][6]    (gripper delta 그대로)
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SRC = Path("~/datasets/ur10_task_v5").expanduser()
DST = Path("~/datasets/ur10_task_v19").expanduser()


def convert_episode(group: pd.DataFrame) -> pd.DataFrame:
    """에피소드 단위로 state/action 변환."""
    group = group.sort_values("frame_index").reset_index(drop=True)
    n = len(group)

    states  = np.stack(group["observation.state"].values)   # (n, 10)
    actions = np.stack(group["action"].values)              # (n, 7)

    new_states  = states[:, :9].astype(np.float32)          # align_z 제거
    new_actions = np.zeros((n, 7), dtype=np.float32)

    for t in range(n):
        if t < n - 1:
            xyz_delta = states[t + 1, 0:3]
            rpy_delta = states[t + 1, 3:6] - states[t, 3:6]
        else:
            # 마지막 프레임: xyz/rpy delta 0
            xyz_delta = np.zeros(3, dtype=np.float32)
            rpy_delta = np.zeros(3, dtype=np.float32)
        gripper_delta = actions[t, 6:7]
        new_actions[t] = np.concatenate([xyz_delta, rpy_delta, gripper_delta])

    result = group.copy()
    result["observation.state"] = list(new_states)
    result["action"]            = list(new_actions)
    return result


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    # ── 데이터 변환 ──────────────────────────────────────────────────────────
    src_data = SRC / "data"
    dst_data = DST / "data"

    all_parquets = sorted(src_data.rglob("*.parquet"))
    print(f"변환 대상 parquet: {len(all_parquets)}개")

    for src_file in tqdm(all_parquets):
        rel = src_file.relative_to(SRC)
        dst_file = DST / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src_file)
        episodes = []
        for ep_idx, group in df.groupby("episode_index"):
            converted_ep = convert_episode(group)
            converted_ep["episode_index"] = ep_idx  # 컬럼 명시 복원
            episodes.append(converted_ep)
        converted = pd.concat(episodes).reset_index(drop=True)
        converted.to_parquet(dst_file, index=False)

    # ── 비디오 복사 (변경 없음) ──────────────────────────────────────────────
    src_videos = SRC / "videos"
    if src_videos.exists():
        print("비디오 복사 중...")
        shutil.copytree(src_videos, DST / "videos")

    # ── 이미지 복사 (변경 없음) ──────────────────────────────────────────────
    src_images = SRC / "images"
    if src_images.exists():
        shutil.copytree(src_images, DST / "images")

    # ── 메타데이터 수정 ──────────────────────────────────────────────────────
    with open(SRC / "meta" / "info.json") as f:
        info = json.load(f)

    info["features"]["observation.state"]["shape"] = [9]
    info["features"]["observation.state"]["names"] = [
        "tcp_x_delta", "tcp_y_delta", "tcp_z_delta",
        "tcp_roll", "tcp_pitch", "tcp_yaw",
        "gripper",
        "align_x", "align_y",
    ]
    info["features"]["action"]["shape"] = [7]
    info["features"]["action"]["names"] = [
        "dx", "dy", "dz",
        "d_roll", "d_pitch", "d_yaw",
        "d_gripper",
    ]

    dst_meta = DST / "meta"
    shutil.copytree(SRC / "meta", dst_meta)
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n완료: {DST}")
    print(f"  state: 10-dim → 9-dim")
    print(f"  action: joint_delta(7) → cartesian_delta(7)  [dx,dy,dz,dr,dp,dyaw,dg]")


if __name__ == "__main__":
    main()
