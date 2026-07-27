"""align_xy → (dx, dy) 지도학습 모델 학습 스크립트."""
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tyro
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Args:
    dataset_dir: str = "~/datasets/ur10_task_v5"
    """학습에 사용할 로컬 데이터셋 디렉터리."""

    output_path: str = "~/checkpoints/align_xy_mlp.pt"
    """학습된 모델 저장 경로."""

    fine_threshold: float = 0.007
    """미세 구간 기준 (ACT xy_mag < 이 값인 프레임만 학습)."""

    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3


def main():
    args = tyro.cli(Args)
    dataset_dir = str(Path(args.dataset_dir).expanduser())
    output_path = str(Path(args.output_path).expanduser())

    # 1. 데이터 로드
    files = sorted(glob.glob(f"{dataset_dir}/data/chunk-000/file-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    states  = np.stack(df["observation.state"].values)  # (N, 10)
    actions = np.stack(df["action"].values)              # (N, 7)

    align_xy  = states[:, 7:9].astype(np.float32)   # align_x, align_y
    action_xy = actions[:, :2].astype(np.float32)   # dx, dy
    xy_mag    = np.linalg.norm(action_xy, axis=1)

    # 미세 구간 + align 검출 성공 프레임만
    mask = (np.linalg.norm(align_xy, axis=1) > 1e-4) & (xy_mag < args.fine_threshold)
    align_xy  = align_xy[mask]
    action_xy = action_xy[mask]
    print(f"전체 프레임: {len(states)}, 미세 구간 학습 샘플: {len(align_xy)}")

    # 2. 정규화
    ax_mean, ax_std = align_xy.mean(0),  align_xy.std(0) + 1e-8
    ay_mean, ay_std = action_xy.mean(0), action_xy.std(0) + 1e-8
    X = (align_xy  - ax_mean) / ax_std
    Y = (action_xy - ay_mean) / ay_std

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    # 3. 모델 정의
    model = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )

    # 4. 학습
    loader    = DataLoader(TensorDataset(X_t, Y_t), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        if epoch % 20 == 0:
            print(f"epoch {epoch:3d}  loss: {total / len(X_t):.6f}")

    # 5. 저장 (모델 + 정규화 파라미터)
    torch.save({
        "model_state": model.state_dict(),
        "ax_mean": ax_mean, "ax_std": ax_std,
        "ay_mean": ay_mean, "ay_std": ay_std,
    }, output_path)
    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()
