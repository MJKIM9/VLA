"""Train ACT policy on collected LeRobot dataset.

Usage:
    python experiments/train_act.py \
        --repo_id koras/ur10_task \
        --dataset_dir ~/datasets/ur10_task \
        --output_dir ~/checkpoints/ur10_act
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    repo_id: str = "koras/ur10_task"
    """Dataset repo id (must match collect_data.py)."""

    dataset_dir: str = "~/datasets/ur10_task"
    """Local dataset directory."""

    output_dir: str = "~/checkpoints/ur10_act"
    """Checkpoint output directory."""

    batch_size: int = 8
    """Training batch size (reduce if OOM)."""

    num_workers: int = 4
    """DataLoader workers."""

    steps: int = 100000
    """Total training steps."""

    save_freq: int = 5000
    """Save checkpoint every N steps."""

    lr: float = 1e-4
    """Learning rate."""

    chunk_size: int = 50
    """ACT action chunk size (frames to predict at once)."""


def main():
    args = tyro.cli(Args)
    dataset_dir = str(Path(args.dataset_dir).expanduser())
    output_dir = str(Path(args.output_dir).expanduser())

    # 설치된 lerobot 버전은 tyro 기반 CLI (--key.subkey 형식)를 사용.
    # 카메라 입력 shape은 데이터셋 메타데이터에서 자동 추론되므로 별도 지정 불필요.
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_dir}",
        "--policy.type=act",
        f"--policy.chunk_size={args.chunk_size}",
        # n_action_steps는 chunk_size 이하여야 함 (ACTConfig 제약) — 기본은 동일하게 맞춤
        f"--policy.n_action_steps={args.chunk_size}",
        f"--output_dir={output_dir}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--optimizer.lr={args.lr}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]

    print("Starting ACT training...")
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_dir}")
    print(f"Steps   : {args.steps}")
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
