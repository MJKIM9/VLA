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

    camera_keys: str = "observation.images.wrist,observation.images.exterior"
    """Comma-separated camera feature keys."""


def main():
    args = tyro.cli(Args)
    dataset_dir = str(Path(args.dataset_dir).expanduser())
    output_dir = str(Path(args.output_dir).expanduser())

    camera_keys = args.camera_keys.split(",")

    # Build LeRobot train command
    # lerobot_train.py uses Hydra config overrides
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--config-name=act",
        f"dataset.repo_id={args.repo_id}",
        f"dataset.root={dataset_dir}",
        f"training.output_dir={output_dir}",
        f"training.batch_size={args.batch_size}",
        f"training.num_workers={args.num_workers}",
        f"training.offline_steps={args.steps}",
        f"training.save_freq={args.save_freq}",
        f"training.lr={args.lr}",
        f"policy.chunk_size={args.chunk_size}",
    ]

    # Add camera image keys to policy config
    for i, key in enumerate(camera_keys):
        cmd.append(f"policy.input_shapes.{key}=[3,480,640]")

    print("Starting ACT training...")
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_dir}")
    print(f"Steps   : {args.steps}")
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
