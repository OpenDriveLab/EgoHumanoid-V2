"""Fine-tune pi0.5 using the official OpenPI PyTorch trainer."""

import argparse
import dataclasses
import os
from pathlib import Path

from eh2_policy.config import make_config
from eh2_policy.upstream import load_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help="Local LeRobot dataset directory"
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="Converted pi0.5 PyTorch checkpoint directory",
    )
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--assets", default="assets/policy")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--action-horizon", type=int, default=100)
    parser.add_argument(
        "--image-augmentation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Global batch size")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset = Path(args.dataset).resolve()
    if not (dataset / "meta/info.json").is_file():
        parser.error("dataset must contain meta/info.json")
    if not (Path(args.pretrained) / "model.safetensors").is_file():
        parser.error("pretrained must contain model.safetensors")
    config = dataclasses.replace(
        make_config(
            str(dataset),
            action_horizon=args.action_horizon,
            image_augmentation=args.image_augmentation,
        ),
        exp_name=args.name,
        assets_base_dir=str(Path(args.assets).resolve()),
        checkpoint_base_dir=str(Path(args.output).resolve()),
        pytorch_weight_path=str(Path(args.pretrained).resolve()),
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        num_workers=args.workers,
        resume=args.resume,
    )
    if args.batch_size < 1 or args.batch_size % int(os.environ.get("WORLD_SIZE", "1")):
        parser.error("global batch size must be positive and divisible by WORLD_SIZE")
    if not (config.assets_dirs / "eh2/norm_stats.json").is_file():
        parser.error("run eh2-norm-stats with the same --assets directory first")
    if args.steps < 1 or args.workers < 0:
        parser.error("steps must be positive and workers nonnegative")
    if (
        config.checkpoint_dir.exists()
        and any(config.checkpoint_dir.iterdir())
        and not args.resume
    ):
        parser.error("experiment directory is not empty; use a new --name or --resume")
    trainer = load_script("train_pytorch")
    if (
        args.resume
        and trainer.get_latest_checkpoint_step(config.checkpoint_dir) is None
    ):
        parser.error("no checkpoint found to resume")
    trainer.init_logging()
    trainer.train_loop(config)


if __name__ == "__main__":
    main()
