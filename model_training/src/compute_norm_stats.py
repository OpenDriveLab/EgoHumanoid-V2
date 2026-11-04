"""Compute state and action normalization statistics from a LeRobot dataset."""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from eh2_policy.config import make_config
from eh2_policy.upstream import load_script
from openpi.shared import normalize
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--assets", default="assets/policy")
    parser.add_argument("--action-horizon", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    dataset = Path(args.dataset).resolve()
    if not (dataset / "meta/info.json").is_file():
        parser.error("dataset must contain meta/info.json")
    config = dataclasses.replace(
        make_config(str(dataset), action_horizon=args.action_horizon),
        assets_base_dir=str(Path(args.assets).resolve()),
    )
    data = config.data.create(config.assets_dirs, config.model)
    loader, count = load_script("compute_norm_stats").create_torch_dataloader(
        data, config.model.action_horizon, args.batch_size, config.model, args.workers
    )
    if count < 1:
        parser.error("dataset is smaller than --batch-size; lower the batch size")
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm(loader, total=count, desc="Normalization statistics"):
        for key, accumulator in stats.items():
            accumulator.update(np.asarray(batch[key]))
    target = config.assets_dirs / data.asset_id
    normalize.save(
        target, {key: value.get_statistics() for key, value in stats.items()}
    )
    print(f"Saved normalization statistics to {target}")


if __name__ == "__main__":
    main()
