"""Task-independent pi0.5 configuration for whole-body and hand actions."""

import dataclasses

from eh2_policy.transforms import ACTION_DIM, ACTION_KEYS, Inputs, Outputs, Repack
from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as upstream


@dataclasses.dataclass(frozen=True)
class DataConfig(upstream.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(inputs=[Repack()]),
            data_transforms=transforms.Group(inputs=[Inputs()], outputs=[Outputs()]),
            model_transforms=upstream.ModelTransformFactory()(model_config),
            action_sequence_keys=ACTION_KEYS,
            prompt_from_task=True,
        )


def make_config(
    dataset: str, *, action_horizon: int = 100, image_augmentation: bool = True
):
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    return upstream.TrainConfig(
        name="eh2",
        model=Pi0Config(
            pi05=True,
            action_dim=ACTION_DIM,
            action_horizon=action_horizon,
            max_token_len=288,
            image_augmentation=image_augmentation,
        ),
        data=DataConfig(repo_id=dataset, assets=upstream.AssetsConfig(asset_id="eh2")),
        batch_size=256,
        num_train_steps=30_000,
        save_interval=1000,
        num_workers=6,
        wandb_enabled=False,
    )
