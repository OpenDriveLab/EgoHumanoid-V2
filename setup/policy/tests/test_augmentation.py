"""Run with the patched OpenPI environment: python -m unittest discover -s setup/policy/tests."""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi.models import model
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch


class AugmentationTest(unittest.TestCase):
    def test_training_switch_and_inference(self):
        image = np.linspace(-1, 1, 2 * 32 * 32 * 3, dtype=np.float32).reshape(
            2, 32, 32, 3
        )
        for backend in ("jax", "torch"):
            convert = jnp.asarray if backend == "jax" else torch.as_tensor
            observation = model.Observation(
                images={"base_0_rgb": convert(image)},
                image_masks={"base_0_rgb": convert(np.ones(2, dtype=bool))},
                state=convert(np.ones((2, 43), dtype=np.float32)),
            )
            for train in (False, True):
                for enabled in (False, True):
                    with self.subTest(backend=backend, train=train, enabled=enabled):
                        kwargs = {
                            "train": train,
                            "image_augmentation": enabled,
                            "image_keys": ("base_0_rgb",),
                            "image_resolution": (32, 32),
                        }
                        if backend == "jax":
                            output = model.preprocess_observation(
                                jax.random.key(42), observation, **kwargs
                            )
                        else:
                            torch.manual_seed(42)
                            output = preprocess_observation_pytorch(
                                observation, **kwargs
                            )
                        result = np.asarray(output.images["base_0_rgb"])
                        self.assertEqual(
                            not np.array_equal(result, image), train and enabled
                        )
                        self.assertEqual(result.shape, image.shape)
                        self.assertTrue(np.isfinite(result).all())
                        np.testing.assert_array_equal(output.state, observation.state)
                        np.testing.assert_array_equal(
                            output.image_masks["base_0_rgb"],
                            observation.image_masks["base_0_rgb"],
                        )


if __name__ == "__main__":
    unittest.main()
