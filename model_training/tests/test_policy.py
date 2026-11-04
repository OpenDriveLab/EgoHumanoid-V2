"""Schema, pretrained-loading, checkpoint and HTTP contract tests."""

import base64
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from eh2_policy.config import make_config
from eh2_policy.pretrained import load_pretrained
from eh2_policy.serve import create_app
from eh2_policy.transforms import Inputs, Repack
from eh2_policy.upstream import load_script
from fastapi.testclient import TestClient
from PIL import Image
from safetensors.torch import save_model


class PolicyTest(unittest.TestCase):
    def test_training_and_serving_share_inputs(self):
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        state = np.arange(43, dtype=np.float32)
        raw = {
            "observation.images.gopro": image.transpose(2, 0, 1).astype(np.float32)
            / 255,
            "observation.state": state,
            "action.motion_token": np.full((4, 64), 1),
            "teleop.left_hand_joints": np.full((4, 6), 2),
            "teleop.right_hand_joints": np.full((4, 6), 3),
            "prompt": "close the laptop",
        }
        training = Inputs()(Repack()(raw))
        serving = Inputs()({"image": image, "state": state, "prompt": raw["prompt"]})
        np.testing.assert_array_equal(
            training["image"]["base_0_rgb"], serving["image"]["base_0_rgb"]
        )
        np.testing.assert_array_equal(training["state"], serving["state"])
        np.testing.assert_array_equal(training["state"][:43], state)
        self.assertTrue((training["state"][43:] == 0).all())
        self.assertEqual(training["actions"].shape, (4, 76))
        for start, end, value in [(0, 64, 1), (64, 70, 2), (70, 76, 3)]:
            self.assertTrue((training["actions"][:, start:end] == value).all())
        self.assertEqual(training["prompt"], raw["prompt"])

    def test_pretrained_backbone_preserved_and_head_initialized(self):
        def model(action_dim):
            result = torch.nn.Module()
            result.backbone = torch.nn.Linear(4, 4)
            result.action_in_proj = torch.nn.Linear(action_dim, 4)
            result.action_out_proj = torch.nn.Linear(4, action_dim)
            return result

        source, target = model(32), model(76)
        initial_head = target.action_out_proj.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_model(source, path)
            load_pretrained(target, path)
            torch.testing.assert_close(target.backbone.weight, source.backbone.weight)
            torch.testing.assert_close(
                target.action_in_proj.bias, source.action_in_proj.bias
            )
            torch.testing.assert_close(target.action_out_proj.weight, initial_head)
            target.backbone = torch.nn.Linear(5, 4)
            with self.assertRaises(ValueError):
                load_pretrained(target, path)

    def test_http_contract(self):
        image = io.BytesIO()
        Image.new("RGB", (16, 12), (10, 20, 30)).save(image, format="JPEG")
        payload = {
            "encode_mode": "jpeg_base64",
            "instruction": "pull the curtain",
            "observation": {
                "ego_view": base64.b64encode(image.getvalue()).decode(),
                "state": [0.0] * 43,
            },
        }
        policy = Mock()
        policy.infer.return_value = {"actions": np.zeros((4, 76))}
        with TestClient(create_app(policy, 4)) as client:
            self.assertEqual(client.get("/health").json()["state_dim"], 43)
            response = client.post("/act", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(np.asarray(response.json()["actions"]).shape, (4, 76))
            self.assertEqual(policy.infer.call_args.args[0]["image"].shape, (12, 16, 3))
            payload["observation"]["state"] = [0.0] * 41
            self.assertEqual(client.post("/act", json=payload).status_code, 400)
            self.assertEqual(policy.infer.call_count, 1)
            payload["observation"]["state"] = [0.0] * 43
            policy.infer.return_value = {"actions": np.full((4, 76), np.nan)}
            self.assertEqual(client.post("/act", json=payload).status_code, 500)

    def test_checkpoint_contains_serving_contract(self):
        trainer = load_script("train_pytorch")
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            config = dataclasses.replace(
                make_config("test", action_horizon=7),
                exp_name="test",
                checkpoint_base_dir=tmp,
                save_interval=1,
            )
            data = Mock(norm_stats=None)
            trainer.save_checkpoint(model, optimizer, 1, config, True, data)
            checkpoint = config.checkpoint_dir / "1"
            metadata = json.loads((checkpoint / "policy.json").read_text())
            self.assertEqual(metadata["action_horizon"], 7)
            self.assertEqual(metadata["action_dim"], 76)
            self.assertTrue((checkpoint / "model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
