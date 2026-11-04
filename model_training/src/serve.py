"""Serve a trained policy through the SONIC OpenPI HTTP interface."""

import argparse
import base64
import binascii
import io
import json
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError


def create_app(policy, action_horizon):
    app = FastAPI()
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "config": "eh2",
            "state_dim": 43,
            "action_dim": 76,
            "requires_state": True,
            "action_horizon": action_horizon,
        }

    @app.post("/act")
    def act(payload: dict):
        try:
            if payload.get("encode_mode") != "jpeg_base64":
                raise ValueError("encode_mode must be jpeg_base64")
            observation = payload["observation"]
            state = np.asarray(observation["state"], dtype=np.float32)
            if state.shape != (43,) or not np.isfinite(state).all():
                raise ValueError("Expected a finite 43-dimensional state")
            prompt = payload["instruction"]
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
                raise ValueError("instruction must contain 1–4096 characters")
            encoded_image = observation["ego_view"]
            if (
                not isinstance(encoded_image, str)
                or len(encoded_image) > 16 * 1024 * 1024
            ):
                raise ValueError("Encoded image exceeds 16 MiB or is not a string")
            encoded = base64.b64decode(encoded_image, validate=True)
            with Image.open(io.BytesIO(encoded)) as source:
                if source.width * source.height > 16_000_000:
                    raise ValueError("Image exceeds 16 megapixels")
                image = np.asarray(source.convert("RGB"))
        except (
            KeyError,
            TypeError,
            ValueError,
            binascii.Error,
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
        ) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with lock:
            result = policy.infer({"image": image, "state": state, "prompt": prompt})
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (action_horizon, 76) or not np.isfinite(actions).all():
            raise HTTPException(
                status_code=500, detail="Policy returned invalid actions"
            )
        return {"actions": actions.tolist()}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    metadata_path = args.checkpoint / "policy.json"
    if not metadata_path.is_file():
        parser.error("checkpoint must contain policy.json saved by eh2-train")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("state_dim") != 43 or metadata.get("action_dim") != 76:
        parser.error("checkpoint does not match the 43-state / 76-action interface")
    import uvicorn
    from eh2_policy.config import make_config
    from openpi.policies.policy_config import create_trained_policy

    config = make_config(
        "unused", action_horizon=metadata["action_horizon"], image_augmentation=False
    )
    policy = create_trained_policy(
        config, args.checkpoint.resolve(), pytorch_device=args.device
    )
    uvicorn.run(
        create_app(policy, config.model.action_horizon), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
