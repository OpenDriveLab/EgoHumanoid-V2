"""Fetch pinned model files and verify their checksums."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(item, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != item["sha256"]:
            raise ValueError(f"Existing model checksum differs: {target}")
        return
    if "repo" in item:
        downloaded = Path(
            hf_hub_download(
                item["repo"],
                item["filename"],
                revision=item["revision"],
                local_dir=target.parent / ".download",
            )
        )
    else:
        downloaded = target.with_suffix(target.suffix + ".partial")
        urllib.request.urlretrieve(item["url"], downloaded)
    if sha256(downloaded) != item["sha256"]:
        raise ValueError(f"Downloaded model checksum differs: {target.name}")
    shutil.move(downloaded, target)
    print(f"Verified {target.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("models.json").read_text())
    fetch(manifest["sam_checkpoint"], args.runtime / "models/sam3.1_multiplex.pt")
    for item in manifest["inpaint_weights"]:
        fetch(item, args.runtime / "diffueraser/weights" / item["path"])


if __name__ == "__main__":
    main()
