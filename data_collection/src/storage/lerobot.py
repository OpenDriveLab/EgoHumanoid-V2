from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from human.devices.gopro import GOPRO_VIDEO_KEY, GOPRO_VIDEO_PATH
from human.devices.pico import PicoSample


PICO_LEROBOT_FEATURES = {
    "observation.pico.body_pose": {
        "dtype": "float32",
        "shape": (24, 7),
        "names": ["body_joint", "xyz_qxyzw"],
    },
    "observation.pico.body_available": {
        "dtype": "bool",
        "shape": (1,),
        "names": ["available"],
    },
    "observation.pico.body_timestamp_ns": {
        "dtype": "int64",
        "shape": (1,),
        "names": ["time_ns"],
    },
    "observation.pico.body_joint_timestamp_ns": {
        "dtype": "int64",
        "shape": (24,),
        "names": ["body_joint"],
    },
    "observation.pico.body_velocity": {
        "dtype": "float32",
        "shape": (24, 6),
        "names": ["body_joint", "linear_angular_velocity"],
    },
    "observation.pico.local_timestamp_ns": {
        "dtype": "int64",
        "shape": (1,),
        "names": ["time_ns"],
    },
    "observation.pico.left_controller_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    },
    "observation.pico.right_controller_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    },
    "observation.pico.left_hand_pose": {
        "dtype": "float32",
        "shape": (26, 7),
        "names": ["hand_joint", "xyz_qxyzw"],
    },
    "observation.pico.right_hand_pose": {
        "dtype": "float32",
        "shape": (26, 7),
        "names": ["hand_joint", "xyz_qxyzw"],
    },
    "observation.pico.left_hand_active": {
        "dtype": "bool",
        "shape": (1,),
        "names": ["active"],
    },
    "observation.pico.right_hand_active": {
        "dtype": "bool",
        "shape": (1,),
        "names": ["active"],
    },
}


DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


@dataclass(frozen=True)
class GoProVideoInfo:
    width: int
    height: int
    fps: float
    codec: str = "mp4v"


@dataclass
class _DatasetMeta:
    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0
    next_episode_index: int = 0


class SimpleLeRobotDataset:
    """Small LeRobot v2.1 on-disk writer for Pico observations plus GoPro video."""

    chunks_size = 1000
    data_path = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

    def __init__(
        self,
        *,
        root: Path,
        repo_id: str,
        fps: int,
        gopro_video: GoProVideoInfo | None,
        append: bool = False,
    ):
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = int(fps)
        self.gopro_video = gopro_video
        self.features = {**PICO_LEROBOT_FEATURES, **DEFAULT_FEATURES}
        if gopro_video is not None:
            self.features.update(_gopro_video_features(gopro_video))
        self.meta = _DatasetMeta()
        self._task_to_index: dict[str, int] = {}

        if append and (self.root / "meta" / "info.json").exists():
            self._load_existing()
            self._validate_existing_features()
            self._write_info()
            return

        (self.root / "meta").mkdir(parents=True, exist_ok=False)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        self._write_info()

    def save_episode(self, samples: Iterable[PicoSample], *, task: str) -> None:
        samples = list(samples)
        if not samples:
            raise ValueError("cannot save an empty episode")

        episode_index = self.meta.next_episode_index
        task_index = self._task_index(task)
        rows = []
        for frame_index, sample in enumerate(samples):
            row = _frame_to_row(sample.to_lerobot_frame(task))
            row.update(
                {
                    "timestamp": np.float32(frame_index / self.fps).item(),
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": self.meta.total_frames + frame_index,
                    "task_index": task_index,
                }
            )
            rows.append(row)

        episode_chunk = episode_index // self.chunks_size
        parquet_path = self.root / self.data_path.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), parquet_path)

        _append_jsonl(
            self.root / "meta" / "episodes.jsonl",
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": len(rows),
                "task_index": task_index,
            },
        )
        _append_jsonl(
            self.root / "meta" / "episodes_stats.jsonl",
            {
                "episode_index": episode_index,
                "stats": _episode_stats(rows, self.features),
            },
        )

        self.meta.total_episodes += 1
        self.meta.total_frames += len(rows)
        self.meta.next_episode_index = episode_index + 1
        self._write_info()

    @property
    def next_episode_index(self) -> int:
        return self.meta.next_episode_index

    def _task_index(self, task: str) -> int:
        if task in self._task_to_index:
            return self._task_to_index[task]
        task_index = len(self._task_to_index)
        self._task_to_index[task] = task_index
        self.meta.total_tasks = len(self._task_to_index)
        _append_jsonl(
            self.root / "meta" / "tasks.jsonl",
            {
                "task_index": task_index,
                "task": task,
            },
        )
        return task_index

    def _write_info(self) -> None:
        total_chunks = (
            (self.meta.total_episodes + self.chunks_size - 1) // self.chunks_size
            if self.meta.total_episodes
            else 0
        )
        info = {
            "codebase_version": "v2.1",
            "robot_type": None,
            "total_episodes": self.meta.total_episodes,
            "total_frames": self.meta.total_frames,
            "total_tasks": self.meta.total_tasks,
            "total_videos": self.meta.total_episodes if self.gopro_video is not None else 0,
            "total_chunks": total_chunks,
            "chunks_size": self.chunks_size,
            "fps": self.fps,
            "splits": {"train": f"0:{self.meta.total_episodes}"},
            "data_path": self.data_path,
            "video_path": GOPRO_VIDEO_PATH if self.gopro_video is not None else None,
            "features": _json_features(self.features),
        }
        _write_json(self.root / "meta" / "info.json", info)

    def _load_existing(self) -> None:
        info_path = self.root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        next_episode_index = _next_episode_index(self.root)
        self.meta = _DatasetMeta(
            total_episodes=max(int(info.get("total_episodes", 0)), next_episode_index),
            total_frames=int(info.get("total_frames", 0)),
            total_tasks=int(info.get("total_tasks", 0)),
            next_episode_index=next_episode_index,
        )
        self._task_to_index = _load_tasks(self.root / "meta" / "tasks.jsonl")
        self.meta.total_tasks = max(self.meta.total_tasks, len(self._task_to_index))

    def _validate_existing_features(self) -> None:
        info_path = self.root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        existing_features = info.get("features", {})
        dataset_is_empty = self.meta.total_episodes == 0 and self.meta.total_frames == 0
        required_features = {**PICO_LEROBOT_FEATURES, **DEFAULT_FEATURES}
        missing_features = sorted(set(required_features) - set(existing_features))
        if missing_features and not dataset_is_empty:
            raise ValueError(
                f"{self.root} LeRobot schema is missing required fields: "
                f"{', '.join(missing_features)}; choose a new --task-name or migrate the dataset"
            )
        for key, expected in required_features.items():
            existing = existing_features.get(key)
            if existing is None:
                continue
            expected_shape = list(expected["shape"])
            if existing.get("dtype") != expected["dtype"] or existing.get("shape") != expected_shape:
                raise ValueError(
                    f"{self.root} LeRobot schema for {key} does not match this collector; "
                    "choose a new --task-name or migrate the dataset"
                )

        existing_has_gopro = GOPRO_VIDEO_KEY in existing_features
        requested_has_gopro = self.gopro_video is not None
        if existing_has_gopro != requested_has_gopro:
            if dataset_is_empty:
                return
            raise ValueError(
                f"{self.root} GoPro video setting does not match this run; "
                "use the same GoPro setting or choose a new --task-name"
            )
        if not existing_has_gopro:
            return

        existing = existing_features[GOPRO_VIDEO_KEY]
        expected = _gopro_video_features(self.gopro_video)[GOPRO_VIDEO_KEY]
        if existing.get("shape") != list(expected["shape"]):
            raise ValueError(
                f"{self.root} GoPro video shape does not match this run; "
                "use the same GoPro resolution or choose a new --task-name"
            )


def create_dataset(
    *,
    root: str | Path,
    repo_id: str,
    fps: int,
    gopro_video: GoProVideoInfo | None = None,
    append: bool = False,
) -> SimpleLeRobotDataset:
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        if append:
            return SimpleLeRobotDataset(
                root=root,
                repo_id=repo_id,
                fps=int(fps),
                gopro_video=gopro_video,
                append=True,
            )
        raise FileExistsError(f"{root} already contains a LeRobot dataset; use a new --task-name")

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} exists and is not an empty LeRobot dataset")

    return SimpleLeRobotDataset(
        root=root,
        repo_id=repo_id,
        fps=int(fps),
        gopro_video=gopro_video,
        append=False,
    )


def save_samples_as_episode(
    dataset: SimpleLeRobotDataset,
    samples: Iterable[PicoSample],
    *,
    task: str,
) -> None:
    dataset.save_episode(samples, task=task)


def _frame_to_row(frame: dict) -> dict:
    row = {}
    for key, value in frame.items():
        if key == "task":
            continue
        arr = np.asarray(value)
        if arr.shape == ():
            row[key] = arr.item()
        elif arr.size == 1:
            row[key] = arr.reshape(-1)[0].item()
        else:
            row[key] = arr.tolist()
    return row


def _episode_stats(rows: list[dict], features: dict) -> dict:
    stats = {}
    for key, feature in features.items():
        if feature["dtype"] == "video":
            continue
        values = np.asarray([row[key] for row in rows])
        if feature["dtype"] == "bool":
            numeric = values.astype(np.float64)
            stats[key] = {
                "min": np.min(values, axis=0).tolist(),
                "max": np.max(values, axis=0).tolist(),
                "mean": np.mean(numeric, axis=0).tolist(),
                "std": np.std(numeric, axis=0).tolist(),
                "count": [len(rows)],
            }
        else:
            numeric = values.astype(np.float64)
            stats[key] = {
                "min": np.min(values, axis=0).tolist(),
                "max": np.max(values, axis=0).tolist(),
                "mean": np.mean(numeric, axis=0).tolist(),
                "std": np.std(numeric, axis=0).tolist(),
                "count": [len(rows)],
            }
    return stats


def _json_features(features: dict) -> dict:
    out = {}
    for key, value in features.items():
        out[key] = dict(value)
        out[key]["shape"] = list(value["shape"])
    return out


def _gopro_video_features(video: GoProVideoInfo) -> dict:
    return {
        GOPRO_VIDEO_KEY: {
            "dtype": "video",
            "shape": (int(video.height), int(video.width), 3),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(video.fps),
                "video.height": int(video.height),
                "video.width": int(video.width),
                "video.channels": 3,
                "video.codec": video.codec,
                "has_audio": False,
            },
        }
    }


def _next_episode_index(root: Path) -> int:
    max_index = -1
    episodes_path = root / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                max_index = max(max_index, int(episode["episode_index"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    for parquet_path in (root / "data").glob("chunk-*/episode_*.parquet"):
        try:
            max_index = max(max_index, int(parquet_path.stem.removeprefix("episode_")))
        except ValueError:
            continue
    return max_index + 1


def _load_tasks(path: Path) -> dict[str, int]:
    tasks: dict[str, int] = {}
    if not path.exists():
        return tasks
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            tasks[str(row["task"])] = int(row["task_index"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return tasks


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(obj), ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(_json_safe(obj), indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def discover_episode_indices(root: str | Path) -> list[int]:
    root = Path(root)
    episodes_path = root / "meta" / "episodes.jsonl"
    indices: set[int] = set()
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                indices.add(int(json.loads(line)["episode_index"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    if not indices:
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            try:
                total = int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                total = 0
            indices.update(range(max(0, total)))
    if not indices:
        for parquet_path in (root / "data").glob("chunk-*/episode_*.parquet"):
            try:
                indices.add(int(parquet_path.stem.removeprefix("episode_")))
            except ValueError:
                continue
    return sorted(indices)
