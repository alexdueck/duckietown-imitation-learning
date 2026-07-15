"""Shared curated start configuration for gym-duckietown tools."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainingPose:
    tile: tuple[int, int]
    position: tuple[float, float, float]
    angle: float
    name: str | None = None

    def as_json(self) -> dict[str, Any]:
        pose = {
            "tile": list(self.tile),
            "position": list(self.position),
            "angle": self.angle,
        }
        if self.name is not None:
            pose["name"] = self.name
        return pose


@dataclass(frozen=True)
class StartConfig:
    source_path: Path
    map_name: str
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    training_poses: tuple[TrainingPose, ...]
    evaluation_poses: tuple[TrainingPose, ...]


@dataclass(frozen=True)
class TrainingStart:
    kind: str
    seed: int | None
    pose: TrainingPose | None = None

    @property
    def name(self) -> str | None:
        return self.pose.name if self.pose is not None else None


def _parse_seed_list(
    data: Any,
    key: str,
    path: Path,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(data, list):
        raise ValueError(f"{path}: {key!r} must be a JSON list")
    if not data and not allow_empty:
        raise ValueError(f"{path}: {key!r} must contain at least one seed")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in data):
        raise ValueError(f"{path}: {key!r} must contain non-negative integers only")
    seeds = tuple(data)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: {key!r} must not contain duplicate seeds")
    return seeds


def _parse_number(value: Any, label: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path}: {label} must be finite")
    return number


def _parse_pose(data: Any, label: str, path: Path) -> TrainingPose:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: {label} must be a JSON object")
    required_keys = {"tile", "position", "angle"}
    allowed_keys = required_keys | {"name"}
    missing_keys = required_keys - data.keys()
    unexpected_keys = data.keys() - allowed_keys
    if missing_keys:
        raise ValueError(f"{path}: {label} is missing keys: {', '.join(sorted(missing_keys))}")
    if unexpected_keys:
        raise ValueError(f"{path}: {label} has unexpected keys: {', '.join(sorted(unexpected_keys))}")

    tile = data["tile"]
    if (
        not isinstance(tile, list)
        or len(tile) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in tile)
    ):
        raise ValueError(f"{path}: {label}.tile must contain two non-negative integers")

    position = data["position"]
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError(f"{path}: {label}.position must contain three numbers")
    parsed_position = tuple(
        _parse_number(value, f"{label}.position[{position_index}]", path)
        for position_index, value in enumerate(position)
    )

    name = data.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"{path}: {label}.name must be a non-empty string or null")

    return TrainingPose(
        tile=(tile[0], tile[1]),
        position=parsed_position,
        angle=_parse_number(data["angle"], f"{label}.angle", path),
        name=name.strip() if isinstance(name, str) else None,
    )


def load_pose_file(path: Path) -> TrainingPose:
    resolved_path = path.expanduser().resolve()
    try:
        data = json.loads(resolved_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in pose file {resolved_path}: {error}") from error
    return _parse_pose(data, "pose", resolved_path)


def apply_env_start_pose(env, pose: TrainingPose) -> None:
    raw_env = getattr(env, "unwrapped", env)
    raw_env.user_tile_start = tuple(pose.tile)
    raw_env.start_pose = [list(pose.position), pose.angle]


def load_start_config(path: Path, expected_map_name: str) -> StartConfig:
    resolved_path = path.expanduser().resolve()
    try:
        data = json.loads(resolved_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in start config {resolved_path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{resolved_path}: top-level JSON value must be an object")

    required_keys = {"map_name", "training_seeds", "evaluation_seeds"}
    allowed_keys = required_keys | {"training_poses", "evaluation_poses"}
    missing_keys = required_keys - data.keys()
    unexpected_keys = data.keys() - allowed_keys
    if missing_keys:
        raise ValueError(f"{resolved_path}: missing keys: {', '.join(sorted(missing_keys))}")
    if unexpected_keys:
        raise ValueError(f"{resolved_path}: unexpected keys: {', '.join(sorted(unexpected_keys))}")

    map_name = data["map_name"]
    if not isinstance(map_name, str) or not map_name:
        raise ValueError(f"{resolved_path}: 'map_name' must be a non-empty string")
    if map_name != expected_map_name:
        raise ValueError(
            f"{resolved_path}: map_name is {map_name!r}, but the environment uses {expected_map_name!r}"
        )

    training_seeds = _parse_seed_list(
        data["training_seeds"],
        "training_seeds",
        resolved_path,
        allow_empty=True,
    )
    evaluation_seeds = _parse_seed_list(
        data["evaluation_seeds"],
        "evaluation_seeds",
        resolved_path,
        allow_empty=True,
    )
    overlap = set(training_seeds) & set(evaluation_seeds)
    if overlap:
        raise ValueError(
            f"{resolved_path}: training_seeds and evaluation_seeds overlap: "
            f"{', '.join(str(seed) for seed in sorted(overlap))}"
        )

    training_poses_data = data.get("training_poses", [])
    if not isinstance(training_poses_data, list):
        raise ValueError(f"{resolved_path}: 'training_poses' must be a JSON list")
    training_poses = tuple(
        _parse_pose(pose, f"training_poses[{index}]", resolved_path)
        for index, pose in enumerate(training_poses_data)
    )
    evaluation_poses_data = data.get("evaluation_poses", [])
    if not isinstance(evaluation_poses_data, list):
        raise ValueError(f"{resolved_path}: 'evaluation_poses' must be a JSON list")
    evaluation_poses = tuple(
        _parse_pose(pose, f"evaluation_poses[{index}]", resolved_path)
        for index, pose in enumerate(evaluation_poses_data)
    )
    if not training_seeds and not training_poses:
        raise ValueError(
            f"{resolved_path}: configure at least one training seed or training pose"
        )
    if not evaluation_seeds and not evaluation_poses:
        raise ValueError(
            f"{resolved_path}: configure at least one evaluation seed or evaluation pose"
        )
    return StartConfig(
        source_path=resolved_path,
        map_name=map_name,
        training_seeds=training_seeds,
        evaluation_seeds=evaluation_seeds,
        training_poses=training_poses,
        evaluation_poses=evaluation_poses,
    )


def write_start_config(config: StartConfig) -> None:
    payload = {
        "map_name": config.map_name,
        "training_seeds": list(config.training_seeds),
        "evaluation_seeds": list(config.evaluation_seeds),
        "training_poses": [pose.as_json() for pose in config.training_poses],
        "evaluation_poses": [pose.as_json() for pose in config.evaluation_poses],
    }
    temporary_path = config.source_path.with_name(f".{config.source_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(config.source_path)


def append_training_pose(path: Path, expected_map_name: str, pose: TrainingPose) -> int:
    config = load_start_config(path, expected_map_name)
    updated_config = StartConfig(
        source_path=config.source_path,
        map_name=config.map_name,
        training_seeds=config.training_seeds,
        evaluation_seeds=config.evaluation_seeds,
        training_poses=(*config.training_poses, pose),
        evaluation_poses=config.evaluation_poses,
    )
    write_start_config(updated_config)
    return len(updated_config.training_poses)


def choose_training_start(
    config: StartConfig | None,
    hard_start_probability: float,
    rng: np.random.Generator,
) -> TrainingStart:
    if config is None:
        return TrainingStart(kind="random", seed=None)

    hard_start_count = len(config.training_seeds) + len(config.training_poses)
    if rng.random() < hard_start_probability:
        hard_start_index = int(rng.integers(0, hard_start_count))
        if hard_start_index < len(config.training_seeds):
            return TrainingStart(kind="hard_seed", seed=config.training_seeds[hard_start_index])
        pose = config.training_poses[hard_start_index - len(config.training_seeds)]
        return TrainingStart(
            kind="hard_pose",
            seed=_draw_random_reset_seed(config, rng),
            pose=pose,
        )

    return TrainingStart(kind="random", seed=_draw_random_reset_seed(config, rng))


def _draw_random_reset_seed(config: StartConfig, rng: np.random.Generator) -> int:
    reserved_seeds = set(config.training_seeds) | set(config.evaluation_seeds)
    while True:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        if seed not in reserved_seeds:
            return seed
