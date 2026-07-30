"""Curated start configuration shared by gym-duckietown tools."""

from __future__ import annotations

from collections.abc import Sequence
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainingPose:
    tile: tuple[int, int]
    position: tuple[float, float, float]
    angle: float
    name: str | None = None
    map_name: str | None = None

    def as_json(self) -> dict[str, Any]:
        pose: dict[str, Any] = {}
        if self.map_name is not None:
            pose["map_name"] = self.map_name
        pose.update(
            {
                "tile": list(self.tile),
                "position": list(self.position),
                "angle": self.angle,
            }
        )
        if self.name is not None:
            pose["name"] = self.name
        return pose


@dataclass(frozen=True)
class StartConfig:
    source_path: Path
    training_poses: tuple[TrainingPose, ...]
    evaluation_poses: tuple[TrainingPose, ...]


@dataclass(frozen=True)
class TrainingStart:
    kind: str
    seed: int | None
    pose: TrainingPose | None = None
    map_name: str | None = None

    @property
    def name(self) -> str | None:
        return self.pose.name if self.pose is not None else None


def _parse_number(value: Any, label: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path}: {label} must be finite")
    return number


def _parse_pose(
    data: Any,
    label: str,
    path: Path,
    *,
    expected_map_names: tuple[str, ...] | None = None,
    require_map_name: bool = True,
) -> TrainingPose:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: {label} must be a JSON object")
    required_keys = {"tile", "position", "angle"}
    allowed_keys = required_keys | {"name", "map_name"}
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

    map_name = data.get("map_name")
    if map_name is not None and (not isinstance(map_name, str) or not map_name.strip()):
        raise ValueError(f"{path}: {label}.map_name must be a non-empty string")
    if map_name is None and require_map_name:
        raise ValueError(
            f"{path}: {label} must contain a non-empty 'map_name'"
        )
    parsed_map_name = map_name.strip() if isinstance(map_name, str) else None
    if expected_map_names is not None and parsed_map_name not in expected_map_names:
        raise ValueError(
            f"{path}: {label}.map_name is {parsed_map_name!r}, but configured maps are "
            f"{expected_map_names!r}"
        )

    return TrainingPose(
        tile=(tile[0], tile[1]),
        position=parsed_position,
        angle=_parse_number(data["angle"], f"{label}.angle", path),
        name=name.strip() if isinstance(name, str) else None,
        map_name=parsed_map_name,
    )


def load_pose_file(path: Path) -> TrainingPose:
    resolved_path = path.expanduser().resolve()
    try:
        data = json.loads(resolved_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in pose file {resolved_path}: {error}") from error
    return _parse_pose(
        data,
        "pose",
        resolved_path,
        require_map_name=False,
    )


def apply_env_start_pose(env, pose: TrainingPose) -> None:
    raw_env = getattr(env, "unwrapped", env)
    raw_env.user_tile_start = tuple(pose.tile)
    raw_env.start_pose = [list(pose.position), pose.angle]


def load_start_config(
    path: Path,
    expected_map_name: str | Sequence[str] | None,
    *,
    require_training_starts: bool = True,
    require_evaluation_scenarios: bool = True,
) -> StartConfig:
    resolved_path = path.expanduser().resolve()
    try:
        data = json.loads(resolved_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in start config {resolved_path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{resolved_path}: top-level JSON value must be an object")

    required_keys = {"training_poses", "evaluation_poses"}
    allowed_keys = required_keys
    missing_keys = required_keys - data.keys()
    unexpected_keys = data.keys() - allowed_keys
    if missing_keys:
        raise ValueError(
            f"{resolved_path}: missing keys: {', '.join(sorted(missing_keys))}"
        )
    if unexpected_keys:
        raise ValueError(
            f"{resolved_path}: unexpected keys: {', '.join(sorted(unexpected_keys))}; "
            "start configs support map-specific training_poses and evaluation_poses only"
        )

    expected_map_names = None
    if expected_map_name is not None:
        expected_map_names = (
            (expected_map_name,)
            if isinstance(expected_map_name, str)
            else tuple(expected_map_name)
        )

    training_poses_data = data["training_poses"]
    if not isinstance(training_poses_data, list):
        raise ValueError(f"{resolved_path}: 'training_poses' must be a JSON list")
    training_poses = tuple(
        _parse_pose(
            pose,
            f"training_poses[{index}]",
            resolved_path,
            expected_map_names=expected_map_names,
        )
        for index, pose in enumerate(training_poses_data)
    )
    evaluation_poses_data = data["evaluation_poses"]
    if not isinstance(evaluation_poses_data, list):
        raise ValueError(f"{resolved_path}: 'evaluation_poses' must be a JSON list")
    evaluation_poses = tuple(
        _parse_pose(
            pose,
            f"evaluation_poses[{index}]",
            resolved_path,
            expected_map_names=expected_map_names,
        )
        for index, pose in enumerate(evaluation_poses_data)
    )
    if require_training_starts and not training_poses:
        raise ValueError(
            f"{resolved_path}: configure at least one training pose"
        )
    if require_evaluation_scenarios and not evaluation_poses:
        raise ValueError(
            f"{resolved_path}: configure at least one evaluation pose"
        )
    return StartConfig(
        source_path=resolved_path,
        training_poses=training_poses,
        evaluation_poses=evaluation_poses,
    )


def write_start_config(config: StartConfig) -> None:
    payload = {
        "training_poses": [pose.as_json() for pose in config.training_poses],
        "evaluation_poses": [pose.as_json() for pose in config.evaluation_poses],
    }
    config.source_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config.source_path.with_name(f".{config.source_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(config.source_path)


def append_pose(
    path: Path,
    map_name: str,
    pose: TrainingPose,
    collection: str,
) -> int:
    resolved_path = path.expanduser().resolve()
    if resolved_path.exists():
        config = load_start_config(
            resolved_path,
            None,
            require_training_starts=False,
            require_evaluation_scenarios=False,
        )
    else:
        config = StartConfig(
            source_path=resolved_path,
            training_poses=(),
            evaluation_poses=(),
        )

    captured_pose = replace(pose, map_name=map_name)
    if collection == "training_poses":
        training_poses = (*config.training_poses, captured_pose)
        evaluation_poses = config.evaluation_poses
        pose_count = len(training_poses)
    elif collection == "evaluation_poses":
        training_poses = config.training_poses
        evaluation_poses = (*config.evaluation_poses, captured_pose)
        pose_count = len(evaluation_poses)
    else:
        raise ValueError(f"unknown pose collection {collection!r}")

    updated_config = StartConfig(
        source_path=config.source_path,
        training_poses=training_poses,
        evaluation_poses=evaluation_poses,
    )
    write_start_config(updated_config)
    return pose_count


def append_training_pose(path: Path, map_name: str, pose: TrainingPose) -> int:
    return append_pose(path, map_name, pose, "training_poses")


def append_evaluation_pose(path: Path, map_name: str, pose: TrainingPose) -> int:
    return append_pose(path, map_name, pose, "evaluation_poses")


def choose_training_start(
    config: StartConfig | None,
    hard_start_probability: float,
    rng: np.random.Generator,
) -> TrainingStart:
    if config is None:
        return TrainingStart(kind="random", seed=None)

    if rng.random() < hard_start_probability:
        pose = config.training_poses[int(rng.integers(0, len(config.training_poses)))]
        if pose.map_name is None:
            raise RuntimeError("Training pose has no associated map")
        return TrainingStart(
            kind="hard_pose",
            seed=_draw_random_reset_seed(rng),
            pose=pose,
            map_name=pose.map_name,
        )

    return TrainingStart(
        kind="random",
        seed=_draw_random_reset_seed(rng),
    )


def _draw_random_reset_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max))
