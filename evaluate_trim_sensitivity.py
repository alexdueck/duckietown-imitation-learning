#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Evaluate gym-duckietown PPO policies over fixed poses and trim values."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import webbrowser
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from dt_utils.cli_completion import parse_args_with_completion
from dt_utils.duckietown_action_control import (
    DuckietownActionControl,
    action_control_from_config,
)
from dt_utils.duckietown_paths import EVALUATION_DATA_DIR
from dt_utils.duckietown_rewards import (
    GymDuckietownRewardCalculator,
    REWARD_FUNCTION_CHOICES,
    format_wheel_action,
    get_lane_metrics,
)
from dt_utils.gym_duckietown_start_config import (
    TrainingPose,
    apply_env_start_pose,
    load_start_config,
)
from dt_utils.temporal_rl import FixedHistory, TemporalTanhGaussianPolicy
from train_imitation_learning import resolve_device, set_seed
from train_rl_ppo_gym_duckietown import (
    configure_gym_duckietown_logging,
    done_reason,
    flatten_reward_breakdown,
    load_environment_map,
    make_env,
    make_transform,
    preprocess,
    reset_raw,
    step_raw,
)


HISTORY_MODES = (
    "native",
    "zero_action_history",
    "current_frame_zero_actions",
    "reversed_action_history",
)
POSE_SOURCES = ("evaluation", "training")
DEFAULT_OUTPUT_DIR = EVALUATION_DATA_DIR / "trim_sensitivity"
CONFIG_KEYS = {
    "checkpoints",
    "trims",
    "start_config",
    "pose_sources",
    "pose_names",
    "history_modes",
    "output_dir",
    "max_steps",
    "repeats",
    "seed",
    "device",
    "reward_function",
    "vd2pp_distance_weight",
    "domain_rand",
    "dynamics_rand",
    "camera_rand",
    "distortion",
    "frame_rate",
    "frame_skip",
    "camera_width",
    "camera_height",
    "robot_speed",
    "accept_start_angle_deg",
    "stochastic",
    "record_step_history",
    "open_report",
    "overwrite",
    "log_level",
}
OUTPUT_FILENAMES = (
    "config.json",
    "episodes.csv",
    "scenario_summary.csv",
    "summary.csv",
    "steps.csv",
    "trim_sensitivity_report.html",
)
COLORS = (
    "#147d92",
    "#d1495b",
    "#2d6a4f",
    "#e09f3e",
    "#6d597a",
    "#577590",
    "#8f2d56",
    "#3a86ff",
)


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class Scenario:
    index: int
    source: str
    name: str
    pose: TrainingPose


@dataclass(frozen=True)
class TrimSensitivityConfig:
    source_path: Path
    checkpoints: tuple[CheckpointSpec, ...]
    trims: tuple[float, ...]
    start_config: Path
    pose_sources: tuple[str, ...]
    pose_names: tuple[str, ...]
    history_modes: tuple[str, ...]
    output_dir: Path
    max_steps: int
    repeats: int
    seed: int
    device: str
    reward_function: str | None
    vd2pp_distance_weight: float | None
    domain_rand: bool
    dynamics_rand: bool
    camera_rand: bool
    distortion: bool
    frame_rate: int | None
    frame_skip: int | None
    camera_width: int | None
    camera_height: int | None
    robot_speed: float | None
    accept_start_angle_deg: float | None
    stochastic: bool
    record_step_history: bool
    open_report: bool
    overwrite: bool
    log_level: str

    def as_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_path"] = str(self.source_path)
        result["start_config"] = str(self.start_config)
        result["output_dir"] = str(self.output_dir)
        result["checkpoints"] = [
            {"name": checkpoint.name, "path": str(checkpoint.path)}
            for checkpoint in self.checkpoints
        ]
        result["trims"] = list(self.trims)
        result["pose_sources"] = list(self.pose_sources)
        result["pose_names"] = list(self.pose_names)
        result["history_modes"] = list(self.history_modes)
        return result


EPISODE_FIELDS = (
    "checkpoint",
    "checkpoint_path",
    "checkpoint_step",
    "model",
    "observation_history_length",
    "action_history_length",
    "history_mode",
    "reward_function",
    "scenario_index",
    "scenario_source",
    "scenario_name",
    "map_name",
    "trim",
    "repeat",
    "reset_seed",
    "steps",
    "selected_return",
    "velopose_return",
    "env_return",
    "reward_per_step",
    "safe",
    "terminated",
    "truncated",
    "done_reason",
    "lane_valid_fraction",
    "mean_lane_distance_m",
    "mean_abs_lane_distance_m",
    "max_abs_lane_distance_m",
    "final_lane_distance_m",
    "lane_distance_slope_mps",
    "abs_lane_distance_slope_mps",
    "mean_abs_heading_error_deg",
    "max_abs_heading_error_deg",
    "mean_speed_mps",
    "mean_forward_speed_mps",
    "mean_normalized_forward_progress",
    "mean_heading_quality",
    "mean_pose_quality",
    "min_pose_quality",
    "mean_wheel_throttle",
    "mean_wheel_steering",
    "std_wheel_steering",
    "mean_abs_wheel_steering",
    "final_x",
    "final_y",
    "final_z",
    "final_angle",
)

STEP_FIELDS = (
    "checkpoint",
    "checkpoint_step",
    "history_mode",
    "reward_function",
    "scenario_index",
    "scenario_source",
    "scenario_name",
    "map_name",
    "trim",
    "repeat",
    "reset_seed",
    "step",
    "timestamp",
    "policy_control_0_name",
    "policy_control_0",
    "policy_control_1_name",
    "policy_control_1",
    "policy_std_0",
    "policy_std_1",
    "wheel_left",
    "wheel_right",
    "wheel_throttle",
    "wheel_steering",
    "selected_reward",
    "velopose_reward",
    "env_reward",
    "selected_return",
    "velopose_return",
    "env_return",
    "lane_valid",
    "lane_distance_m",
    "lane_dot_dir",
    "lane_angle_deg",
    "speed_mps",
    "forward_speed_mps",
    "normalized_forward_progress",
    "heading_quality",
    "target_heading_offset_deg",
    "signed_scaled_lane_distance",
    "scaled_abs_lane_distance",
    "pose_quality",
    "x",
    "y",
    "z",
    "angle",
    "terminated",
    "truncated",
    "done_reason",
    "selected_reward_components_json",
)

SUMMARY_METRICS = (
    "selected_return",
    "velopose_return",
    "reward_per_step",
    "steps",
    "lane_valid_fraction",
    "mean_abs_lane_distance_m",
    "max_abs_lane_distance_m",
    "lane_distance_slope_mps",
    "abs_lane_distance_slope_mps",
    "mean_abs_heading_error_deg",
    "mean_speed_mps",
    "mean_forward_speed_mps",
    "mean_heading_quality",
    "mean_pose_quality",
    "mean_wheel_steering",
    "std_wheel_steering",
    "mean_abs_wheel_steering",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO checkpoint sensitivity to Duckiebot trim.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Override the checkpoint list from JSON; repeat for multiple policies.",
    )
    parser.add_argument("--trim", action="append", type=float, default=None)
    parser.add_argument(
        "--history-mode",
        action="append",
        choices=HISTORY_MODES,
        default=None,
    )
    parser.add_argument(
        "--pose-source",
        action="append",
        choices=POSE_SOURCES,
        default=None,
    )
    parser.add_argument("--pose-name", action="append", default=None)
    parser.add_argument("--start-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--reward-function", choices=REWARD_FUNCTION_CHOICES, default=None)
    parser.add_argument("--vd2pp-distance-weight", type=float, default=None)
    parser.add_argument(
        "--domain-rand",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--dynamics-rand",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--camera-rand",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--distortion",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--frame-rate", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--robot-speed", type=float, default=None)
    parser.add_argument("--accept-start-angle-deg", type=float, default=None)
    parser.add_argument(
        "--stochastic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--record-step-history",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--open-report",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
    )
    return parse_args_with_completion(parser)


def _load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Trim-sensitivity config not found: {resolved}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{resolved}: top-level value must be a JSON object")
    unexpected = set(value) - CONFIG_KEYS
    if unexpected:
        raise ValueError(
            f"{resolved}: unknown options: {', '.join(sorted(unexpected))}"
        )
    return value


def _value(args: argparse.Namespace, data: dict[str, Any], name: str, default: Any) -> Any:
    cli_value = getattr(args, name, None)
    return cli_value if cli_value is not None else data.get(name, default)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_positive_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)


def _optional_positive_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _string_list(value: Any, name: str, allowed: Sequence[str] | None = None) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    result = tuple(item.strip() for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    if allowed is not None:
        invalid = [item for item in result if item not in allowed]
        if invalid:
            raise ValueError(f"{name} contains unsupported values: {invalid}")
    return result


def _checkpoint_specs(value: Any) -> tuple[CheckpointSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("checkpoints must be a non-empty JSON list")
    checkpoints = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"name", "path"}:
            raise ValueError(
                f"checkpoints[{index}] must contain exactly 'name' and 'path'"
            )
        name = item["name"]
        path = item["path"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"checkpoints[{index}].name must be a non-empty string")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"checkpoints[{index}].path must be a non-empty string")
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"Checkpoint not found: {resolved}")
        checkpoints.append(CheckpointSpec(name=name.strip(), path=resolved))
    names = [checkpoint.name for checkpoint in checkpoints]
    if len(set(names)) != len(names):
        raise ValueError("checkpoint names must be unique")
    return tuple(checkpoints)


def _cli_checkpoint_specs(values: list[str]) -> tuple[CheckpointSpec, ...]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--checkpoint expects NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        parsed.append({"name": name, "path": path})
    return _checkpoint_specs(parsed)


def load_config(args: argparse.Namespace) -> TrimSensitivityConfig:
    source_path = args.config.expanduser().resolve()
    data = _load_json(source_path)
    checkpoints = (
        _cli_checkpoint_specs(args.checkpoint)
        if args.checkpoint is not None
        else _checkpoint_specs(data.get("checkpoints"))
    )

    trims_value = args.trim if args.trim is not None else data.get("trims")
    if trims_value is None:
        raise ValueError("Configure at least one trim value")
    if not isinstance(trims_value, list):
        trims_value = list(trims_value)
    trims = tuple(sorted({_finite_number(value, "trim") for value in trims_value}))
    if not trims:
        raise ValueError("Configure at least one trim value")

    start_config_value = _value(args, data, "start_config", None)
    if not isinstance(start_config_value, (str, Path)):
        raise ValueError("start_config must be a path")
    start_config = Path(start_config_value).expanduser().resolve()
    if not start_config.is_file():
        raise ValueError(f"Start config not found: {start_config}")

    pose_sources_value = (
        args.pose_source
        if args.pose_source is not None
        else data.get("pose_sources", ["evaluation"])
    )
    pose_sources = _string_list(pose_sources_value, "pose_sources", POSE_SOURCES)
    pose_names_value = (
        args.pose_name if args.pose_name is not None else data.get("pose_names", [])
    )
    if not isinstance(pose_names_value, list) or not all(
        isinstance(item, str) and item.strip() for item in pose_names_value
    ):
        raise ValueError("pose_names must be a JSON list of non-empty strings")
    pose_names = tuple(item.strip() for item in pose_names_value)

    history_modes_value = (
        args.history_mode
        if args.history_mode is not None
        else data.get("history_modes", ["native"])
    )
    history_modes = _string_list(history_modes_value, "history_modes", HISTORY_MODES)

    output_value = _value(args, data, "output_dir", str(DEFAULT_OUTPUT_DIR))
    if not isinstance(output_value, (str, Path)):
        raise ValueError("output_dir must be a path")

    reward_function = _value(args, data, "reward_function", None)
    if reward_function is not None and reward_function not in REWARD_FUNCTION_CHOICES:
        raise ValueError(f"Unsupported reward_function {reward_function!r}")
    vd2pp_distance_weight = _value(args, data, "vd2pp_distance_weight", None)
    if vd2pp_distance_weight is not None:
        vd2pp_distance_weight = _finite_number(
            vd2pp_distance_weight,
            "vd2pp_distance_weight",
        )
        if vd2pp_distance_weight < 0.0:
            raise ValueError("vd2pp_distance_weight must be non-negative")

    device = _value(args, data, "device", "auto")
    if device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported device {device!r}")
    log_level = _value(args, data, "log_level", "WARNING")
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ValueError(f"Unsupported log_level {log_level!r}")

    config = TrimSensitivityConfig(
        source_path=source_path,
        checkpoints=checkpoints,
        trims=trims,
        start_config=start_config,
        pose_sources=pose_sources,
        pose_names=pose_names,
        history_modes=history_modes,
        output_dir=Path(output_value).expanduser().resolve(),
        max_steps=_positive_integer(_value(args, data, "max_steps", 500), "max_steps"),
        repeats=_positive_integer(_value(args, data, "repeats", 1), "repeats"),
        seed=_integer(_value(args, data, "seed", 42), "seed"),
        device=device,
        reward_function=reward_function,
        vd2pp_distance_weight=vd2pp_distance_weight,
        domain_rand=_boolean(_value(args, data, "domain_rand", False), "domain_rand"),
        dynamics_rand=_boolean(
            _value(args, data, "dynamics_rand", False),
            "dynamics_rand",
        ),
        camera_rand=_boolean(_value(args, data, "camera_rand", False), "camera_rand"),
        distortion=_boolean(_value(args, data, "distortion", False), "distortion"),
        frame_rate=_optional_positive_integer(
            _value(args, data, "frame_rate", None),
            "frame_rate",
        ),
        frame_skip=_optional_positive_integer(
            _value(args, data, "frame_skip", None),
            "frame_skip",
        ),
        camera_width=_optional_positive_integer(
            _value(args, data, "camera_width", None),
            "camera_width",
        ),
        camera_height=_optional_positive_integer(
            _value(args, data, "camera_height", None),
            "camera_height",
        ),
        robot_speed=_optional_positive_number(
            _value(args, data, "robot_speed", None),
            "robot_speed",
        ),
        accept_start_angle_deg=_optional_positive_number(
            _value(args, data, "accept_start_angle_deg", None),
            "accept_start_angle_deg",
        ),
        stochastic=_boolean(_value(args, data, "stochastic", False), "stochastic"),
        record_step_history=_boolean(
            _value(args, data, "record_step_history", True),
            "record_step_history",
        ),
        open_report=_boolean(
            _value(args, data, "open_report", False),
            "open_report",
        ),
        overwrite=_boolean(_value(args, data, "overwrite", False), "overwrite"),
        log_level=log_level,
    )
    if config.camera_rand and not config.distortion:
        raise ValueError("camera_rand requires distortion")
    if config.dynamics_rand:
        raise ValueError(
            "dynamics_rand must be false: this evaluator applies each requested "
            "trim explicitly after reset"
        )
    return config


def load_policy(
    checkpoint: CheckpointSpec,
    device: torch.device,
) -> tuple[TemporalTanhGaussianPolicy, DuckietownActionControl, dict[str, Any], int]:
    payload = torch.load(checkpoint.path, map_location=device)
    if "policy_state_dict" not in payload:
        raise ValueError(f"{checkpoint.path} has no policy_state_dict")
    if payload.get("env_backend") not in (None, "gym-duckietown"):
        raise ValueError(f"{checkpoint.path} is not a gym-duckietown checkpoint")
    checkpoint_config = payload.get("config", {})
    action_control = action_control_from_config(checkpoint_config)
    observation_history_length = int(
        checkpoint_config.get("observation_history_length", 1)
    )
    action_history_length = int(
        checkpoint_config.get(
            "action_history_length",
            observation_history_length - 1,
        )
    )
    policy = TemporalTanhGaussianPolicy(
        checkpoint_config.get("model", "mobilenet_v3_small"),
        action_dim=action_control.policy_action_dim,
        pretrained=False,
        observation_history_length=observation_history_length,
        action_history_length=action_history_length,
        temporal_hidden_dim=int(checkpoint_config.get("temporal_hidden_dim", 256)),
        temporal_head_mode=checkpoint_config.get("temporal_head_mode", "residual"),
    )
    policy.load_state_dict(payload["policy_state_dict"])
    policy.to(device)
    policy.eval()
    return policy, action_control, checkpoint_config, int(payload.get("step", 0))


def checkpoint_value(
    override: Any,
    checkpoint_config: dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    return checkpoint_config.get(name, default) if override is None else override


def environment_args(
    config: TrimSensitivityConfig,
    checkpoint_config: dict[str, Any],
    map_name: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        seed=config.seed,
        map_name=map_name,
        simulator_max_steps=config.max_steps,
        max_episode_steps=config.max_steps,
        robot_speed=checkpoint_value(
            config.robot_speed,
            checkpoint_config,
            "robot_speed",
            None,
        ),
        domain_rand=config.domain_rand,
        dynamics_rand=False,
        camera_rand=config.camera_rand,
        distortion=config.distortion,
        frame_rate=int(
            checkpoint_value(config.frame_rate, checkpoint_config, "frame_rate", 30)
        ),
        frame_skip=int(
            checkpoint_value(config.frame_skip, checkpoint_config, "frame_skip", 1)
        ),
        camera_width=int(
            checkpoint_value(
                config.camera_width,
                checkpoint_config,
                "camera_width",
                640,
            )
        ),
        camera_height=int(
            checkpoint_value(
                config.camera_height,
                checkpoint_config,
                "camera_height",
                480,
            )
        ),
        accept_start_angle_deg=float(
            checkpoint_value(
                config.accept_start_angle_deg,
                checkpoint_config,
                "accept_start_angle_deg",
                4.0,
            )
        ),
    )


def selected_scenarios(config: TrimSensitivityConfig) -> tuple[Scenario, ...]:
    starts = load_start_config(
        config.start_config,
        None,
        require_training_starts=False,
        require_evaluation_scenarios=False,
    )
    candidates: list[tuple[str, TrainingPose]] = []
    if "evaluation" in config.pose_sources:
        candidates.extend(("evaluation", pose) for pose in starts.evaluation_poses)
    if "training" in config.pose_sources:
        candidates.extend(("training", pose) for pose in starts.training_poses)
    if config.pose_names:
        wanted = set(config.pose_names)
        candidates = [item for item in candidates if item[1].name in wanted]
        found = {pose.name for _, pose in candidates}
        missing = wanted - found
        if missing:
            raise ValueError(
                "Requested pose names were not found in selected sources: "
                + ", ".join(sorted(missing))
            )
    if not candidates:
        raise ValueError("No start poses selected")

    scenarios = []
    used_names: set[str] = set()
    for index, (source, pose) in enumerate(candidates, start=1):
        if pose.map_name is None:
            raise ValueError(f"Selected pose {pose.name or index!r} has no map_name")
        base_name = pose.name or f"{source}_{index:02d}"
        name = base_name
        if name in used_names:
            name = f"{source}:{base_name}"
        if name in used_names:
            name = f"{source}:{index:02d}:{base_name}"
        used_names.add(name)
        scenarios.append(Scenario(index=index, source=source, name=name, pose=pose))
    return tuple(scenarios)


def episode_seed(master_seed: int, scenario_index: int, repeat: int) -> int:
    sequence = np.random.SeedSequence(
        [
            int(master_seed) % (2**32),
            int(scenario_index) % (2**32),
            int(repeat) % (2**32),
            0x5452494D,
        ]
    )
    return int(np.random.default_rng(sequence).integers(0, np.iinfo(np.int32).max))


def apply_trim_override(env, trim: float) -> None:
    import geometry
    from duckietown_world import get_DB18_uncalibrated

    raw_env = getattr(env, "unwrapped", env)
    dynamics = get_DB18_uncalibrated(delay=0.15, trim=float(trim))
    configuration = raw_env.cartesian_from_weird(raw_env.cur_pos, raw_env.cur_angle)
    velocity = geometry.se2_from_linear_angular(np.zeros(2), 0)
    raw_env.state = dynamics.initialize(c0=(configuration, velocity), t0=0)
    raw_env.last_action = np.zeros(2, dtype=np.float32)
    raw_env.wheelVels = np.zeros(2, dtype=np.float32)


def history_inputs(
    history: FixedHistory,
    history_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations, actions = history.batched(device)
    if history_mode == "native":
        return observations, actions
    if history_mode == "zero_action_history":
        return observations, torch.zeros_like(actions)
    if history_mode == "current_frame_zero_actions":
        current = observations[:, -1:, ...]
        repeated = current.repeat(1, observations.shape[1], 1, 1, 1)
        return repeated, torch.zeros_like(actions)
    if history_mode == "reversed_action_history":
        return observations, torch.flip(actions, dims=(1,))
    raise ValueError(f"Unknown history mode {history_mode!r}")


@torch.no_grad()
def predict_action(
    policy: TemporalTanhGaussianPolicy,
    action_control: DuckietownActionControl,
    history: FixedHistory,
    history_mode: str,
    stochastic: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations, actions = history_inputs(history, history_mode, device)
    mean, log_std = policy(observations, actions)
    distribution = torch.distributions.Normal(mean, log_std.exp())
    raw_action = distribution.sample() if stochastic else distribution.mean
    policy_action = torch.tanh(raw_action)
    wheel_action = action_control.to_wheels_tensor(policy_action)
    return (
        policy_action.squeeze(0).cpu().numpy().astype(np.float32),
        format_wheel_action(wheel_action.squeeze(0).cpu().numpy()),
        log_std.exp().squeeze(0).cpu().numpy().astype(np.float32),
    )


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return fmean(finite) if finite else math.nan


def finite_max(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else math.nan


def finite_min(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else math.nan


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    points = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(points) < 2:
        return math.nan
    mean_x = finite_mean(x for x, _ in points)
    mean_y = finite_mean(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 0.0:
        return math.nan
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def value_or_nan(values: dict[str, float], name: str) -> float:
    return float(values.get(name, math.nan))


def run_episode(
    *,
    env,
    policy: TemporalTanhGaussianPolicy,
    action_control: DuckietownActionControl,
    checkpoint: CheckpointSpec,
    checkpoint_step: int,
    checkpoint_config: dict[str, Any],
    config: TrimSensitivityConfig,
    scenario: Scenario,
    trim: float,
    repeat: int,
    history_mode: str,
    device: torch.device,
    transform,
    step_writer: csv.DictWriter | None,
) -> dict[str, Any]:
    raw_env = getattr(env, "unwrapped", env)
    if getattr(raw_env, "map_name", None) != scenario.pose.map_name:
        load_environment_map(env, scenario.pose.map_name)
    apply_env_start_pose(env, scenario.pose)
    reset_seed = episode_seed(config.seed, scenario.index, repeat)
    observation, _ = reset_raw(env, seed=reset_seed)
    apply_trim_override(env, trim)
    if not raw_env._valid_pose(raw_env.cur_pos, raw_env.cur_angle):
        raise ValueError(
            f"Pose {scenario.name!r} is invalid on map {scenario.pose.map_name!r}"
        )

    reward_function = config.reward_function or checkpoint_config.get(
        "reward_function",
        "velopose",
    )
    gamma = float(checkpoint_config.get("gamma", 0.99))
    vd2pp_weight = (
        config.vd2pp_distance_weight
        if config.vd2pp_distance_weight is not None
        else float(checkpoint_config.get("vd2pp_distance_weight", 1.0))
    )
    selected_calculator = GymDuckietownRewardCalculator(
        reward_function,
        gamma=gamma,
        vd2pp_distance_weight=vd2pp_weight,
    )
    velopose_calculator = (
        selected_calculator
        if reward_function == "velopose"
        else GymDuckietownRewardCalculator("velopose", gamma=gamma)
    )
    selected_calculator.reset(env)
    if velopose_calculator is not selected_calculator:
        velopose_calculator.reset(env)

    observation_history_length = int(
        checkpoint_config.get("observation_history_length", 1)
    )
    action_history_length = int(
        checkpoint_config.get(
            "action_history_length",
            observation_history_length - 1,
        )
    )
    image_size = int(checkpoint_config.get("image_size", 224))
    crop_y_start = int(checkpoint_config.get("crop_y_start", 0))
    channel_order = checkpoint_config.get("source_observation_channel_order", "rgb")
    history = FixedHistory(
        observation_history_length,
        action_history_length,
        action_control.policy_action_dim,
    )
    history.reset(preprocess(observation, crop_y_start, image_size, channel_order, transform))

    selected_return = 0.0
    velopose_return = 0.0
    env_return = 0.0
    lane_valid: list[float] = []
    lane_distances: list[float] = []
    lane_times: list[float] = []
    heading_errors: list[float] = []
    speeds: list[float] = []
    forward_speeds: list[float] = []
    normalized_progress: list[float] = []
    heading_qualities: list[float] = []
    pose_qualities: list[float] = []
    wheel_throttles: list[float] = []
    wheel_steerings: list[float] = []
    terminated = False
    truncated = False
    reason = "time_limit"

    for step_index in range(1, config.max_steps + 1):
        if config.stochastic:
            torch.manual_seed(reset_seed + step_index)
        policy_action, wheel_action, policy_std = predict_action(
            policy,
            action_control,
            history,
            history_mode,
            config.stochastic,
            device,
        )
        observation, env_reward, terminated, truncated, info = step_raw(
            env,
            wheel_action,
        )
        local_time_limit = step_index >= config.max_steps and not (
            terminated or truncated
        )
        reason = done_reason(terminated, truncated, local_time_limit, info)
        done_code = reason if terminated or truncated or local_time_limit else "in-progress"
        selected_breakdown = selected_calculator.compute_breakdown(
            env,
            float(env_reward),
            done_code=done_code,
        )
        velopose_breakdown = (
            selected_breakdown
            if velopose_calculator is selected_calculator
            else velopose_calculator.compute_breakdown(
                env,
                float(env_reward),
                done_code=done_code,
            )
        )
        selected_flat = flatten_reward_breakdown(selected_breakdown)
        velopose_flat = flatten_reward_breakdown(velopose_breakdown)
        selected_reward = float(selected_breakdown["total"])
        velopose_reward = float(velopose_breakdown["total"])
        selected_return += selected_reward
        velopose_return += velopose_reward
        env_return += float(env_reward)

        lane = get_lane_metrics(env)
        timestamp = float(getattr(raw_env, "timestamp", step_index))
        lane_valid.append(float(bool(lane["lane_valid"])))
        if lane["lane_valid"]:
            lane_distances.append(float(lane["dist"]))
            lane_times.append(timestamp)
            heading_errors.append(abs(float(lane["angle_deg"])))
        speed = float(lane["speed"])
        forward_speed = value_or_nan(velopose_flat, "Velocity.ForwardSpeedMps")
        normalized = value_or_nan(
            velopose_flat,
            "Velocity.NormalizedForwardProgress",
        )
        heading_quality = value_or_nan(velopose_flat, "Pose.HeadingQuality")
        pose_quality = value_or_nan(velopose_flat, "Pose.PoseQuality")
        wheel_throttle = 0.5 * float(wheel_action[0] + wheel_action[1])
        wheel_steering = 0.5 * float(wheel_action[1] - wheel_action[0])
        speeds.append(speed)
        forward_speeds.append(forward_speed)
        normalized_progress.append(normalized)
        heading_qualities.append(heading_quality)
        pose_qualities.append(pose_quality)
        wheel_throttles.append(wheel_throttle)
        wheel_steerings.append(wheel_steering)

        if step_writer is not None:
            position = np.asarray(raw_env.cur_pos, dtype=np.float64)
            control_names = tuple(action_control.control_names)
            second_control_name = control_names[1] if len(control_names) > 1 else ""
            second_control = (
                float(policy_action[1]) if policy_action.size > 1 else math.nan
            )
            second_std = float(policy_std[1]) if policy_std.size > 1 else math.nan
            step_writer.writerow(
                {
                    "checkpoint": checkpoint.name,
                    "checkpoint_step": checkpoint_step,
                    "history_mode": history_mode,
                    "reward_function": reward_function,
                    "scenario_index": scenario.index,
                    "scenario_source": scenario.source,
                    "scenario_name": scenario.name,
                    "map_name": scenario.pose.map_name,
                    "trim": trim,
                    "repeat": repeat,
                    "reset_seed": reset_seed,
                    "step": step_index,
                    "timestamp": timestamp,
                    "policy_control_0_name": control_names[0],
                    "policy_control_0": float(policy_action[0]),
                    "policy_control_1_name": second_control_name,
                    "policy_control_1": second_control,
                    "policy_std_0": float(policy_std[0]),
                    "policy_std_1": second_std,
                    "wheel_left": float(wheel_action[0]),
                    "wheel_right": float(wheel_action[1]),
                    "wheel_throttle": wheel_throttle,
                    "wheel_steering": wheel_steering,
                    "selected_reward": selected_reward,
                    "velopose_reward": velopose_reward,
                    "env_reward": float(env_reward),
                    "selected_return": selected_return,
                    "velopose_return": velopose_return,
                    "env_return": env_return,
                    "lane_valid": int(bool(lane["lane_valid"])),
                    "lane_distance_m": float(lane["dist"]) if lane["lane_valid"] else math.nan,
                    "lane_dot_dir": float(lane["dot_dir"]) if lane["lane_valid"] else math.nan,
                    "lane_angle_deg": float(lane["angle_deg"]) if lane["lane_valid"] else math.nan,
                    "speed_mps": speed,
                    "forward_speed_mps": forward_speed,
                    "normalized_forward_progress": normalized,
                    "heading_quality": heading_quality,
                    "target_heading_offset_deg": value_or_nan(
                        velopose_flat,
                        "Pose.TargetHeadingOffsetDeg",
                    ),
                    "signed_scaled_lane_distance": value_or_nan(
                        velopose_flat,
                        "Pose.SignedScaledLaneDistance",
                    ),
                    "scaled_abs_lane_distance": value_or_nan(
                        velopose_flat,
                        "Pose.ScaledAbsLaneDistance",
                    ),
                    "pose_quality": pose_quality,
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "angle": float(raw_env.cur_angle),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "done_reason": reason,
                    "selected_reward_components_json": json.dumps(
                        selected_flat,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

        if terminated or truncated or local_time_limit:
            break
        history.append(
            policy_action,
            preprocess(
                observation,
                crop_y_start,
                image_size,
                channel_order,
                transform,
            ),
        )

    steps = step_index
    position = np.asarray(raw_env.cur_pos, dtype=np.float64)
    safe = int(steps >= config.max_steps and not terminated)
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_step": checkpoint_step,
        "model": checkpoint_config.get("model", "mobilenet_v3_small"),
        "observation_history_length": observation_history_length,
        "action_history_length": action_history_length,
        "history_mode": history_mode,
        "reward_function": reward_function,
        "scenario_index": scenario.index,
        "scenario_source": scenario.source,
        "scenario_name": scenario.name,
        "map_name": scenario.pose.map_name,
        "trim": trim,
        "repeat": repeat,
        "reset_seed": reset_seed,
        "steps": steps,
        "selected_return": selected_return,
        "velopose_return": velopose_return,
        "env_return": env_return,
        "reward_per_step": selected_return / steps,
        "safe": safe,
        "terminated": int(terminated),
        "truncated": int(truncated),
        "done_reason": reason,
        "lane_valid_fraction": finite_mean(lane_valid),
        "mean_lane_distance_m": finite_mean(lane_distances),
        "mean_abs_lane_distance_m": finite_mean(abs(value) for value in lane_distances),
        "max_abs_lane_distance_m": finite_max(abs(value) for value in lane_distances),
        "final_lane_distance_m": lane_distances[-1] if lane_distances else math.nan,
        "lane_distance_slope_mps": linear_slope(lane_times, lane_distances),
        "abs_lane_distance_slope_mps": linear_slope(
            lane_times,
            [abs(value) for value in lane_distances],
        ),
        "mean_abs_heading_error_deg": finite_mean(heading_errors),
        "max_abs_heading_error_deg": finite_max(heading_errors),
        "mean_speed_mps": finite_mean(speeds),
        "mean_forward_speed_mps": finite_mean(forward_speeds),
        "mean_normalized_forward_progress": finite_mean(normalized_progress),
        "mean_heading_quality": finite_mean(heading_qualities),
        "mean_pose_quality": finite_mean(pose_qualities),
        "min_pose_quality": finite_min(pose_qualities),
        "mean_wheel_throttle": finite_mean(wheel_throttles),
        "mean_wheel_steering": finite_mean(wheel_steerings),
        "std_wheel_steering": float(np.std(wheel_steerings)) if wheel_steerings else math.nan,
        "mean_abs_wheel_steering": finite_mean(abs(value) for value in wheel_steerings),
        "final_x": float(position[0]),
        "final_y": float(position[1]),
        "final_z": float(position[2]),
        "final_angle": float(raw_env.cur_angle),
    }


def aggregate_rows(
    episodes: Sequence[dict[str, Any]],
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode[key] for key in group_keys)].append(episode)
    result = []
    for key, rows in groups.items():
        aggregate = dict(zip(group_keys, key))
        aggregate.update(
            {
                "episodes": len(rows),
                "safe_rate": finite_mean(float(row["safe"]) for row in rows),
                "invalid_pose_rate": finite_mean(
                    float(row["done_reason"] == "invalid-pose") for row in rows
                ),
            }
        )
        for metric in SUMMARY_METRICS:
            aggregate[f"mean_{metric}"] = finite_mean(
                float(row[metric]) for row in rows
            )
        result.append(aggregate)
    return sorted(
        result,
        key=lambda row: tuple(str(row[key]) for key in group_keys[:-1])
        + (float(row[group_keys[-1]]),)
        if group_keys[-1] == "trim"
        else tuple(str(row[key]) for key in group_keys),
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_fields(group_keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(group_keys) + (
        "episodes",
        "safe_rate",
        "invalid_pose_rate",
    ) + tuple(f"mean_{metric}" for metric in SUMMARY_METRICS)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def line_chart(
    summary: Sequence[dict[str, Any]],
    metric: str,
    title: str,
    y_label: str,
) -> str:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in summary:
        value = float(row.get(metric, math.nan))
        trim = float(row["trim"])
        if math.isfinite(value):
            grouped[f"{row['checkpoint']} [{row['history_mode']}]"] .append(
                (trim, value)
            )
    if not grouped:
        return f"<section><h2>{html.escape(title)}</h2><p>No data.</p></section>"
    for points in grouped.values():
        points.sort()
    all_points = [point for points in grouped.values() for point in points]
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        min_x -= 0.01
        max_x += 0.01
    if min_y == max_y:
        min_y -= 0.5
        max_y += 0.5
    padding = 0.08 * (max_y - min_y)
    min_y -= padding
    max_y += padding
    width, height = 960, 340
    left, right, top, bottom = 72, 20, 24, 52
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_width

    def sy(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * plot_height

    parts = [
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_height}' class='axis'/>",
        f"<line x1='{left}' y1='{top + plot_height}' x2='{left + plot_width}' y2='{top + plot_height}' class='axis'/>",
    ]
    for index in range(6):
        x_value = min_x + index * (max_x - min_x) / 5
        y_value = min_y + index * (max_y - min_y) / 5
        x = sx(x_value)
        y = sy(y_value)
        parts.append(
            f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_height}' class='grid'/>"
        )
        parts.append(
            f"<text x='{x:.1f}' y='{height - 24}' text-anchor='middle'>{x_value:.2f}</text>"
        )
        parts.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_width}' y2='{y:.1f}' class='grid'/>"
        )
        parts.append(
            f"<text x='{left - 10}' y='{y + 4:.1f}' text-anchor='end'>{y_value:.2f}</text>"
        )
    legend = []
    for series_index, (label, points) in enumerate(sorted(grouped.items())):
        color = COLORS[series_index % len(COLORS)]
        coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(
            f"<polyline points='{coordinates}' fill='none' stroke='{color}' stroke-width='2.5'/>"
        )
        for x, y in points:
            parts.append(
                f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3.5' fill='{color}'><title>{html.escape(label)} trim={x:+.3f} value={y:.5f}</title></circle>"
            )
        legend.append(
            f"<span><i style='background:{color}'></i>{html.escape(label)}</span>"
        )
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<div class='legend'>{''.join(legend)}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img'>"
        f"<text x='{left + plot_width / 2:.1f}' y='{height - 4}' text-anchor='middle'>trim</text>"
        f"<text transform='translate(16 {top + plot_height / 2:.1f}) rotate(-90)' text-anchor='middle'>{html.escape(y_label)}</text>"
        f"{''.join(parts)}</svg></section>"
    )


def html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def write_report(
    path: Path,
    config: TrimSensitivityConfig,
    checkpoint_details: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    scenario_summary: Sequence[dict[str, Any]],
) -> None:
    summary_rows = [
        (
            row["checkpoint"],
            row["history_mode"],
            f"{float(row['trim']):+.3f}",
            row["episodes"],
            fmt(100 * float(row["safe_rate"]), 1) + "%",
            fmt(100 * float(row["invalid_pose_rate"]), 1) + "%",
            fmt(row["mean_reward_per_step"]),
            fmt(row["mean_mean_abs_lane_distance_m"]),
            fmt(row["mean_mean_abs_heading_error_deg"], 2),
            fmt(row["mean_mean_wheel_steering"]),
        )
        for row in summary
    ]
    scenario_rows = [
        (
            row["checkpoint"],
            row["history_mode"],
            row["scenario_name"],
            row["map_name"],
            f"{float(row['trim']):+.3f}",
            fmt(100 * float(row["safe_rate"]), 1) + "%",
            fmt(row["mean_reward_per_step"]),
            fmt(row["mean_mean_abs_lane_distance_m"]),
            fmt(row["mean_abs_lane_distance_slope_mps"]),
        )
        for row in scenario_summary
    ]
    charts = "".join(
        (
            line_chart(summary, "safe_rate", "Safe episode rate", "fraction"),
            line_chart(summary, "invalid_pose_rate", "Invalid-pose rate", "fraction"),
            line_chart(summary, "mean_reward_per_step", "Reward per step", "reward / step"),
            line_chart(
                summary,
                "mean_mean_abs_lane_distance_m",
                "Mean absolute lane distance",
                "meters",
            ),
            line_chart(
                summary,
                "mean_abs_lane_distance_slope_mps",
                "Absolute lane-distance drift rate",
                "meters / second",
            ),
            line_chart(
                summary,
                "mean_mean_abs_heading_error_deg",
                "Mean absolute heading error",
                "degrees",
            ),
            line_chart(
                summary,
                "mean_mean_wheel_steering",
                "Mean steering compensation",
                "(right - left) / 2",
            ),
        )
    )
    payload = html.escape(json.dumps(config.as_json(), indent=2))
    checkpoint_payload = html.escape(json.dumps(list(checkpoint_details), indent=2))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duckietown trim sensitivity</title>
<style>
:root{{--bg:#f5f7f8;--panel:#fff;--text:#172126;--muted:#607078;--line:#ccd5d9;--accent:#147d92}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px}} h1{{margin:0 0 8px}} h2{{margin-top:0;font-size:20px}}
.muted{{color:var(--muted)}} section{{background:var(--panel);margin:18px 0;padding:20px;border:1px solid var(--line);border-radius:6px}}
.links a{{margin-right:18px;color:var(--accent)}} .legend{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;font-size:13px}}
.legend i{{display:inline-block;width:13px;height:3px;margin:0 6px 3px 0}} svg{{width:100%;height:auto}}
svg text{{font-size:12px;fill:#526169}} .axis{{stroke:#53636b;stroke-width:1.2}} .grid{{stroke:#dfe5e8;stroke-width:1}}
.table-wrap{{overflow:auto;max-height:620px}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #e1e6e8;padding:7px 9px;text-align:right;white-space:nowrap}} th{{position:sticky;top:0;background:#eef2f3}}
th:first-child,td:first-child{{text-align:left}} details pre{{overflow:auto;background:#f0f3f4;padding:14px}}
</style></head><body><main>
<h1>Duckietown trim sensitivity</h1>
<p class="muted">Deterministic fixed-pose comparison across explicit Duckiebot trim values.</p>
<section class="links"><h2>Artifacts</h2>
<a href="episodes.csv">episodes.csv</a><a href="scenario_summary.csv">scenario_summary.csv</a>
<a href="summary.csv">summary.csv</a>{'<a href="steps.csv">steps.csv</a>' if config.record_step_history else ''}
<a href="config.json">config.json</a></section>
{charts}
<section><h2>Aggregate results</h2>{html_table(
        ("Checkpoint", "History", "Trim", "Episodes", "Safe", "Invalid", "Reward/step", "|lane dist| m", "|heading| deg", "Mean steering"),
        summary_rows,
    )}</section>
<section><h2>Per-scenario results</h2>{html_table(
        ("Checkpoint", "History", "Scenario", "Map", "Trim", "Safe", "Reward/step", "|lane dist| m", "|lane| slope m/s"),
        scenario_rows,
    )}</section>
<section><details><summary>Resolved configuration</summary><pre>{payload}</pre></details>
<details><summary>Checkpoint architectures</summary><pre>{checkpoint_payload}</pre></details></section>
</main></body></html>"""
    path.write_text(document)


def prepare_output_directory(config: TrimSensitivityConfig) -> None:
    existing = [
        config.output_dir / filename
        for filename in OUTPUT_FILENAMES
        if (config.output_dir / filename).exists()
    ]
    if existing and not config.overwrite:
        raise ValueError(
            f"Output directory already contains evaluator artifacts: {config.output_dir}. "
            "Choose another output_dir or set overwrite=true."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        for path in existing:
            path.unlink()


def modes_for_checkpoint(
    configured_modes: Sequence[str],
    observation_history_length: int,
) -> tuple[str, ...]:
    if observation_history_length > 1:
        return tuple(configured_modes)
    return ("native",)


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args)
        scenarios = selected_scenarios(config)
        prepare_output_directory(config)
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error

    configure_gym_duckietown_logging(config.log_level)
    device = resolve_device(config.device)
    set_seed(config.seed)
    transform = make_transform()
    config_path = config.output_dir / "config.json"
    config_path.write_text(json.dumps(config.as_json(), indent=2) + "\n")
    episodes: list[dict[str, Any]] = []
    checkpoint_details: list[dict[str, Any]] = []
    steps_file = None
    step_writer = None
    if config.record_step_history:
        steps_file = (config.output_dir / "steps.csv").open("w", newline="")
        step_writer = csv.DictWriter(steps_file, fieldnames=STEP_FIELDS)
        step_writer.writeheader()

    try:
        for checkpoint in config.checkpoints:
            policy, action_control, checkpoint_config, checkpoint_step = load_policy(
                checkpoint,
                device,
            )
            observation_history_length = int(
                checkpoint_config.get("observation_history_length", 1)
            )
            action_history_length = int(
                checkpoint_config.get(
                    "action_history_length",
                    observation_history_length - 1,
                )
            )
            history_modes = modes_for_checkpoint(
                config.history_modes,
                observation_history_length,
            )
            checkpoint_details.append(
                {
                    "name": checkpoint.name,
                    "path": str(checkpoint.path),
                    "step": checkpoint_step,
                    "model": checkpoint_config.get("model", "mobilenet_v3_small"),
                    "observation_history_length": observation_history_length,
                    "action_history_length": action_history_length,
                    "temporal_head_mode": checkpoint_config.get(
                        "temporal_head_mode",
                        "residual",
                    ),
                    "action_mode": action_control.mode,
                    "history_modes_evaluated": list(history_modes),
                }
            )
            env_args = environment_args(
                config,
                checkpoint_config,
                scenarios[0].pose.map_name,
            )
            env = make_env(env_args, seed=config.seed)
            try:
                for history_mode in history_modes:
                    for trim in config.trims:
                        for scenario in scenarios:
                            for repeat in range(1, config.repeats + 1):
                                row = run_episode(
                                    env=env,
                                    policy=policy,
                                    action_control=action_control,
                                    checkpoint=checkpoint,
                                    checkpoint_step=checkpoint_step,
                                    checkpoint_config=checkpoint_config,
                                    config=config,
                                    scenario=scenario,
                                    trim=trim,
                                    repeat=repeat,
                                    history_mode=history_mode,
                                    device=device,
                                    transform=transform,
                                    step_writer=step_writer,
                                )
                                episodes.append(row)
                                print(
                                    f"checkpoint={checkpoint.name} history={history_mode} "
                                    f"trim={trim:+.3f} pose={scenario.name} repeat={repeat} "
                                    f"return={row['selected_return']:+.3f} "
                                    f"steps={row['steps']} reason={row['done_reason']}",
                                    flush=True,
                                )
            finally:
                env.close()
            del policy
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if steps_file is not None:
            steps_file.close()

    summary_keys = ("checkpoint", "history_mode", "trim")
    scenario_keys = (
        "checkpoint",
        "history_mode",
        "scenario_source",
        "scenario_name",
        "map_name",
        "trim",
    )
    summary = aggregate_rows(episodes, summary_keys)
    scenario_summary = aggregate_rows(episodes, scenario_keys)
    write_csv(config.output_dir / "episodes.csv", EPISODE_FIELDS, episodes)
    write_csv(
        config.output_dir / "summary.csv",
        summary_fields(summary_keys),
        summary,
    )
    write_csv(
        config.output_dir / "scenario_summary.csv",
        summary_fields(scenario_keys),
        scenario_summary,
    )
    report_path = config.output_dir / "trim_sensitivity_report.html"
    write_report(report_path, config, checkpoint_details, summary, scenario_summary)
    print(f"Results: {config.output_dir}", flush=True)
    print(f"Report:  {report_path}", flush=True)
    if config.open_report:
        try:
            webbrowser.open(report_path.resolve().as_uri(), new=2)
        except webbrowser.Error as error:
            print(f"Could not open report automatically: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
