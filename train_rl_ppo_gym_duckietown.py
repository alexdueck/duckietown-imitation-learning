#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Train a PPO policy from camera images in gym-duckietown.

This trainer intentionally lives beside the Duckiematrix trainer instead of
adding another backend to it. Policies can emit direct wheel commands or
semantic throttle/steering controls that are converted to wheel commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from duckietown_rewards import (
    GymDuckietownRewardCalculator,
    MAX_STEPS_DONE_CODE,
    REWARD_FUNCTION_CHOICES,
    format_wheel_action,
    gym_duckietown_done_code,
    patch_duckietown_world_dynamics,
    reward_source,
)
from cli_completion import parse_args_with_completion
from duckietown_action_control import (
    ACTION_MODE_CHOICES,
    DuckietownActionControl,
)
from duckietown_paths import RL_PPO_GYM_DUCKIETOWN_CHECKPOINT_DIR
from gym_duckietown_start_config import (
    StartConfig,
    TrainingPose,
    TrainingStart,
    apply_env_start_pose,
    choose_training_start,
    load_start_config,
)
from rl_models import TanhGaussianPolicy, load_imitation_actor, tanh_normal_log_prob
from train_imitation_learning import IMAGENET_MEAN, IMAGENET_STD, build_model, resolve_device, set_seed
from velopose_reward import (
    VELOPPOSE_HEADING_CORRECTION_GAIN,
    VELOPPOSE_HEADING_MAX_CORRECTION_DEG,
    VELOPPOSE_INVALID_POSE_PENALTY,
    VELOPPOSE_POSE_WEIGHT,
    VELOPPOSE_VELOCITY_WEIGHT,
)


@dataclass
class PPOConfig:
    output_dir: str
    map_name: str
    reward_function: str
    model: str
    imitation_checkpoint: str | None
    resume_checkpoint: str | None
    action_mode: str
    fixed_throttle: float | None
    max_throttle: float
    max_steering: float
    total_steps: int
    max_episode_steps: int
    reset_random_warmup_steps: int
    reset_random_warmup_retries: int
    reset_random_action_scale: float
    start_seeds_config: str | None
    hard_start_probability: float
    training_start_seeds: tuple[int, ...]
    training_start_poses: tuple[TrainingPose, ...]
    evaluation_start_poses: tuple[TrainingPose, ...]
    eval_interval_rollouts: int
    eval_steps: int
    eval_seeds: tuple[int, ...]
    eval_deterministic: bool
    initial_log_std: float | None
    min_log_std: float
    max_log_std: float
    rollout_steps: int
    epochs: int
    batch_size: int
    gamma: float
    gae_lambda: float
    clip_ratio: float
    policy_lr: float
    value_lr: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    image_size: int
    crop_y_start: int
    source_observation_channel_order: str
    domain_rand: bool
    distortion: bool
    frame_skip: int
    frame_rate: int
    robot_speed: float | None
    accept_start_angle_deg: float
    simulator_max_steps: int | None
    camera_width: int
    camera_height: int
    render_training: bool
    log_level: str
    debug_initial_action: bool
    seed: int
    device: str


@dataclass(frozen=True)
class EnvironmentStartDefaults:
    user_tile_start: Any
    start_pose: Any


class ValueNetwork(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = False) -> None:
        super().__init__()
        from rl_models import build_encoder

        self.encoder, features_dim = build_encoder(model_name, pretrained=pretrained)
        self.value = nn.Linear(features_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value(self.encoder(observations)).squeeze(1)


def parse_eval_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--eval-seeds must be a comma-separated list of integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("--eval-seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--eval-seeds must not contain duplicates")
    return seeds


def parse_probability(value: str) -> float:
    try:
        probability = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("probability must be a number between 0 and 1") from error
    if not 0.0 <= probability <= 1.0:
        raise argparse.ArgumentTypeError("probability must be between 0 and 1")
    return probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO in gym-duckietown.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RL_PPO_GYM_DUCKIETOWN_CHECKPOINT_DIR,
    )
    parser.add_argument("--map-name", default="loop_empty")
    parser.add_argument(
        "--reward-function",
        choices=REWARD_FUNCTION_CHOICES,
        default="posangle",
        help="Reward seen by PPO. Non-default options follow kaland313/Duckietown-RL reward wrappers.",
    )
    parser.add_argument("--model", choices=("mobilenet_v3_small", "resnet18"), default="mobilenet_v3_small")
    parser.add_argument("--imitation-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume PPO training from an RL checkpoint such as last.pt, best_return.pt, or best_safe.pt.",
    )
    parser.add_argument(
        "--action-mode",
        choices=ACTION_MODE_CHOICES,
        default="wheel",
        help=(
            "Policy controls: direct left/right wheel commands (wheel), or "
            "non-negative throttle plus symmetric steering (throttle_steering)."
        ),
    )
    parser.add_argument(
        "--fixed-throttle",
        type=float,
        default=None,
        help=(
            "Fix throttle to this wheel-command value in [0, 1]. With "
            "throttle_steering, the policy then learns steering only."
        ),
    )
    parser.add_argument(
        "--max-throttle",
        type=float,
        default=1.0,
        help="Maximum learned throttle in throttle_steering mode.",
    )
    parser.add_argument(
        "--max-steering",
        type=float,
        default=0.5,
        help="Maximum wheel differential contributed by steering.",
    )
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1024,
        help="Reset and log an episode after this many steps; set to 0 to rely only on environment termination.",
    )
    parser.add_argument("--reset-random-warmup-steps", type=int, default=0)
    parser.add_argument("--reset-random-warmup-retries", type=int, default=3)
    parser.add_argument("--reset-random-action-scale", type=float, default=0.6)
    parser.add_argument(
        "--start-seeds-config",
        type=Path,
        default=None,
        help=(
            "JSON config containing training_seeds/training_poses and "
            "evaluation_seeds/evaluation_poses. Its evaluation scenarios replace --eval-seeds."
        ),
    )
    parser.add_argument(
        "--hard-start-probability",
        type=parse_probability,
        default=0.5,
        help="Probability of selecting a configured training seed or pose at each episode reset.",
    )
    parser.add_argument("--eval-interval-rollouts", type=int, default=10)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="Maximum steps for each fixed evaluation scenario.",
    )
    parser.add_argument(
        "--eval-seeds",
        type=parse_eval_seeds,
        default=(10042, 10043, 10044, 10045, 10046),
        help="Comma-separated reset seeds defining the fixed evaluation scenarios.",
    )
    parser.add_argument("--eval-stochastic", action="store_false", dest="eval_deterministic")
    parser.add_argument("--initial-log-std", type=float, default=None)
    parser.add_argument("--min-log-std", type=float, default=-5.0)
    parser.add_argument("--max-log-std", type=float, default=-1.0)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--policy-lr", type=float, default=1e-5)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-y-start", type=int, default=0)
    parser.add_argument(
        "--source-observation-channel-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="gym-duckietown observations are expected to be RGB.",
    )
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument("--distortion", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--robot-speed", type=float, default=None)
    parser.add_argument("--accept-start-angle-deg", type=float, default=4.0)
    parser.add_argument(
        "--simulator-max-steps",
        type=int,
        default=None,
        help="gym-duckietown Simulator max_steps. Defaults to --max-episode-steps when positive.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--render-training",
        action="store_true",
        help="Open gym-duckietown's human-view window and update it after every training step.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level for gym-duckietown and its Duckietown dependencies.",
    )
    parser.add_argument("--debug-initial-action", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parse_args_with_completion(parser)


def observation_to_rgb(observation: np.ndarray, channel_order: str) -> np.ndarray:
    image = np.ascontiguousarray(observation)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected observation shape (H, W, 3), got {image.shape}")
    if channel_order == "rgb":
        return image
    if channel_order == "bgr":
        return np.ascontiguousarray(image[:, :, [2, 1, 0]])
    raise ValueError(f"Unknown channel order {channel_order!r}")


def make_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def preprocess(
    observation: np.ndarray,
    crop_y_start: int,
    image_size: int,
    channel_order: str,
    transform: transforms.Compose,
) -> torch.Tensor:
    image = Image.fromarray(observation_to_rgb(observation, channel_order)).convert("RGB")
    width, height = image.size
    crop_y_start = max(0, min(crop_y_start, height - 1))
    image = image.crop((0, crop_y_start, width, height)).resize((image_size, image_size), Image.BILINEAR)
    return transform(image)


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for step in reversed(range(rewards.size(0))):
        next_value = last_value if step == rewards.size(0) - 1 else values[step + 1]
        not_done = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * not_done - values[step]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[step] = gae
    returns = advantages + values
    return advantages, returns


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def flatten_reward_breakdown(breakdown: dict[str, Any]) -> dict[str, float]:
    flattened = {"Reward": float(breakdown["total"])}

    def visit(components: dict[str, Any], prefix: str = "") -> None:
        for name, value in components.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            if isinstance(value, dict):
                if "total" in value:
                    flattened[path] = float(value["total"])
                nested = value.get("components")
                if isinstance(nested, dict):
                    visit(nested, path)
            elif isinstance(value, (int, float, np.number)):
                flattened[path] = float(value)

    components = breakdown.get("components")
    if isinstance(components, dict):
        visit(components)
    return flattened


@dataclass
class RewardComponentAccumulator:
    steps: int = 0
    sums: dict[str, float] = field(default_factory=dict)
    present_counts: dict[str, int] = field(default_factory=dict)

    def add(self, breakdown: dict[str, Any]) -> float:
        flattened = flatten_reward_breakdown(breakdown)
        self.steps += 1
        for name, value in flattened.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
            self.present_counts[name] = self.present_counts.get(name, 0) + 1
        return flattened["Reward"]

    def merge(self, other: "RewardComponentAccumulator") -> None:
        self.steps += other.steps
        for name, value in other.sums.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
            self.present_counts[name] = self.present_counts.get(name, 0) + other.present_counts[name]


REWARD_COMPONENT_FIELDS = [
    "phase",
    "train_step",
    "train_rollout",
    "eval_index",
    "scenario_index",
    "scenario_seed",
    "component",
    "component_sum",
    "component_mean_per_step",
    "component_mean_when_present",
    "present_count",
    "step_count",
]


def write_reward_components(
    path: Path,
    accumulator: RewardComponentAccumulator,
    *,
    phase: str,
    train_step: int,
    train_rollout: int,
    eval_index: int | None = None,
    scenario_index: int | None = None,
    scenario_seed: int | None = None,
) -> None:
    if accumulator.steps == 0:
        return
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REWARD_COMPONENT_FIELDS)
        for component in sorted(accumulator.sums):
            component_sum = accumulator.sums[component]
            present_count = accumulator.present_counts[component]
            writer.writerow({
                "phase": phase,
                "train_step": train_step,
                "train_rollout": train_rollout,
                "eval_index": "" if eval_index is None else eval_index,
                "scenario_index": "" if scenario_index is None else scenario_index,
                "scenario_seed": "" if scenario_seed is None else scenario_seed,
                "component": component,
                "component_sum": component_sum,
                "component_mean_per_step": component_sum / accumulator.steps,
                "component_mean_when_present": component_sum / present_count,
                "present_count": present_count,
                "step_count": accumulator.steps,
            })


def ensure_gym_duckietown_available() -> None:
    try:
        import gym_duckietown.simulator  # noqa: F401
        patch_duckietown_world_dynamics()
    except ImportError as error:
        raise SystemExit(
            "gym-duckietown is not installed in this Python environment. "
            "Install it first, then rerun train_rl_ppo_gym_duckietown.py."
        ) from error


def configure_gym_duckietown_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper())
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level)
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)

    for logger_name in (
        "gym-duckietown",
        "duckietown_world",
        "geometry",
        "typing",
        "commons",
        "nodes",
        "aido_schemas",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def make_env(args: argparse.Namespace, seed: int | None = None):
    from gym_duckietown.simulator import DEFAULT_ROBOT_SPEED, Simulator

    patch_duckietown_world_dynamics()

    simulator_max_steps = args.simulator_max_steps
    if simulator_max_steps is None:
        simulator_max_steps = args.max_episode_steps if args.max_episode_steps > 0 else 100_000_000

    robot_speed = DEFAULT_ROBOT_SPEED if args.robot_speed is None else args.robot_speed
    return Simulator(
        seed=args.seed if seed is None else seed,
        map_name=args.map_name,
        max_steps=simulator_max_steps,
        draw_curve=False,
        draw_bbox=False,
        domain_rand=args.domain_rand,
        frame_rate=args.frame_rate,
        frame_skip=args.frame_skip,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        robot_speed=robot_speed,
        accept_start_angle_deg=args.accept_start_angle_deg,
        full_transparency=True,
        distortion=args.distortion,
    )


def reset_raw(env, seed: int | None = None):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    return result, {}


def step_raw(env, action: np.ndarray):
    result = env.step(format_wheel_action(action))
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated), bool(truncated), info
    observation, reward, done, info = result
    done_code = gym_duckietown_done_code(bool(done), info)
    truncated = bool(done and done_code == MAX_STEPS_DONE_CODE)
    terminated = bool(done and not truncated)
    return observation, reward, terminated, truncated, info


def render_training_environment(env) -> None:
    raw_env = getattr(env, "unwrapped", env)
    window = getattr(raw_env, "window", None)
    if window is not None:
        from pyglet import gl

        window.switch_to()
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_LIGHT0)
        gl.glColor4ub(255, 255, 255, 255)
    env.render(mode="human")


def sample_random_action(env, rng: np.random.Generator, action_scale: float) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    action = rng.uniform(low=low, high=high).astype(np.float32)
    return np.clip(action * float(np.clip(action_scale, 0.0, 1.0)), low, high).astype(np.float32)


def reset_environment(
    env,
    args: argparse.Namespace,
    rng: np.random.Generator,
    reward_calculator: GymDuckietownRewardCalculator,
    seed: int | None = None,
    use_random_warmup: bool = True,
):
    observation, info = reset_raw(env, seed=seed)
    reward_calculator.reset(env)
    warmup_steps = max(0, args.reset_random_warmup_steps) if use_random_warmup else 0
    if warmup_steps == 0:
        return observation, info

    retries = max(1, args.reset_random_warmup_retries)
    for _ in range(retries):
        warmup_observation = observation
        warmup_info = info
        warmup_done = False
        for _ in range(warmup_steps):
            action = sample_random_action(env, rng, args.reset_random_action_scale)
            warmup_observation, reward, terminated, truncated, warmup_info = step_raw(env, action)
            done_code = gym_duckietown_done_code(bool(terminated or truncated), warmup_info)
            reward_calculator.compute(env, float(reward), done_code)
            warmup_done = bool(terminated or truncated)
            if warmup_done:
                break
        if not warmup_done:
            return warmup_observation, warmup_info
        observation, info = reset_raw(env)
        reward_calculator.reset(env)

    return observation, info


def capture_environment_start_defaults(env) -> EnvironmentStartDefaults:
    raw_env = getattr(env, "unwrapped", env)
    return EnvironmentStartDefaults(
        user_tile_start=deepcopy(getattr(raw_env, "user_tile_start", None)),
        start_pose=deepcopy(getattr(raw_env, "start_pose", None)),
    )


def apply_training_start(env, training_start: TrainingStart, defaults: EnvironmentStartDefaults) -> None:
    raw_env = getattr(env, "unwrapped", env)
    if training_start.pose is None:
        raw_env.user_tile_start = deepcopy(defaults.user_tile_start)
        raw_env.start_pose = deepcopy(defaults.start_pose)
        return

    apply_env_start_pose(env, training_start.pose)


def reset_training_environment(
    env,
    args: argparse.Namespace,
    rng: np.random.Generator,
    reward_calculator: GymDuckietownRewardCalculator,
    training_start: TrainingStart,
    defaults: EnvironmentStartDefaults,
):
    apply_training_start(env, training_start, defaults)
    observation, info = reset_environment(
        env,
        args,
        rng,
        reward_calculator,
        seed=training_start.seed,
        use_random_warmup=training_start.pose is None,
    )
    if training_start.pose is not None:
        raw_env = getattr(env, "unwrapped", env)
        if not raw_env._valid_pose(raw_env.cur_pos, raw_env.cur_angle):
            pose_label = training_start.name or "unnamed"
            raise ValueError(
                f"Training pose {pose_label!r} from {args.start_seeds_config} is not valid "
                f"on map {args.map_name!r}"
            )
    return observation, info


def done_reason(
    terminated: bool,
    truncated: bool,
    time_limit_done: bool,
    info: dict[str, Any] | None = None,
) -> str:
    if terminated:
        code = gym_duckietown_done_code(True, info)
        return code if code != "terminated" else "terminated"
    if truncated:
        code = gym_duckietown_done_code(True, info)
        return code if code == MAX_STEPS_DONE_CODE else "truncated"
    if time_limit_done:
        return "time_limit"
    return "unknown"


def write_training_episode(
    metrics_file: Path,
    metrics_fields: list[str],
    step: int,
    episode: int,
    episode_return: float,
    episode_length: int,
    reason: str,
    training_start: TrainingStart,
) -> None:
    with metrics_file.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics_fields)
        writer.writerow({
            "step": step,
            "episode": episode,
            "episode_return": episode_return,
            "episode_length": episode_length,
            "episode_return_per_step": episode_return / max(1, episode_length),
            "done_reason": reason,
            "start_type": training_start.kind,
            "start_seed": training_start.seed,
            "start_name": training_start.name,
        })


def evaluate_policy(
    env,
    policy: TanhGaussianPolicy,
    action_control: DuckietownActionControl,
    reward_calculator: GymDuckietownRewardCalculator,
    args: argparse.Namespace,
    evaluation_poses: tuple[TrainingPose, ...],
    start_defaults: EnvironmentStartDefaults,
    transform: transforms.Compose,
    device: torch.device,
) -> dict[str, Any]:
    total_return = 0.0
    total_steps = 0
    terminated_count = 0
    truncated_count = 0
    time_limit_count = 0
    scenario_results: list[dict[str, Any]] = []
    scenario_components: list[RewardComponentAccumulator] = []
    aggregate_components = RewardComponentAccumulator()

    evaluation_starts = [
        TrainingStart(kind="eval_seed", seed=seed)
        for seed in args.eval_seeds
    ]
    evaluation_starts.extend(
        TrainingStart(
            kind="eval_pose",
            seed=args.seed + 100_000 + pose_index,
            pose=pose,
        )
        for pose_index, pose in enumerate(evaluation_poses)
    )

    policy.eval()
    for scenario_index, evaluation_start in enumerate(evaluation_starts, start=1):
        apply_training_start(env, evaluation_start, start_defaults)
        scenario_seed = evaluation_start.seed
        if scenario_seed is None:
            raise RuntimeError("Evaluation scenarios must have a deterministic reset seed")
        scenario_rng = np.random.default_rng(scenario_seed)
        observation, info = reset_environment(
            env,
            args,
            scenario_rng,
            reward_calculator,
            seed=scenario_seed,
            use_random_warmup=False,
        )
        raw_env = getattr(env, "unwrapped", env)
        if evaluation_start.pose is not None and not raw_env._valid_pose(
            raw_env.cur_pos,
            raw_env.cur_angle,
        ):
            pose_label = evaluation_start.name or "unnamed"
            raise ValueError(
                f"Evaluation pose {pose_label!r} from {args.start_seeds_config} is not valid "
                f"on map {args.map_name!r}"
            )
        start_position = np.asarray(raw_env.cur_pos, dtype=np.float64).copy()
        start_angle = float(raw_env.cur_angle)
        scenario_return = 0.0
        scenario_steps = 0
        scenario_terminated = False
        scenario_truncated = False
        last_info = info
        components = RewardComponentAccumulator()

        for _ in range(args.eval_steps):
            obs_tensor = preprocess(
                observation,
                args.crop_y_start,
                args.image_size,
                args.source_observation_channel_order,
                transform,
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_controls = policy.act(
                    obs_tensor,
                    deterministic=args.eval_deterministic,
                )
                wheel_action = action_control.to_wheels_tensor(policy_controls)
                action = format_wheel_action(
                    wheel_action.squeeze(0).cpu().numpy()
                )
                observation, env_reward, terminated, truncated, last_info = step_raw(env, action)
            done_code = gym_duckietown_done_code(bool(terminated or truncated), last_info)
            breakdown = reward_calculator.compute_breakdown(env, float(env_reward), done_code)
            reward = components.add(breakdown)
            total_return += reward
            scenario_return += reward
            total_steps += 1
            scenario_steps += 1
            scenario_terminated = bool(terminated)
            scenario_truncated = bool(truncated)
            if scenario_terminated or scenario_truncated:
                break

        scenario_time_limit = not scenario_terminated and not scenario_truncated
        terminated_count += int(scenario_terminated)
        truncated_count += int(scenario_truncated)
        time_limit_count += int(scenario_time_limit)
        aggregate_components.merge(components)
        scenario_components.append(components)
        scenario_results.append({
            "scenario_index": scenario_index,
            "scenario_type": evaluation_start.kind,
            "scenario_seed": scenario_seed,
            "scenario_name": evaluation_start.name,
            "start_x": float(start_position[0]),
            "start_y": float(start_position[1]),
            "start_z": float(start_position[2]),
            "start_angle": start_angle,
            "scenario_return": scenario_return,
            "scenario_mean_reward": scenario_return / max(1, scenario_steps),
            "scenario_steps": scenario_steps,
            "terminated": int(scenario_terminated),
            "truncated": int(scenario_truncated),
            "time_limit": int(scenario_time_limit),
            "done_reason": done_reason(
                scenario_terminated,
                scenario_truncated,
                scenario_time_limit,
                last_info,
            ),
        })

    scenario_lengths = [int(result["scenario_steps"]) for result in scenario_results]
    scenario_count = len(scenario_results)

    return {
        "eval_return": total_return,
        "eval_mean_scenario_return": total_return / max(1, scenario_count),
        "eval_mean_reward": total_return / max(1, total_steps),
        "eval_steps": total_steps,
        "eval_scenarios": scenario_count,
        "eval_safe_scenarios": scenario_count - terminated_count,
        "eval_mean_scenario_length": float(np.mean(scenario_lengths)) if scenario_lengths else 0.0,
        "eval_min_scenario_length": min(scenario_lengths, default=0),
        "eval_max_scenario_length": max(scenario_lengths, default=0),
        "eval_terminated": terminated_count,
        "eval_truncated": truncated_count,
        "eval_time_limit": time_limit_count,
        "scenario_results": scenario_results,
        "scenario_components": scenario_components,
        "reward_components": aggregate_components,
    }


def save_checkpoint(path, policy, value, policy_optimizer, value_optimizer, config, step):
    action_control = DuckietownActionControl(
        mode=config.action_mode,
        fixed_throttle=config.fixed_throttle,
        max_throttle=config.max_throttle,
        max_steering=config.max_steering,
    )
    torch.save({
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "value_optimizer_state_dict": value_optimizer.state_dict(),
        "config": asdict(config),
        "env_backend": "gym-duckietown",
        "action_space": {
            "mode": action_control.mode,
            "policy_controls": action_control.control_names,
            "policy_action_dim": action_control.policy_action_dim,
            "fixed_throttle": action_control.fixed_throttle,
            "max_throttle": action_control.max_throttle,
            "max_steering": action_control.max_steering,
            "environment_actions": "left/right wheel commands clipped to [-1, 1]",
        },
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
    }, path)


def load_rl_checkpoint(path, policy, value, policy_optimizer, value_optimizer, device):
    checkpoint = torch.load(path.expanduser(), map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    value.load_state_dict(checkpoint["value_state_dict"])
    policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
    value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
    return checkpoint


def load_reference_imitation_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path.expanduser(), map_location=device)
    config = checkpoint.get("config", {})
    model = build_model(
        model_name=config.get("model", "mobilenet_v3_small"),
        pretrained=False,
        train_backbone=bool(config.get("train_backbone", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.resume_checkpoint is not None and args.imitation_checkpoint is not None:
        raise ValueError("Use either --resume-checkpoint or --imitation-checkpoint, not both.")
    if args.resume_checkpoint is not None:
        resume_preview = torch.load(
            args.resume_checkpoint.expanduser(),
            map_location="cpu",
        )
        resume_config = resume_preview.get("config", {})
        args.action_mode = resume_config.get("action_mode", "wheel")
        args.fixed_throttle = resume_config.get("fixed_throttle")
        args.max_throttle = float(resume_config.get("max_throttle", 1.0))
        args.max_steering = float(resume_config.get("max_steering", 0.5))
    action_control = DuckietownActionControl(
        mode=args.action_mode,
        fixed_throttle=args.fixed_throttle,
        max_throttle=args.max_throttle,
        max_steering=args.max_steering,
    )
    if args.imitation_checkpoint is not None and args.action_mode != "wheel":
        raise ValueError(
            "--imitation-checkpoint currently requires --action-mode wheel because "
            "the imitation head predicts left/right wheel commands."
        )
    start_config = None
    if args.start_seeds_config is not None:
        start_config = load_start_config(args.start_seeds_config, args.map_name)
        args.start_seeds_config = start_config.source_path
        args.eval_seeds = start_config.evaluation_seeds
    configure_gym_duckietown_logging(args.log_level)
    ensure_gym_duckietown_available()
    configure_gym_duckietown_logging(args.log_level)

    set_seed(args.seed)
    reset_rng = np.random.default_rng(args.seed + 1)
    start_rng = np.random.default_rng(args.seed + 2)
    device = resolve_device(args.device)
    run_dir = args.output_dir.expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S_ppo_gym_duckietown")
    run_dir.mkdir(parents=True, exist_ok=False)
    config_values = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k != "device"
    }
    config_values["training_start_seeds"] = (
        start_config.training_seeds if start_config is not None else ()
    )
    config_values["training_start_poses"] = (
        start_config.training_poses if start_config is not None else ()
    )
    config_values["evaluation_start_poses"] = (
        start_config.evaluation_poses if start_config is not None else ()
    )
    config = PPOConfig(**config_values, device=str(device))
    config_json = {
        **asdict(config),
        "env_backend": "gym-duckietown",
        "reward_metadata": {
            "name": args.reward_function,
            "source": reward_source(args.reward_function),
            "supported": REWARD_FUNCTION_CHOICES,
            "invalid_pose_penalty": (
                VELOPPOSE_INVALID_POSE_PENALTY
                if args.reward_function == "velopose"
                else 0.0
            ),
            "velocity_weight": (
                VELOPPOSE_VELOCITY_WEIGHT
                if args.reward_function == "velopose"
                else None
            ),
            "pose_weight": (
                VELOPPOSE_POSE_WEIGHT
                if args.reward_function == "velopose"
                else None
            ),
            "heading_max_correction_deg": (
                VELOPPOSE_HEADING_MAX_CORRECTION_DEG
                if args.reward_function == "velopose"
                else None
            ),
            "heading_correction_gain": (
                VELOPPOSE_HEADING_CORRECTION_GAIN
                if args.reward_function == "velopose"
                else None
            ),
        },
        "action_control": {
            "mode": action_control.mode,
            "policy_controls": action_control.control_names,
            "policy_action_dim": action_control.policy_action_dim,
            "fixed_throttle": action_control.fixed_throttle,
            "max_throttle": action_control.max_throttle,
            "max_steering": action_control.max_steering,
        },
        "curated_start_oversampling": {
            "enabled": start_config is not None,
            "source_path": (
                str(start_config.source_path) if start_config is not None else None
            ),
            "map_name": start_config.map_name if start_config is not None else args.map_name,
            "hard_start_probability": args.hard_start_probability,
            "training_seeds": (
                list(start_config.training_seeds) if start_config is not None else []
            ),
            "training_poses": (
                [pose.as_json() for pose in start_config.training_poses]
                if start_config is not None
                else []
            ),
            "evaluation_seeds": list(args.eval_seeds),
            "evaluation_poses": (
                [pose.as_json() for pose in start_config.evaluation_poses]
                if start_config is not None
                else []
            ),
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")

    transform = make_transform()
    policy = TanhGaussianPolicy(
        args.model,
        action_dim=action_control.policy_action_dim,
        pretrained=args.imitation_checkpoint is None and args.resume_checkpoint is None,
    ).to(device)
    if args.imitation_checkpoint is not None:
        load_imitation_actor(policy, args.imitation_checkpoint)
        policy.to(device)
    if args.resume_checkpoint is None:
        initial_log_std = args.initial_log_std
        if initial_log_std is None:
            initial_log_std = -2.0 if args.imitation_checkpoint is not None else -0.5
        initial_log_std = float(np.clip(initial_log_std, args.min_log_std, args.max_log_std))
        with torch.no_grad():
            policy.log_std.fill_(initial_log_std)
        print(f"Initial policy log_std={initial_log_std:.3f} std={np.exp(initial_log_std):.3f}", flush=True)
    value = ValueNetwork(args.model, pretrained=args.resume_checkpoint is None).to(device)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=args.policy_lr)
    value_optimizer = torch.optim.AdamW(value.parameters(), lr=args.value_lr)
    resumed_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = load_rl_checkpoint(args.resume_checkpoint, policy, value, policy_optimizer, value_optimizer, device)
        for parameter_group in policy_optimizer.param_groups:
            parameter_group["lr"] = args.policy_lr
        for parameter_group in value_optimizer.param_groups:
            parameter_group["lr"] = args.value_lr
        resumed_step = int(checkpoint.get("step", 0))
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_model = checkpoint_config.get("model")
        if checkpoint_model is not None and checkpoint_model != args.model:
            raise ValueError(
                f"Checkpoint model is {checkpoint_model!r}, but --model is {args.model!r}. "
                "Use the same model architecture when resuming."
            )
        with torch.no_grad():
            policy.log_std.clamp_(args.min_log_std, args.max_log_std)
        print(f"Resumed RL checkpoint {args.resume_checkpoint.expanduser()} at step={resumed_step}", flush=True)

    # PPO log-probability ratios require a deterministic network for fixed
    # observations and parameters. Eval mode freezes BatchNorm while gradients
    # through convolutional and linear parameters remain enabled.
    policy.eval()
    value.eval()

    metrics_file = run_dir / "history.csv"
    metrics_fields = [
        "step",
        "episode",
        "episode_return",
        "episode_length",
        "episode_return_per_step",
        "done_reason",
        "start_type",
        "start_seed",
        "start_name",
        "policy_loss",
        "value_loss",
        "entropy",
    ]
    with metrics_file.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=metrics_fields).writeheader()
    rollout_metrics_file = run_dir / "rollout_history.csv"
    rollout_metrics_fields = [
        "step",
        "rollout",
        "rollout_steps",
        "rollout_return",
        "rollout_reward_per_step",
        "rollout_seconds",
        "preprocess_seconds",
        "policy_value_inference_seconds",
        "env_step_seconds",
        "reward_and_reset_seconds",
        "rollout_overhead_seconds",
        "update_seconds",
        "rollout_update_seconds",
        "environment_steps_per_second",
        "cycle_steps_per_second",
        "overall_steps_per_second",
        "progress_percent",
        "elapsed_seconds",
        "eta_seconds",
        "policy_loss",
        "value_loss",
        "entropy",
    ]
    with rollout_metrics_file.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=rollout_metrics_fields).writeheader()
    ppo_diagnostics_file = run_dir / "ppo_diagnostics.csv"
    ppo_diagnostics_fields = [
        "step",
        "rollout",
        "action_mode",
        "policy_control_0_name",
        "policy_control_1_name",
        "pre_update_mean_abs_log_ratio",
        "pre_update_max_abs_log_ratio",
        "approx_kl",
        "clip_fraction",
        "ratio_mean",
        "ratio_min",
        "ratio_max",
        "log_std_left",
        "log_std_right",
        "std_left",
        "std_right",
        "sampled_policy_control_0_mean",
        "sampled_policy_control_1_mean",
        "sampled_policy_control_0_std",
        "sampled_policy_control_1_std",
        "deterministic_policy_control_0_mean",
        "deterministic_policy_control_1_mean",
        "policy_control_noise_0_std",
        "policy_control_noise_1_std",
        "sampled_action_left_mean",
        "sampled_action_right_mean",
        "sampled_action_left_std",
        "sampled_action_right_std",
        "deterministic_action_left_mean",
        "deterministic_action_right_mean",
        "action_noise_left_std",
        "action_noise_right_std",
        "action_noise_steering_std",
        "sampled_steering_std",
        "sampled_action_saturation_fraction",
        "deterministic_action_saturation_fraction",
        "sampled_policy_control_saturation_fraction",
        "squashed_entropy_estimate",
    ]
    with ppo_diagnostics_file.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=ppo_diagnostics_fields).writeheader()
    reward_components_file = run_dir / "reward_components_history.csv"
    with reward_components_file.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=REWARD_COMPONENT_FIELDS).writeheader()
    eval_metrics_file = run_dir / "eval_history.csv"
    eval_metrics_fields = [
        "train_step",
        "train_rollout",
        "eval_index",
        "eval_return",
        "eval_mean_scenario_return",
        "eval_mean_reward",
        "eval_steps",
        "eval_scenarios",
        "eval_safe_scenarios",
        "eval_mean_scenario_length",
        "eval_min_scenario_length",
        "eval_max_scenario_length",
        "eval_terminated",
        "eval_truncated",
        "eval_time_limit",
    ]
    with eval_metrics_file.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=eval_metrics_fields).writeheader()
    eval_scenarios_file = run_dir / "eval_scenarios.csv"
    eval_scenario_fields = [
        "train_step",
        "train_rollout",
        "eval_index",
        "scenario_index",
        "scenario_type",
        "scenario_seed",
        "scenario_name",
        "start_x",
        "start_y",
        "start_z",
        "start_angle",
        "scenario_return",
        "scenario_mean_reward",
        "scenario_steps",
        "terminated",
        "truncated",
        "time_limit",
        "done_reason",
    ]
    with eval_scenarios_file.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=eval_scenario_fields).writeheader()

    env = None
    eval_env = None
    eval_start_defaults = None
    reward_calculator = GymDuckietownRewardCalculator(args.reward_function)
    eval_reward_calculator = GymDuckietownRewardCalculator(args.reward_function)
    try:
        env = make_env(args, seed=args.seed)
        training_start_defaults = capture_environment_start_defaults(env)
        if args.eval_interval_rollouts > 0 and args.eval_steps > 0:
            initial_eval_seed = args.eval_seeds[0] if args.eval_seeds else args.seed + 100_000
            eval_env = make_env(args, seed=initial_eval_seed)
            eval_start_defaults = capture_environment_start_defaults(eval_env)
        print(f"Environment: gym-duckietown map={args.map_name}", flush=True)
        print(f"Reward function: {args.reward_function}", flush=True)
        print(
            f"Action control: mode={action_control.mode} "
            f"policy_controls={action_control.control_names} "
            f"fixed_throttle={action_control.fixed_throttle} "
            f"max_throttle={action_control.max_throttle:.3f} "
            f"max_steering={action_control.max_steering:.3f}",
            flush=True,
        )
        if start_config is not None:
            print(
                f"Hard starts: seeds={len(start_config.training_seeds)} "
                f"poses={len(start_config.training_poses)} "
                f"probability={args.hard_start_probability:.3f} "
                f"config={start_config.source_path}",
                flush=True,
            )
        evaluation_pose_count = (
            len(start_config.evaluation_poses) if start_config is not None else 0
        )
        print(
            f"Evaluation scenarios: seeds={len(args.eval_seeds)} "
            f"poses={evaluation_pose_count}",
            flush=True,
        )
        if start_config is None:
            training_start = TrainingStart(kind="random", seed=args.seed)
        else:
            training_start = choose_training_start(
                start_config,
                args.hard_start_probability,
                start_rng,
            )
        observation, info = reset_training_environment(
            env,
            args,
            reset_rng,
            reward_calculator,
            training_start,
            training_start_defaults,
        )
        if args.render_training:
            render_training_environment(env)
        if args.debug_initial_action:
            with torch.no_grad():
                obs_tensor = preprocess(
                    observation,
                    args.crop_y_start,
                    args.image_size,
                    args.source_observation_channel_order,
                    transform,
                ).unsqueeze(0).to(device)
                mean, log_std = policy(obs_tensor)
                deterministic_controls_tensor = torch.tanh(mean)
                deterministic_controls = (
                    deterministic_controls_tensor.squeeze(0).cpu().numpy()
                )
                deterministic_action = (
                    action_control.to_wheels_tensor(deterministic_controls_tensor)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                raw_mean = mean.squeeze(0).cpu().numpy()
                std = log_std.exp().squeeze(0).cpu().numpy()
                reference_action = None
                if args.imitation_checkpoint is not None:
                    reference_model = load_reference_imitation_model(args.imitation_checkpoint, device)
                    reference_action = reference_model(obs_tensor).squeeze(0).cpu().numpy()
            print(
                "debug_initial_action "
                f"action_mode={action_control.mode} "
                f"controls={np.array2string(deterministic_controls, precision=4)} "
                f"raw_mean={np.array2string(raw_mean, precision=4)} "
                f"action_left={deterministic_action[0]:+.4f} action_right={deterministic_action[1]:+.4f} "
                f"std={np.array2string(std, precision=4)}",
                flush=True,
            )
            if reference_action is not None:
                print(
                    "debug_initial_action_compare "
                    f"il_left={reference_action[0]:+.4f} il_right={reference_action[1]:+.4f} "
                    f"rl_raw_minus_il_left={raw_mean[0] - reference_action[0]:+.6f} "
                    f"rl_raw_minus_il_right={raw_mean[1] - reference_action[1]:+.6f} "
                    f"rl_action_minus_il_left={deterministic_action[0] - reference_action[0]:+.6f} "
                    f"rl_action_minus_il_right={deterministic_action[1] - reference_action[1]:+.6f}",
                    flush=True,
                )

        episode_return = 0.0
        episode_length = 0
        episode = 0
        global_step = resumed_step
        rollout = 0
        eval_index = 0
        best_eval_return = -float("inf")
        best_safe_score: tuple[int, float, float] | None = None
        training_started_at = perf_counter()
        training_start_step = global_step

        while global_step < args.total_steps:
            synchronize_device(device)
            rollout_started_at = perf_counter()
            obs_buf, action_buf, raw_action_buf = [], [], []
            wheel_action_buf, deterministic_action_buf = [], []
            deterministic_wheel_action_buf = []
            logp_buf, reward_buf, done_buf, value_buf = [], [], [], []
            rollout_reward_components = RewardComponentAccumulator()
            preprocess_seconds = 0.0
            policy_value_inference_seconds = 0.0
            env_step_seconds = 0.0
            reward_and_reset_seconds = 0.0
            for _ in range(args.rollout_steps):
                phase_started_at = perf_counter()
                obs_tensor = preprocess(
                    observation,
                    args.crop_y_start,
                    args.image_size,
                    args.source_observation_channel_order,
                    transform,
                ).to(device)
                preprocess_seconds += perf_counter() - phase_started_at

                phase_started_at = perf_counter()
                with torch.no_grad():
                    action_tensor, raw_action_tensor, logp_tensor, deterministic_action_tensor = (
                        policy.sample_with_raw(obs_tensor.unsqueeze(0))
                    )
                    wheel_action_tensor = action_control.to_wheels_tensor(action_tensor)
                    deterministic_wheel_action_tensor = action_control.to_wheels_tensor(
                        deterministic_action_tensor
                    )
                    value_tensor = value(obs_tensor.unsqueeze(0))
                action = format_wheel_action(
                    wheel_action_tensor.squeeze(0).cpu().numpy()
                )
                policy_value_inference_seconds += perf_counter() - phase_started_at

                phase_started_at = perf_counter()
                next_observation, reward, terminated, truncated, info = step_raw(env, action)
                env_step_seconds += perf_counter() - phase_started_at
                if args.render_training:
                    render_training_environment(env)

                phase_started_at = perf_counter()
                done_code = gym_duckietown_done_code(bool(terminated or truncated), info)
                reward_breakdown = reward_calculator.compute_breakdown(env, float(reward), done_code)
                reward = rollout_reward_components.add(reward_breakdown)
                reward_and_reset_seconds += perf_counter() - phase_started_at

                obs_buf.append(obs_tensor.cpu())
                action_buf.append(action_tensor.squeeze(0).cpu())
                raw_action_buf.append(raw_action_tensor.squeeze(0).cpu())
                deterministic_action_buf.append(deterministic_action_tensor.squeeze(0).cpu())
                wheel_action_buf.append(wheel_action_tensor.squeeze(0).cpu())
                deterministic_wheel_action_buf.append(
                    deterministic_wheel_action_tensor.squeeze(0).cpu()
                )
                logp_buf.append(logp_tensor.squeeze(0).cpu())
                value_buf.append(value_tensor.squeeze(0).cpu())
                episode_return += reward
                episode_length += 1
                global_step += 1
                env_done = bool(terminated or truncated)
                time_limit_done = args.max_episode_steps > 0 and episode_length >= args.max_episode_steps
                done = env_done or time_limit_done
                reward_buf.append(reward)
                done_buf.append(float(done))
                observation = next_observation

                if done:
                    reason = done_reason(terminated, truncated, time_limit_done, info)
                    write_training_episode(
                        metrics_file,
                        metrics_fields,
                        global_step,
                        episode,
                        episode_return,
                        episode_length,
                        reason,
                        training_start,
                    )
                    print(
                        f"step={global_step} episode={episode} return={episode_return:.3f} "
                        f"length={episode_length} reward_per_step={episode_return / max(1, episode_length):.4f} "
                        f"done_reason={reason} start_type={training_start.kind} "
                        f"start_seed={training_start.seed if training_start.seed is not None else 'continued_rng'} "
                        f"start_name={training_start.name or '-'}",
                        flush=True,
                    )
                    phase_started_at = perf_counter()
                    training_start = choose_training_start(
                        start_config,
                        args.hard_start_probability,
                        start_rng,
                    )
                    observation, info = reset_training_environment(
                        env,
                        args,
                        reset_rng,
                        reward_calculator,
                        training_start,
                        training_start_defaults,
                    )
                    if args.render_training:
                        render_training_environment(env)
                    reward_and_reset_seconds += perf_counter() - phase_started_at
                    episode += 1
                    episode_return = 0.0
                    episode_length = 0
                if global_step >= args.total_steps:
                    break

            synchronize_device(device)
            rollout_finished_at = perf_counter()
            rollout_seconds = rollout_finished_at - rollout_started_at
            measured_rollout_seconds = (
                preprocess_seconds
                + policy_value_inference_seconds
                + env_step_seconds
                + reward_and_reset_seconds
            )
            rollout_overhead_seconds = max(0.0, rollout_seconds - measured_rollout_seconds)
            update_started_at = rollout_finished_at

            obs = torch.stack(obs_buf).to(device)
            actions = torch.stack(action_buf).to(device)
            wheel_actions = torch.stack(wheel_action_buf).to(device)
            raw_actions = torch.stack(raw_action_buf).to(device)
            deterministic_actions = torch.stack(deterministic_action_buf).to(device)
            deterministic_wheel_actions = torch.stack(
                deterministic_wheel_action_buf
            ).to(device)
            old_logp = torch.stack(logp_buf).to(device)
            rewards = torch.tensor(reward_buf, dtype=torch.float32, device=device)
            dones = torch.tensor(done_buf, dtype=torch.float32, device=device)
            values = torch.stack(value_buf).to(device)
            with torch.no_grad():
                last_obs = preprocess(
                    observation,
                    args.crop_y_start,
                    args.image_size,
                    args.source_observation_channel_order,
                    transform,
                ).unsqueeze(0).to(device)
                last_value = value(last_obs).squeeze(0)
                advantages, returns = compute_gae(rewards, dones, values, last_value, args.gamma, args.gae_lambda)
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

                pre_update_distribution = policy.distribution(obs)
                pre_update_actions = torch.tanh(raw_actions)
                pre_update_logp = tanh_normal_log_prob(
                    pre_update_distribution,
                    raw_actions,
                    pre_update_actions,
                )
                pre_update_log_ratio = pre_update_logp - old_logp
                pre_update_mean_abs_log_ratio = pre_update_log_ratio.abs().mean().item()
                pre_update_max_abs_log_ratio = pre_update_log_ratio.abs().max().item()

            last_policy_loss = last_value_loss = last_entropy = 0.0
            indices = torch.arange(obs.size(0), device=device)
            for _ in range(args.epochs):
                permutation = indices[torch.randperm(obs.size(0), device=device)]
                for start in range(0, obs.size(0), args.batch_size):
                    batch = permutation[start:start + args.batch_size]
                    distribution = policy.distribution(obs[batch])
                    batch_raw_actions = raw_actions[batch]
                    batch_actions = torch.tanh(batch_raw_actions)
                    logp = tanh_normal_log_prob(distribution, batch_raw_actions, batch_actions)
                    ratio = torch.exp(logp - old_logp[batch])
                    clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * advantages[batch]
                    policy_loss = -torch.min(ratio * advantages[batch], clipped).mean()
                    entropy = distribution.entropy().sum(dim=1).mean()
                    values_pred = value(obs[batch])
                    value_loss = nn.functional.mse_loss(values_pred, returns[batch])

                    policy_optimizer.zero_grad(set_to_none=True)
                    (policy_loss - args.entropy_coef * entropy).backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    policy_optimizer.step()
                    with torch.no_grad():
                        policy.log_std.clamp_(args.min_log_std, args.max_log_std)

                    value_optimizer.zero_grad(set_to_none=True)
                    (args.value_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(value.parameters(), args.max_grad_norm)
                    value_optimizer.step()
                    last_policy_loss, last_value_loss, last_entropy = policy_loss.item(), value_loss.item(), entropy.item()

            with torch.no_grad():
                final_distribution = policy.distribution(obs)
                final_actions = torch.tanh(raw_actions)
                final_logp = tanh_normal_log_prob(final_distribution, raw_actions, final_actions)
                final_log_ratio = final_logp - old_logp
                final_ratio = final_log_ratio.exp()
                approx_kl = ((final_ratio - 1.0) - final_log_ratio).mean().item()
                clip_fraction = ((final_ratio - 1.0).abs() > args.clip_ratio).float().mean().item()
                policy_control_noise = actions - deterministic_actions
                action_noise = wheel_actions - deterministic_wheel_actions
                sampled_steering = 0.5 * (
                    wheel_actions[:, 1] - wheel_actions[:, 0]
                )
                action_noise_steering = 0.5 * (
                    action_noise[:, 1] - action_noise[:, 0]
                )
                effective_log_std = policy.log_std.clamp(args.min_log_std, args.max_log_std)
                effective_std = effective_log_std.exp()

                def vector_item(values: torch.Tensor, index: int):
                    return values[index].item() if values.numel() > index else None

                def column_mean(values: torch.Tensor, index: int):
                    return values[:, index].mean().item() if values.shape[1] > index else None

                def column_std(values: torch.Tensor, index: int):
                    if values.shape[1] <= index:
                        return None
                    return values[:, index].std(unbiased=False).item()

                ppo_diagnostics = {
                    "action_mode": action_control.mode,
                    "policy_control_0_name": action_control.control_names[0],
                    "policy_control_1_name": (
                        action_control.control_names[1]
                        if len(action_control.control_names) > 1
                        else None
                    ),
                    "pre_update_mean_abs_log_ratio": pre_update_mean_abs_log_ratio,
                    "pre_update_max_abs_log_ratio": pre_update_max_abs_log_ratio,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "ratio_mean": final_ratio.mean().item(),
                    "ratio_min": final_ratio.min().item(),
                    "ratio_max": final_ratio.max().item(),
                    "log_std_left": vector_item(effective_log_std, 0),
                    "log_std_right": vector_item(effective_log_std, 1),
                    "std_left": vector_item(effective_std, 0),
                    "std_right": vector_item(effective_std, 1),
                    "sampled_policy_control_0_mean": column_mean(actions, 0),
                    "sampled_policy_control_1_mean": column_mean(actions, 1),
                    "sampled_policy_control_0_std": column_std(actions, 0),
                    "sampled_policy_control_1_std": column_std(actions, 1),
                    "deterministic_policy_control_0_mean": column_mean(
                        deterministic_actions,
                        0,
                    ),
                    "deterministic_policy_control_1_mean": column_mean(
                        deterministic_actions,
                        1,
                    ),
                    "policy_control_noise_0_std": column_std(
                        policy_control_noise,
                        0,
                    ),
                    "policy_control_noise_1_std": column_std(
                        policy_control_noise,
                        1,
                    ),
                    "sampled_action_left_mean": wheel_actions[:, 0].mean().item(),
                    "sampled_action_right_mean": wheel_actions[:, 1].mean().item(),
                    "sampled_action_left_std": wheel_actions[:, 0].std(unbiased=False).item(),
                    "sampled_action_right_std": wheel_actions[:, 1].std(unbiased=False).item(),
                    "deterministic_action_left_mean": deterministic_wheel_actions[:, 0].mean().item(),
                    "deterministic_action_right_mean": deterministic_wheel_actions[:, 1].mean().item(),
                    "action_noise_left_std": action_noise[:, 0].std(unbiased=False).item(),
                    "action_noise_right_std": action_noise[:, 1].std(unbiased=False).item(),
                    "action_noise_steering_std": action_noise_steering.std(unbiased=False).item(),
                    "sampled_steering_std": sampled_steering.std(unbiased=False).item(),
                    "sampled_action_saturation_fraction": (
                        wheel_actions.abs() > 0.95
                    ).float().mean().item(),
                    "deterministic_action_saturation_fraction": (
                        deterministic_wheel_actions.abs() > 0.95
                    ).float().mean().item(),
                    "sampled_policy_control_saturation_fraction": (
                        actions.abs() > 0.95
                    ).float().mean().item(),
                    "squashed_entropy_estimate": (-old_logp).mean().item(),
                }

            synchronize_device(device)
            update_finished_at = perf_counter()
            update_seconds = update_finished_at - update_started_at
            rollout_update_seconds = update_finished_at - rollout_started_at
            rollout_step_count = len(reward_buf)
            environment_steps_per_second = rollout_step_count / max(rollout_seconds, 1e-12)
            rollout_return = sum(reward_buf)
            rollout_reward_per_step = rollout_return / max(1, rollout_step_count)
            cycle_steps_per_second = rollout_step_count / max(rollout_update_seconds, 1e-12)

            save_checkpoint(run_dir / "last.pt", policy, value, policy_optimizer, value_optimizer, config, global_step)
            metrics_recorded_at = perf_counter()
            elapsed_seconds = metrics_recorded_at - training_started_at
            completed_training_steps = global_step - training_start_step
            overall_steps_per_second = completed_training_steps / max(elapsed_seconds, 1e-12)
            progress_percent = 100.0 * global_step / max(1, args.total_steps)
            remaining_steps = max(0, args.total_steps - global_step)
            eta_seconds = remaining_steps / max(overall_steps_per_second, 1e-12)
            rollout += 1
            with rollout_metrics_file.open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rollout_metrics_fields)
                writer.writerow({
                    "step": global_step,
                    "rollout": rollout,
                    "rollout_steps": rollout_step_count,
                    "rollout_return": rollout_return,
                    "rollout_reward_per_step": rollout_reward_per_step,
                    "rollout_seconds": rollout_seconds,
                    "preprocess_seconds": preprocess_seconds,
                    "policy_value_inference_seconds": policy_value_inference_seconds,
                    "env_step_seconds": env_step_seconds,
                    "reward_and_reset_seconds": reward_and_reset_seconds,
                    "rollout_overhead_seconds": rollout_overhead_seconds,
                    "update_seconds": update_seconds,
                    "rollout_update_seconds": rollout_update_seconds,
                    "environment_steps_per_second": environment_steps_per_second,
                    "cycle_steps_per_second": cycle_steps_per_second,
                    "overall_steps_per_second": overall_steps_per_second,
                    "progress_percent": progress_percent,
                    "elapsed_seconds": elapsed_seconds,
                    "eta_seconds": eta_seconds,
                    "policy_loss": last_policy_loss,
                    "value_loss": last_value_loss,
                    "entropy": last_entropy,
                })
            with ppo_diagnostics_file.open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=ppo_diagnostics_fields)
                writer.writerow({
                    "step": global_step,
                    "rollout": rollout,
                    **ppo_diagnostics,
                })
            write_reward_components(
                reward_components_file,
                rollout_reward_components,
                phase="train_rollout",
                train_step=global_step,
                train_rollout=rollout,
            )
            print(
                f"update step={global_step} rollout={rollout} rollout_return={rollout_return:.3f} "
                f"rollout_reward_per_step={rollout_reward_per_step:.4f} "
                f"current_episode_return={episode_return:.3f} current_episode_length={episode_length} "
                f"policy_loss={last_policy_loss:.4f} value_loss={last_value_loss:.4f} entropy={last_entropy:.4f} "
                f"rollout_seconds={rollout_seconds:.3f} update_seconds={update_seconds:.3f} "
                f"rollout_update_seconds={rollout_update_seconds:.3f} "
                f"preprocess_seconds={preprocess_seconds:.3f} "
                f"policy_value_inference_seconds={policy_value_inference_seconds:.3f} "
                f"env_step_seconds={env_step_seconds:.3f} "
                f"reward_and_reset_seconds={reward_and_reset_seconds:.3f} "
                f"rollout_overhead_seconds={rollout_overhead_seconds:.3f} "
                f"environment_steps_per_second={environment_steps_per_second:.2f} "
                f"cycle_steps_per_second={cycle_steps_per_second:.2f} "
                f"overall_steps_per_second={overall_steps_per_second:.2f} "
                f"progress={progress_percent:.2f}% elapsed={format_duration(elapsed_seconds)} "
                f"eta={format_duration(eta_seconds)}",
                flush=True,
            )

            should_eval = (
                args.eval_interval_rollouts > 0
                and args.eval_steps > 0
                and rollout % args.eval_interval_rollouts == 0
            )
            if should_eval:
                if eval_env is None:
                    raise RuntimeError("Evaluation environment was not initialized")
                if eval_start_defaults is None:
                    raise RuntimeError("Evaluation environment defaults were not captured")
                eval_result = evaluate_policy(
                    eval_env,
                    policy,
                    action_control,
                    eval_reward_calculator,
                    args,
                    start_config.evaluation_poses if start_config is not None else (),
                    eval_start_defaults,
                    transform,
                    device,
                )
                eval_index += 1
                eval_row = {
                    "train_step": global_step,
                    "train_rollout": rollout,
                    "eval_index": eval_index,
                    **{field: eval_result[field] for field in eval_metrics_fields if field in eval_result},
                }
                with eval_metrics_file.open("a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=eval_metrics_fields)
                    writer.writerow(eval_row)
                with eval_scenarios_file.open("a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=eval_scenario_fields)
                    for scenario_result in eval_result["scenario_results"]:
                        writer.writerow({
                            "train_step": global_step,
                            "train_rollout": rollout,
                            "eval_index": eval_index,
                            **scenario_result,
                        })
                write_reward_components(
                    reward_components_file,
                    eval_result["reward_components"],
                    phase="eval",
                    train_step=global_step,
                    train_rollout=rollout,
                    eval_index=eval_index,
                )
                for scenario_result, scenario_component_values in zip(
                    eval_result["scenario_results"],
                    eval_result["scenario_components"],
                ):
                    write_reward_components(
                        reward_components_file,
                        scenario_component_values,
                        phase="eval_scenario",
                        train_step=global_step,
                        train_rollout=rollout,
                        eval_index=eval_index,
                        scenario_index=int(scenario_result["scenario_index"]),
                        scenario_seed=int(scenario_result["scenario_seed"]),
                    )
                print(
                    f"eval step={global_step} rollout={rollout} eval_index={eval_index} "
                    f"return={eval_result['eval_return']:.3f} "
                    f"mean_scenario_return={eval_result['eval_mean_scenario_return']:.3f} "
                    f"steps={eval_result['eval_steps']} scenarios={eval_result['eval_scenarios']} "
                    f"mean_reward={eval_result['eval_mean_reward']:.4f} "
                    f"safe_scenarios={eval_result['eval_safe_scenarios']} "
                    f"terminated={eval_result['eval_terminated']} "
                    f"mean_scenario_length={eval_result['eval_mean_scenario_length']:.1f}",
                    flush=True,
                )
                eval_checkpoint = (
                    run_dir / f"eval_{eval_index:04d}_step_{global_step:010d}.pt"
                )
                save_checkpoint(
                    eval_checkpoint,
                    policy,
                    value,
                    policy_optimizer,
                    value_optimizer,
                    config,
                    global_step,
                )
                print(f"eval_checkpoint step={global_step} checkpoint={eval_checkpoint}", flush=True)

                mean_scenario_return = float(eval_result["eval_mean_scenario_return"])
                if mean_scenario_return > best_eval_return:
                    best_eval_return = mean_scenario_return
                    save_checkpoint(
                        run_dir / "best_return.pt",
                        policy,
                        value,
                        policy_optimizer,
                        value_optimizer,
                        config,
                        global_step,
                    )
                    print(
                        f"new_best_return step={global_step} mean_scenario_return={best_eval_return:.3f} "
                        f"checkpoint={run_dir / 'best_return.pt'}",
                        flush=True,
                    )
                safe_score = (
                    int(eval_result["eval_safe_scenarios"]),
                    float(eval_result["eval_mean_scenario_length"]),
                    mean_scenario_return,
                )
                if best_safe_score is None or safe_score > best_safe_score:
                    best_safe_score = safe_score
                    save_checkpoint(
                        run_dir / "best_safe.pt",
                        policy,
                        value,
                        policy_optimizer,
                        value_optimizer,
                        config,
                        global_step,
                    )
                    print(
                        f"new_best_safe step={global_step} safe_scenarios={safe_score[0]} "
                        f"mean_scenario_length={safe_score[1]:.1f} "
                        f"mean_scenario_return={safe_score[2]:.3f} "
                        f"checkpoint={run_dir / 'best_safe.pt'}",
                        flush=True,
                    )
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
