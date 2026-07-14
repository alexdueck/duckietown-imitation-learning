#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Train a PPO policy from camera images in gym-duckietown.

This trainer intentionally lives beside the Duckiematrix trainer instead of
adding another backend to it. The policy still outputs direct continuous
left/right wheel commands in [-1, 1], matching gym-duckietown's Simulator API.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

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
from duckietown_paths import RL_PPO_GYM_DUCKIETOWN_CHECKPOINT_DIR
from rl_models import TanhGaussianPolicy, load_imitation_actor
from train_imitation_learning import IMAGENET_MEAN, IMAGENET_STD, build_model, resolve_device, set_seed
from velopose_reward import VELOPPOSE_INVALID_POSE_PENALTY


@dataclass
class PPOConfig:
    output_dir: str
    map_name: str
    reward_function: str
    model: str
    imitation_checkpoint: str | None
    resume_checkpoint: str | None
    total_steps: int
    max_episode_steps: int
    reset_random_warmup_steps: int
    reset_random_warmup_retries: int
    reset_random_action_scale: float
    eval_interval_rollouts: int
    eval_steps: int
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
    log_level: str
    debug_initial_action: bool
    seed: int
    device: str


class ValueNetwork(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = False) -> None:
        super().__init__()
        from rl_models import build_encoder

        self.encoder, features_dim = build_encoder(model_name, pretrained=pretrained)
        self.value = nn.Linear(features_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value(self.encoder(observations)).squeeze(1)


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
        help="Resume PPO training from an RL checkpoint such as last.pt or best_eval.pt.",
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
    parser.add_argument("--eval-interval-rollouts", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
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
    parser.add_argument("--policy-lr", type=float, default=3e-4)
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
        })


def evaluate_policy(
    env,
    policy: TanhGaussianPolicy,
    reward_calculator: GymDuckietownRewardCalculator,
    args: argparse.Namespace,
    transform: transforms.Compose,
    device: torch.device,
    reset_rng: np.random.Generator,
) -> dict[str, float | int]:
    observation, info = reset_environment(env, args, reset_rng, reward_calculator, use_random_warmup=False)
    total_return = 0.0
    episode_return = 0.0
    episode_length = 0
    completed_episodes = 0
    terminated_count = 0
    truncated_count = 0
    time_limit_count = 0

    was_training = policy.training
    policy.eval()
    try:
        for _ in range(args.eval_steps):
            obs_tensor = preprocess(
                observation,
                args.crop_y_start,
                args.image_size,
                args.source_observation_channel_order,
                transform,
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                action_tensor = policy.act(obs_tensor, deterministic=args.eval_deterministic)
                action = format_wheel_action(action_tensor.squeeze(0).cpu().numpy())
                observation, reward, terminated, truncated, info = step_raw(env, action)
            done_code = gym_duckietown_done_code(bool(terminated or truncated), info)
            reward = reward_calculator.compute(env, float(reward), done_code)
            total_return += reward
            episode_return += reward
            episode_length += 1

            time_limit_done = args.max_episode_steps > 0 and episode_length >= args.max_episode_steps
            done = bool(terminated or truncated or time_limit_done)
            if done:
                completed_episodes += 1
                terminated_count += int(bool(terminated))
                truncated_count += int(bool(truncated))
                time_limit_count += int(time_limit_done and not terminated and not truncated)
                observation, info = reset_environment(env, args, reset_rng, reward_calculator, use_random_warmup=False)
                episode_return = 0.0
                episode_length = 0
    finally:
        if was_training:
            policy.train()

    return {
        "eval_return": total_return,
        "eval_mean_reward": total_return / max(1, args.eval_steps),
        "eval_steps": args.eval_steps,
        "eval_completed_episodes": completed_episodes,
        "eval_open_episode_return": episode_return,
        "eval_open_episode_length": episode_length,
        "eval_terminated": terminated_count,
        "eval_truncated": truncated_count,
        "eval_time_limit": time_limit_count,
    }


def save_checkpoint(path, policy, value, policy_optimizer, value_optimizer, config, step):
    torch.save({
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "value_optimizer_state_dict": value_optimizer.state_dict(),
        "config": asdict(config),
        "env_backend": "gym-duckietown",
        "action_space": "continuous gym-duckietown left/right wheel commands in [-1, 1]",
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
    configure_gym_duckietown_logging(args.log_level)
    ensure_gym_duckietown_available()
    configure_gym_duckietown_logging(args.log_level)

    set_seed(args.seed)
    reset_rng = np.random.default_rng(args.seed + 1)
    device = resolve_device(args.device)
    run_dir = args.output_dir.expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S_ppo_gym_duckietown")
    run_dir.mkdir(parents=True, exist_ok=False)
    config_values = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k != "device"
    }
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
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")

    transform = make_transform()
    policy = TanhGaussianPolicy(args.model, pretrained=args.imitation_checkpoint is None and args.resume_checkpoint is None).to(device)
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

    metrics_file = run_dir / "history.csv"
    metrics_fields = [
        "step",
        "episode",
        "episode_return",
        "episode_length",
        "episode_return_per_step",
        "done_reason",
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
    eval_metrics_file = run_dir / "eval_history.csv"
    eval_metrics_fields = [
        "train_step",
        "train_rollout",
        "eval_index",
        "eval_return",
        "eval_mean_reward",
        "eval_steps",
        "eval_completed_episodes",
        "eval_open_episode_return",
        "eval_open_episode_length",
        "eval_terminated",
        "eval_truncated",
        "eval_time_limit",
    ]
    with eval_metrics_file.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=eval_metrics_fields).writeheader()

    env = None
    reward_calculator = GymDuckietownRewardCalculator(args.reward_function)
    try:
        env = make_env(args, seed=args.seed)
        print(f"Environment: gym-duckietown map={args.map_name}", flush=True)
        print(f"Reward function: {args.reward_function}", flush=True)
        observation, info = reset_environment(env, args, reset_rng, reward_calculator, seed=args.seed)
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
                deterministic_action = torch.tanh(mean).squeeze(0).cpu().numpy()
                raw_mean = mean.squeeze(0).cpu().numpy()
                std = log_std.exp().squeeze(0).cpu().numpy()
                reference_action = None
                if args.imitation_checkpoint is not None:
                    reference_model = load_reference_imitation_model(args.imitation_checkpoint, device)
                    reference_action = reference_model(obs_tensor).squeeze(0).cpu().numpy()
            print(
                "debug_initial_action "
                f"raw_mean_left={raw_mean[0]:+.4f} raw_mean_right={raw_mean[1]:+.4f} "
                f"action_left={deterministic_action[0]:+.4f} action_right={deterministic_action[1]:+.4f} "
                f"std_left={std[0]:.4f} std_right={std[1]:.4f}",
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
        training_started_at = perf_counter()
        training_start_step = global_step

        while global_step < args.total_steps:
            synchronize_device(device)
            rollout_started_at = perf_counter()
            obs_buf, action_buf, logp_buf, reward_buf, done_buf, value_buf = [], [], [], [], [], []
            for _ in range(args.rollout_steps):
                obs_tensor = preprocess(
                    observation,
                    args.crop_y_start,
                    args.image_size,
                    args.source_observation_channel_order,
                    transform,
                ).to(device)
                with torch.no_grad():
                    action_tensor, logp_tensor, _ = policy.sample(obs_tensor.unsqueeze(0))
                    value_tensor = value(obs_tensor.unsqueeze(0))
                action = format_wheel_action(action_tensor.squeeze(0).cpu().numpy())
                next_observation, reward, terminated, truncated, info = step_raw(env, action)
                done_code = gym_duckietown_done_code(bool(terminated or truncated), info)
                reward = reward_calculator.compute(env, float(reward), done_code)

                obs_buf.append(obs_tensor.cpu())
                action_buf.append(torch.from_numpy(action))
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
                    )
                    print(
                        f"step={global_step} episode={episode} return={episode_return:.3f} "
                        f"length={episode_length} reward_per_step={episode_return / max(1, episode_length):.4f} "
                        f"done_reason={reason}",
                        flush=True,
                    )
                    observation, info = reset_environment(env, args, reset_rng, reward_calculator)
                    episode += 1
                    episode_return = 0.0
                    episode_length = 0
                if global_step >= args.total_steps:
                    break

            synchronize_device(device)
            rollout_finished_at = perf_counter()
            rollout_seconds = rollout_finished_at - rollout_started_at
            update_started_at = rollout_finished_at

            obs = torch.stack(obs_buf).to(device)
            actions = torch.stack(action_buf).to(device)
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

            last_policy_loss = last_value_loss = last_entropy = 0.0
            indices = torch.arange(obs.size(0), device=device)
            for _ in range(args.epochs):
                permutation = indices[torch.randperm(obs.size(0), device=device)]
                for start in range(0, obs.size(0), args.batch_size):
                    batch = permutation[start:start + args.batch_size]
                    distribution = policy.distribution(obs[batch])
                    raw_actions = torch.atanh(actions[batch].clamp(-0.999, 0.999))
                    logp = (distribution.log_prob(raw_actions) - torch.log(1.0 - actions[batch].pow(2) + 1e-6)).sum(dim=1)
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
            print(
                f"update step={global_step} rollout={rollout} rollout_return={rollout_return:.3f} "
                f"rollout_reward_per_step={rollout_reward_per_step:.4f} "
                f"current_episode_return={episode_return:.3f} current_episode_length={episode_length} "
                f"policy_loss={last_policy_loss:.4f} value_loss={last_value_loss:.4f} entropy={last_entropy:.4f} "
                f"rollout_seconds={rollout_seconds:.3f} update_seconds={update_seconds:.3f} "
                f"rollout_update_seconds={rollout_update_seconds:.3f} "
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
                if episode_length > 0:
                    write_training_episode(
                        metrics_file,
                        metrics_fields,
                        global_step,
                        episode,
                        episode_return,
                        episode_length,
                        "eval_interrupt",
                    )
                    print(
                        f"step={global_step} episode={episode} return={episode_return:.3f} "
                        f"length={episode_length} reward_per_step={episode_return / max(1, episode_length):.4f} "
                        f"done_reason=eval_interrupt",
                        flush=True,
                    )
                    episode += 1
                    episode_return = 0.0
                    episode_length = 0

                eval_result = evaluate_policy(env, policy, reward_calculator, args, transform, device, reset_rng)
                eval_index += 1
                eval_row = {
                    "train_step": global_step,
                    "train_rollout": rollout,
                    "eval_index": eval_index,
                    **eval_result,
                }
                with eval_metrics_file.open("a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=eval_metrics_fields)
                    writer.writerow(eval_row)
                print(
                    f"eval step={global_step} rollout={rollout} eval_index={eval_index} "
                    f"return={eval_result['eval_return']:.3f} steps={eval_result['eval_steps']} "
                    f"mean_reward={eval_result['eval_mean_reward']:.4f} "
                    f"completed_episodes={eval_result['eval_completed_episodes']} "
                    f"open_episode_return={eval_result['eval_open_episode_return']:.3f} "
                    f"open_episode_length={eval_result['eval_open_episode_length']}",
                    flush=True,
                )
                if float(eval_result["eval_return"]) > best_eval_return:
                    best_eval_return = float(eval_result["eval_return"])
                    save_checkpoint(
                        run_dir / "best_eval.pt",
                        policy,
                        value,
                        policy_optimizer,
                        value_optimizer,
                        config,
                        global_step,
                    )
                    print(
                        f"new_best_eval step={global_step} return={best_eval_return:.3f} "
                        f"checkpoint={run_dir / 'best_eval.pt'}",
                        flush=True,
                    )
                observation, info = reset_environment(env, args, reset_rng, reward_calculator)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
