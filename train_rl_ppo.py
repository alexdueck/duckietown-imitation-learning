#!/usr/bin/env python3
"""Train a PPO policy from camera images in gym-duckiematrix.

Recommended first RL approach: on-policy PPO with a squashed Gaussian actor. It is
stable for continuous left/right wheel commands, simple to debug in simulation,
and can optionally warm-start the actor from an imitation-learning checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from duckietown.sdk.middleware.dtps.base import DTPS
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv

from live_eval_imitation_policy import observation_to_rgb, shutdown_dtps
from rl_models import TanhGaussianPolicy, load_imitation_actor
from train_imitation_learning import IMAGENET_MEAN, IMAGENET_STD, resolve_device, set_seed


@dataclass
class PPOConfig:
    output_dir: str
    map_name: str | None
    entity_name: str
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
    parser = argparse.ArgumentParser(description="Train PPO in Duckiematrix.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/rl_ppo"))
    parser.add_argument("--entity-name", default="map_0/vehicle_0")
    parser.add_argument("--map-name", default=None, help="Recorded in config; select the map in Duckiematrix itself.")
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
    parser.add_argument(
        "--reset-random-warmup-steps",
        type=int,
        default=0,
        help="After each reset, take this many random untrained actions before starting the next PPO episode.",
    )
    parser.add_argument(
        "--reset-random-warmup-retries",
        type=int,
        default=3,
        help="Retry random reset warmup if the environment terminates during the warmup.",
    )
    parser.add_argument(
        "--reset-random-action-scale",
        type=float,
        default=0.6,
        help="Scale for random warmup actions in the continuous left/right action space.",
    )
    parser.add_argument(
        "--eval-interval-rollouts",
        type=int,
        default=10,
        help="Run evaluation after this many completed PPO rollouts/updates; set to 0 to disable.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Number of environment steps to run for each evaluation.",
    )
    parser.add_argument(
        "--eval-stochastic",
        action="store_false",
        dest="eval_deterministic",
        help="Sample from the policy during evaluation instead of using the deterministic mean action.",
    )
    parser.add_argument(
        "--initial-log-std",
        type=float,
        default=None,
        help="Initial actor log standard deviation. Defaults to -2.0 for imitation warm-starts and -0.5 otherwise.",
    )
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
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def make_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def preprocess(observation: np.ndarray, crop_y_start: int, image_size: int, transform: transforms.Compose) -> torch.Tensor:
    image = Image.fromarray(observation_to_rgb(observation)).convert("RGB")
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


def sample_random_action(env, rng: np.random.Generator, action_scale: float) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    action_scale = float(np.clip(action_scale, 0.0, 1.0))
    action = rng.uniform(low=low, high=high).astype(np.float32)
    return np.clip(action * action_scale, low, high).astype(np.float32)


def reset_environment(
    env,
    args: argparse.Namespace,
    rng: np.random.Generator,
    seed: int | None = None,
    use_random_warmup: bool = True,
):
    observation, info = env.reset(seed=seed)
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
            warmup_observation, _, terminated, truncated, warmup_info = env.step(action)
            warmup_done = bool(terminated or truncated)
            if warmup_done:
                break
        if not warmup_done:
            return warmup_observation, warmup_info
        observation, info = env.reset()

    return observation, info


def done_reason(terminated: bool, truncated: bool, time_limit_done: bool) -> str:
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
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
            "done_reason": reason,
        })


def evaluate_policy(
    env,
    policy: TanhGaussianPolicy,
    args: argparse.Namespace,
    transform: transforms.Compose,
    device: torch.device,
    reset_rng: np.random.Generator,
) -> dict[str, float | int]:
    observation, info = reset_environment(env, args, reset_rng, use_random_warmup=False)
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
            obs_tensor = preprocess(observation, args.crop_y_start, args.image_size, transform).unsqueeze(0).to(device)
            with torch.no_grad():
                action_tensor = policy.act(obs_tensor, deterministic=args.eval_deterministic)
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)

            reward = float(reward)
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
                observation, info = reset_environment(env, args, reset_rng, use_random_warmup=False)
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
        "action_space": "continuous left/right in [-1, 1]",
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


def main() -> None:
    args = parse_args()
    if args.resume_checkpoint is not None and args.imitation_checkpoint is not None:
        raise ValueError("Use either --resume-checkpoint or --imitation-checkpoint, not both.")

    set_seed(args.seed)
    reset_rng = np.random.default_rng(args.seed + 1)
    device = resolve_device(args.device)
    run_dir = args.output_dir.expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S_ppo")
    run_dir.mkdir(parents=True, exist_ok=False)
    config_values = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"camera_width", "camera_height", "device"}
    }
    config = PPOConfig(**config_values, device=str(device))
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    transform = make_transform()
    policy = TanhGaussianPolicy(args.model, pretrained=args.imitation_checkpoint is None and args.resume_checkpoint is None).to(device)
    if args.imitation_checkpoint is not None:
        load_imitation_actor(policy, args.imitation_checkpoint)
        policy.to(device)
    if args.resume_checkpoint is None:
        initial_log_std = args.initial_log_std
        if initial_log_std is None:
            initial_log_std = -2.0 if args.imitation_checkpoint is not None else -0.5
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
        print(f"Resumed RL checkpoint {args.resume_checkpoint.expanduser()} at step={resumed_step}", flush=True)

    metrics_file = run_dir / "history.csv"
    metrics_fields = [
        "step",
        "episode",
        "episode_return",
        "episode_length",
        "done_reason",
        "policy_loss",
        "value_loss",
        "entropy",
    ]
    with metrics_file.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=metrics_fields).writeheader()
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
    try:
        env = DuckiematrixDB21JEnv(entity_name=args.entity_name, headless=True, camera_width=args.camera_width, camera_height=args.camera_height)
        observation, info = reset_environment(env, args, reset_rng, seed=args.seed)
        episode_return = 0.0
        episode_length = 0
        episode = 0
        global_step = resumed_step
        rollout = 0
        eval_index = 0
        best_eval_return = -float("inf")

        while global_step < args.total_steps:
            obs_buf, action_buf, logp_buf, reward_buf, done_buf, value_buf = [], [], [], [], [], []
            for _ in range(args.rollout_steps):
                obs_tensor = preprocess(observation, args.crop_y_start, args.image_size, transform).to(device)
                with torch.no_grad():
                    action_tensor, logp_tensor, _ = policy.sample(obs_tensor.unsqueeze(0))
                    value_tensor = value(obs_tensor.unsqueeze(0))
                action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                next_observation, reward, terminated, truncated, info = env.step(action)

                obs_buf.append(obs_tensor.cpu())
                action_buf.append(action_tensor.squeeze(0).cpu())
                logp_buf.append(logp_tensor.squeeze(0).cpu())
                value_buf.append(value_tensor.squeeze(0).cpu())
                episode_return += float(reward)
                episode_length += 1
                global_step += 1
                env_done = bool(terminated or truncated)
                time_limit_done = args.max_episode_steps > 0 and episode_length >= args.max_episode_steps
                done = env_done or time_limit_done
                reward_buf.append(float(reward))
                done_buf.append(float(done))
                observation = next_observation

                if done:
                    reason = done_reason(terminated, truncated, time_limit_done)
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
                        f"length={episode_length} done_reason={reason}",
                        flush=True,
                    )
                    observation, info = reset_environment(env, args, reset_rng)
                    episode += 1
                    episode_return = 0.0
                    episode_length = 0
                if global_step >= args.total_steps:
                    break

            obs = torch.stack(obs_buf).to(device)
            actions = torch.stack(action_buf).to(device)
            old_logp = torch.stack(logp_buf).to(device)
            rewards = torch.tensor(reward_buf, dtype=torch.float32, device=device)
            dones = torch.tensor(done_buf, dtype=torch.float32, device=device)
            values = torch.stack(value_buf).to(device)
            with torch.no_grad():
                last_obs = preprocess(observation, args.crop_y_start, args.image_size, transform).unsqueeze(0).to(device)
                last_value = value(last_obs).squeeze(0)
                advantages, returns = compute_gae(rewards, dones, values, last_value, args.gamma, args.gae_lambda)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

                    value_optimizer.zero_grad(set_to_none=True)
                    (args.value_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(value.parameters(), args.max_grad_norm)
                    value_optimizer.step()
                    last_policy_loss, last_value_loss, last_entropy = policy_loss.item(), value_loss.item(), entropy.item()

            save_checkpoint(run_dir / "last.pt", policy, value, policy_optimizer, value_optimizer, config, global_step)
            rollout += 1
            print(
                f"update step={global_step} rollout={rollout} rollout_return={sum(reward_buf):.3f} "
                f"current_episode_return={episode_return:.3f} current_episode_length={episode_length} "
                f"policy_loss={last_policy_loss:.4f} value_loss={last_value_loss:.4f} entropy={last_entropy:.4f}",
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
                        f"length={episode_length} done_reason=eval_interrupt",
                        flush=True,
                    )
                    episode += 1
                    episode_return = 0.0
                    episode_length = 0

                eval_result = evaluate_policy(env, policy, args, transform, device, reset_rng)
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
                observation, info = reset_environment(env, args, reset_rng)
    finally:
        if env is not None:
            env.close()
        shutdown_dtps()


if __name__ == "__main__":
    main()
