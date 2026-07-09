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
    total_steps: int
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
    parser.add_argument("--total-steps", type=int, default=100_000)
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
    parser.add_argument("--crop-y-start", type=int, default=200)
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
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
    policy = TanhGaussianPolicy(args.model, pretrained=args.imitation_checkpoint is None).to(device)
    if args.imitation_checkpoint is not None:
        load_imitation_actor(policy, args.imitation_checkpoint)
        policy.to(device)
    value = ValueNetwork(args.model, pretrained=True).to(device)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=args.policy_lr)
    value_optimizer = torch.optim.AdamW(value.parameters(), lr=args.value_lr)

    metrics_file = run_dir / "history.csv"
    with metrics_file.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=["step", "episode", "episode_return", "episode_length", "policy_loss", "value_loss", "entropy"]).writeheader()

    env = None
    try:
        env = DuckiematrixDB21JEnv(entity_name=args.entity_name, headless=True, camera_width=args.camera_width, camera_height=args.camera_height)
        observation, info = env.reset(seed=args.seed)
        episode_return = 0.0
        episode_length = 0
        episode = 0
        global_step = 0

        while global_step < args.total_steps:
            obs_buf, action_buf, logp_buf, reward_buf, done_buf, value_buf = [], [], [], [], [], []
            for _ in range(args.rollout_steps):
                obs_tensor = preprocess(observation, args.crop_y_start, args.image_size, transform).to(device)
                with torch.no_grad():
                    action_tensor, logp_tensor, _ = policy.sample(obs_tensor.unsqueeze(0))
                    value_tensor = value(obs_tensor.unsqueeze(0))
                action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                obs_buf.append(obs_tensor.cpu())
                action_buf.append(action_tensor.squeeze(0).cpu())
                logp_buf.append(logp_tensor.squeeze(0).cpu())
                reward_buf.append(float(reward))
                done_buf.append(float(done))
                value_buf.append(value_tensor.squeeze(0).cpu())
                episode_return += float(reward)
                episode_length += 1
                global_step += 1
                observation = next_observation

                if done:
                    with metrics_file.open("a", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=["step", "episode", "episode_return", "episode_length", "policy_loss", "value_loss", "entropy"])
                        writer.writerow({"step": global_step, "episode": episode, "episode_return": episode_return, "episode_length": episode_length})
                    print(f"step={global_step} episode={episode} return={episode_return:.3f} length={episode_length}", flush=True)
                    observation, info = env.reset()
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
            print(f"update step={global_step} policy_loss={last_policy_loss:.4f} value_loss={last_value_loss:.4f} entropy={last_entropy:.4f}", flush=True)
    finally:
        if env is not None:
            env.close()
        shutdown_dtps()


if __name__ == "__main__":
    main()
