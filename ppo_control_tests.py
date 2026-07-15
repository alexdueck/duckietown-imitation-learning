#!/usr/bin/env python3
"""Small continuous-control checks for the PPO math used by the trainers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl_models import TanhGaussianPolicy, tanh_normal_log_prob
from train_rl_ppo_gym_duckietown import compute_gae


class SquashedGaussianActor(nn.Module):
    def __init__(self, action_dim: int, initial_log_std: float = -0.5) -> None:
        super().__init__()
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def mean(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean(observations)
        return torch.distributions.Normal(mean, self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean))

    def sample_with_raw(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = tanh_normal_log_prob(distribution, raw_action, action)
        return action, raw_action, log_prob, torch.tanh(distribution.mean)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean(observations))


class MLPActor(SquashedGaussianActor):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__(action_dim)
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def mean(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class MLPValue(nn.Module):
    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(1)


class ImageActor(SquashedGaussianActor):
    def __init__(self) -> None:
        super().__init__(action_dim=2)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(32, 2)

    def mean(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(observations))


class ImageValue(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(1)


@dataclass
class PPOBatch:
    observations: torch.Tensor
    raw_actions: torch.Tensor
    old_logp: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    last_value: torch.Tensor


def ppo_update(
    actor: SquashedGaussianActor,
    value: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    *,
    epochs: int,
    batch_size: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.0,
) -> tuple[float, float]:
    with torch.no_grad():
        advantages, returns = compute_gae(
            batch.rewards,
            batch.dones,
            batch.values,
            batch.last_value,
            gamma,
            gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        before_distribution = actor.distribution(batch.observations)
        before_actions = torch.tanh(batch.raw_actions)
        before_logp = tanh_normal_log_prob(before_distribution, batch.raw_actions, before_actions)
        invariant_error = (before_logp - batch.old_logp).abs().max().item()
        if invariant_error > 1e-4:
            raise RuntimeError(f"Pre-update log-probability invariant failed: {invariant_error:.3e}")

    indices = torch.arange(batch.observations.size(0))
    for _ in range(epochs):
        permutation = indices[torch.randperm(indices.numel())]
        for start in range(0, indices.numel(), batch_size):
            selected = permutation[start:start + batch_size]
            distribution = actor.distribution(batch.observations[selected])
            raw_actions = batch.raw_actions[selected]
            actions = torch.tanh(raw_actions)
            logp = tanh_normal_log_prob(distribution, raw_actions, actions)
            ratio = (logp - batch.old_logp[selected]).exp()
            unclipped = ratio * advantages[selected]
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[selected]
            policy_loss = -torch.min(unclipped, clipped).mean()
            entropy = distribution.entropy().sum(dim=1).mean()

            actor_optimizer.zero_grad(set_to_none=True)
            (policy_loss - entropy_coef * entropy).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optimizer.step()
            with torch.no_grad():
                actor.log_std.clamp_(-5.0, 1.0)

            value_loss = nn.functional.mse_loss(value(batch.observations[selected]), returns[selected])
            value_optimizer.zero_grad(set_to_none=True)
            (0.5 * value_loss).backward()
            nn.utils.clip_grad_norm_(value.parameters(), 0.5)
            value_optimizer.step()

    with torch.no_grad():
        distribution = actor.distribution(batch.observations)
        actions = torch.tanh(batch.raw_actions)
        log_ratio = tanh_normal_log_prob(distribution, batch.raw_actions, actions) - batch.old_logp
        ratio = log_ratio.exp()
        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
    return approx_kl, clip_fraction


def run_policy_invariant_test(seed: int) -> bool:
    torch.manual_seed(seed)
    policy = TanhGaussianPolicy("mobilenet_v3_small", pretrained=False)
    policy.eval()
    with torch.no_grad():
        policy.mean.weight.zero_()
        policy.mean.bias.fill_(5.0)
        policy.log_std.fill_(-1.5)
        observations = torch.randn(8, 3, 64, 64)
        actions, raw_actions, old_logp, _ = policy.sample_with_raw(observations)
        distribution = policy.distribution(observations)
        new_logp = tanh_normal_log_prob(distribution, raw_actions, torch.tanh(raw_actions))
    max_logp_error = (new_logp - old_logp).abs().max().item()
    dropout_disabled = all(
        module.p == 0.0
        for module in policy.modules()
        if isinstance(module, nn.Dropout)
    )
    batch_norm_frozen = all(
        not module.training
        for module in policy.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    saturated = (actions.abs() > 0.95).all().item()
    passed = max_logp_error < 1e-4 and dropout_disabled and batch_norm_frozen and saturated
    print(
        f"policy_invariant max_logp_error={max_logp_error:.3e} "
        f"dropout_disabled={dropout_disabled} batch_norm_frozen={batch_norm_frozen} "
        f"saturated_actions={saturated} passed={passed}",
        flush=True,
    )
    return bool(passed)


def reset_env(env, seed: int | None = None) -> np.ndarray:
    result = env.reset(seed=seed)
    return np.asarray(result[0] if isinstance(result, tuple) else result, dtype=np.float32)


def step_env(env, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, _ = result
        done = bool(terminated or truncated)
    else:
        observation, reward, done, _ = result
    return np.asarray(observation, dtype=np.float32), float(reward), bool(done)


def evaluate_pendulum(env, actor: MLPActor, seeds: tuple[int, ...]) -> float:
    returns = []
    actor.eval()
    for seed in seeds:
        observation = reset_env(env, seed)
        episode_return = 0.0
        done = False
        while not done:
            with torch.no_grad():
                normalized_action = actor.act(torch.from_numpy(observation).unsqueeze(0))[0].numpy()
            observation, reward, done = step_env(env, normalized_action * 2.0)
            episode_return += reward
        returns.append(episode_return)
    return float(np.mean(returns))


def run_pendulum_test(total_steps: int, seed: int) -> bool:
    try:
        import gym
        env = gym.make("Pendulum-v1")
        eval_env = gym.make("Pendulum-v1")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("Pendulum-v1 requires Gym classic-control dependencies, including pygame") from error

    torch.manual_seed(seed)
    np.random.seed(seed)
    actor = MLPActor(3, 1)
    value = MLPValue(3)
    actor.eval()
    value.eval()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    evaluation_seeds = (1001, 1002, 1003, 1004, 1005)
    initial_return = evaluate_pendulum(eval_env, actor, evaluation_seeds)
    observation = reset_env(env, seed)
    completed_steps = 0
    rollout_steps = 1024
    update_index = 0

    while completed_steps < total_steps:
        observations, raw_actions, old_logp = [], [], []
        rewards, dones, values = [], [], []
        for _ in range(min(rollout_steps, total_steps - completed_steps)):
            observation_tensor = torch.from_numpy(observation)
            with torch.no_grad():
                action, raw_action, logp, _ = actor.sample_with_raw(observation_tensor.unsqueeze(0))
                value_prediction = value(observation_tensor.unsqueeze(0))[0]
            next_observation, reward, done = step_env(env, action[0].numpy() * 2.0)
            observations.append(observation_tensor)
            raw_actions.append(raw_action[0])
            old_logp.append(logp[0])
            rewards.append(0.1 * reward)
            dones.append(float(done))
            values.append(value_prediction)
            observation = reset_env(env) if done else next_observation
            completed_steps += 1

        with torch.no_grad():
            last_value = value(torch.from_numpy(observation).unsqueeze(0))[0]
        approx_kl, clip_fraction = ppo_update(
            actor,
            value,
            actor_optimizer,
            value_optimizer,
            PPOBatch(
                observations=torch.stack(observations),
                raw_actions=torch.stack(raw_actions),
                old_logp=torch.stack(old_logp),
                rewards=torch.tensor(rewards, dtype=torch.float32),
                dones=torch.tensor(dones, dtype=torch.float32),
                values=torch.stack(values),
                last_value=last_value,
            ),
            epochs=4,
            batch_size=64,
            entropy_coef=0.01,
        )
        update_index += 1
        if update_index % 25 == 0:
            interim_return = evaluate_pendulum(eval_env, actor, evaluation_seeds[:2])
            print(
                f"pendulum_progress steps={completed_steps} mean_return={interim_return:.2f} "
                f"approx_kl={approx_kl:.5f} clip_fraction={clip_fraction:.3f} "
                f"log_std={actor.log_std.item():.3f}",
                flush=True,
            )

    final_return = evaluate_pendulum(eval_env, actor, evaluation_seeds)
    env.close()
    eval_env.close()
    improvement = final_return - initial_return
    passed = improvement > 300.0 and final_return > -900.0
    print(
        f"pendulum initial_mean_return={initial_return:.2f} "
        f"final_mean_return={final_return:.2f} improvement={improvement:.2f} passed={passed}",
        flush=True,
    )
    return passed


def targets_to_images(targets: torch.Tensor) -> torch.Tensor:
    images = torch.zeros(targets.size(0), 3, 16, 16)
    images[:, 0] = (targets[:, 0] + 1.0).view(-1, 1, 1) * 0.5
    images[:, 1] = (targets[:, 1] + 1.0).view(-1, 1, 1) * 0.5
    horizontal = torch.linspace(0.0, 1.0, 16).view(1, 1, 16)
    images[:, 2] = horizontal
    return images


def image_policy_mse(actor: ImageActor, targets: torch.Tensor) -> float:
    actor.eval()
    with torch.no_grad():
        predictions = actor.act(targets_to_images(targets))
    return nn.functional.mse_loss(predictions, targets).item()


def run_image_test(rollouts: int, seed: int) -> bool:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    actor = ImageActor()
    value = ImageValue()
    actor.eval()
    value.eval()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    evaluation_targets = torch.rand(512, 2, generator=generator) * 1.6 - 0.8
    initial_mse = image_policy_mse(actor, evaluation_targets)

    for _ in range(rollouts):
        targets = torch.rand(256, 2, generator=generator) * 1.6 - 0.8
        observations = targets_to_images(targets)
        with torch.no_grad():
            actions, raw_actions, old_logp, _ = actor.sample_with_raw(observations)
            value_predictions = value(observations)
            rewards = 1.0 - 2.0 * (actions - targets).pow(2).mean(dim=1)
        ppo_update(
            actor,
            value,
            actor_optimizer,
            value_optimizer,
            PPOBatch(
                observations=observations,
                raw_actions=raw_actions,
                old_logp=old_logp,
                rewards=rewards,
                dones=torch.ones_like(rewards),
                values=value_predictions,
                last_value=torch.tensor(0.0),
            ),
            epochs=4,
            batch_size=64,
            gamma=0.0,
            gae_lambda=0.0,
            entropy_coef=0.0,
        )

    final_mse = image_policy_mse(actor, evaluation_targets)
    passed = final_mse < 0.08 and final_mse < 0.4 * initial_mse
    print(
        f"image_control initial_mse={initial_mse:.4f} final_mse={final_mse:.4f} passed={passed}",
        flush=True,
    )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        choices=("all", "invariant", "pendulum", "image"),
        default="all",
    )
    parser.add_argument("--pendulum-steps", type=int, default=300_000)
    parser.add_argument("--image-rollouts", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    passed = True
    if args.test in ("all", "invariant"):
        passed = run_policy_invariant_test(args.seed) and passed
    if args.test in ("all", "pendulum"):
        passed = run_pendulum_test(args.pendulum_steps, args.seed) and passed
    if args.test in ("all", "image"):
        passed = run_image_test(args.image_rollouts, args.seed) and passed
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
