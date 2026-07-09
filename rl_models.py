#!/usr/bin/env python3
"""Shared image policies for Duckietown reinforcement learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from train_imitation_learning import TARGET_COLUMNS


@dataclass(frozen=True)
class PolicySpec:
    model_name: str = "mobilenet_v3_small"
    image_size: int = 224
    action_dim: int = len(TARGET_COLUMNS)


class TanhGaussianPolicy(nn.Module):
    """CNN policy that samples continuous wheel commands in [-1, 1]."""

    def __init__(self, model_name: str, action_dim: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        self.model_name = model_name
        self.action_dim = action_dim
        self.encoder, features_dim = build_encoder(model_name, pretrained=pretrained)
        self.mean = nn.Linear(features_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        mean = self.mean(features)
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        return mean, log_std

    def distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean, log_std = self(observations)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = tanh_normal_log_prob(distribution, raw_action, action)
        return action, log_prob, torch.tanh(distribution.mean)

    def act(self, observations: torch.Tensor, deterministic: bool) -> torch.Tensor:
        distribution = self.distribution(observations)
        raw_action = distribution.mean if deterministic else distribution.sample()
        return torch.tanh(raw_action)


class QNetwork(nn.Module):
    """Image-action critic for offline RL or actor-critic extensions."""

    def __init__(self, model_name: str, action_dim: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder, features_dim = build_encoder(model_name, pretrained=pretrained)
        self.q = nn.Sequential(
            nn.Linear(features_dim + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(observations)
        return self.q(torch.cat([features, actions], dim=1))


def build_encoder(model_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        features_dim = model.classifier[-1].in_features
        encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten(), *model.classifier[:-1])
        return encoder, features_dim

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        features_dim = model.fc.in_features
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return encoder, features_dim

    raise ValueError(f"Unsupported model: {model_name}")


def tanh_normal_log_prob(
    distribution: torch.distributions.Normal,
    raw_action: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    correction = torch.log(1.0 - action.pow(2) + 1e-6)
    return (distribution.log_prob(raw_action) - correction).sum(dim=1)


def load_imitation_actor(policy: TanhGaussianPolicy, checkpoint_path: Path) -> None:
    """Initialize a stochastic RL actor from a deterministic imitation checkpoint."""
    checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    if policy.model_name == "mobilenet_v3_small":
        translated = {}
        for key, value in state_dict.items():
            if key.startswith("features.") or key.startswith("avgpool."):
                translated[f"encoder.{key}"] = value
            elif key.startswith("classifier.") and not key.startswith("classifier.3"):
                translated[f"encoder.3.{key.removeprefix('classifier.')}"] = value
            elif key == "classifier.3.weight":
                translated["mean.weight"] = value
            elif key == "classifier.3.bias":
                translated["mean.bias"] = value
        policy.load_state_dict(translated, strict=False)
        return

    if policy.model_name == "resnet18":
        translated = {}
        for key, value in state_dict.items():
            if key.startswith("fc."):
                translated[key.replace("fc.", "mean.")] = value
            else:
                translated[f"encoder.{key}"] = value
        policy.load_state_dict(translated, strict=False)
        return

    raise ValueError(f"Unsupported model: {policy.model_name}")
