"""Fixed observation/action histories for image-based reinforcement learning."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from dt_utils.rl_models import build_encoder, tanh_normal_log_prob


HISTORY_INITIALIZATION = {
    "observations": "repeat_first_observation",
    "actions": "zeros",
}
TEMPORAL_HEAD_MODE_MLP = "temporal_mlp"
TEMPORAL_HEAD_MODE_RESIDUAL = "residual"
TEMPORAL_HEAD_MODES = (
    TEMPORAL_HEAD_MODE_MLP,
    TEMPORAL_HEAD_MODE_RESIDUAL,
)


def history_model_description(temporal_head_mode: str) -> str:
    validate_temporal_head_mode(temporal_head_mode)
    if temporal_head_mode == TEMPORAL_HEAD_MODE_MLP:
        return "shared CNN encoder + direct temporal MLP"
    return "shared CNN encoder + current-frame head + temporal residual MLP"


def validate_temporal_head_mode(temporal_head_mode: str) -> None:
    if temporal_head_mode not in TEMPORAL_HEAD_MODES:
        raise ValueError(
            f"temporal_head_mode must be one of {TEMPORAL_HEAD_MODES}, "
            f"got {temporal_head_mode!r}"
        )


def validate_history_lengths(
    observation_history_length: int,
    action_history_length: int,
) -> None:
    if observation_history_length <= 0:
        raise ValueError("observation_history_length must be positive")
    if action_history_length != observation_history_length - 1:
        raise ValueError(
            "action_history_length must equal observation_history_length - 1, "
            f"got {action_history_length} and {observation_history_length}"
        )


class FixedHistory:
    """Per-environment history reset deterministically at episode boundaries."""

    def __init__(
        self,
        observation_history_length: int,
        action_history_length: int,
        action_dim: int,
    ) -> None:
        validate_history_lengths(observation_history_length, action_history_length)
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.observation_history_length = observation_history_length
        self.action_history_length = action_history_length
        self.action_dim = action_dim
        self._observations: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []

    def reset(self, first_observation: torch.Tensor) -> None:
        observation = first_observation.detach().cpu().clone()
        self._observations = [
            observation.clone() for _ in range(self.observation_history_length)
        ]
        self._actions = [
            torch.zeros(self.action_dim, dtype=torch.float32)
            for _ in range(self.action_history_length)
        ]

    def append(
        self,
        action: torch.Tensor | Sequence[float],
        observation: torch.Tensor,
    ) -> None:
        self._require_initialized()
        action_tensor = torch.as_tensor(action, dtype=torch.float32).detach().cpu().clone()
        if action_tensor.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action shape {(self.action_dim,)}, got {tuple(action_tensor.shape)}"
            )
        observation_tensor = observation.detach().cpu().clone()
        if observation_tensor.shape != self._observations[-1].shape:
            raise ValueError(
                "Observation shape changed within an episode: "
                f"{tuple(self._observations[-1].shape)} -> {tuple(observation_tensor.shape)}"
            )
        self._observations = self._observations[1:] + [observation_tensor]
        if self.action_history_length:
            self._actions = self._actions[1:] + [action_tensor]

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_initialized()
        observations = torch.stack(self._observations)
        if self.action_history_length:
            actions = torch.stack(self._actions)
        else:
            actions = torch.empty((0, self.action_dim), dtype=torch.float32)
        return observations, actions

    def batched(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        observations, actions = self.snapshot()
        return observations.unsqueeze(0).to(device), actions.unsqueeze(0).to(device)

    def _require_initialized(self) -> None:
        if len(self._observations) != self.observation_history_length:
            raise RuntimeError("History must be reset with the first observation before use")


def _encode_history(
    encoder: nn.Module,
    observations: torch.Tensor,
    observation_history_length: int,
) -> torch.Tensor:
    if observation_history_length == 1 and observations.ndim == 4:
        return encoder(observations).unsqueeze(1)
    if observations.ndim != 5:
        raise ValueError(
            "Temporal observations must have shape [batch, history, channels, height, width], "
            f"got {tuple(observations.shape)}"
        )
    if observations.shape[1] != observation_history_length:
        raise ValueError(
            f"Expected {observation_history_length} observations, got {observations.shape[1]}"
        )
    batch_size, history_length = observations.shape[:2]
    flat = observations.reshape(batch_size * history_length, *observations.shape[2:])
    return encoder(flat).reshape(batch_size, history_length, -1)


def _concatenate_temporal_features(
    features: torch.Tensor,
    action_history: torch.Tensor | None,
    action_history_length: int,
    action_dim: int,
) -> torch.Tensor:
    expected_shape = (features.shape[0], action_history_length, action_dim)
    if action_history is None or tuple(action_history.shape) != expected_shape:
        actual = None if action_history is None else tuple(action_history.shape)
        raise ValueError(f"Expected action history shape {expected_shape}, got {actual}")
    return torch.cat(
        [features.flatten(start_dim=1), action_history.flatten(start_dim=1)],
        dim=1,
    )


def _build_temporal_head(
    features_dim: int,
    observation_history_length: int,
    action_history_length: int,
    action_dim: int,
    hidden_dim: int,
    output_dim: int,
    zero_initialize_output: bool,
) -> nn.Sequential:
    if hidden_dim <= 0:
        raise ValueError("temporal_hidden_dim must be positive")
    input_dim = features_dim * observation_history_length + action_dim * action_history_length
    head = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )
    if zero_initialize_output:
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    return head


class TemporalTanhGaussianPolicy(nn.Module):
    """Squashed Gaussian policy with a shared CNN for every history frame."""

    def __init__(
        self,
        model_name: str,
        action_dim: int = 2,
        pretrained: bool = False,
        observation_history_length: int = 5,
        action_history_length: int = 4,
        temporal_hidden_dim: int = 256,
        temporal_head_mode: str = TEMPORAL_HEAD_MODE_MLP,
    ) -> None:
        super().__init__()
        validate_history_lengths(observation_history_length, action_history_length)
        validate_temporal_head_mode(temporal_head_mode)
        self.model_name = model_name
        self.action_dim = action_dim
        self.observation_history_length = observation_history_length
        self.action_history_length = action_history_length
        self.temporal_hidden_dim = temporal_hidden_dim
        self.temporal_head_mode = temporal_head_mode
        self.encoder, features_dim = build_encoder(model_name, pretrained=pretrained)
        self.mean = (
            nn.Linear(features_dim, action_dim)
            if temporal_head_mode == TEMPORAL_HEAD_MODE_RESIDUAL
            else None
        )
        self.temporal_head = (
            _build_temporal_head(
                features_dim,
                observation_history_length,
                action_history_length,
                action_dim,
                temporal_hidden_dim,
                output_dim=action_dim,
                zero_initialize_output=temporal_head_mode == TEMPORAL_HEAD_MODE_RESIDUAL,
            )
            if temporal_head_mode == TEMPORAL_HEAD_MODE_MLP
            or observation_history_length > 1
            else None
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(
        self,
        observations: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = _encode_history(
            self.encoder,
            observations,
            self.observation_history_length,
        )
        temporal_output = None
        if self.temporal_head is not None:
            temporal_input = _concatenate_temporal_features(
                features,
                action_history,
                self.action_history_length,
                self.action_dim,
            )
            temporal_output = self.temporal_head(temporal_input)
        if self.temporal_head_mode == TEMPORAL_HEAD_MODE_MLP:
            if temporal_output is None:
                raise RuntimeError("Direct temporal policy requires a temporal head")
            mean = temporal_output
        else:
            if self.mean is None:
                raise RuntimeError("Residual policy requires a current-frame mean head")
            mean = self.mean(features[:, -1])
            if temporal_output is not None:
                mean = mean + temporal_output
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        return mean, log_std

    def distribution(
        self,
        observations: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> torch.distributions.Normal:
        mean, log_std = self(observations, action_history)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample_with_raw(
        self,
        observations: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations, action_history)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = tanh_normal_log_prob(distribution, raw_action, action)
        return action, raw_action, log_prob, torch.tanh(distribution.mean)

    def act(
        self,
        observations: torch.Tensor,
        deterministic: bool,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        distribution = self.distribution(observations, action_history)
        raw_action = distribution.mean if deterministic else distribution.sample()
        return torch.tanh(raw_action)


class TemporalValueNetwork(nn.Module):
    """Value network over the same fixed history used by the policy."""

    def __init__(
        self,
        model_name: str,
        action_dim: int,
        pretrained: bool = False,
        observation_history_length: int = 5,
        action_history_length: int = 4,
        temporal_hidden_dim: int = 256,
        temporal_head_mode: str = TEMPORAL_HEAD_MODE_MLP,
    ) -> None:
        super().__init__()
        validate_history_lengths(observation_history_length, action_history_length)
        validate_temporal_head_mode(temporal_head_mode)
        self.model_name = model_name
        self.action_dim = action_dim
        self.observation_history_length = observation_history_length
        self.action_history_length = action_history_length
        self.temporal_hidden_dim = temporal_hidden_dim
        self.temporal_head_mode = temporal_head_mode
        self.encoder, features_dim = build_encoder(model_name, pretrained=pretrained)
        self.value = (
            nn.Linear(features_dim, 1)
            if temporal_head_mode == TEMPORAL_HEAD_MODE_RESIDUAL
            else None
        )
        self.temporal_head = (
            _build_temporal_head(
                features_dim,
                observation_history_length,
                action_history_length,
                action_dim,
                temporal_hidden_dim,
                output_dim=1,
                zero_initialize_output=temporal_head_mode == TEMPORAL_HEAD_MODE_RESIDUAL,
            )
            if temporal_head_mode == TEMPORAL_HEAD_MODE_MLP
            or observation_history_length > 1
            else None
        )

    def forward(
        self,
        observations: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = _encode_history(
            self.encoder,
            observations,
            self.observation_history_length,
        )
        temporal_output = None
        if self.temporal_head is not None:
            temporal_input = _concatenate_temporal_features(
                features,
                action_history,
                self.action_history_length,
                self.action_dim,
            )
            temporal_output = self.temporal_head(temporal_input)
        if self.temporal_head_mode == TEMPORAL_HEAD_MODE_MLP:
            if temporal_output is None:
                raise RuntimeError("Direct temporal value network requires a temporal head")
            value = temporal_output
        else:
            if self.value is None:
                raise RuntimeError("Residual value network requires a current-frame value head")
            value = self.value(features[:, -1])
            if temporal_output is not None:
                value = value + temporal_output
        return value.squeeze(1)
