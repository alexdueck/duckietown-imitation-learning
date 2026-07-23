#!/usr/bin/env python3
"""Policy-control to wheel-command mappings for gym-duckietown."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np
    import torch


ACTION_MODE_CHOICES = ("wheel", "throttle_steering")


@dataclass(frozen=True)
class DuckietownActionControl:
    mode: str = "wheel"
    fixed_throttle: float | None = None
    max_throttle: float = 1.0
    max_steering: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in ACTION_MODE_CHOICES:
            raise ValueError(
                f"Unsupported action mode {self.mode!r}; choose from {ACTION_MODE_CHOICES}"
            )
        if not 0.0 < self.max_throttle <= 1.0:
            raise ValueError("--max-throttle must be in (0, 1]")
        if not 0.0 < self.max_steering <= 1.0:
            raise ValueError("--max-steering must be in (0, 1]")
        if self.fixed_throttle is not None:
            if self.mode != "throttle_steering":
                raise ValueError(
                    "--fixed-throttle requires --action-mode throttle_steering"
                )
            if not 0.0 <= self.fixed_throttle <= 1.0:
                raise ValueError("--fixed-throttle must be in [0, 1]")

    @property
    def policy_action_dim(self) -> int:
        if self.mode == "wheel":
            return 2
        return 1 if self.fixed_throttle is not None else 2

    @property
    def control_names(self) -> tuple[str, ...]:
        if self.mode == "wheel":
            return ("left_wheel", "right_wheel")
        if self.fixed_throttle is not None:
            return ("steering",)
        return ("throttle", "steering")

    def to_wheels_numpy(self, controls: np.ndarray) -> np.ndarray:
        import numpy as np

        controls = np.asarray(controls, dtype=np.float32)
        self._check_last_dimension(controls.shape[-1])
        if self.mode == "wheel":
            return self._scale_wheels_numpy(controls)

        if self.fixed_throttle is None:
            throttle = 0.5 * (controls[..., 0] + 1.0) * self.max_throttle
            steering_control = controls[..., 1]
        else:
            throttle = np.full(
                controls.shape[:-1],
                self.fixed_throttle,
                dtype=np.float32,
            )
            steering_control = controls[..., 0]
        steering = steering_control * self.max_steering
        wheels = np.stack((throttle - steering, throttle + steering), axis=-1)
        return self._scale_wheels_numpy(wheels)

    def to_wheels_tensor(self, controls: torch.Tensor) -> torch.Tensor:
        import torch

        self._check_last_dimension(controls.shape[-1])
        if self.mode == "wheel":
            return self._scale_wheels_tensor(controls)

        if self.fixed_throttle is None:
            throttle = 0.5 * (controls[..., 0] + 1.0) * self.max_throttle
            steering_control = controls[..., 1]
        else:
            throttle = torch.full_like(controls[..., 0], self.fixed_throttle)
            steering_control = controls[..., 0]
        steering = steering_control * self.max_steering
        wheels = torch.stack((throttle - steering, throttle + steering), dim=-1)
        return self._scale_wheels_tensor(wheels)

    def to_wheels_pair(self, controls: Sequence[float]) -> tuple[float, float]:
        """Map one policy-control sample to normalized wheel commands.

        This scalar variant has no NumPy or PyTorch dependency and is intended
        for deployment adapters, safety checks, and lightweight tests.
        """

        values = tuple(float(value) for value in controls)
        self._check_last_dimension(len(values))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Policy controls must contain finite values only")

        if self.mode == "wheel":
            left, right = values
        else:
            if self.fixed_throttle is None:
                throttle = 0.5 * (values[0] + 1.0) * self.max_throttle
                steering_control = values[1]
            else:
                throttle = self.fixed_throttle
                steering_control = values[0]
            steering = steering_control * self.max_steering
            left = throttle - steering
            right = throttle + steering

        scale = max(1.0, abs(left), abs(right))
        return left / scale, right / scale

    @staticmethod
    def _scale_wheels_numpy(wheels: np.ndarray) -> np.ndarray:
        import numpy as np

        scale = np.maximum(
            1.0,
            np.max(np.abs(wheels), axis=-1, keepdims=True),
        )
        return (wheels / scale).astype(np.float32, copy=False)

    @staticmethod
    def _scale_wheels_tensor(wheels: torch.Tensor) -> torch.Tensor:
        import torch

        scale = torch.maximum(
            torch.ones_like(wheels[..., :1]),
            wheels.abs().amax(dim=-1, keepdim=True),
        )
        return wheels / scale

    def _check_last_dimension(self, actual: int) -> None:
        if actual != self.policy_action_dim:
            raise ValueError(
                f"{self.mode} expects {self.policy_action_dim} policy control(s), "
                f"received {actual}"
            )


def action_control_from_config(config: dict) -> DuckietownActionControl:
    return DuckietownActionControl(
        mode=config.get("action_mode", "wheel"),
        fixed_throttle=config.get("fixed_throttle"),
        max_throttle=float(config.get("max_throttle", 1.0)),
        max_steering=float(config.get("max_steering", 0.5)),
    )
