"""Adaptive sampling of random and curated gym-duckietown starts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from dt_utils.gym_duckietown_start_config import (
    StartConfig,
    TrainingPose,
    TrainingStart,
)


# EMA update coefficient. A value of 0.15 reacts roughly like a window over
# the latest 12 observations while avoiding a per-pose episode buffer.
START_SUCCESS_EMA_LAMBDA = 0.15

# Neutral success estimate used before a start type or pose has observations.
INITIAL_SUCCESS_RATE = 0.5

# Adaptive hard-start probability range. The lower bound keeps curated starts
# in circulation; the upper bound preserves random-start coverage.
HARD_START_PROBABILITY_MIN = 0.20
HARD_START_PROBABILITY_MAX = 0.80

# A completely unsuccessful pose receives 1 + HARD_POSE_DIFFICULTY_STRENGTH
# times the raw sampling weight of a completely successful pose.
HARD_POSE_DIFFICULTY_STRENGTH = 5.0

# Prevents a zero denominator when both smoothed failure rates are zero.
RELATIVE_DIFFICULTY_EPSILON = 1e-8

SAMPLER_STATE_SCHEMA_VERSION = 1
SUCCESSFUL_EPISODE_DONE_REASONS = frozenset(
    ("time_limit", "max-steps-reached")
)


@dataclass(frozen=True)
class AdaptiveStartSamplingSettings:
    """Resolved per-run settings for adaptive curated-start sampling."""

    ema_lambda: float = START_SUCCESS_EMA_LAMBDA
    initial_success_rate: float = INITIAL_SUCCESS_RATE
    hard_probability_min: float = HARD_START_PROBABILITY_MIN
    hard_probability_max: float = HARD_START_PROBABILITY_MAX
    pose_difficulty_strength: float = HARD_POSE_DIFFICULTY_STRENGTH
    relative_difficulty_epsilon: float = RELATIVE_DIFFICULTY_EPSILON

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_lambda <= 1.0:
            raise ValueError("adaptive start EMA lambda must be in (0, 1]")
        if not 0.0 <= self.initial_success_rate <= 1.0:
            raise ValueError("initial start success rate must be in [0, 1]")
        if not 0.0 <= self.hard_probability_min <= 1.0:
            raise ValueError("minimum hard-start probability must be in [0, 1]")
        if not 0.0 <= self.hard_probability_max <= 1.0:
            raise ValueError("maximum hard-start probability must be in [0, 1]")
        if self.hard_probability_min > self.hard_probability_max:
            raise ValueError(
                "minimum hard-start probability must not exceed the maximum"
            )
        if self.pose_difficulty_strength < 0.0:
            raise ValueError("hard-pose difficulty strength must be non-negative")
        if self.relative_difficulty_epsilon <= 0.0:
            raise ValueError("relative-difficulty epsilon must be positive")


@dataclass
class EmaSuccessRate:
    value: float = INITIAL_SUCCESS_RATE
    observations: int = 0
    ema_lambda: float = START_SUCCESS_EMA_LAMBDA

    def update(self, success: bool) -> None:
        outcome = float(bool(success))
        self.value = (
            (1.0 - self.ema_lambda) * self.value
            + self.ema_lambda * outcome
        )
        self.observations += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": float(self.value),
            "observations": int(self.observations),
        }

    def load_dict(self, state: dict[str, Any]) -> None:
        value = float(state["value"])
        observations = int(state["observations"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"EMA success rate must be in [0, 1], got {value}")
        if observations < 0:
            raise ValueError("EMA observation count must be non-negative")
        self.value = value
        self.observations = observations


def adaptive_start_sampling_configuration(
    settings: AdaptiveStartSamplingSettings | None = None,
) -> dict[str, Any]:
    settings = settings or AdaptiveStartSamplingSettings()
    return {
        "enabled_with_start_config": True,
        "successful_done_reasons": sorted(SUCCESSFUL_EPISODE_DONE_REASONS),
        "ema_lambda": settings.ema_lambda,
        "initial_success_rate": settings.initial_success_rate,
        "hard_start_probability_min": settings.hard_probability_min,
        "hard_start_probability_max": settings.hard_probability_max,
        "hard_pose_difficulty_strength": settings.pose_difficulty_strength,
        "relative_difficulty_epsilon": settings.relative_difficulty_epsilon,
        "hard_probability_formula": (
            "p_min + (p_max - p_min) * max(0, "
            "(hard_failure - random_failure) / "
            "(hard_failure + random_failure + epsilon))"
        ),
        "hard_pose_weight_formula": "1 + difficulty_strength * pose_failure",
    }


def episode_was_successful(done_reason: str) -> bool:
    return done_reason in SUCCESSFUL_EPISODE_DONE_REASONS


def training_pose_key(pose: TrainingPose) -> str:
    if pose.map_name is None:
        raise ValueError("adaptive training poses require map_name")
    if pose.name is not None:
        return f"{pose.map_name}:{pose.name}"
    tile = ",".join(str(value) for value in pose.tile)
    position = ",".join(f"{value:.12g}" for value in pose.position)
    return f"{pose.map_name}:unnamed:tile={tile}:position={position}:angle={pose.angle:.12g}"


class AdaptiveStartSampler:
    def __init__(
        self,
        config: StartConfig | None,
        rng: np.random.Generator,
        initial_hard_start_probability: float,
        settings: AdaptiveStartSamplingSettings | None = None,
    ) -> None:
        if not 0.0 <= initial_hard_start_probability <= 1.0:
            raise ValueError("initial hard-start probability must be in [0, 1]")
        self.config = config
        self.rng = rng
        self.settings = settings or AdaptiveStartSamplingSettings()
        self.initial_hard_start_probability = float(initial_hard_start_probability)
        statistic_defaults = {
            "value": self.settings.initial_success_rate,
            "ema_lambda": self.settings.ema_lambda,
        }
        self.hard_success = EmaSuccessRate(**statistic_defaults)
        self.random_success = EmaSuccessRate(**statistic_defaults)
        self.poses = tuple(config.training_poses) if config is not None else ()
        self.pose_keys = tuple(training_pose_key(pose) for pose in self.poses)
        if len(set(self.pose_keys)) != len(self.pose_keys):
            raise ValueError(
                "training pose identities must be unique; assign unique names within each map"
            )
        self.pose_success = {
            pose_key: EmaSuccessRate(**statistic_defaults)
            for pose_key in self.pose_keys
        }

    @property
    def enabled(self) -> bool:
        return bool(self.poses)

    def hard_start_probability(self) -> float:
        if not self.enabled:
            return 0.0
        if self.settings.hard_probability_min == self.settings.hard_probability_max:
            return self.settings.hard_probability_min
        if self.hard_success.observations == 0 or self.random_success.observations == 0:
            return self.initial_hard_start_probability

        hard_failure = 1.0 - self.hard_success.value
        random_failure = 1.0 - self.random_success.value
        relative_excess_failure = max(
            0.0,
            (hard_failure - random_failure)
            / (
                hard_failure
                + random_failure
                + self.settings.relative_difficulty_epsilon
            ),
        )
        probability = self.settings.hard_probability_min + (
            self.settings.hard_probability_max
            - self.settings.hard_probability_min
        ) * relative_excess_failure
        return float(np.clip(
            probability,
            self.settings.hard_probability_min,
            self.settings.hard_probability_max,
        ))

    def hard_pose_probabilities(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        weights = np.asarray([
            1.0
            + self.settings.pose_difficulty_strength
            * (1.0 - self.pose_success[key].value)
            for key in self.pose_keys
        ], dtype=np.float64)
        probabilities = weights / weights.sum()
        return {
            key: float(probability)
            for key, probability in zip(self.pose_keys, probabilities)
        }

    def choose(self) -> TrainingStart:
        if not self.enabled:
            return TrainingStart(kind="random", seed=self._draw_reset_seed())

        if self.rng.random() < self.hard_start_probability():
            probabilities = self.hard_pose_probabilities()
            pose_index = int(self.rng.choice(
                len(self.poses),
                p=[probabilities[key] for key in self.pose_keys],
            ))
            pose = self.poses[pose_index]
            return TrainingStart(
                kind="hard_pose",
                seed=self._draw_reset_seed(),
                pose=pose,
                map_name=pose.map_name,
            )

        return TrainingStart(kind="random", seed=self._draw_reset_seed())

    def update(self, training_start: TrainingStart, success: bool) -> None:
        if training_start.kind == "random":
            self.random_success.update(success)
            return
        if training_start.kind != "hard_pose" or training_start.pose is None:
            raise ValueError(f"unsupported training start kind {training_start.kind!r}")
        pose_key = training_pose_key(training_start.pose)
        if pose_key not in self.pose_success:
            raise ValueError(f"completed hard pose {pose_key!r} is not in the sampler config")
        self.hard_success.update(success)
        self.pose_success[pose_key].update(success)

    def pose_success_rate(self, training_start: TrainingStart) -> float | None:
        if training_start.pose is None:
            return None
        return self.pose_success[training_pose_key(training_start.pose)].value

    def pose_sampling_probability(self, training_start: TrainingStart) -> float | None:
        if training_start.pose is None:
            return None
        return self.hard_pose_probabilities()[training_pose_key(training_start.pose)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SAMPLER_STATE_SCHEMA_VERSION,
            "configuration": adaptive_start_sampling_configuration(self.settings),
            "initial_hard_start_probability": self.initial_hard_start_probability,
            "hard_success": self.hard_success.as_dict(),
            "random_success": self.random_success.as_dict(),
            "pose_success": {
                key: statistic.as_dict()
                for key, statistic in self.pose_success.items()
            },
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> dict[str, list[str]]:
        schema_version = int(state.get("schema_version", 0))
        if schema_version != SAMPLER_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported adaptive sampler schema {schema_version}; "
                f"expected {SAMPLER_STATE_SCHEMA_VERSION}"
            )
        saved_configuration = state.get("configuration", {})
        current_configuration = adaptive_start_sampling_configuration(self.settings)
        configuration_changes = sorted(
            key
            for key in set(saved_configuration) | set(current_configuration)
            if saved_configuration.get(key) != current_configuration.get(key)
        )

        # Sampling constants and the cold-start probability are run settings,
        # not model architecture. Resume the learned statistics while honoring
        # the values selected for the new run.
        self.hard_success.load_dict(state["hard_success"])
        self.random_success.load_dict(state["random_success"])

        saved_pose_success = state.get("pose_success", {})
        restored: list[str] = []
        added: list[str] = []
        for pose_key, statistic in self.pose_success.items():
            if pose_key in saved_pose_success:
                statistic.load_dict(saved_pose_success[pose_key])
                restored.append(pose_key)
            else:
                added.append(pose_key)
        removed = sorted(set(saved_pose_success) - set(self.pose_success))
        if "rng_state" in state:
            self.rng.bit_generator.state = deepcopy(state["rng_state"])
        return {
            "restored": sorted(restored),
            "added": sorted(added),
            "removed": removed,
            "configuration_changes": configuration_changes,
        }

    def _draw_reset_seed(self) -> int:
        return int(self.rng.integers(0, np.iinfo(np.int32).max))
