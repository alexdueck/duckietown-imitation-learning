from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


REWARD_FUNCTION_CHOICES = (
    "default",
    "default_clipped",
    "posangle",
    "target_orientation",
    "lane_distance",
)

DISPLAY_REWARD_FUNCTIONS = (
    "default",
    "posangle",
    "target_orientation",
    "lane_distance",
)


def pose_position(pose: dict[str, Any]) -> np.ndarray:
    position = pose["position"]
    return np.array(
        [
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
        ],
        dtype=np.float64,
    )


def pose_yaw(pose: dict[str, Any]) -> float:
    from gym_duckiematrix.utils import quaternion_to_euler

    rotation = pose["rotation"]
    quat_rot = [
        rotation["w"],
        rotation["x"],
        rotation["y"],
        rotation["z"],
    ]
    return float(quaternion_to_euler(quat_rot)[-1])


def safe_float(value: float) -> float:
    value = float(value)
    if math.isnan(value):
        return 0.0
    return value


def velocity_reward(action: np.ndarray | None) -> float:
    if action is None:
        return 0.0
    return safe_float(float(np.max(np.asarray(action, dtype=np.float64))) * 0.25)


class KalaposRewardCalculator:
    """Duckietown-RL reward functions adapted to gym-duckiematrix pose APIs."""

    def __init__(self, name: str) -> None:
        if name not in REWARD_FUNCTION_CHOICES:
            raise ValueError(f"Unknown reward function {name!r}")
        self.name = name

    def compute(
        self,
        env,
        action: np.ndarray | None,
        previous_pose: dict[str, Any] | None,
        env_reward: float,
    ) -> float:
        breakdown = self.compute_breakdown(
            env=env,
            action=action,
            previous_pose=previous_pose,
            env_reward=env_reward,
        )
        return safe_float(breakdown["total"])

    def compute_breakdown(
        self,
        env,
        action: np.ndarray | None,
        previous_pose: dict[str, Any] | None,
        env_reward: float,
    ) -> dict[str, float | dict[str, float]]:
        if self.name == "default":
            total = float(env_reward)
            return {"total": total, "components": {"gym_duckiematrix": total}}
        if self.name == "default_clipped":
            total = float(np.clip(safe_float(env_reward), -2.0, 2.0))
            return {"total": total, "components": {"gym_duckiematrix_clipped": total}}
        if self.name == "posangle":
            return self._posangle_breakdown(env, action, target_orientation_only=False)
        if self.name == "target_orientation":
            return self._posangle_breakdown(env, action, target_orientation_only=True)
        if self.name == "lane_distance":
            total = safe_float(self._lane_distance_reward(env, previous_pose))
            return {"total": total, "components": {"DtRewardDistanceTravelled": total}}
        raise AssertionError(f"Unhandled reward function {self.name!r}")

    def _current_lane_position(self, env):
        current_pose = getattr(env, "last_pose", None)
        if current_pose is None or not hasattr(env, "lp_cal"):
            raise ValueError("Lane position is unavailable")
        pos = pose_position(current_pose)
        yaw = pose_yaw(current_pose)
        return env.lp_cal.get_lane_pos2(pos, yaw)

    @staticmethod
    def _leaky_cosine(x: float) -> float:
        slope = 0.05
        if abs(x) < math.pi:
            return math.cos(x)
        return -1.0 - slope * (abs(x) - math.pi)

    @classmethod
    def _target_angle_reward(cls, lp_dist: float, lp_angle_deg: float, max_dev_deg: float) -> float:
        max_lp_dist = 0.05
        target_angle_deg_at_edge = 45.0
        normed_lp_dist = float(lp_dist) / max_lp_dist
        target_angle = -float(np.clip(normed_lp_dist, -1.0, 1.0)) * target_angle_deg_at_edge
        return 0.5 + 0.5 * cls._leaky_cosine(
            math.pi * (target_angle - float(lp_angle_deg)) / max_dev_deg
        )

    def _posangle_breakdown(
        self,
        env,
        action: np.ndarray | None,
        target_orientation_only: bool,
    ) -> dict[str, float | dict[str, float]]:
        velocity_component = velocity_reward(action)
        try:
            lane_position = self._current_lane_position(env)
        except Exception:
            component_name = "DtRewardTargetOrientation" if target_orientation_only else "DtRewardPosAngle"
            orientation_reward = -10.0
            return {
                "total": orientation_reward + velocity_component,
                "components": {
                    component_name: orientation_reward,
                    "DtRewardVelocity": velocity_component,
                },
            }

        if target_orientation_only:
            component_name = "DtRewardTargetOrientation"
            orientation_reward = self._target_angle_reward(
                lane_position.dist,
                lane_position.angle_deg,
                max_dev_deg=50.0,
            )
        else:
            component_name = "DtRewardPosAngle"
            narrow_reward = self._target_angle_reward(
                lane_position.dist,
                lane_position.angle_deg,
                max_dev_deg=10.0,
            )
            wide_reward = self._target_angle_reward(
                lane_position.dist,
                lane_position.angle_deg,
                max_dev_deg=50.0,
            )
            orientation_reward = 0.5 * (narrow_reward + wide_reward)

        return {
            "total": orientation_reward + velocity_component,
            "components": {
                component_name: orientation_reward,
                "DtRewardVelocity": velocity_component,
            },
        }

    def _lane_distance_reward(self, env, previous_pose: dict[str, Any] | None) -> float:
        current_pose = getattr(env, "last_pose", None)
        if previous_pose is None or current_pose is None or not hasattr(env, "lp_cal"):
            return 0.0

        pos = pose_position(current_pose)
        prev_pos = pose_position(previous_pose)
        yaw = pose_yaw(current_pose)

        try:
            curve_point, tangent = env.lp_cal.closest_curve_point(pos, yaw)
            prev_curve_point, _ = env.lp_cal.closest_curve_point(prev_pos, yaw)
            lane_position = env.lp_cal.get_lane_pos2(pos, yaw)
        except Exception:
            return 0.0

        if curve_point is None or prev_curve_point is None or tangent is None:
            return 0.0

        diff = curve_point - prev_curve_point
        distance = float(np.linalg.norm(diff))

        # Same threshold as DtRewardWrapperDistanceTravelled in kaland313/Duckietown-RL.
        if float(lane_position.dist) < -0.05:
            return 0.0
        if float(np.dot(tangent, diff)) < 0.0:
            return 0.0

        return 50.0 * distance


@dataclass(frozen=True)
class RewardMetadata:
    name: str
    source: str = "kaland313/Duckietown-RL reward_wrappers.py"
    supported: tuple[str, ...] = REWARD_FUNCTION_CHOICES


def compute_reward_breakdowns(
    env,
    action: np.ndarray | None,
    previous_pose: dict[str, Any] | None,
    env_reward: float,
    names: tuple[str, ...] = DISPLAY_REWARD_FUNCTIONS,
) -> dict[str, dict[str, float | dict[str, float]]]:
    return {
        name: KalaposRewardCalculator(name).compute_breakdown(
            env=env,
            action=action,
            previous_pose=previous_pose,
            env_reward=env_reward,
        )
        for name in names
    }
