#!/usr/bin/env python3
"""Shared velocity-and-pose reward calculation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


# Nominal DB18 dynamics: (u_alpha_right + u_alpha_left) / u1 = 3 / 5 m/s.
VELOPPOSE_REFERENCE_SPEED_MPS = 0.6
VELOPPOSE_VELOCITY_WEIGHT = 1.0
VELOPPOSE_POSE_WEIGHT = 0.2
# Duckietown's right-lane centerline sits 0.2 tile widths from either boundary.
VELOPPOSE_LANE_HALF_WIDTH_FACTOR = 0.2


@dataclass(frozen=True)
class DirectedLaneReference:
    curve_point: np.ndarray
    tangent: np.ndarray
    robot_forward: np.ndarray
    lane_distance: float


class DirectedLaneTracker:
    """Follow one directed lane without using the robot heading to reselect it."""

    def __init__(self) -> None:
        self._tangent: np.ndarray | None = None

    def reset(self) -> None:
        self._tangent = None

    @staticmethod
    def _direction_from_yaw(yaw: float) -> np.ndarray:
        return np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float64)

    @staticmethod
    def _yaw_from_direction(direction: np.ndarray) -> float:
        return math.atan2(-float(direction[2]), float(direction[0]))

    def update(
        self,
        position: np.ndarray,
        robot_yaw: float,
        closest_curve_point: Callable[[np.ndarray, float], tuple[Any, Any]],
    ) -> DirectedLaneReference | None:
        position = np.asarray(position, dtype=np.float64)
        selection_yaw = (
            float(robot_yaw)
            if self._tangent is None
            else self._yaw_from_direction(self._tangent)
        )
        curve_point, tangent = closest_curve_point(position, selection_yaw)
        if curve_point is None or tangent is None:
            return None

        curve_point = np.asarray(curve_point, dtype=np.float64).copy()
        tangent = np.asarray(tangent, dtype=np.float64).copy()
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            return None
        tangent /= tangent_norm

        if self._tangent is not None and float(np.dot(tangent, self._tangent)) < 0.0:
            tangent = -tangent
        self._tangent = tangent.copy()

        robot_forward = self._direction_from_yaw(float(robot_yaw))
        right = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
        lane_distance = float(np.dot(position - curve_point, right))
        return DirectedLaneReference(
            curve_point=curve_point,
            tangent=tangent,
            robot_forward=robot_forward,
            lane_distance=lane_distance,
        )


def compute_velopose_breakdown(
    *,
    current_position: np.ndarray,
    previous_position: np.ndarray,
    lane_tangent: np.ndarray,
    robot_forward: np.ndarray,
    delta_time: float,
    lane_distance: float,
    lane_half_width: float,
    reference_speed: float = VELOPPOSE_REFERENCE_SPEED_MPS,
) -> dict[str, Any]:
    current_position = np.asarray(current_position, dtype=np.float64)
    previous_position = np.asarray(previous_position, dtype=np.float64)
    tangent = np.asarray(lane_tangent, dtype=np.float64)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        raise ValueError("lane_tangent must be non-zero")
    tangent /= tangent_norm

    robot_forward = np.asarray(robot_forward, dtype=np.float64)
    robot_forward_norm = float(np.linalg.norm(robot_forward))
    if robot_forward_norm <= 1e-12:
        raise ValueError("robot_forward must be non-zero")
    robot_forward /= robot_forward_norm

    delta_time = float(delta_time)
    reference_speed = float(reference_speed)
    lane_half_width = float(lane_half_width)
    if reference_speed <= 0.0:
        raise ValueError("reference_speed must be positive")
    if lane_half_width <= 0.0:
        raise ValueError("lane_half_width must be positive")

    progress = float(np.dot(current_position - previous_position, tangent))
    forward_speed = progress / delta_time if delta_time > 0.0 else 0.0
    normalized_forward_progress = float(
        np.clip(forward_speed / reference_speed, -1.0, 1.0)
    )

    heading_quality = float(np.clip(np.dot(robot_forward, tangent), -1.0, 1.0))
    normalized_abs_lane_distance = float(
        np.clip(abs(float(lane_distance)) / lane_half_width, 0.0, 1.0)
    )
    lane_center_quality = 1.0 - normalized_abs_lane_distance
    aligned_heading_contribution = lane_center_quality * heading_quality
    lane_distance_penalty = -normalized_abs_lane_distance
    pose_quality = aligned_heading_contribution + lane_distance_penalty

    velocity_contribution = VELOPPOSE_VELOCITY_WEIGHT * normalized_forward_progress
    pose_contribution = VELOPPOSE_POSE_WEIGHT * pose_quality
    total = velocity_contribution + pose_contribution

    return {
        "total": float(total),
        "components": {
            "Velocity": {
                "total": float(velocity_contribution),
                "components": {
                    "ProgressM": progress,
                    "DeltaTimeS": delta_time,
                    "ForwardSpeedMps": float(forward_speed),
                    "ReferenceSpeedMps": reference_speed,
                    "NormalizedForwardProgress": normalized_forward_progress,
                    "Weight": VELOPPOSE_VELOCITY_WEIGHT,
                },
            },
            "Pose": {
                "total": float(pose_contribution),
                "components": {
                    "HeadingQuality": heading_quality,
                    "NormalizedAbsLaneDistance": normalized_abs_lane_distance,
                    "LaneCenterQuality": lane_center_quality,
                    "AlignedHeadingContribution": aligned_heading_contribution,
                    "LaneDistancePenalty": lane_distance_penalty,
                    "PoseQuality": float(pose_quality),
                    "Weight": VELOPPOSE_POSE_WEIGHT,
                },
            },
        },
    }


def invalid_velopose_breakdown() -> dict[str, Any]:
    pose_quality = -1.0
    pose_contribution = VELOPPOSE_POSE_WEIGHT * pose_quality
    return {
        "total": pose_contribution,
        "components": {
            "Velocity": {
                "total": 0.0,
                "components": {
                    "NormalizedForwardProgress": 0.0,
                    "Weight": VELOPPOSE_VELOCITY_WEIGHT,
                },
            },
            "Pose": {
                "total": pose_contribution,
                "components": {
                    "PoseQuality": pose_quality,
                    "LaneValid": 0.0,
                    "Weight": VELOPPOSE_POSE_WEIGHT,
                },
            },
        },
    }
