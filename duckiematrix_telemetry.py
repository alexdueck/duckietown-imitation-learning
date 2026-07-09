from __future__ import annotations

import math
from typing import Any

import numpy as np


TELEMETRY_COLUMNS = [
    "reward",
    "speed",
    "lane_dot_dir",
    "lane_dist",
    "lane_angle_deg",
    "lane_angle_rad",
    "lane_position_valid",
    "terminated",
    "truncated",
]


def empty_step_telemetry(
    reward: float = math.nan,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, float | bool]:
    return {
        "reward": float(reward),
        "speed": math.nan,
        "lane_dot_dir": math.nan,
        "lane_dist": math.nan,
        "lane_angle_deg": math.nan,
        "lane_angle_rad": math.nan,
        "lane_position_valid": False,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _pose_position(pose: dict[str, Any]) -> tuple[float, float, float]:
    position = pose["position"]
    return float(position["x"]), float(position["y"]), float(position["z"])


def _pose_timestamp(pose: dict[str, Any]) -> float:
    return float(pose["header"]["timestamp"])


def _pose_yaw(pose: dict[str, Any]) -> float:
    from gym_duckiematrix.utils import quaternion_to_euler

    rotation = pose["rotation"]
    quat_rot = [
        rotation["w"],
        rotation["x"],
        rotation["y"],
        rotation["z"],
    ]
    return float(quaternion_to_euler(quat_rot)[-1])


def compute_speed(previous_pose: dict[str, Any] | None, current_pose: dict[str, Any] | None) -> float:
    if previous_pose is None or current_pose is None:
        return math.nan

    previous_t = _pose_timestamp(previous_pose)
    current_t = _pose_timestamp(current_pose)
    delta_t = current_t - previous_t
    if delta_t <= 0:
        return 0.0

    previous_position = np.array(_pose_position(previous_pose), dtype=np.float64)
    current_position = np.array(_pose_position(current_pose), dtype=np.float64)
    return float(np.linalg.norm(current_position - previous_position) / delta_t)


def collect_step_telemetry(
    env,
    previous_pose: dict[str, Any] | None,
    reward: float | None = None,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, float | bool]:
    telemetry = empty_step_telemetry(
        reward=math.nan if reward is None else reward,
        terminated=terminated,
        truncated=truncated,
    )
    current_pose = getattr(env, "last_pose", None)
    telemetry["speed"] = compute_speed(previous_pose, current_pose)

    if current_pose is None or not hasattr(env, "lp_cal"):
        return telemetry

    try:
        x, y, z = _pose_position(current_pose)
        yaw = _pose_yaw(current_pose)
        lane_position = env.lp_cal.get_lane_pos2(np.array([x, y, z]), yaw)
    except Exception:
        return telemetry

    telemetry.update(
        {
            "lane_dot_dir": float(lane_position.dot_dir),
            "lane_dist": float(lane_position.dist),
            "lane_angle_deg": float(lane_position.angle_deg),
            "lane_angle_rad": float(lane_position.angle_rad),
            "lane_position_valid": True,
        }
    )
    if reward is None:
        telemetry["reward"] = telemetry["speed"] * telemetry["lane_dot_dir"] - 10.0 * abs(telemetry["lane_dist"])
    return telemetry


def collect_state_telemetry(
    env,
    previous_pose: dict[str, Any] | None,
    reward: float = math.nan,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, float | bool]:
    return collect_step_telemetry(
        env=env,
        previous_pose=previous_pose,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )


def format_telemetry_value(value: float | bool) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"

    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.9f}"
