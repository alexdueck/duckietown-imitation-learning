from __future__ import annotations

import math
from typing import Any

import numpy as np

from velopose_reward import (
    DirectedLaneTracker,
    POSEPOT_DEFAULT_GAMMA,
    VD2PP_DISTANCE_SQUARED_WEIGHT,
    VELOPPOSE_INVALID_POSE_PENALTY,
    VELOPPOSE_LANE_HALF_WIDTH_FACTOR,
    compute_posepot_breakdown,
    compute_vd2pp_breakdown,
    compute_velopose_breakdown,
    invalid_velopose_breakdown,
)


REWARD_FUNCTION_CHOICES = (
    "default",
    "default_clipped",
    "posangle",
    "target_orientation",
    "distance_travelled",
    "velopose",
    "posepot",
    "vd2pp",
)

DISPLAY_REWARD_FUNCTIONS = (
    "default",
    "posangle",
    "target_orientation",
    "distance_travelled",
    "velopose",
    "posepot",
    "vd2pp",
)

INVALID_POSE_DONE_CODE = "invalid-pose"
MAX_STEPS_DONE_CODE = "max-steps-reached"


def reward_source(name: str) -> str:
    if name == "velopose":
        return "Custom signed forward-velocity and lane-pose reward"
    if name == "posepot":
        return "Custom signed forward-velocity reward with potential-based lane-pose shaping"
    if name == "vd2pp":
        return (
            "Custom signed forward-velocity reward with direct squared lane-distance "
            "cost and potential-based lane-pose shaping"
        )
    return "kaland313/Duckietown-RL reward_wrappers.py on gym-duckietown Simulator"


def safe_float(value: float) -> float:
    value = float(value)
    if math.isnan(value):
        return 0.0
    return value


def gym_duckietown_done_code(done: bool, info: dict[str, Any] | None) -> str:
    simulator_info = info.get("Simulator", {}) if isinstance(info, dict) else {}
    code = simulator_info.get("done_code")
    if code:
        return str(code)

    message = str(simulator_info.get("msg", "")).lower()
    if "invalid pose" in message:
        return INVALID_POSE_DONE_CODE
    if "max_steps" in message or "max steps" in message:
        return MAX_STEPS_DONE_CODE
    return "terminated" if done else "in-progress"


def unwrapped_env(env):
    return getattr(env, "unwrapped", env)


def format_wheel_action(action) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.size != 2:
        raise ValueError(f"Expected a two-element action, got shape {np.asarray(action).shape}")
    return np.ascontiguousarray(np.clip(action_array, -1.0, 1.0), dtype=np.float32)


def patch_duckietown_world_dynamics() -> None:
    """Keep old gym-duckietown dynamics working with newer duckietown-world."""
    try:
        from duckietown_world.world_duckietown import pwm_dynamics
    except ImportError:
        return

    original_integrate = pwm_dynamics.DynamicModel.integrate
    if getattr(original_integrate, "_rl_scalar_compatible", False):
        return

    def scalar_compatible_integrate(self, dt: float, commands):
        linear_prev, angular_prev = pwm_dynamics.geo.linear_angular_from_se2(self.v0)
        linear_prev = np.asarray(linear_prev, dtype=np.float64).reshape(-1)
        longitudinal_prev = float(linear_prev[0])
        angular_prev = float(angular_prev)

        acceleration = self.model(commands, self.parameters, u=longitudinal_prev, w=angular_prev)
        acceleration = np.asarray(acceleration, dtype=np.float64).reshape(-1)
        longitudinal = float(longitudinal_prev + dt * acceleration[0])
        angular = float(angular_prev + dt * acceleration[1])

        commands_se2 = pwm_dynamics.geo.se2_from_linear_angular([longitudinal, 0.0], angular)
        next_kinematics = pwm_dynamics.GenericKinematicsSE2.integrate(self, dt, commands_se2)

        wheel_distance = self.parameters.wheel_distance
        radius_right = self.parameters.wheel_radius_right
        radius_left = self.parameters.wheel_radius_left
        wheel_matrix = np.array(
            [[radius_right / wheel_distance, -radius_left / wheel_distance], [radius_right / 2, radius_left / 2]]
        )
        wheel_rates = (np.linalg.inv(wheel_matrix) @ np.array([angular, longitudinal])).reshape(-1)
        right_wheel_rate = float(wheel_rates[0])
        left_wheel_rate = float(wheel_rates[1])

        return pwm_dynamics.DynamicModel(
            self.parameters,
            (next_kinematics.q0, next_kinematics.v0),
            next_kinematics.t0,
            axis_left_rad=self.axis_left_rad + left_wheel_rate * dt,
            axis_right_rad=self.axis_right_rad + right_wheel_rate * dt,
        )

    scalar_compatible_integrate._rl_scalar_compatible = True
    pwm_dynamics.DynamicModel.integrate = scalar_compatible_integrate


class GymDuckietownRewardCalculator:
    """Project reward functions on top of gym-duckietown Simulator state."""

    def __init__(
        self,
        name: str,
        *,
        gamma: float = POSEPOT_DEFAULT_GAMMA,
        vd2pp_distance_weight: float = VD2PP_DISTANCE_SQUARED_WEIGHT,
    ) -> None:
        if name not in REWARD_FUNCTION_CHOICES:
            raise ValueError(f"Unknown reward function {name!r}")
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if float(vd2pp_distance_weight) < 0.0:
            raise ValueError("vd2pp distance weight must be non-negative")
        self.name = name
        self.gamma = float(gamma)
        self.vd2pp_distance_weight = float(vd2pp_distance_weight)
        self.prev_pos: np.ndarray | None = None
        self.prev_timestamp: float | None = None
        self.previous_pose_potential: float | None = None
        self.velopose_lane_tracker = DirectedLaneTracker()
        self.orientation_reward = 0.0
        self.velocity_reward = 0.0

    def reset(self, env=None) -> None:
        self.prev_pos = None
        self.prev_timestamp = None
        self.previous_pose_potential = None
        self.velopose_lane_tracker.reset()
        self.orientation_reward = 0.0
        self.velocity_reward = 0.0
        if env is not None and self.name in ("velopose", "posepot", "vd2pp"):
            raw_env = unwrapped_env(env)
            self.prev_pos = np.asarray(raw_env.cur_pos, dtype=np.float64).copy()
            self.prev_timestamp = float(raw_env.timestamp)
            try:
                lane_reference = self.velopose_lane_tracker.update(
                    position=self.prev_pos,
                    robot_yaw=float(raw_env.cur_angle),
                    closest_curve_point=raw_env.closest_curve_point,
                )
                if self.name in ("posepot", "vd2pp") and lane_reference is not None:
                    initial_breakdown = compute_velopose_breakdown(
                        current_position=self.prev_pos,
                        previous_position=self.prev_pos,
                        lane_tangent=lane_reference.tangent,
                        robot_forward=lane_reference.robot_forward,
                        delta_time=0.0,
                        lane_distance=lane_reference.lane_distance,
                        lane_half_width=(
                            VELOPPOSE_LANE_HALF_WIDTH_FACTOR
                            * float(raw_env.road_tile_size)
                        ),
                    )
                    self.previous_pose_potential = float(
                        initial_breakdown["components"]["Pose"]["components"][
                            "PoseQuality"
                        ]
                    )
            except Exception:
                pass

    def compute(self, env, env_reward: float, done_code: str | None = None) -> float:
        return safe_float(float(self.compute_breakdown(env, env_reward, done_code)["total"]))

    def compute_breakdown(
        self,
        env,
        env_reward: float,
        done_code: str | None = None,
    ) -> dict[str, float | dict[str, float]]:
        if self.name == "default":
            total = safe_float(env_reward)
            return {"total": total, "components": {"gym_duckietown": total}}
        if self.name == "default_clipped":
            total = float(np.clip(safe_float(env_reward), -2.0, 2.0))
            return {"total": total, "components": {"gym_duckietown_clipped": total}}
        if self.name == "posangle":
            return self._posangle_breakdown(env, target_orientation_only=False)
        if self.name == "target_orientation":
            return self._posangle_breakdown(env, target_orientation_only=True)
        if self.name == "distance_travelled":
            total = safe_float(self._distance_travelled_reward(env))
            return {"total": total, "components": {"DtRewardDistanceTravelled": total}}
        if self.name == "velopose":
            breakdown = self._velopose_breakdown(env)
            if done_code == INVALID_POSE_DONE_CODE:
                components = dict(breakdown["components"])
                components["InvalidPosePenalty"] = VELOPPOSE_INVALID_POSE_PENALTY
                return {
                    "total": float(breakdown["total"]) + VELOPPOSE_INVALID_POSE_PENALTY,
                    "components": components,
                }
            return breakdown
        if self.name in ("posepot", "vd2pp"):
            terminal = done_code not in (None, "in-progress")
            velopose_breakdown = self._velopose_breakdown(env)
            current_pose_quality = float(
                velopose_breakdown["components"]["Pose"]["components"].get(
                    "PoseQuality",
                    velopose_breakdown["components"]["Pose"]["total"],
                )
            )
            if self.name == "posepot":
                breakdown = compute_posepot_breakdown(
                    velopose_breakdown,
                    previous_pose_quality=self.previous_pose_potential,
                    gamma=self.gamma,
                    terminal=terminal,
                )
            else:
                breakdown = compute_vd2pp_breakdown(
                    velopose_breakdown,
                    previous_pose_quality=self.previous_pose_potential,
                    gamma=self.gamma,
                    terminal=terminal,
                    distance_squared_weight=self.vd2pp_distance_weight,
                )
            self.previous_pose_potential = current_pose_quality
            if done_code == INVALID_POSE_DONE_CODE:
                components = dict(breakdown["components"])
                components["InvalidPosePenalty"] = VELOPPOSE_INVALID_POSE_PENALTY
                return {
                    "total": float(breakdown["total"])
                    + VELOPPOSE_INVALID_POSE_PENALTY,
                    "components": components,
                }
            return breakdown
        raise AssertionError(f"Unhandled reward function {self.name!r}")

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

    def _posangle_breakdown(self, env, target_orientation_only: bool) -> dict[str, float | dict[str, float]]:
        raw_env = unwrapped_env(env)
        try:
            lane_position = raw_env.get_lane_pos2(raw_env.cur_pos, raw_env.cur_angle)
        except Exception:
            self.orientation_reward = -10.0
            self.velocity_reward = self._velocity_reward(raw_env)
        else:
            if target_orientation_only:
                self.orientation_reward = self._target_angle_reward(
                    lane_position.dist,
                    lane_position.angle_deg,
                    max_dev_deg=50.0,
                )
            else:
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
                self.orientation_reward = 0.5 * (narrow_reward + wide_reward)

            self.velocity_reward = self._velocity_reward(raw_env)

        total = safe_float(self.orientation_reward + self.velocity_reward)
        if target_orientation_only:
            components = {
                "DtRewardTargetOrientation": safe_float(self.orientation_reward),
                "DtRewardVelocity": safe_float(self.velocity_reward),
            }
        else:
            components = {
                "DtRewardPosAngle": safe_float(self.orientation_reward),
                "DtRewardVelocity": safe_float(self.velocity_reward),
            }
        return {"total": total, "components": components}

    def _velocity_reward(self, raw_env) -> float:
        wheel_vels = getattr(raw_env, "wheelVels", np.array([0.0, 0.0]))
        return safe_float(float(np.max(np.asarray(wheel_vels, dtype=np.float64))) * 0.25)

    def _distance_travelled_reward(self, env) -> float:
        raw_env = unwrapped_env(env)
        pos = np.asarray(raw_env.cur_pos, dtype=np.float64).copy()
        prev_pos = None if self.prev_pos is None else self.prev_pos.copy()
        self.prev_pos = pos
        if prev_pos is None:
            return 0.0

        angle = float(raw_env.cur_angle)
        try:
            curve_point, tangent = raw_env.closest_curve_point(pos, angle)
            prev_curve_point, _ = raw_env.closest_curve_point(prev_pos, angle)
            lane_position = raw_env.get_lane_pos2(pos, angle)
        except Exception:
            return 0.0

        if curve_point is None or prev_curve_point is None or tangent is None:
            return 0.0

        diff = curve_point - prev_curve_point
        distance = float(np.linalg.norm(diff))

        if float(lane_position.dist) < -0.05:
            return 0.0
        if float(np.dot(tangent, diff)) < 0.0:
            return 0.0

        return safe_float(50.0 * distance)

    def _velopose_breakdown(self, env) -> dict[str, Any]:
        raw_env = unwrapped_env(env)
        pos = np.asarray(raw_env.cur_pos, dtype=np.float64).copy()
        timestamp = float(raw_env.timestamp)
        previous_pos = pos if self.prev_pos is None else self.prev_pos.copy()
        previous_timestamp = timestamp if self.prev_timestamp is None else self.prev_timestamp
        self.prev_pos = pos
        self.prev_timestamp = timestamp

        try:
            lane_reference = self.velopose_lane_tracker.update(
                position=pos,
                robot_yaw=float(raw_env.cur_angle),
                closest_curve_point=raw_env.closest_curve_point,
            )
            if lane_reference is None:
                return invalid_velopose_breakdown()
            lane_half_width = (
                VELOPPOSE_LANE_HALF_WIDTH_FACTOR * float(raw_env.road_tile_size)
            )
            return compute_velopose_breakdown(
                current_position=pos,
                previous_position=previous_pos,
                lane_tangent=lane_reference.tangent,
                robot_forward=lane_reference.robot_forward,
                delta_time=timestamp - previous_timestamp,
                lane_distance=lane_reference.lane_distance,
                lane_half_width=lane_half_width,
            )
        except Exception:
            return invalid_velopose_breakdown()


def create_reward_calculators(
    names: tuple[str, ...] = DISPLAY_REWARD_FUNCTIONS,
    *,
    posepot_gamma: float = POSEPOT_DEFAULT_GAMMA,
    vd2pp_distance_weight: float = VD2PP_DISTANCE_SQUARED_WEIGHT,
) -> dict[str, GymDuckietownRewardCalculator]:
    return {
        name: GymDuckietownRewardCalculator(
            name,
            gamma=posepot_gamma,
            vd2pp_distance_weight=vd2pp_distance_weight,
        )
        for name in names
    }


def reset_reward_calculators(
    calculators: dict[str, GymDuckietownRewardCalculator],
    env=None,
) -> None:
    for calculator in calculators.values():
        calculator.reset(env)


def compute_reward_breakdowns(
    env,
    env_reward: float,
    calculators: dict[str, GymDuckietownRewardCalculator],
    done_code: str | None = None,
) -> dict[str, dict[str, float | dict[str, float]]]:
    return {
        name: calculator.compute_breakdown(env, env_reward, done_code)
        for name, calculator in calculators.items()
    }


def get_lane_metrics(env) -> dict[str, Any]:
    raw_env = unwrapped_env(env)
    metrics: dict[str, Any] = {
        "lane_valid": False,
        "dist": 0.0,
        "dot_dir": 0.0,
        "angle_deg": 0.0,
        "speed": safe_float(getattr(raw_env, "speed", 0.0)),
    }
    try:
        lane_position = raw_env.get_lane_pos2(raw_env.cur_pos, raw_env.cur_angle)
    except Exception:
        return metrics

    metrics.update({
        "lane_valid": True,
        "dist": safe_float(lane_position.dist),
        "dot_dir": safe_float(lane_position.dot_dir),
        "angle_deg": safe_float(lane_position.angle_deg),
    })
    return metrics
