#!/usr/bin/env python3
"""Safe policy-action to physical Duckiebot chassis-command adaptation.

This module deliberately has no ROS dependency.  A runtime can turn the
returned ``linear_velocity`` and ``angular_velocity`` values into a
``duckietown_msgs/Twist2DStamped`` only after all checks in this layer pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from time import monotonic
from typing import Any, Sequence

from duckietown_action_control import (
    DuckietownActionControl,
    action_control_from_config,
)


@dataclass(frozen=True)
class PhysicalControlLimits:
    """Physical command limits; conservative defaults for initial testing."""

    max_linear_velocity: float = 0.10
    max_angular_velocity: float = 1.50
    max_linear_acceleration: float = 0.25
    max_angular_acceleration: float = 3.00
    command_timeout: float = 0.50
    max_frame_age: float = 0.50
    nominal_control_period: float = 0.10
    forward_only: bool = True

    def __post_init__(self) -> None:
        positive_fields = (
            "max_linear_velocity",
            "max_angular_velocity",
            "max_linear_acceleration",
            "max_angular_acceleration",
            "command_timeout",
            "max_frame_age",
            "nominal_control_period",
        )
        for field_name in positive_fields:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")


@dataclass(frozen=True)
class ChassisCommand:
    """Auditable output of the physical-control safety layer."""

    linear_velocity: float
    angular_velocity: float
    target_linear_velocity: float
    target_angular_velocity: float
    normalized_left_wheel: float
    normalized_right_wheel: float
    policy_controls: tuple[float, ...]
    timestamp: float
    frame_age: float | None
    reason: str
    armed: bool
    emergency_stop_latched: bool
    linear_rate_limited: bool = False
    angular_rate_limited: bool = False
    coupled_limit_applied: bool = False

    @property
    def stopped(self) -> bool:
        return self.linear_velocity == 0.0 and self.angular_velocity == 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "stopped": self.stopped}


class PhysicalDuckiebotControl:
    """Stateful, fail-closed adapter from policy controls to ``v``/``omega``."""

    def __init__(
        self,
        action_control: DuckietownActionControl,
        limits: PhysicalControlLimits | None = None,
    ) -> None:
        self.action_control = action_control
        self.limits = limits or PhysicalControlLimits()
        self._armed = False
        self._emergency_stop_latched = False
        self._last_update_at: float | None = None
        self._current_linear_velocity = 0.0
        self._current_angular_velocity = 0.0
        self._last_command = self._stop_command(
            timestamp=monotonic(),
            reason="disarmed",
        )

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def emergency_stop_latched(self) -> bool:
        return self._emergency_stop_latched

    @property
    def last_command(self) -> ChassisCommand:
        return self._last_command

    def arm(self, timestamp: float | None = None) -> ChassisCommand:
        now = self._timestamp(timestamp)
        if self._emergency_stop_latched:
            return self._set_stop(now, "emergency_stop_latched")
        self._armed = True
        self._last_update_at = now
        return self._set_stop(now, "armed_waiting_for_command")

    def disarm(self, timestamp: float | None = None) -> ChassisCommand:
        now = self._timestamp(timestamp)
        self._armed = False
        self._last_update_at = None
        return self._set_stop(now, "disarmed")

    def engage_emergency_stop(
        self,
        timestamp: float | None = None,
    ) -> ChassisCommand:
        now = self._timestamp(timestamp)
        self._emergency_stop_latched = True
        self._armed = False
        self._last_update_at = None
        return self._set_stop(now, "emergency_stop_latched")

    def clear_emergency_stop(
        self,
        timestamp: float | None = None,
    ) -> ChassisCommand:
        now = self._timestamp(timestamp)
        self._emergency_stop_latched = False
        self._armed = False
        self._last_update_at = None
        return self._set_stop(now, "emergency_stop_cleared_disarmed")

    def update(
        self,
        policy_controls: Sequence[float],
        *,
        timestamp: float | None = None,
        frame_age: float | None = None,
    ) -> ChassisCommand:
        """Map one policy output to a bounded chassis command.

        ``timestamp`` must use a monotonic clock. ``frame_age`` must already be
        computed by the runtime because ROS wall-clock stamps and monotonic
        process time are different clock domains.
        """

        now = self._timestamp(timestamp)
        if self._emergency_stop_latched:
            return self._set_stop(now, "emergency_stop_latched", frame_age=frame_age)
        if not self._armed:
            return self._set_stop(now, "disarmed", frame_age=frame_age)
        if self._last_update_at is not None and now < self._last_update_at:
            return self._fault_stop(
                now,
                "non_monotonic_timestamp",
                frame_age=frame_age,
            )
        if frame_age is not None:
            if not math.isfinite(frame_age) or frame_age < 0.0:
                return self._fault_stop(
                    now,
                    "invalid_frame_age",
                    frame_age=frame_age,
                )
            if frame_age > self.limits.max_frame_age:
                return self._fault_stop(now, "stale_frame", frame_age=frame_age)

        try:
            controls = tuple(float(value) for value in policy_controls)
            left, right = self.action_control.to_wheels_pair(controls)
        except (TypeError, ValueError, OverflowError):
            return self._fault_stop(
                now,
                "invalid_policy_controls",
                frame_age=frame_age,
            )

        normalized_linear = 0.5 * (left + right)
        normalized_angular = 0.5 * (right - left)
        if self.limits.forward_only:
            normalized_linear = max(0.0, normalized_linear)

        target_linear = normalized_linear * self.limits.max_linear_velocity
        target_angular = normalized_angular * self.limits.max_angular_velocity
        delta_time = self._delta_time(now)
        linear, linear_limited = self._move_towards(
            self._current_linear_velocity,
            target_linear,
            self.limits.max_linear_acceleration * delta_time,
        )
        angular, angular_limited = self._move_towards(
            self._current_angular_velocity,
            target_angular,
            self.limits.max_angular_acceleration * delta_time,
        )
        linear, angular, coupled_limited = self._enforce_coupled_limit(
            linear,
            angular,
        )

        self._current_linear_velocity = linear
        self._current_angular_velocity = angular
        self._last_update_at = now
        self._last_command = ChassisCommand(
            linear_velocity=linear,
            angular_velocity=angular,
            target_linear_velocity=target_linear,
            target_angular_velocity=target_angular,
            normalized_left_wheel=left,
            normalized_right_wheel=right,
            policy_controls=controls,
            timestamp=now,
            frame_age=frame_age,
            reason="active",
            armed=True,
            emergency_stop_latched=False,
            linear_rate_limited=linear_limited,
            angular_rate_limited=angular_limited,
            coupled_limit_applied=coupled_limited,
        )
        return self._last_command

    def watchdog(self, timestamp: float | None = None) -> ChassisCommand:
        """Return the current command, or stop after a command timeout."""

        now = self._timestamp(timestamp)
        if self._emergency_stop_latched:
            return self._set_stop(now, "emergency_stop_latched")
        if not self._armed:
            return self._set_stop(now, "disarmed")
        if self._last_update_at is None:
            return self._fault_stop(now, "no_command")
        if now < self._last_update_at:
            return self._fault_stop(now, "non_monotonic_timestamp")
        if now - self._last_update_at > self.limits.command_timeout:
            return self._fault_stop(now, "watchdog_timeout")
        return self._last_command

    def _delta_time(self, now: float) -> float:
        if self._last_update_at is None:
            return self.limits.nominal_control_period
        return now - self._last_update_at

    def _enforce_coupled_limit(
        self,
        linear: float,
        angular: float,
    ) -> tuple[float, float, bool]:
        load = (
            abs(linear) / self.limits.max_linear_velocity
            + abs(angular) / self.limits.max_angular_velocity
        )
        if load <= 1.0:
            return linear, angular, False
        return linear / load, angular / load, True

    def _set_stop(
        self,
        timestamp: float,
        reason: str,
        *,
        frame_age: float | None = None,
    ) -> ChassisCommand:
        self._current_linear_velocity = 0.0
        self._current_angular_velocity = 0.0
        self._last_command = self._stop_command(
            timestamp=timestamp,
            reason=reason,
            frame_age=frame_age,
        )
        return self._last_command

    def _fault_stop(
        self,
        timestamp: float,
        reason: str,
        *,
        frame_age: float | None = None,
    ) -> ChassisCommand:
        # Safety faults require an explicit arm() before movement can resume.
        self._armed = False
        self._last_update_at = None
        return self._set_stop(timestamp, reason, frame_age=frame_age)

    def _stop_command(
        self,
        *,
        timestamp: float,
        reason: str,
        frame_age: float | None = None,
    ) -> ChassisCommand:
        return ChassisCommand(
            linear_velocity=0.0,
            angular_velocity=0.0,
            target_linear_velocity=0.0,
            target_angular_velocity=0.0,
            normalized_left_wheel=0.0,
            normalized_right_wheel=0.0,
            policy_controls=(),
            timestamp=timestamp,
            frame_age=frame_age,
            reason=reason,
            armed=self._armed,
            emergency_stop_latched=self._emergency_stop_latched,
        )

    @staticmethod
    def _timestamp(value: float | None) -> float:
        timestamp = monotonic() if value is None else float(value)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative")
        return timestamp

    @staticmethod
    def _move_towards(
        current: float,
        target: float,
        maximum_delta: float,
    ) -> tuple[float, bool]:
        delta = target - current
        if abs(delta) <= maximum_delta:
            return target, False
        return current + math.copysign(maximum_delta, delta), True


def hardware_control_from_checkpoint_config(
    checkpoint_config: dict[str, Any],
    limits: PhysicalControlLimits | None = None,
) -> PhysicalDuckiebotControl:
    """Construct the adapter using the action semantics stored in a checkpoint."""

    return PhysicalDuckiebotControl(
        action_control=action_control_from_config(checkpoint_config),
        limits=limits,
    )
