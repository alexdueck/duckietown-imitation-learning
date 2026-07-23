#!/usr/bin/env python3
"""Unit tests for physical Duckiebot action adaptation and safety state."""

from __future__ import annotations

import math
import unittest

from duckietown_action_control import DuckietownActionControl
from duckiebot_hardware_control import (
    PhysicalControlLimits,
    PhysicalDuckiebotControl,
    hardware_control_from_checkpoint_config,
)


FAST_LIMITS = PhysicalControlLimits(
    max_linear_velocity=0.10,
    max_angular_velocity=1.50,
    max_linear_acceleration=10.0,
    max_angular_acceleration=100.0,
    command_timeout=0.50,
    max_frame_age=0.25,
    nominal_control_period=0.10,
    forward_only=True,
)


class ActionControlScalarTests(unittest.TestCase):
    def test_direct_wheel_scaling_preserves_ratio(self) -> None:
        control = DuckietownActionControl(mode="wheel")
        self.assertEqual(control.to_wheels_pair((2.0, 1.0)), (1.0, 0.5))

    def test_throttle_steering_mapping(self) -> None:
        control = DuckietownActionControl(
            mode="throttle_steering",
            max_throttle=0.5,
            max_steering=0.25,
        )
        left, right = control.to_wheels_pair((1.0, 1.0))
        self.assertAlmostEqual(left, 0.25)
        self.assertAlmostEqual(right, 0.75)

    def test_fixed_throttle_mapping(self) -> None:
        control = DuckietownActionControl(
            mode="throttle_steering",
            fixed_throttle=0.4,
            max_steering=0.2,
        )
        left, right = control.to_wheels_pair((-1.0,))
        self.assertAlmostEqual(left, 0.6)
        self.assertAlmostEqual(right, 0.2)

    def test_nonfinite_control_is_rejected(self) -> None:
        control = DuckietownActionControl(mode="wheel")
        with self.assertRaises(ValueError):
            control.to_wheels_pair((math.nan, 0.0))


class PhysicalDuckiebotControlTests(unittest.TestCase):
    def make_control(self, limits: PhysicalControlLimits = FAST_LIMITS):
        return PhysicalDuckiebotControl(
            DuckietownActionControl(mode="wheel"),
            limits,
        )

    def test_starts_disarmed(self) -> None:
        control = self.make_control()
        command = control.update((1.0, 1.0), timestamp=1.0, frame_age=0.0)
        self.assertTrue(command.stopped)
        self.assertEqual(command.reason, "disarmed")

    def test_equal_wheels_map_to_forward_chassis_velocity(self) -> None:
        control = self.make_control()
        control.arm(timestamp=0.0)
        command = control.update((0.5, 0.5), timestamp=0.1, frame_age=0.01)
        self.assertAlmostEqual(command.linear_velocity, 0.05)
        self.assertAlmostEqual(command.angular_velocity, 0.0)

    def test_positive_right_minus_left_maps_to_positive_omega(self) -> None:
        control = self.make_control()
        control.arm(timestamp=0.0)
        command = control.update((0.0, 1.0), timestamp=0.1, frame_age=0.01)
        self.assertAlmostEqual(command.target_linear_velocity, 0.05)
        self.assertAlmostEqual(command.target_angular_velocity, 0.75)

    def test_forward_only_blocks_reverse(self) -> None:
        control = self.make_control()
        control.arm(timestamp=0.0)
        command = control.update((-1.0, -1.0), timestamp=0.1, frame_age=0.0)
        self.assertEqual(command.linear_velocity, 0.0)

    def test_slew_rate_limits_acceleration(self) -> None:
        limits = PhysicalControlLimits(
            max_linear_velocity=0.10,
            max_angular_velocity=1.50,
            max_linear_acceleration=0.20,
            max_angular_acceleration=3.00,
            command_timeout=0.50,
            max_frame_age=0.25,
            nominal_control_period=0.10,
        )
        control = self.make_control(limits)
        control.arm(timestamp=0.0)
        command = control.update((1.0, 1.0), timestamp=0.1, frame_age=0.0)
        self.assertAlmostEqual(command.linear_velocity, 0.02)
        self.assertTrue(command.linear_rate_limited)

    def test_stale_frame_stops_immediately(self) -> None:
        control = self.make_control()
        control.arm(timestamp=0.0)
        command = control.update((1.0, 1.0), timestamp=0.1, frame_age=0.30)
        self.assertTrue(command.stopped)
        self.assertEqual(command.reason, "stale_frame")
        self.assertFalse(command.armed)
        self.assertEqual(
            control.update((1.0, 1.0), timestamp=0.2, frame_age=0.0).reason,
            "disarmed",
        )

    def test_invalid_policy_control_stops_immediately(self) -> None:
        control = self.make_control()
        control.arm(timestamp=0.0)
        command = control.update((math.nan, 0.0), timestamp=0.1, frame_age=0.0)
        self.assertTrue(command.stopped)
        self.assertEqual(command.reason, "invalid_policy_controls")

    def test_watchdog_stops_after_timeout(self) -> None:
        control = self.make_control()
        control.arm(timestamp=0.0)
        control.update((1.0, 1.0), timestamp=0.1, frame_age=0.0)
        self.assertFalse(control.watchdog(timestamp=0.5).stopped)
        command = control.watchdog(timestamp=0.61)
        self.assertTrue(command.stopped)
        self.assertEqual(command.reason, "watchdog_timeout")
        self.assertFalse(command.armed)

    def test_emergency_stop_is_latched_and_requires_rearming(self) -> None:
        control = self.make_control()
        control.arm(timestamp=0.0)
        control.update((1.0, 1.0), timestamp=0.1, frame_age=0.0)
        self.assertTrue(control.engage_emergency_stop(timestamp=0.2).stopped)
        self.assertEqual(control.arm(timestamp=0.3).reason, "emergency_stop_latched")
        self.assertEqual(
            control.clear_emergency_stop(timestamp=0.4).reason,
            "emergency_stop_cleared_disarmed",
        )
        self.assertEqual(
            control.update((1.0, 1.0), timestamp=0.5).reason,
            "disarmed",
        )

    def test_checkpoint_action_configuration_is_used(self) -> None:
        control = hardware_control_from_checkpoint_config(
            {
                "action_mode": "throttle_steering",
                "fixed_throttle": 0.4,
                "max_throttle": 0.5,
                "max_steering": 0.2,
            },
            FAST_LIMITS,
        )
        control.arm(timestamp=0.0)
        command = control.update((0.0,), timestamp=0.1, frame_age=0.0)
        self.assertAlmostEqual(command.normalized_left_wheel, 0.4)
        self.assertAlmostEqual(command.normalized_right_wheel, 0.4)
        self.assertAlmostEqual(command.linear_velocity, 0.04)


if __name__ == "__main__":
    unittest.main(verbosity=2)
