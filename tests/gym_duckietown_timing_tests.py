"""Tests for real-time gym-duckietown viewer scheduling."""

from __future__ import annotations

import unittest

from manual_control_gym_duckietown import environment_step_interval_seconds


class GymDuckietownTimingTests(unittest.TestCase):
    def test_frame_skip_scales_wall_time_between_environment_steps(self) -> None:
        self.assertAlmostEqual(
            environment_step_interval_seconds(frame_rate=30, frame_skip=3),
            0.1,
        )

    def test_single_frame_step_uses_physics_interval(self) -> None:
        self.assertAlmostEqual(
            environment_step_interval_seconds(frame_rate=30, frame_skip=1),
            1.0 / 30.0,
        )

    def test_invalid_timing_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            environment_step_interval_seconds(frame_rate=0, frame_skip=1)
        with self.assertRaises(ValueError):
            environment_step_interval_seconds(frame_rate=30, frame_skip=0)


if __name__ == "__main__":
    unittest.main()
