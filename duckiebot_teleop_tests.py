#!/usr/bin/env python3
"""Unit tests for device-neutral teleop and physical dataset recording."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from duckiebot_dataset_recorder import PhysicalDatasetRecorder
from duckiebot_teleop_input import (
    ActionMixer,
    DriveProfile,
    InputState,
    SDLControllerInput,
    input_state_from_json,
)
from physical_duckiebot_teleop import _effective_wheels
from duckiebot_hardware_control import ChassisCommand, PhysicalControlLimits


class ActionMixerTests(unittest.TestCase):
    def test_keyboard_and_joystick_semantics_share_mixer(self) -> None:
        mixer = ActionMixer(
            DriveProfile(throttle_rate=100.0, steering_rate=100.0)
        )
        action = mixer.update(InputState(throttle=1.0, steering=1.0), 0.1)
        self.assertAlmostEqual(action[0], 0.0)
        self.assertAlmostEqual(action[1], 1.0)

    def test_deadzone_suppresses_stick_drift(self) -> None:
        mixer = ActionMixer()
        action = mixer.update(InputState(throttle=0.02, steering=-0.03), 0.1)
        self.assertEqual(action, (0.0, 0.0))

    def test_deadzone_does_not_weaken_values_above_threshold(self) -> None:
        mixer = ActionMixer(
            DriveProfile(
                throttle_rate=100.0,
                steering_rate=100.0,
                auto_center_rate=100.0,
            )
        )
        action = mixer.update(InputState(throttle=0.10), 0.1)
        self.assertAlmostEqual(action[0], 0.10)
        self.assertAlmostEqual(action[1], 0.10)

    def test_analog_control_bypasses_ramps(self) -> None:
        mixer = ActionMixer(
            DriveProfile(throttle_rate=0.01, steering_rate=0.01)
        )
        action = mixer.update(
            InputState(throttle=-1.0, steering=0.5),
            0.001,
            smooth=False,
        )
        self.assertAlmostEqual(action[0], -1.0)
        self.assertAlmostEqual(action[1], -1.0 / 3.0)

    def test_bridge_message_is_validated_and_clamped(self) -> None:
        state = input_state_from_json(
            '{"throttle":2,"steering":-2,"recording_toggle":true}'
        )
        self.assertEqual(state.throttle, 1.0)
        self.assertEqual(state.steering, -1.0)
        self.assertTrue(state.recording_toggle)

    def test_bridge_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValueError):
            input_state_from_json('{"throttle":0,"surprise":true}')

    def test_ps4_button_polling_emits_only_rising_edge(self) -> None:
        class FakePygame:
            QUIT = 1
            KEYDOWN = 2
            K_ESCAPE = 27

        class FakeJoystick:
            buttons = [False] * 7

            def get_numbuttons(self):
                return len(self.buttons)

            def get_button(self, index):
                return self.buttons[index]

            def get_numaxes(self):
                return 2

            def get_axis(self, index):
                return 0.0

        from duckiebot_teleop_input import PS4Input

        joystick = FakeJoystick()
        controller = PS4Input(FakePygame(), joystick)
        joystick.buttons[0] = True
        self.assertTrue(controller.poll([]).arm_toggle)
        self.assertFalse(controller.poll([]).arm_toggle)
        joystick.buttons[0] = False
        controller.poll([])
        joystick.buttons[0] = True
        self.assertTrue(controller.poll([]).arm_toggle)

    def test_standard_sdl_controller_normalizes_signed_axes(self) -> None:
        class FakePygame:
            QUIT = 1
            KEYDOWN = 2
            K_ESCAPE = 27
            CONTROLLER_AXIS_MAX = 6
            CONTROLLER_BUTTON_MAX = 21

        class FakeController:
            axes = {0: 16384, 1: -32768}
            buttons = [False] * 21

            def get_axis(self, index):
                return self.axes[index]

            def get_button(self, index):
                return self.buttons[index]

        state = SDLControllerInput(FakePygame(), FakeController()).poll([])
        self.assertEqual(state.throttle, 1.0)
        self.assertAlmostEqual(state.steering, -16384 / 32767)


class EffectiveWheelTests(unittest.TestCase):
    def test_labels_describe_sent_rate_limited_command(self) -> None:
        limits = PhysicalControlLimits()
        command = ChassisCommand(
            linear_velocity=0.05,
            angular_velocity=0.75,
            target_linear_velocity=0.1,
            target_angular_velocity=1.5,
            normalized_left_wheel=0.0,
            normalized_right_wheel=1.0,
            policy_controls=(0.0, 1.0),
            timestamp=1.0,
            frame_age=0.0,
            reason="active",
            armed=True,
            emergency_stop_latched=False,
        )
        self.assertEqual(_effective_wheels(command, limits), (0.0, 1.0))


class RecorderTests(unittest.TestCase):
    def test_writes_training_compatible_rows_and_skips_duplicate_frame(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            recorder = PhysicalDatasetRecorder(
                Path(temporary_dir), {"env": "test"}
            )
            arguments = {
                "payload": b"\xff\xd8\xfftest",
                "image_suffix": ".jpg",
                "camera_seq": 7,
                "camera_stamp": 12.5,
                "frame_age": 0.02,
                "left_action": 0.1,
                "right_action": 0.2,
                "linear_velocity": 0.03,
                "angular_velocity": 0.1,
            }
            self.assertTrue(recorder.record(**arguments))
            self.assertFalse(recorder.record(**arguments))
            run_dir = recorder.run_dir
            recorder.close()

            with (run_dir / "actions.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image"], "run_001_000000.jpg")
            self.assertEqual(rows[0]["left_action"], "0.100000000")
            self.assertTrue((run_dir / "images" / rows[0]["image"]).exists())
            metadata = json.loads((run_dir / "meta.json").read_text())
            self.assertEqual(metadata["num_samples"], 1)
            self.assertFalse(metadata["recording"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
