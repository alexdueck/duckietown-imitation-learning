"""Unit tests for unified physical control without robot or network access."""

from __future__ import annotations

import argparse
import base64
import csv
from io import BytesIO
import json
import os
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from threading import Event, Lock
from time import monotonic, sleep
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import pygame

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from dt_utils.duckiebot_hardware_control import (
    ChassisCommand,
)
from dt_utils.duckiebot_teleop_input import InputState
from dt_utils.duckiebot_rosbridge import (
    RosbridgeCameraFrame,
    RosbridgeTwistPublisher,
    decode_rosbridge_camera_publish,
)
from physical_duckiebot_control import (
    InferenceResult,
    InferenceWorker,
    ManualControlDisplay,
    ModelActionRecorder,
    PublishedInference,
    alternate_control_mode,
    alternate_input_name,
    decode_frame_image,
    draw_status,
    effective_command_timeout,
    effective_wheel_actions,
    merge_keyboard_control_events,
    validate_args,
)
from view_model_actions_on_images import Prediction


def encoded_image(
    image_format: str = "PNG",
    *,
    color: tuple[int, int, int] = (12, 34, 56),
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 1), color=color).save(buffer, format=image_format)
    return buffer.getvalue()


def camera_frame(
    receive_id: int,
    *,
    payload: bytes | None = None,
) -> RosbridgeCameraFrame:
    return RosbridgeCameraFrame(
        payload=payload or encoded_image(),
        suffix=".png",
        message_format="png",
        seq=receive_id + 100,
        stamp=receive_id + 0.25,
        received_at=monotonic(),
        receive_id=receive_id,
    )


def prediction(
    policy_controls: tuple[float, ...] = (0.1, 0.2),
    wheels: tuple[float, float] = (0.3, 0.4),
) -> Prediction:
    return Prediction(
        policy_controls=np.asarray(policy_controls, dtype=np.float32),
        wheel_commands=np.asarray(wheels, dtype=np.float32),
        policy_std=None,
    )


def result_for(
    frame: RosbridgeCameraFrame,
    *,
    policy_controls: tuple[float, ...] = (0.1, 0.2),
    wheels: tuple[float, float] = (0.3, 0.4),
    epoch: int = 1,
) -> InferenceResult:
    return InferenceResult(
        frame=frame,
        image=Image.new("RGB", (2, 1)),
        prediction=prediction(policy_controls, wheels),
        inference_seconds=0.0125,
        epoch=epoch,
    )


def active_command(
    *,
    linear_velocity: float,
    angular_velocity: float,
    frame_age: float,
    wheel_speeds: tuple[float, float] = (0.0, 0.0),
    normalized_wheels: tuple[float, float] = (0.0, 0.0),
) -> ChassisCommand:
    return ChassisCommand(
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        wheel_speed_left=wheel_speeds[0],
        wheel_speed_right=wheel_speeds[1],
        normalized_left_wheel=normalized_wheels[0],
        normalized_right_wheel=normalized_wheels[1],
        policy_controls=(),
        timestamp=monotonic(),
        frame_age=frame_age,
        reason="active",
        armed=True,
        emergency_stop_latched=False,
    )


def wait_for_result(worker: InferenceWorker, timeout: float = 2.0) -> InferenceResult:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        try:
            return worker.get_nowait()
        except Empty:
            sleep(0.005)
    raise AssertionError("timed out waiting for inference result")


class CameraDecodeTests(unittest.TestCase):
    def test_decodes_base64_and_ros1_header(self) -> None:
        payload = encoded_image("JPEG")
        frame = decode_rosbridge_camera_publish(
            {
                "op": "publish",
                "msg": {
                    "format": "jpeg",
                    "data": base64.b64encode(payload).decode("ascii"),
                    "header": {
                        "seq": 17,
                        "stamp": {"secs": 12, "nsecs": 345_000_000},
                    },
                },
            },
            receive_id=9,
            received_at=100.5,
        )

        self.assertEqual(frame.payload, payload)
        self.assertEqual(frame.suffix, ".jpg")
        self.assertEqual(frame.seq, 17)
        self.assertAlmostEqual(frame.stamp, 12.345)
        self.assertEqual(frame.received_at, 100.5)
        self.assertEqual(frame.receive_id, 9)

    def test_decodes_byte_array_and_ros2_header_names(self) -> None:
        payload = encoded_image("PNG")
        frame = decode_rosbridge_camera_publish(
            {
                "op": "publish",
                "msg": {
                    "format": "",
                    "data": list(payload),
                    "header": {
                        "seq": 3,
                        "stamp": {"sec": 4, "nanosec": 500_000_000},
                    },
                },
            },
            receive_id=2,
            received_at=7.0,
        )

        self.assertEqual(frame.payload, payload)
        self.assertEqual(frame.suffix, ".png")
        self.assertEqual(frame.seq, 3)
        self.assertAlmostEqual(frame.stamp, 4.5)

    def test_rejects_invalid_base64_and_empty_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid base64"):
            decode_rosbridge_camera_publish(
                {"op": "publish", "msg": {"format": "jpeg", "data": "@@@"}},
                receive_id=1,
            )
        with self.assertRaisesRegex(ValueError, "empty"):
            decode_rosbridge_camera_publish(
                {"op": "publish", "msg": {"format": "jpeg", "data": []}},
                receive_id=1,
            )

    def test_rgb_decode_keeps_channels_and_bgr_option_swaps_them(self) -> None:
        frame = camera_frame(
            1,
            payload=encoded_image("PNG", color=(10, 20, 30)),
        )
        self.assertEqual(
            decode_frame_image(frame, "rgb").getpixel((0, 0)),
            (10, 20, 30),
        )
        self.assertEqual(
            decode_frame_image(frame, "bgr").getpixel((0, 0)),
            (30, 20, 10),
        )


class EffectiveWheelTests(unittest.TestCase):
    def test_returns_wheel_actions_represented_by_command(self) -> None:
        command = active_command(
            linear_velocity=0.0325,
            angular_velocity=0.0375,
            frame_age=0.01,
            normalized_wheels=(0.30, 0.35),
        )
        left, right = effective_wheel_actions(command)
        self.assertAlmostEqual(left, 0.30)
        self.assertAlmostEqual(right, 0.35)


class GuiRenderTests(unittest.TestCase):
    def test_manual_and_model_panels_render(self) -> None:
        pygame.font.init()
        screen = pygame.Surface((1080, 820))
        fonts = {
            "large": pygame.font.Font(None, 38),
            "normal": pygame.font.Font(None, 25),
            "small": pygame.font.Font(None, 20),
        }
        command = active_command(
            linear_velocity=0.02,
            angular_velocity=0.15,
            frame_age=0.01,
        )
        common = {
            "screen": screen,
            "fonts": fonts,
            "input_name": "keyboard",
            "camera_image": Image.new("RGB", (640, 480)),
            "current_command": command,
            "armed": True,
            "emergency_stop": False,
            "camera_age": 0.01,
            "wheel_speed_scale": 0.1,
            "wheel_baseline": 0.102,
            "sample_count": 3,
            "recording_text": "REC ON",
            "policy_control_names": ("left_wheel", "right_wheel"),
            "model_available": True,
            "notice": None,
        }
        draw_status(
            **common,
            mode="manual",
            published=None,
            manual=ManualControlDisplay(
                state=SimpleNamespace(throttle=0.3, steering=-0.1),
                requested_wheels=(0.4, 0.2),
                effective_wheels=(0.1, 0.3),
                command=command,
            ),
        )
        model_result = result_for(camera_frame(1))
        draw_status(
            **common,
            mode="model",
            manual=None,
            published=PublishedInference(
                result=model_result,
                effective_wheels=(0.1, 0.3),
                command=command,
            ),
        )


class RuntimeConfigurationTests(unittest.TestCase):
    @staticmethod
    def args(**overrides: float | None) -> argparse.Namespace:
        values = {
            "rosbridge_port": 9001,
            "wheel_speed_scale": 0.10,
            "wheel_baseline": 0.102,
            "max_inference_rate": 0.0,
            "status_period": 1.0,
            "command_timeout": None,
            "control_rate": 20.0,
            "deadzone": 0.08,
            "controller_index": 0,
            "forward_target": 0.45,
            "backward_target": 0.30,
            "turn_target": 0.22,
            "throttle_rate": 2.0,
            "steering_rate": 0.75,
            "auto_center_rate": 0.55,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_low_rate_automatically_extends_command_timeout(self) -> None:
        args = self.args(max_inference_rate=1.0)
        validate_args(args)
        self.assertEqual(effective_command_timeout(args), 1.25)
        self.assertEqual(
            effective_command_timeout(self.args(max_inference_rate=10.0)),
            0.50,
        )

    def test_rejects_nonfinite_rates_and_incompatible_explicit_timeout(self) -> None:
        for args in (
            self.args(max_inference_rate=float("nan")),
            self.args(status_period=float("inf")),
            self.args(max_inference_rate=1.0, command_timeout=0.5),
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    validate_args(args)


class ModeAndInputTests(unittest.TestCase):
    def test_mode_switch_requires_model_and_toggles_both_directions(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "--checkpoint"):
            alternate_control_mode("manual", model_available=False)
        self.assertEqual(
            alternate_control_mode("manual", model_available=True),
            "model",
        )
        self.assertEqual(
            alternate_control_mode("model", model_available=True),
            "manual",
        )

    def test_input_switch_toggles_keyboard_and_ps4(self) -> None:
        self.assertEqual(alternate_input_name("keyboard"), "ps4")
        self.assertEqual(alternate_input_name("ps4"), "keyboard")

    def test_keyboard_estop_remains_available_with_ps4_input(self) -> None:
        fake_pygame = SimpleNamespace(
            QUIT=1,
            KEYDOWN=2,
            K_ESCAPE=27,
            K_RETURN=13,
            K_r=ord("r"),
            K_SPACE=32,
            K_c=ord("c"),
        )
        state = merge_keyboard_control_events(
            InputState(throttle=0.5),
            [SimpleNamespace(type=2, key=32)],
            fake_pygame,
        )
        self.assertEqual(state.throttle, 0.5)
        self.assertTrue(state.emergency_stop)


class PublisherTests(unittest.TestCase):
    def test_stored_status_error_blocks_motion_but_still_attempts_zero(self) -> None:
        publisher = object.__new__(RosbridgeTwistPublisher)
        publisher._lock = Lock()
        publisher._error = RuntimeError("advertise rejected")
        publisher._sequence = 0
        publisher._topic = "/robot/joy_mapper_node/car_cmd"
        publisher._advertisement_id = "test-command"
        sent: list[dict[str, object]] = []
        publisher._send = sent.append

        with self.assertRaisesRegex(RuntimeError, "command rosbridge failed"):
            publisher.publish(
                active_command(
                    linear_velocity=0.05,
                    angular_velocity=0.0,
                    frame_age=0.01,
                )
            )
        publisher.publish(
            active_command(
                linear_velocity=0.0,
                angular_velocity=0.0,
                frame_age=0.01,
            )
        )
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["msg"]["v"], 0.0)
        self.assertEqual(sent[0]["msg"]["omega"], 0.0)


class InferenceWorkerTests(unittest.TestCase):
    @staticmethod
    def bundle() -> SimpleNamespace:
        return SimpleNamespace(
            preprocess=SimpleNamespace(file_channel_order="rgb"),
        )

    @staticmethod
    def decoder(frame: RosbridgeCameraFrame, channel_order: str) -> Image.Image:
        image = Image.new("RGB", (1, 1))
        image.info["receive_id"] = frame.receive_id
        image.info["channel_order"] = channel_order
        return image

    def test_rate_window_keeps_only_latest_pending_frame(self) -> None:
        started_at: list[float] = []
        inferred_ids: list[int] = []
        lock = Lock()

        def infer(bundle: object, image: Image.Image) -> Prediction:
            del bundle
            with lock:
                started_at.append(monotonic())
                inferred_ids.append(int(image.info["receive_id"]))
            return prediction()

        worker = InferenceWorker(
            self.bundle(),
            max_rate=5.0,
            inference_function=infer,
            decode_function=self.decoder,
        )
        try:
            worker.set_state(enabled=True, epoch=1)
            worker.submit(camera_frame(1))
            first = wait_for_result(worker)
            worker.submit(camera_frame(3))
            worker.submit(camera_frame(2))
            second = wait_for_result(worker)

            self.assertEqual(first.frame.receive_id, 1)
            self.assertEqual(second.frame.receive_id, 3)
            self.assertEqual(inferred_ids, [1, 3])
            self.assertGreaterEqual(started_at[1] - started_at[0], 0.17)
        finally:
            worker.close()

    def test_epoch_change_discards_in_flight_result_after_emergency_stop(self) -> None:
        inference_started = Event()
        release_inference = Event()
        inference_finished = Event()

        def blocking_infer(bundle: object, image: Image.Image) -> Prediction:
            del bundle, image
            inference_started.set()
            if not release_inference.wait(timeout=2.0):
                raise RuntimeError("test did not release inference")
            inference_finished.set()
            return prediction()

        worker = InferenceWorker(
            self.bundle(),
            max_rate=0.0,
            inference_function=blocking_infer,
            decode_function=self.decoder,
        )
        try:
            worker.set_state(enabled=True, epoch=10)
            worker.submit(camera_frame(1))
            self.assertTrue(inference_started.wait(timeout=1.0))

            # This is the runtime's E-stop transition: disable and bump epoch
            # while the previous forward pass is still running.
            worker.set_state(enabled=False, epoch=11)
            release_inference.set()
            self.assertTrue(inference_finished.wait(timeout=1.0))
            sleep(0.05)
            with self.assertRaises(Empty):
                worker.get_nowait()

            # Clearing/rearming under a fresh epoch leaves the worker usable.
            worker.set_state(enabled=True, epoch=12)
            worker.submit(camera_frame(2))
            self.assertEqual(wait_for_result(worker).frame.receive_id, 2)
        finally:
            release_inference.set()
            worker.close()


class ModelActionRecorderTests(unittest.TestCase):
    def test_pairs_each_raw_frame_with_its_model_and_published_action(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            recorder = ModelActionRecorder(
                Path(temporary_dir),
                {"env": "test", "checkpoint": "checkpoint.pt"},
            )
            first_payload = b"\xff\xd8\xfffirst"
            second_payload = b"\xff\xd8\xffsecond"
            first = result_for(
                camera_frame(11, payload=first_payload),
                policy_controls=(0.1, -0.2),
                wheels=(0.3, -0.4),
            )
            second = result_for(
                camera_frame(12, payload=second_payload),
                policy_controls=(0.5,),
                wheels=(0.6, 0.7),
            )
            recorder.record(
                first,
                command=active_command(
                    linear_velocity=-0.0025,
                    angular_velocity=-0.2625,
                    frame_age=0.01,
                    wheel_speeds=(0.015, -0.020),
                ),
            )
            recorder.record(
                second,
                command=active_command(
                    linear_velocity=0.0325,
                    angular_velocity=0.0375,
                    frame_age=0.02,
                    wheel_speeds=(0.030, 0.035),
                ),
            )
            run_dir = recorder.run_dir
            recorder.close()

            with (run_dir / "actions.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["camera_receive_id"] for row in rows], ["11", "12"])
            self.assertEqual(rows[0]["policy_control_1"], "-0.200000003")
            self.assertEqual(rows[1]["policy_control_1"], "")
            self.assertEqual(rows[0]["wheel_speed_left"], "0.015000000")
            self.assertEqual(rows[1]["published_linear_velocity"], "0.032500000")
            self.assertEqual(
                (run_dir / "images" / rows[0]["image"]).read_bytes(),
                first_payload,
            )
            self.assertEqual(
                (run_dir / "images" / rows[1]["image"]).read_bytes(),
                second_payload,
            )
            metadata = json.loads((run_dir / "meta.json").read_text())
            self.assertEqual(metadata["num_samples"], 2)
            self.assertFalse(metadata["recording"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
