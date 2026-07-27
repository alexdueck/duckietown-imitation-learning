#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Manually or autonomously control a physical Duckiebot from macOS."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Condition, Thread
from time import monotonic
from typing import Any, Callable

import numpy as np
import pygame
from PIL import Image, ImageOps

from dt_utils.cli_completion import parse_args_with_completion
from dt_utils.duckiebot_dataset_recorder import PhysicalDatasetRecorder
from dt_utils.duckiebot_hardware_control import PhysicalControlLimits, PhysicalDuckiebotControl
from dt_utils.duckiebot_rosbridge import (
    RosbridgeCameraFrame,
    RosbridgeCameraSubscriber,
    RosbridgeTwistPublisher,
    import_websocket,
    normalize_robot_name,
    resolve_robot_ip,
)
from dt_utils.duckiebot_teleop_input import (
    ActionMixer,
    DriveProfile,
    InputState,
    KeyboardInput,
    SDLControllerInput,
)
from dt_utils.duckietown_action_control import DuckietownActionControl
from dt_utils.duckietown_paths import (
    IMITATION_LEARNING_TRAIN_DATA_DIR,
    PHYSICAL_CONTROL_DATA_DIR,
)
from view_model_actions_on_images import (
    PolicyBundle,
    Prediction,
    load_policy_bundle,
    predict,
)


MODE_MANUAL = "manual"
MODE_MODEL = "model"
INPUT_KEYBOARD = "keyboard"
INPUT_PS4 = "ps4"
DEFAULT_MANUAL_OUTPUT_DIR = IMITATION_LEARNING_TRAIN_DATA_DIR
DEFAULT_MODEL_OUTPUT_DIR = PHYSICAL_CONTROL_DATA_DIR
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 820
PANEL_HEIGHT = 290
BACKGROUND = (20, 22, 24)
PANEL = (34, 38, 42)
TEXT = (235, 238, 240)
MUTED = (165, 172, 178)
ACCENT = (79, 189, 186)
GOOD = (92, 184, 92)
BAD = (230, 92, 92)
BAR_BACKGROUND = (62, 68, 74)
LEFT_COLOR = (92, 184, 92)
RIGHT_COLOR = (235, 137, 88)


@dataclass(frozen=True)
class InferenceResult:
    frame: RosbridgeCameraFrame
    image: Image.Image
    prediction: Prediction
    inference_seconds: float
    epoch: int


@dataclass(frozen=True)
class PublishedInference:
    result: InferenceResult
    scaled_wheels: tuple[float, float]
    effective_wheels: tuple[float, float]
    command: Any


@dataclass(frozen=True)
class ManualControlDisplay:
    state: InputState
    requested_wheels: tuple[float, float]
    effective_wheels: tuple[float, float]
    command: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Control a physical Duckiebot manually or with an IL/PPO model. "
            "Camera and bounded Twist2DStamped commands use rosbridge."
        )
    )
    parser.add_argument(
        "robot_name",
        nargs="?",
        default=os.environ.get("VEHICLE_NAME"),
        help=(
            "Duckiebot hostname without .local; defaults to $VEHICLE_NAME."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional IL or PPO checkpoint (.pt). Without one, model mode is "
            "unavailable and the script remains a manual controller."
        ),
    )
    parser.add_argument(
        "--input",
        choices=(INPUT_KEYBOARD, INPUT_PS4),
        default=INPUT_KEYBOARD,
        help="Initial manual input device; press I in the GUI to switch.",
    )
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--throttle-axis", type=int, default=1)
    parser.add_argument("--steering-axis", type=int, default=0)
    parser.add_argument("--arm-button", type=int, default=0)
    parser.add_argument("--record-button", type=int, default=6)
    parser.add_argument("--emergency-button", type=int, default=1)
    parser.add_argument("--clear-button", type=int, default=3)
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.08,
        help="Zero manual input axes below this absolute value.",
    )
    parser.add_argument(
        "--rate-limit-analog",
        action="store_true",
        help="Apply keyboard-style input ramps to PS4 input.",
    )
    parser.add_argument("--control-rate", type=float, default=20.0)
    parser.add_argument(
        "--robot-ip",
        default=os.environ.get("ROBOT_IP"),
        help=(
            "Numeric robot IP; defaults to $ROBOT_IP, then ROBOT_NAME.local "
            "is resolved on macOS."
        ),
    )
    parser.add_argument("--rosbridge-port", type=int, default=9001)
    parser.add_argument("--camera-topic", default=None)
    parser.add_argument(
        "--command-topic",
        default=None,
        help="Defaults to /ROBOT_NAME/joy_mapper_node/car_cmd.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--crop-y-start", type=int, default=None)
    parser.add_argument(
        "--jpeg-stage",
        choices=("auto", "none", "before-resize", "after-resize"),
        default="auto",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--file-channel-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="Physical compressed camera images decode to RGB by default.",
    )
    parser.add_argument(
        "--wheel-action-scale",
        type=float,
        default=1.0,
        help=(
            "Common zero-centered scale applied to both normalized wheel "
            "actions after checkpoint mapping; 0.5 gives range [-0.5, 0.5]."
        ),
    )
    parser.add_argument("--max-linear-velocity", type=float, default=0.10)
    parser.add_argument("--max-angular-velocity", type=float, default=1.50)
    parser.add_argument("--max-linear-acceleration", type=float, default=0.25)
    parser.add_argument("--max-angular-acceleration", type=float, default=3.0)
    parser.add_argument(
        "--rate-limit-commands",
        action="store_true",
        help=(
            "Enable independent v/omega slew limits. Disabled by default so "
            "transient wheel ratios are not altered."
        ),
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Block negative chassis velocity in both control modes.",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=None,
        help=(
            "Seconds without a published model command before stopping. "
            "Defaults to 0.5, or 1.25/max-inference-rate for slower caps."
        ),
    )
    parser.add_argument("--max-frame-age", type=float, default=0.50)
    parser.add_argument(
        "--max-inference-rate",
        type=float,
        default=0.0,
        help=(
            "Maximum processed frames per second; 0 processes every new frame "
            "that inference can keep up with."
        ),
    )
    parser.add_argument(
        "--manual-output-dir",
        type=Path,
        default=DEFAULT_MANUAL_OUTPUT_DIR,
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT_DIR,
    )
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help=(
            "Disable manual recording and automatic model-action recording."
        ),
    )
    parser.add_argument(
        "--status-period",
        type=float,
        default=1.0,
        help="Print status every N seconds; 0 disables periodic output.",
    )
    return parse_args_with_completion(parser)


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.rosbridge_port <= 65535:
        raise ValueError("--rosbridge-port must be in [1, 65535]")
    if not 0.0 < args.wheel_action_scale <= 1.0:
        raise ValueError("--wheel-action-scale must be in (0, 1]")
    if not math.isfinite(args.control_rate) or args.control_rate <= 0.0:
        raise ValueError("--control-rate must be finite and positive")
    if not math.isfinite(args.deadzone) or not 0.0 <= args.deadzone < 1.0:
        raise ValueError("--deadzone must be finite and in [0, 1)")
    if args.controller_index < 0:
        raise ValueError("--controller-index must be non-negative")
    if not math.isfinite(args.max_inference_rate) or args.max_inference_rate < 0.0:
        raise ValueError("--max-inference-rate must be finite and non-negative")
    if not math.isfinite(args.status_period) or args.status_period < 0.0:
        raise ValueError("--status-period must be finite and non-negative")
    if args.command_timeout is not None:
        if not math.isfinite(args.command_timeout) or args.command_timeout <= 0.0:
            raise ValueError("--command-timeout must be finite and positive")
        if (
            args.max_inference_rate > 0.0
            and args.command_timeout <= 1.0 / args.max_inference_rate
        ):
            raise ValueError(
                "--command-timeout must be greater than the period selected by "
                "--max-inference-rate"
            )


def effective_command_timeout(args: argparse.Namespace) -> float:
    if args.command_timeout is not None:
        return float(args.command_timeout)
    if args.max_inference_rate > 0.0:
        return max(0.50, 1.25 / args.max_inference_rate)
    return 0.50


def decode_frame_image(
    frame: RosbridgeCameraFrame,
    file_channel_order: str,
) -> Image.Image:
    with Image.open(BytesIO(frame.payload)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if file_channel_order == "bgr":
        array = np.asarray(image)
        image = Image.fromarray(np.ascontiguousarray(array[:, :, ::-1]))
    return image


def scale_wheel_actions(
    wheels: Any,
    scale: float,
) -> tuple[float, float]:
    values = np.asarray(wheels, dtype=np.float64).reshape(-1)
    if values.size != 2 or not np.all(np.isfinite(values)):
        raise ValueError("wheel actions must contain two finite values")
    if not 0.0 < scale <= 1.0:
        raise ValueError("wheel action scale must be in (0, 1]")
    left, right = (float(value) * float(scale) for value in values)
    return left, right


def effective_wheel_actions(
    command: Any,
    limits: PhysicalControlLimits,
) -> tuple[float, float]:
    """Invert this runtime's v/omega scaling for display and diagnostics."""

    normalized_linear = (
        float(command.linear_velocity) / limits.max_linear_velocity
    )
    normalized_angular = (
        float(command.angular_velocity) / limits.max_angular_velocity
    )
    return (
        normalized_linear - normalized_angular,
        normalized_linear + normalized_angular,
    )


class InferenceWorker:
    def __init__(
        self,
        bundle: PolicyBundle,
        *,
        max_rate: float,
        inference_function: Callable[[PolicyBundle, Image.Image], Prediction] = predict,
        decode_function: Callable[
            [RosbridgeCameraFrame, str],
            Image.Image,
        ] = decode_frame_image,
    ) -> None:
        self._bundle = bundle
        self._interval = 0.0 if max_rate == 0.0 else 1.0 / max_rate
        self._inference_function = inference_function
        self._decode_function = decode_function
        self._condition = Condition()
        self._stop = False
        self._enabled = False
        self._epoch = 0
        self._pending: tuple[RosbridgeCameraFrame, int] | None = None
        self._results: Queue[InferenceResult] = Queue()
        self._error: Exception | None = None
        self._last_started_at: float | None = None
        self._last_submitted_receive_id = 0
        self._thread = Thread(
            target=self._run,
            name="model-inference",
            daemon=True,
        )
        self._thread.start()

    def set_state(self, *, enabled: bool, epoch: int) -> None:
        with self._condition:
            if enabled and (not self._enabled or epoch != self._epoch):
                self._last_submitted_receive_id = 0
                self._last_started_at = None
            self._enabled = bool(enabled)
            self._epoch = int(epoch)
            if not enabled:
                self._pending = None
            self._condition.notify_all()

    def submit(self, frame: RosbridgeCameraFrame) -> None:
        with self._condition:
            if not self._enabled:
                return
            if frame.receive_id <= self._last_submitted_receive_id:
                return
            self._last_submitted_receive_id = frame.receive_id
            self._pending = (frame, self._epoch)
            self._condition.notify_all()

    def get_nowait(self) -> InferenceResult:
        return self._results.get_nowait()

    @property
    def error(self) -> Exception | None:
        with self._condition:
            return self._error

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._stop
                    and (not self._enabled or self._pending is None)
                ):
                    self._condition.wait()
                if self._stop:
                    return

                if self._interval > 0.0 and self._last_started_at is not None:
                    remaining = self._interval - (
                        monotonic() - self._last_started_at
                    )
                    if remaining > 0.0:
                        self._condition.wait(timeout=remaining)
                        continue
                frame, epoch = self._pending
                self._pending = None
                self._last_started_at = monotonic()

            try:
                image = self._decode_function(
                    frame,
                    self._bundle.preprocess.file_channel_order,
                )
                started_at = monotonic()
                prediction = self._inference_function(self._bundle, image)
                inference_seconds = monotonic() - started_at
            except Exception as error:
                with self._condition:
                    if self._enabled and epoch == self._epoch:
                        self._error = error
                        self._enabled = False
                continue

            with self._condition:
                if self._stop:
                    return
                if not self._enabled or epoch != self._epoch:
                    continue
            self._results.put(
                InferenceResult(
                    frame=frame,
                    image=image,
                    prediction=prediction,
                    inference_seconds=inference_seconds,
                    epoch=epoch,
                )
            )


class ModelActionRecorder:
    FIELDNAMES = (
        "step_idx",
        "timestamp_seconds",
        "image",
        "camera_receive_id",
        "camera_seq",
        "camera_stamp",
        "frame_age_seconds",
        "inference_seconds",
        "policy_control_0",
        "policy_control_1",
        "model_wheel_left",
        "model_wheel_right",
        "scaled_wheel_left",
        "scaled_wheel_right",
        "published_linear_velocity",
        "published_angular_velocity",
        "target_linear_velocity",
        "target_angular_velocity",
        "command_reason",
    )

    def __init__(self, output_dir: Path, metadata: dict[str, Any]) -> None:
        output_dir = output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_ids = []
        for path in output_dir.glob("run_*"):
            candidate = path.name.removeprefix("run_").split("_", 1)[0]
            if candidate.isdigit():
                run_ids.append(int(candidate))
        run_id = max(run_ids, default=0) + 1
        started = datetime.now(timezone.utc)
        self.run_prefix = f"run_{run_id:03d}"
        self.run_dir = output_dir / (
            f"{self.run_prefix}_{started.strftime('%Y%m%d_%H%M%S')}"
        )
        self.images_dir = self.run_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=False)
        self._started_at = monotonic()
        self._sample_count = 0
        self._metadata = {
            **metadata,
            "run_id": run_id,
            "created_at": started.isoformat(),
        }
        self._csv_file = (self.run_dir / "actions.csv").open("w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._csv_file.flush()
        self._write_metadata(recording=True)

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def record(
        self,
        result: InferenceResult,
        *,
        scaled_wheels: tuple[float, float],
        command: Any,
    ) -> None:
        sample_idx = self._sample_count
        image_name = f"{self.run_prefix}_{sample_idx:06d}{result.frame.suffix}"
        image_path = self.images_dir / image_name
        temporary_path = image_path.with_name(f".{image_path.name}.tmp")
        temporary_path.write_bytes(result.frame.payload)
        temporary_path.replace(image_path)

        controls = result.prediction.policy_controls.reshape(-1)
        model_wheels = result.prediction.wheel_commands.reshape(-1)
        self._writer.writerow(
            {
                "step_idx": sample_idx,
                "timestamp_seconds": f"{monotonic() - self._started_at:.9f}",
                "image": image_name,
                "camera_receive_id": result.frame.receive_id,
                "camera_seq": result.frame.seq,
                "camera_stamp": f"{result.frame.stamp:.9f}",
                "frame_age_seconds": (
                    ""
                    if command.frame_age is None
                    else f"{command.frame_age:.9f}"
                ),
                "inference_seconds": f"{result.inference_seconds:.9f}",
                "policy_control_0": f"{float(controls[0]):.9f}",
                "policy_control_1": (
                    "" if controls.size < 2 else f"{float(controls[1]):.9f}"
                ),
                "model_wheel_left": f"{float(model_wheels[0]):.9f}",
                "model_wheel_right": f"{float(model_wheels[1]):.9f}",
                "scaled_wheel_left": f"{scaled_wheels[0]:.9f}",
                "scaled_wheel_right": f"{scaled_wheels[1]:.9f}",
                "published_linear_velocity": f"{command.linear_velocity:.9f}",
                "published_angular_velocity": f"{command.angular_velocity:.9f}",
                "target_linear_velocity": f"{command.target_linear_velocity:.9f}",
                "target_angular_velocity": f"{command.target_angular_velocity:.9f}",
                "command_reason": command.reason,
            }
        )
        self._csv_file.flush()
        self._sample_count += 1

    def close(self) -> None:
        if self._csv_file.closed:
            return
        self._csv_file.flush()
        self._csv_file.close()
        self._write_metadata(recording=False)

    def _write_metadata(self, *, recording: bool) -> None:
        path = self.run_dir / "meta.json"
        temporary_path = path.with_name(".meta.json.tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    **self._metadata,
                    "num_samples": self._sample_count,
                    "recording": recording,
                },
                indent=2,
            )
            + "\n"
        )
        temporary_path.replace(path)


def model_loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        crop_y_start=args.crop_y_start,
        jpeg_stage=args.jpeg_stage,
        jpeg_quality=args.jpeg_quality,
        file_channel_order=args.file_channel_order,
    )


def warm_up_model(bundle: PolicyBundle) -> None:
    dummy = Image.new("RGB", (640, 480), color=(0, 0, 0))
    predict(bundle, dummy)
    if bundle.device.type == "mps":
        import torch

        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def alternate_input_name(input_name: str) -> str:
    if input_name == INPUT_KEYBOARD:
        return INPUT_PS4
    if input_name == INPUT_PS4:
        return INPUT_KEYBOARD
    raise ValueError(f"Unknown input device: {input_name!r}")


def alternate_control_mode(mode: str, *, model_available: bool) -> str:
    if mode == MODE_MANUAL:
        if not model_available:
            raise RuntimeError(
                "Model mode is unavailable; start with --checkpoint CHECKPOINT"
            )
        return MODE_MODEL
    if mode == MODE_MODEL:
        return MODE_MANUAL
    raise ValueError(f"Unknown control mode: {mode!r}")


def merge_keyboard_control_events(
    state: InputState,
    events: list[Any],
    pygame_module: Any,
) -> InputState:
    """Keep keyboard safety controls available with either manual input."""

    arm = recording = emergency = clear = quit_requested = False
    for event in events:
        if event.type == pygame_module.QUIT:
            quit_requested = True
        elif event.type == pygame_module.KEYDOWN:
            quit_requested |= event.key == pygame_module.K_ESCAPE
            arm |= event.key == pygame_module.K_RETURN
            recording |= event.key == pygame_module.K_r
            emergency |= event.key == pygame_module.K_SPACE
            clear |= event.key == pygame_module.K_c
    return InputState(
        throttle=state.throttle,
        steering=state.steering,
        arm_toggle=state.arm_toggle or arm,
        recording_toggle=state.recording_toggle or recording,
        emergency_stop=state.emergency_stop or emergency,
        clear_emergency_stop=state.clear_emergency_stop or clear,
        quit=state.quit or quit_requested,
    )


def open_manual_input(
    pygame_module,
    input_name: str,
    args: argparse.Namespace,
):
    if input_name == INPUT_KEYBOARD:
        return KeyboardInput(pygame_module), None
    if input_name != INPUT_PS4:
        raise ValueError(f"Unknown input device: {input_name!r}")
    try:
        from pygame._sdl2 import controller as sdl_controller
    except ImportError as error:
        raise RuntimeError(
            "pygame with SDL2 controller support is required for PS4 input"
        ) from error
    sdl_controller.init()
    device_indices = [
        index
        for index in range(sdl_controller.get_count())
        if sdl_controller.is_controller(index)
    ]
    if not 0 <= args.controller_index < len(device_indices):
        raise RuntimeError(
            f"Controller index {args.controller_index} is unavailable; "
            f"SDL found {len(device_indices)} controller(s)"
        )
    controller = sdl_controller.Controller(
        device_indices[args.controller_index]
    )
    input_device = SDLControllerInput(
        pygame_module,
        controller,
        throttle_axis=args.throttle_axis,
        steering_axis=args.steering_axis,
        arm_button=args.arm_button,
        recording_button=args.record_button,
        emergency_button=args.emergency_button,
        clear_button=args.clear_button,
    )
    return input_device, controller


def close_controller(controller: Any | None) -> None:
    if controller is None:
        return
    close = getattr(controller, "quit", None)
    if close is not None:
        close()


def fit_rect(image_size: tuple[int, int], bounds: pygame.Rect) -> pygame.Rect:
    width, height = image_size
    scale = min(bounds.width / width, bounds.height / height)
    target_width = max(1, int(width * scale))
    target_height = max(1, int(height * scale))
    return pygame.Rect(
        bounds.x + (bounds.width - target_width) // 2,
        bounds.y + (bounds.height - target_height) // 2,
        target_width,
        target_height,
    )


def render_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text_value: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = TEXT,
) -> None:
    screen.blit(font.render(text_value, True, color), position)


def draw_signal_bar(
    screen: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    color: tuple[int, int, int],
) -> None:
    pygame.draw.rect(screen, BAR_BACKGROUND, rect, border_radius=4)
    center_x = rect.centerx
    pygame.draw.line(screen, MUTED, (center_x, rect.y), (center_x, rect.bottom), 1)
    clipped = max(-1.0, min(1.0, float(value)))
    half_width = rect.width // 2 - 2
    fill_width = int(abs(clipped) * half_width)
    fill = pygame.Rect(
        center_x if clipped >= 0.0 else center_x - fill_width,
        rect.y + 2,
        fill_width,
        rect.height - 4,
    )
    pygame.draw.rect(screen, color, fill, border_radius=4)


def point_tuple(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_direction_arrow(
    screen: pygame.Surface,
    center: tuple[int, int],
    wheels: tuple[float, float],
) -> None:
    pygame.draw.circle(screen, BAR_BACKGROUND, center, 62)
    pygame.draw.circle(screen, MUTED, center, 62, 1)
    pygame.draw.line(
        screen,
        MUTED,
        (center[0], center[1] + 18),
        (center[0], center[1] - 18),
        3,
    )

    left, right = wheels
    normalized_v = 0.5 * (left + right)
    normalized_omega = 0.5 * (right - left)
    vector = np.array(
        [-normalized_omega, -normalized_v],
        dtype=np.float32,
    )
    vector_length = float(np.linalg.norm(vector))
    magnitude = min(1.0, vector_length)
    if magnitude < 1e-6:
        pygame.draw.circle(screen, ACCENT, center, 5)
        return

    direction = vector / vector_length
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    origin = np.asarray(center, dtype=np.float32)
    tip = origin + direction * (24.0 + 28.0 * magnitude)
    head_left = tip - direction * 14.0 + perpendicular * 8.0
    head_right = tip - direction * 14.0 - perpendicular * 8.0
    pygame.draw.line(screen, ACCENT, center, point_tuple(tip), 5)
    pygame.draw.polygon(
        screen,
        ACCENT,
        [point_tuple(tip), point_tuple(head_left), point_tuple(head_right)],
    )


def format_policy_controls(
    values: np.ndarray,
    names: tuple[str, ...],
) -> str:
    flattened = np.asarray(values).reshape(-1)
    parts = []
    for index, value in enumerate(flattened):
        name = names[index] if index < len(names) else f"action_{index}"
        parts.append(f"{name}={float(value):+.3f}")
    return "   ".join(parts)


def draw_status(
    screen: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    *,
    mode: str,
    input_name: str,
    camera_image: Image.Image | None,
    published: PublishedInference | None,
    manual: ManualControlDisplay | None,
    current_command: Any,
    armed: bool,
    emergency_stop: bool,
    camera_age: float | None,
    wheel_scale: float,
    sample_count: int,
    recording_text: str,
    policy_control_names: tuple[str, ...],
    model_available: bool,
    notice: str | None,
) -> None:
    screen.fill(BACKGROUND)
    image_area = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT - PANEL_HEIGHT)
    image = (
        published.result.image
        if mode == MODE_MODEL and published is not None
        else camera_image
    )
    if image is not None:
        surface = pygame.image.fromstring(
            image.tobytes(),
            image.size,
            "RGB",
        )
        target = fit_rect(surface.get_size(), image_area.inflate(-24, -24))
        screen.blit(pygame.transform.smoothscale(surface, target.size), target)
    else:
        label = fonts["large"].render(
            "Waiting for a processed camera frame",
            True,
            MUTED,
        )
        screen.blit(label, label.get_rect(center=image_area.center))
    if notice:
        notice_surface = fonts["normal"].render(notice, True, BAD)
        notice_background = notice_surface.get_rect()
        notice_background.inflate_ip(24, 14)
        notice_background.midtop = (image_area.centerx, 14)
        pygame.draw.rect(
            screen,
            PANEL,
            notice_background,
            border_radius=6,
        )
        screen.blit(
            notice_surface,
            notice_surface.get_rect(center=notice_background.center),
        )

    panel = pygame.Rect(0, WINDOW_HEIGHT - PANEL_HEIGHT, WINDOW_WIDTH, PANEL_HEIGHT)
    pygame.draw.rect(screen, PANEL, panel)
    if emergency_stop:
        state_text, state_color = "EMERGENCY STOP", BAD
    elif armed:
        state_text, state_color = "ARMED", GOOD
    else:
        state_text, state_color = "DISARMED", MUTED
    y = panel.y + 12
    render_text(screen, fonts["large"], state_text, (20, y), state_color)
    render_text(
        screen,
        fonts["normal"],
        f"MODE {mode.upper()}   |   INPUT {input_name.upper()}",
        (245, y + 4),
        ACCENT,
    )
    render_text(
        screen,
        fonts["small"],
        f"{recording_text}   |   samples {sample_count}",
        (800, y + 7),
        GOOD if recording_text != "REC OFF" else MUTED,
    )
    camera_text = "none" if camera_age is None else f"{camera_age:.3f}s"
    render_text(
        screen,
        fonts["small"],
        f"current Twist: v={current_command.linear_velocity:+.3f} m/s, "
        f"omega={current_command.angular_velocity:+.3f} rad/s   |   "
        f"camera age {camera_text}   |   "
        f"{current_command.reason}",
        (20, y + 37),
        MUTED,
    )

    if mode == MODE_MODEL and published is not None:
        result = published.result
        prediction = result.prediction
        model_left, model_right = (
            float(value) for value in prediction.wheel_commands
        )
        sent_left, sent_right = published.effective_wheels
        sent_v_norm = 0.5 * (sent_left + sent_right)
        sent_omega_norm = 0.5 * (sent_right - sent_left)

        y += 67
        render_text(
            screen,
            fonts["small"],
            "Model output:",
            (20, y),
            MUTED,
        )
        render_text(
            screen,
            fonts["small"],
            format_policy_controls(
                prediction.policy_controls,
                policy_control_names,
            ),
            (165, y),
        )

        y += 32
        render_text(screen, fonts["small"], "Gym wheels", (20, y))
        render_text(
            screen,
            fonts["small"],
            f"left {model_left:+.3f}",
            (190, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(300, y, 190, 18),
            model_left,
            LEFT_COLOR,
        )
        render_text(
            screen,
            fonts["small"],
            f"right {model_right:+.3f}",
            (515, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(635, y, 185, 18),
            model_right,
            RIGHT_COLOR,
        )

        y += 32
        render_text(screen, fonts["small"], "Sent wheel-equiv.", (20, y))
        render_text(
            screen,
            fonts["small"],
            f"left {sent_left:+.3f}",
            (190, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(300, y, 190, 18),
            sent_left,
            LEFT_COLOR,
        )
        render_text(
            screen,
            fonts["small"],
            f"right {sent_right:+.3f}",
            (515, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(635, y, 185, 18),
            sent_right,
            RIGHT_COLOR,
        )

        draw_direction_arrow(
            screen,
            (955, y + 8),
            published.effective_wheels,
        )
        render_text(screen, fonts["small"], "Movement", (920, y - 62), MUTED)

        y += 31
        command = published.command
        render_text(
            screen,
            fonts["small"],
            f"Twist sent for image: v={command.linear_velocity:+.3f} m/s   "
            f"omega={command.angular_velocity:+.3f} rad/s   "
            f"|   normalized v={sent_v_norm:+.3f}, "
            f"omega={sent_omega_norm:+.3f}",
            (20, y),
        )
        y += 29
        render_text(
            screen,
            fonts["small"],
            f"requested scaled wheels "
            f"[{published.scaled_wheels[0]:+.3f}, "
            f"{published.scaled_wheels[1]:+.3f}]   |   "
            f"wheel scale {wheel_scale:.2f}   |   "
            f"inference {result.inference_seconds * 1000:.1f} ms   |   "
            f"{command.reason}",
            (20, y),
            MUTED,
        )
    elif mode == MODE_MANUAL and manual is not None:
        requested_left, requested_right = manual.requested_wheels
        sent_left, sent_right = manual.effective_wheels
        y += 67
        render_text(
            screen,
            fonts["small"],
            f"Manual input: throttle={manual.state.throttle:+.3f}   "
            f"steering={manual.state.steering:+.3f}",
            (20, y),
        )

        y += 32
        render_text(screen, fonts["small"], "Requested wheels", (20, y))
        render_text(
            screen,
            fonts["small"],
            f"left {requested_left:+.3f}",
            (190, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(300, y, 190, 18),
            requested_left,
            LEFT_COLOR,
        )
        render_text(
            screen,
            fonts["small"],
            f"right {requested_right:+.3f}",
            (515, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(635, y, 185, 18),
            requested_right,
            RIGHT_COLOR,
        )

        y += 32
        render_text(screen, fonts["small"], "Sent wheel-equiv.", (20, y))
        render_text(
            screen,
            fonts["small"],
            f"left {sent_left:+.3f}",
            (190, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(300, y, 190, 18),
            sent_left,
            LEFT_COLOR,
        )
        render_text(
            screen,
            fonts["small"],
            f"right {sent_right:+.3f}",
            (515, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(635, y, 185, 18),
            sent_right,
            RIGHT_COLOR,
        )
        draw_direction_arrow(
            screen,
            (955, y + 8),
            manual.effective_wheels,
        )
        render_text(screen, fonts["small"], "Movement", (920, y - 62), MUTED)

        y += 35
        render_text(
            screen,
            fonts["small"],
            f"Twist sent: v={manual.command.linear_velocity:+.3f} m/s   "
            f"omega={manual.command.angular_velocity:+.3f} rad/s",
            (20, y),
            MUTED,
        )
    else:
        message = (
            "Arm to run model inference on the latest frame."
            if mode == MODE_MODEL
            else "Arm to enable manual wheel commands."
        )
        render_text(
            screen,
            fonts["normal"],
            message,
            (20, panel.y + 80),
            MUTED,
        )
        if mode == MODE_MANUAL and not model_available:
            render_text(
                screen,
                fonts["small"],
                "Model mode requires --checkpoint CHECKPOINT.",
                (20, panel.y + 115),
                MUTED,
            )

    render_text(
        screen,
        fonts["small"],
        "M: mode   I: input   Enter/Cross: arm   R/Options: record   "
        "Space/Circle: E-stop   C/Triangle: clear   Esc: quit",
        (20, panel.bottom - 35),
        ACCENT,
    )


def close_recorder(recorder: Any | None, label: str) -> None:
    if recorder is None:
        return
    recorder.close()
    print(
        f"{label} recording saved: {recorder.sample_count} samples in "
        f"{recorder.run_dir}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    subscriber = None
    publisher = None
    worker = None
    manual_recorder = None
    model_recorder = None
    control = None
    input_device = None
    controller_device = None
    pygame_initialized = False
    try:
        validate_args(args)
        import_websocket()
        robot_name = normalize_robot_name(args.robot_name)
        robot_ip = resolve_robot_ip(robot_name, args.robot_ip)
        camera_topic = (
            args.camera_topic
            or f"/{robot_name}/camera_node/image/compressed"
        )
        command_topic = args.command_topic or f"/{robot_name}/joy_mapper_node/car_cmd"
        rosbridge_url = f"ws://{robot_ip}:{args.rosbridge_port}"

        bundle: PolicyBundle | None = None
        if args.checkpoint is not None:
            print(
                f"Loading checkpoint: {args.checkpoint.expanduser().resolve()}",
                flush=True,
            )
            bundle = load_policy_bundle(model_loader_args(args))
            print(
                f"Checkpoint type: {bundle.checkpoint_type}; "
                f"device: {bundle.device}"
            )
            print(
                "Preprocess: "
                f"crop_y_start={bundle.preprocess.crop_y_start}, "
                f"image_size={bundle.preprocess.image_size}, "
                f"jpeg_stage={bundle.preprocess.jpeg_stage}, "
                f"file_channel_order={bundle.preprocess.file_channel_order}"
            )
            if bundle.preprocess.inference_note:
                print(f"Preprocess note: {bundle.preprocess.inference_note}")
            print("Warming up model...", flush=True)
            warm_up_model(bundle)

        limits = PhysicalControlLimits(
            max_linear_velocity=args.max_linear_velocity,
            max_angular_velocity=args.max_angular_velocity,
            max_linear_acceleration=args.max_linear_acceleration,
            max_angular_acceleration=args.max_angular_acceleration,
            command_timeout=effective_command_timeout(args),
            max_frame_age=args.max_frame_age,
            nominal_control_period=1.0 / args.control_rate,
            forward_only=args.forward_only,
            rate_limit_commands=args.rate_limit_commands,
        )
        control = PhysicalDuckiebotControl(
            DuckietownActionControl(mode="wheel"),
            limits=limits,
        )
        if bundle is not None:
            worker = InferenceWorker(
                bundle,
                max_rate=args.max_inference_rate,
            )

        print(f"Connecting to {rosbridge_url}...", flush=True)
        subscriber = RosbridgeCameraSubscriber(
            rosbridge_url,
            camera_topic,
            frame_callback=None if worker is None else worker.submit,
            client_id="physical-control",
        )
        publisher = RosbridgeTwistPublisher(
            rosbridge_url,
            command_topic,
            client_id="physical-control",
        )
        publisher.publish(control.disarm())

        policy_control_names = (
            tuple(bundle.action_control.control_names)
            if bundle is not None and bundle.action_control is not None
            else ("left_wheel", "right_wheel")
        )
        model_metadata = None
        if bundle is not None and args.checkpoint is not None:
            model_metadata = {
                "env": "physical_duckiebot",
                "runtime": "physical_duckiebot_control.py",
                "mode": MODE_MODEL,
                "robot_name": robot_name,
                "robot_ip": robot_ip,
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_type": bundle.checkpoint_type,
                "checkpoint_model": bundle.config.get(
                    "model",
                    "mobilenet_v3_small",
                ),
                "checkpoint_action_mode": bundle.config.get(
                    "action_mode",
                    "wheel",
                ),
                "policy_control_names": list(policy_control_names),
                "preprocess": asdict(bundle.preprocess),
                "wheel_action_scale": args.wheel_action_scale,
                "physical_control_limits": asdict(limits),
                "max_inference_rate": args.max_inference_rate,
                "camera_topic": camera_topic,
                "command_topic": command_topic,
                "rosbridge_url": rosbridge_url,
                "alignment": (
                    "Each saved raw camera frame is paired with the "
                    "deterministic model output and Twist2DStamped command "
                    "published from it."
                ),
            }

        pygame.init()
        pygame_initialized = True
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Physical Duckiebot control — focus here")
        fonts = {
            "large": pygame.font.Font(None, 38),
            "normal": pygame.font.Font(None, 25),
            "small": pygame.font.Font(None, 20),
        }
        clock = pygame.time.Clock()
        input_name = args.input
        input_device, controller_device = open_manual_input(
            pygame,
            input_name,
            args,
        )
        if controller_device is not None:
            print(f"Controller: {controller_device.name}")
        mixer = ActionMixer(DriveProfile(deadzone=args.deadzone))

        mode = MODE_MANUAL
        epoch = 0
        last_published: PublishedInference | None = None
        manual_display: ManualControlDisplay | None = None
        camera_image: Image.Image | None = None
        displayed_receive_id = 0
        command = control.last_command
        next_status_at = monotonic()
        next_manual_command_at = monotonic()
        last_manual_command_at = monotonic()
        last_manual_recorded_receive_id = 0
        camera_decode_warning_shown = False
        recording_enabled = False
        notice: str | None = None
        notice_until = 0.0
        running = True
        print(
            f"Camera: {camera_topic}; command: {command_topic}; "
            f"mode={mode}; input={input_name}; "
            f"model={'loaded' if bundle is not None else 'not loaded'}"
        )
        print(
            f"Limits: v=±{limits.max_linear_velocity:.3f} m/s, "
            f"omega=±{limits.max_angular_velocity:.3f} rad/s, "
            f"wheel range=[-{args.wheel_action_scale:.3f}, "
            f"+{args.wheel_action_scale:.3f}], "
            f"reverse={'off' if limits.forward_only else 'on'}, "
            f"rate limit={'on' if limits.rate_limit_commands else 'off'}, "
            f"watchdog={limits.command_timeout:.3f}s"
        )
        print(
            "M: mode  I: input  Enter/Cross: arm  R/Options: record  "
            "Space/Circle: E-stop  C/Triangle: clear  Escape: quit"
        )

        while running:
            now = monotonic()
            events = pygame.event.get()
            mode_toggle = False
            input_toggle = False
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_m:
                        mode_toggle = True
                    elif event.key == pygame.K_i:
                        input_toggle = True

            if not running:
                break
            if subscriber.error is not None:
                raise RuntimeError(f"camera rosbridge failed: {subscriber.error}")
            decode_error_count, decode_error = subscriber.decode_diagnostics
            if decode_error_count > 0 and not camera_decode_warning_shown:
                print(
                    "warning: ignoring malformed camera messages; first error: "
                    f"{decode_error}",
                    file=sys.stderr,
                    flush=True,
                )
                camera_decode_warning_shown = True
            if publisher.error is not None:
                raise RuntimeError(f"command rosbridge failed: {publisher.error}")
            if worker is not None and worker.error is not None:
                command = control.engage_emergency_stop(now)
                try:
                    publisher.publish(command)
                finally:
                    raise RuntimeError(f"model inference failed: {worker.error}")

            frame = subscriber.latest()
            camera_age = (
                None if frame is None else max(0.0, now - frame.received_at)
            )
            if frame is not None and frame.receive_id != displayed_receive_id:
                try:
                    camera_image = decode_frame_image(frame, "rgb")
                    displayed_receive_id = frame.receive_id
                except (OSError, ValueError) as error:
                    if not camera_decode_warning_shown:
                        print(
                            f"warning: cannot decode camera frame: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        camera_decode_warning_shown = True

            state = merge_keyboard_control_events(
                input_device.poll(events),
                events,
                pygame,
            )
            if state.quit:
                break

            transition_happened = False
            if mode_toggle:
                try:
                    new_mode = alternate_control_mode(
                        mode,
                        model_available=bundle is not None,
                    )
                except RuntimeError as error:
                    notice = str(error)
                    notice_until = now + 4.0
                    print(notice, flush=True)
                else:
                    command = control.disarm(now)
                    publisher.publish(command)
                    epoch += 1
                    if worker is not None:
                        worker.set_state(enabled=False, epoch=epoch)
                    mixer.reset()
                    if mode == MODE_MANUAL:
                        close_recorder(manual_recorder, "Manual")
                        manual_recorder = None
                    else:
                        close_recorder(model_recorder, "Model")
                        model_recorder = None
                    mode = new_mode
                    recording_enabled = (
                        mode == MODE_MODEL and not args.no_recording
                    )
                    manual_display = None
                    last_published = None
                    transition_happened = True
                    notice = f"Mode switched to {mode.upper()}; controller disarmed"
                    notice_until = now + 3.0
                    print(notice, flush=True)

            if input_toggle:
                requested_input = alternate_input_name(input_name)
                try:
                    new_input, new_controller = open_manual_input(
                        pygame,
                        requested_input,
                        args,
                    )
                except (RuntimeError, ValueError) as error:
                    notice = f"Input switch failed: {error}"
                    notice_until = now + 4.0
                    print(notice, flush=True)
                else:
                    command = control.disarm(now)
                    publisher.publish(command)
                    epoch += 1
                    if worker is not None:
                        worker.set_state(enabled=False, epoch=epoch)
                    mixer.reset()
                    close_controller(controller_device)
                    input_device = new_input
                    controller_device = new_controller
                    input_name = requested_input
                    if manual_recorder is not None:
                        close_recorder(manual_recorder, "Manual")
                        manual_recorder = None
                        recording_enabled = False
                    transition_happened = True
                    notice = (
                        f"Input switched to {input_name.upper()}; "
                        "controller disarmed"
                    )
                    notice_until = now + 3.0
                    print(notice, flush=True)
                    if controller_device is not None:
                        print(f"Controller: {controller_device.name}")

            if transition_happened:
                state = InputState()

            if state.emergency_stop:
                epoch += 1
                if worker is not None:
                    worker.set_state(enabled=False, epoch=epoch)
                command = control.engage_emergency_stop(now)
                publisher.publish(command)
                mixer.reset()
                print("EMERGENCY STOP LATCHED", flush=True)
            elif state.clear_emergency_stop:
                epoch += 1
                if worker is not None:
                    worker.set_state(enabled=False, epoch=epoch)
                command = control.clear_emergency_stop(now)
                publisher.publish(command)
                mixer.reset()
                print(
                    "Emergency stop cleared; controller remains DISARMED",
                    flush=True,
                )
            elif state.arm_toggle:
                if control.armed:
                    epoch += 1
                    if worker is not None:
                        worker.set_state(enabled=False, epoch=epoch)
                    command = control.disarm(now)
                    publisher.publish(command)
                    mixer.reset()
                    print("Controller DISARMED", flush=True)
                elif control.emergency_stop_latched:
                    notice = "Clear E-stop with C/Triangle before arming"
                    notice_until = now + 3.0
                    print(notice, flush=True)
                elif camera_age is None or camera_age > args.max_frame_age:
                    notice = "Cannot arm: no fresh camera frame"
                    notice_until = now + 3.0
                    print(notice, flush=True)
                else:
                    epoch += 1
                    command = control.arm(now)
                    publisher.publish(command)
                    if mode == MODE_MODEL and worker is not None and frame is not None:
                        worker.set_state(enabled=True, epoch=epoch)
                        worker.submit(frame)
                    print(
                        f"Controller ARMED in {mode.upper()} mode",
                        flush=True,
                    )

            if state.recording_toggle:
                if args.no_recording:
                    notice = "Recording is disabled by --no-recording"
                    notice_until = now + 3.0
                    print(notice, flush=True)
                elif recording_enabled:
                    recording_enabled = False
                    if mode == MODE_MANUAL:
                        close_recorder(manual_recorder, "Manual")
                        manual_recorder = None
                    else:
                        close_recorder(model_recorder, "Model")
                        model_recorder = None
                else:
                    if camera_age is None or camera_age > args.max_frame_age:
                        notice = "Recording not started: no fresh camera frame"
                        notice_until = now + 3.0
                        print(notice, flush=True)
                    else:
                        recording_enabled = True
                        print(
                            f"{mode.capitalize()} recording enabled",
                            flush=True,
                        )

            if (
                mode == MODE_MANUAL
                and recording_enabled
                and manual_recorder is None
            ):
                manual_recorder = PhysicalDatasetRecorder(
                    args.manual_output_dir,
                    {
                        "env": "physical_duckiebot",
                        "runtime": "physical_duckiebot_control.py",
                        "mode": MODE_MANUAL,
                        "robot_name": robot_name,
                        "robot_ip": robot_ip,
                        "camera_topic": camera_topic,
                        "command_topic": command_topic,
                        "rosbridge_url": rosbridge_url,
                        "source_observation_channel_order": "camera-compressed",
                        "saved_image_channel_order": "camera-encoded",
                        "controller": {
                            "type": input_name,
                            "drive_profile": asdict(mixer.profile),
                        },
                        "physical_control_limits": asdict(limits),
                        "sample_period_seconds": 1.0 / args.control_rate,
                    },
                )
                last_manual_recorded_receive_id = 0
                print(f"Manual recording -> {manual_recorder.run_dir}")

            if mode == MODE_MANUAL and now >= next_manual_command_at:
                delta_time = max(0.0, now - last_manual_command_at)
                last_manual_command_at = now
                smooth_input = (
                    input_name == INPUT_KEYBOARD or args.rate_limit_analog
                )
                requested_wheels = (
                    mixer.update(state, delta_time, smooth=smooth_input)
                    if control.armed
                    else (0.0, 0.0)
                )
                command = control.update(
                    requested_wheels,
                    timestamp=now,
                    frame_age=camera_age,
                )
                publisher.publish(command)
                effective_wheels = effective_wheel_actions(command, limits)
                manual_display = ManualControlDisplay(
                    state=state,
                    requested_wheels=requested_wheels,
                    effective_wheels=effective_wheels,
                    command=command,
                )
                if (
                    manual_recorder is not None
                    and frame is not None
                    and frame.receive_id != last_manual_recorded_receive_id
                    and command.reason == "active"
                ):
                    manual_recorder.record(
                        payload=frame.payload,
                        image_suffix=frame.suffix,
                        camera_seq=frame.seq,
                        camera_stamp=frame.stamp,
                        frame_age=camera_age or 0.0,
                        left_action=effective_wheels[0],
                        right_action=effective_wheels[1],
                        linear_velocity=command.linear_velocity,
                        angular_velocity=command.angular_velocity,
                    )
                    last_manual_recorded_receive_id = frame.receive_id
                next_manual_command_at += 1.0 / args.control_rate
                if next_manual_command_at <= now:
                    next_manual_command_at = now + 1.0 / args.control_rate
                if not control.armed and command.reason not in {
                    "disarmed",
                    "emergency_stop_latched",
                }:
                    mixer.reset()

            if mode == MODE_MODEL and worker is not None:
                while True:
                    try:
                        result = worker.get_nowait()
                    except Empty:
                        break
                    if result.epoch != epoch or not control.armed:
                        continue
                    publish_at = monotonic()
                    scaled_wheels = scale_wheel_actions(
                        result.prediction.wheel_commands,
                        args.wheel_action_scale,
                    )
                    result_frame_age = max(
                        0.0,
                        publish_at - result.frame.received_at,
                    )
                    command = control.update(
                        scaled_wheels,
                        timestamp=publish_at,
                        frame_age=result_frame_age,
                    )
                    publisher.publish(command)
                    effective_wheels = effective_wheel_actions(command, limits)
                    if (
                        recording_enabled
                        and model_recorder is None
                        and model_metadata is not None
                    ):
                        model_recorder = ModelActionRecorder(
                            args.model_output_dir,
                            model_metadata,
                        )
                        print(f"Model recording -> {model_recorder.run_dir}")
                    if model_recorder is not None:
                        model_recorder.record(
                            result,
                            scaled_wheels=scaled_wheels,
                            command=command,
                        )
                    last_published = PublishedInference(
                        result=result,
                        scaled_wheels=scaled_wheels,
                        effective_wheels=effective_wheels,
                        command=command,
                    )
                    if not control.armed:
                        epoch += 1
                        worker.set_state(enabled=False, epoch=epoch)
                        print(
                            f"Controller stopped and disarmed: {command.reason}",
                            flush=True,
                        )

            if control.armed and mode == MODE_MODEL:
                watchdog_command = control.watchdog(monotonic())
                if watchdog_command.reason != command.reason or (
                    watchdog_command.stopped and not command.stopped
                ):
                    command = watchdog_command
                    publisher.publish(command)
                    if not control.armed:
                        epoch += 1
                        if worker is not None:
                            worker.set_state(enabled=False, epoch=epoch)
                        print(
                            f"Controller stopped and disarmed: {command.reason}",
                            flush=True,
                        )

            if args.status_period > 0.0 and now >= next_status_at:
                camera_text = "none" if camera_age is None else f"{camera_age:.3f}s"
                active_recorder = (
                    manual_recorder if mode == MODE_MANUAL else model_recorder
                )
                print(
                    f"status mode={mode} input={input_name} "
                    f"armed={int(control.armed)} "
                    f"estop={int(control.emergency_stop_latched)} "
                    f"recording={int(recording_enabled)} "
                    f"camera_age={camera_text} reason={command.reason} "
                    f"v={command.linear_velocity:+.3f} "
                    f"omega={command.angular_velocity:+.3f} "
                    f"samples={0 if active_recorder is None else active_recorder.sample_count}",
                    flush=True,
                )
                next_status_at = now + args.status_period

            active_recorder = (
                manual_recorder if mode == MODE_MANUAL else model_recorder
            )
            if args.no_recording:
                recording_text = "REC DISABLED"
            elif recording_enabled and active_recorder is None:
                recording_text = "REC READY"
            elif recording_enabled:
                recording_text = "REC ON"
            else:
                recording_text = "REC OFF"
            if notice is not None and now >= notice_until:
                notice = None
            draw_status(
                screen,
                fonts,
                mode=mode,
                input_name=input_name,
                camera_image=camera_image,
                published=last_published,
                manual=manual_display,
                current_command=command,
                armed=control.armed,
                emergency_stop=control.emergency_stop_latched,
                camera_age=camera_age,
                wheel_scale=args.wheel_action_scale,
                sample_count=(
                    0 if active_recorder is None else active_recorder.sample_count
                ),
                recording_text=recording_text,
                policy_control_names=policy_control_names,
                model_available=bundle is not None,
                notice=notice,
            )
            pygame.display.set_caption(
                f"Duckiebot control | {mode.upper()} | {input_name.upper()} | "
                f"{'ARMED' if control.armed else command.reason}"
            )
            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if control is not None and publisher is not None:
            try:
                publisher.publish(control.disarm())
            except Exception:
                pass
        if worker is not None:
            worker.set_state(enabled=False, epoch=sys.maxsize)
            worker.close()
        close_recorder(manual_recorder, "Manual")
        close_recorder(model_recorder, "Model")
        if subscriber is not None:
            try:
                subscriber.close()
            except Exception:
                pass
        if publisher is not None:
            try:
                publisher.close()
            except Exception:
                pass
        close_controller(controller_device)
        if pygame_initialized:
            pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
