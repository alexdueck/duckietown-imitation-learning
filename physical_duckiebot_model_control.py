#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Drive a physical Duckiebot from an IL or PPO camera policy on macOS."""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
import ipaddress
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import socket
import sys
from threading import Condition, Event, Lock, Thread
from time import monotonic, time
from typing import Any, Callable

import numpy as np
import pygame
from PIL import Image, ImageOps

from capture_duckiebot_camera import compressed_format, normalize_robot_name
from dt_utils.cli_completion import parse_args_with_completion
from dt_utils.duckiebot_hardware_control import PhysicalControlLimits, PhysicalDuckiebotControl
from dt_utils.duckietown_action_control import DuckietownActionControl
from dt_utils.duckietown_paths import PHYSICAL_MODEL_CONTROL_DATA_DIR
from view_model_actions_on_images import (
    PolicyBundle,
    Prediction,
    load_policy_bundle,
    predict,
)


DEFAULT_OUTPUT_DIR = PHYSICAL_MODEL_CONTROL_DATA_DIR
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
class RosbridgeCameraFrame:
    payload: bytes
    suffix: str
    message_format: str
    seq: int
    stamp: float
    received_at: float
    receive_id: int


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an IL or PPO checkpoint on a physical Duckiebot camera stream "
            "and publish bounded Twist2DStamped commands through rosbridge."
        )
    )
    parser.add_argument("robot_name", help="Duckiebot hostname without .local.")
    parser.add_argument("checkpoint", type=Path, help="IL or PPO checkpoint (.pt).")
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="Do not save processed images and their published actions.",
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


def resolve_robot_ip(robot_name: str, explicit_ip: str | None) -> str:
    if explicit_ip is not None:
        try:
            address = ipaddress.ip_address(explicit_ip)
        except ValueError as error:
            raise ValueError(f"Invalid --robot-ip: {explicit_ip!r}") from error
        if address.version != 4:
            raise ValueError("--robot-ip must currently be an IPv4 address")
        return str(address)

    hostname = f"{robot_name}.local"
    try:
        results = socket.getaddrinfo(
            hostname,
            None,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as error:
        raise RuntimeError(
            f"Could not resolve {hostname}; pass the current address with --robot-ip"
        ) from error
    addresses = [result[4][0] for result in results]
    if not addresses:
        raise RuntimeError(
            f"Could not resolve an IPv4 address for {hostname}; use --robot-ip"
        )
    return str(addresses[0])


def ros_stamp(header: Any) -> float:
    if not isinstance(header, dict):
        return 0.0
    stamp = header.get("stamp", {})
    if not isinstance(stamp, dict):
        return 0.0
    seconds = stamp.get("secs", stamp.get("sec", 0))
    nanoseconds = stamp.get("nsecs", stamp.get("nanosec", 0))
    try:
        return float(seconds) + float(nanoseconds) / 1_000_000_000
    except (TypeError, ValueError):
        return 0.0


def decode_rosbridge_camera_publish(
    message: dict[str, Any],
    *,
    receive_id: int,
    received_at: float | None = None,
) -> RosbridgeCameraFrame:
    if message.get("op") != "publish":
        raise ValueError("rosbridge camera message must use op=publish")
    payload_message = message.get("msg")
    if not isinstance(payload_message, dict):
        raise ValueError("rosbridge camera publish is missing msg")
    encoded = payload_message.get("data")
    if isinstance(encoded, str):
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError("camera data is not valid base64") from error
    elif isinstance(encoded, list):
        try:
            payload = bytes(encoded)
        except (TypeError, ValueError) as error:
            raise ValueError("camera byte array is invalid") from error
    else:
        raise ValueError("camera data must be base64 text or a byte array")
    if not payload:
        raise ValueError("camera payload is empty")

    message_format = str(payload_message.get("format", ""))
    image_format = compressed_format(message_format, payload)
    header = payload_message.get("header", {})
    seq = int(header.get("seq", 0)) if isinstance(header, dict) else 0
    return RosbridgeCameraFrame(
        payload=payload,
        suffix=".jpg" if image_format == "jpeg" else ".png",
        message_format=message_format,
        seq=seq,
        stamp=ros_stamp(header),
        received_at=monotonic() if received_at is None else float(received_at),
        receive_id=int(receive_id),
    )


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


class RosbridgeCameraSubscriber:
    def __init__(
        self,
        url: str,
        topic: str,
        *,
        frame_callback: Callable[[RosbridgeCameraFrame], None] | None = None,
    ) -> None:
        websocket = import_websocket()
        self._websocket = websocket
        self._url = url
        self._topic = topic
        self._frame_callback = frame_callback
        self._lock = Lock()
        self._send_lock = Lock()
        self._stop = Event()
        self._latest: RosbridgeCameraFrame | None = None
        self._error: Exception | None = None
        self._decode_error_count = 0
        self._last_decode_error: Exception | None = None
        self._receive_id = 0
        try:
            self._socket = websocket.create_connection(
                url,
                timeout=5.0,
                enable_multithread=True,
            )
            self._socket.settimeout(1.0)
            self._send(
                {
                    "op": "subscribe",
                    "id": "physical-model-camera",
                    "topic": topic,
                    "type": "sensor_msgs/CompressedImage",
                    "compression": "none",
                    "queue_length": 1,
                    "throttle_rate": 0,
                }
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not subscribe to {topic} through {url}: {error}"
            ) from error
        self._thread = Thread(
            target=self._receive_loop,
            name="rosbridge-camera",
            daemon=True,
        )
        self._thread.start()

    def latest(self) -> RosbridgeCameraFrame | None:
        with self._lock:
            return self._latest

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    @property
    def decode_diagnostics(self) -> tuple[int, Exception | None]:
        with self._lock:
            return self._decode_error_count, self._last_decode_error

    def close(self) -> None:
        if self._stop.is_set():
            return
        try:
            self._send(
                {
                    "op": "unsubscribe",
                    "id": "physical-model-camera",
                    "topic": self._topic,
                }
            )
        except Exception:
            pass
        self._stop.set()
        try:
            self._socket.close()
        finally:
            self._thread.join(timeout=2.0)

    def _send(self, payload: dict[str, Any]) -> None:
        with self._send_lock:
            self._socket.send(json.dumps(payload, separators=(",", ":")))

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw_message = self._socket.recv()
            except self._websocket.WebSocketTimeoutException:
                continue
            except Exception as error:
                if not self._stop.is_set():
                    self._set_error(error)
                return
            if not raw_message:
                if not self._stop.is_set():
                    self._set_error(RuntimeError("camera rosbridge connection closed"))
                return
            try:
                message = json.loads(raw_message)
                if not isinstance(message, dict):
                    continue
                if message.get("op") == "status":
                    if message.get("level") == "error":
                        self._set_error(
                            RuntimeError(
                                f"camera rosbridge error: {message.get('msg', message)}"
                            )
                        )
                        return
                    continue
                if (
                    message.get("op") != "publish"
                    or message.get("topic") != self._topic
                ):
                    continue
                self._receive_id += 1
                frame = decode_rosbridge_camera_publish(
                    message,
                    receive_id=self._receive_id,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                with self._lock:
                    self._decode_error_count += 1
                    self._last_decode_error = error
                continue
            with self._lock:
                self._latest = frame
            if self._frame_callback is not None:
                try:
                    self._frame_callback(frame)
                except Exception as error:
                    self._set_error(
                        RuntimeError(f"camera frame callback failed: {error}")
                    )
                    return

    def _set_error(self, error: Exception) -> None:
        with self._lock:
            self._error = error


class RosbridgeTwistPublisher:
    def __init__(self, url: str, topic: str) -> None:
        websocket = import_websocket()
        self._websocket = websocket
        self._url = url
        self._topic = topic
        self._lock = Lock()
        self._send_lock = Lock()
        self._stop = Event()
        self._error: Exception | None = None
        self._sequence = 0
        try:
            self._socket = websocket.create_connection(
                url,
                timeout=5.0,
                enable_multithread=True,
            )
            self._socket.settimeout(1.0)
            self._send(
                {
                    "op": "advertise",
                    "id": "physical-model-command",
                    "topic": topic,
                    "type": "duckietown_msgs/Twist2DStamped",
                    "queue_size": 1,
                }
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not advertise {topic} through {url}: {error}"
            ) from error
        self._thread = Thread(
            target=self._receive_loop,
            name="rosbridge-command-status",
            daemon=True,
        )
        self._thread.start()

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def publish(self, command: Any) -> None:
        if self.error is not None and not command.stopped:
            raise RuntimeError(f"command rosbridge failed: {self.error}")
        now = time()
        seconds = int(now)
        self._send(
            {
                "op": "publish",
                "id": f"physical-model-command-{self._sequence}",
                "topic": self._topic,
                "msg": {
                    "header": {
                        "seq": self._sequence,
                        "stamp": {
                            "secs": seconds,
                            "nsecs": int((now - seconds) * 1_000_000_000),
                        },
                        "frame_id": "",
                    },
                    "v": float(command.linear_velocity),
                    "omega": float(command.angular_velocity),
                },
            }
        )
        self._sequence += 1

    def close(self) -> None:
        if self._stop.is_set():
            return
        try:
            self._send(
                {
                    "op": "unadvertise",
                    "id": "physical-model-command",
                    "topic": self._topic,
                }
            )
        except Exception:
            pass
        self._stop.set()
        try:
            self._socket.close()
        finally:
            self._thread.join(timeout=2.0)

    def _send(self, payload: dict[str, Any]) -> None:
        with self._send_lock:
            self._socket.send(json.dumps(payload, separators=(",", ":")))

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw_message = self._socket.recv()
            except self._websocket.WebSocketTimeoutException:
                continue
            except Exception as error:
                if not self._stop.is_set():
                    self._set_error(error)
                return
            if not raw_message:
                if not self._stop.is_set():
                    self._set_error(RuntimeError("command rosbridge connection closed"))
                return
            try:
                message = json.loads(raw_message)
            except (TypeError, json.JSONDecodeError):
                continue
            if (
                isinstance(message, dict)
                and message.get("op") == "status"
                and message.get("level") == "error"
            ):
                self._set_error(
                    RuntimeError(
                        f"command rosbridge error: {message.get('msg', message)}"
                    )
                )
                return

    def _set_error(self, error: Exception) -> None:
        with self._lock:
            self._error = error


def import_websocket():
    try:
        import websocket
    except ImportError as error:
        raise RuntimeError(
            "websocket-client is required; install requirements/gym-duckietown.txt "
            "inside gymdt39_venv"
        ) from error
    if not hasattr(websocket, "create_connection"):
        raise RuntimeError(
            "The installed 'websocket' module is not websocket-client; install "
            "requirements/gym-duckietown.txt inside gymdt39_venv"
        )
    return websocket


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
    published: PublishedInference | None,
    current_command: Any,
    armed: bool,
    emergency_stop: bool,
    camera_age: float | None,
    wheel_scale: float,
    sample_count: int,
    policy_control_names: tuple[str, ...],
) -> None:
    screen.fill(BACKGROUND)
    image_area = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT - PANEL_HEIGHT)
    if published is not None:
        image = published.result.image
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
    camera_text = "none" if camera_age is None else f"{camera_age:.3f}s"
    render_text(
        screen,
        fonts["small"],
        f"current Twist: v={current_command.linear_velocity:+.3f} m/s, "
        f"omega={current_command.angular_velocity:+.3f} rad/s   |   "
        f"camera age {camera_text}   |   samples {sample_count}   |   "
        f"{current_command.reason}",
        (245, y + 7),
        MUTED,
    )

    if published is not None:
        result = published.result
        prediction = result.prediction
        model_left, model_right = (
            float(value) for value in prediction.wheel_commands
        )
        sent_left, sent_right = published.effective_wheels
        sent_v_norm = 0.5 * (sent_left + sent_right)
        sent_omega_norm = 0.5 * (sent_right - sent_left)

        y += 45
        render_text(
            screen,
            fonts["normal"],
            "Model output:",
            (20, y),
            MUTED,
        )
        render_text(
            screen,
            fonts["normal"],
            format_policy_controls(
                prediction.policy_controls,
                policy_control_names,
            ),
            (165, y),
        )

        y += 39
        render_text(screen, fonts["normal"], "Gym wheels", (20, y))
        render_text(
            screen,
            fonts["normal"],
            f"left {model_left:+.3f}",
            (190, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(300, y + 3, 190, 20),
            model_left,
            LEFT_COLOR,
        )
        render_text(
            screen,
            fonts["normal"],
            f"right {model_right:+.3f}",
            (515, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(635, y + 3, 185, 20),
            model_right,
            RIGHT_COLOR,
        )

        y += 39
        render_text(screen, fonts["normal"], "Sent wheel-equiv.", (20, y))
        render_text(
            screen,
            fonts["normal"],
            f"left {sent_left:+.3f}",
            (190, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(300, y + 3, 190, 20),
            sent_left,
            LEFT_COLOR,
        )
        render_text(
            screen,
            fonts["normal"],
            f"right {sent_right:+.3f}",
            (515, y),
        )
        draw_signal_bar(
            screen,
            pygame.Rect(635, y + 3, 185, 20),
            sent_right,
            RIGHT_COLOR,
        )

        draw_direction_arrow(
            screen,
            (955, y + 15),
            published.effective_wheels,
        )
        render_text(screen, fonts["small"], "Movement", (920, y - 65), MUTED)

        y += 40
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
        y += 34
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
    else:
        render_text(
            screen,
            fonts["normal"],
            "Arm the controller to run inference on the latest camera frame.",
            (20, panel.y + 80),
            MUTED,
        )

    render_text(
        screen,
        fonts["normal"],
        "Enter: arm/disarm    Space: E-stop    C: clear E-stop    Esc: quit",
        (20, panel.bottom - 35),
        ACCENT,
    )


def main() -> int:
    args = parse_args()
    subscriber = None
    publisher = None
    worker = None
    recorder = None
    control = None
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

        print(
            f"Loading checkpoint: {args.checkpoint.expanduser().resolve()}",
            flush=True,
        )
        bundle = load_policy_bundle(model_loader_args(args))
        print(f"Checkpoint type: {bundle.checkpoint_type}; device: {bundle.device}")
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
            nominal_control_period=(
                1.0 / args.max_inference_rate
                if args.max_inference_rate > 0.0
                else 1.0 / 30.0
            ),
            forward_only=False,
            rate_limit_commands=args.rate_limit_commands,
        )
        control = PhysicalDuckiebotControl(
            DuckietownActionControl(mode="wheel"),
            limits=limits,
        )
        worker = InferenceWorker(
            bundle,
            max_rate=args.max_inference_rate,
        )

        print(f"Connecting to {rosbridge_url}...", flush=True)
        subscriber = RosbridgeCameraSubscriber(
            rosbridge_url,
            camera_topic,
            frame_callback=worker.submit,
        )
        publisher = RosbridgeTwistPublisher(rosbridge_url, command_topic)
        publisher.publish(control.disarm())

        policy_control_names = (
            tuple(bundle.action_control.control_names)
            if bundle.action_control is not None
            else ("left_wheel", "right_wheel")
        )
        metadata = {
            "env": "physical_duckiebot",
            "runtime": "physical_duckiebot_model_control.py",
            "robot_name": robot_name,
            "robot_ip": robot_ip,
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_type": bundle.checkpoint_type,
            "checkpoint_model": bundle.config.get("model", "mobilenet_v3_small"),
            "checkpoint_action_mode": bundle.config.get("action_mode", "wheel"),
            "policy_control_names": list(policy_control_names),
            "preprocess": asdict(bundle.preprocess),
            "wheel_action_scale": args.wheel_action_scale,
            "physical_control_limits": asdict(limits),
            "max_inference_rate": args.max_inference_rate,
            "camera_topic": camera_topic,
            "command_topic": command_topic,
            "rosbridge_url": rosbridge_url,
            "alignment": (
                "Each saved raw camera frame is paired with the deterministic "
                "model output and Twist2DStamped command published from it."
            ),
        }
        if not args.no_recording:
            recorder = ModelActionRecorder(args.output_dir, metadata)
            print(f"Recording automatically to {recorder.run_dir}")

        pygame.init()
        pygame_initialized = True
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Physical Duckiebot model control — focus here")
        fonts = {
            "large": pygame.font.Font(None, 38),
            "normal": pygame.font.Font(None, 25),
            "small": pygame.font.Font(None, 20),
        }
        clock = pygame.time.Clock()

        epoch = 0
        last_published: PublishedInference | None = None
        command = control.last_command
        next_status_at = monotonic()
        camera_decode_warning_shown = False
        running = True
        print(
            f"Camera: {camera_topic}; command: {command_topic}; "
            f"recording={'off' if recorder is None else 'on'}"
        )
        print(
            f"Limits: v=±{limits.max_linear_velocity:.3f} m/s, "
            f"omega=±{limits.max_angular_velocity:.3f} rad/s, "
            f"wheel range=[-{args.wheel_action_scale:.3f}, "
            f"+{args.wheel_action_scale:.3f}], reverse=on, "
            f"rate limit={'on' if limits.rate_limit_commands else 'off'}, "
            f"watchdog={limits.command_timeout:.3f}s"
        )
        print("Enter: arm/disarm  Space: E-stop  C: clear  Escape: quit")

        while running:
            now = monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        epoch += 1
                        worker.set_state(enabled=False, epoch=epoch)
                        command = control.engage_emergency_stop(now)
                        publisher.publish(command)
                        print("EMERGENCY STOP LATCHED", flush=True)
                    elif event.key == pygame.K_c:
                        epoch += 1
                        worker.set_state(enabled=False, epoch=epoch)
                        command = control.clear_emergency_stop(now)
                        publisher.publish(command)
                        print(
                            "Emergency stop cleared; controller remains DISARMED",
                            flush=True,
                        )
                    elif event.key == pygame.K_RETURN:
                        if control.armed:
                            epoch += 1
                            worker.set_state(enabled=False, epoch=epoch)
                            command = control.disarm(now)
                            publisher.publish(command)
                            print("Controller DISARMED", flush=True)
                        elif control.emergency_stop_latched:
                            print("Clear E-stop with C before arming.", flush=True)
                        else:
                            frame = subscriber.latest()
                            frame_age = (
                                None
                                if frame is None
                                else max(0.0, now - frame.received_at)
                            )
                            if frame_age is None or frame_age > args.max_frame_age:
                                print(
                                    "Cannot arm: no fresh camera frame.",
                                    flush=True,
                                )
                            else:
                                epoch += 1
                                command = control.arm(now)
                                publisher.publish(command)
                                worker.set_state(enabled=True, epoch=epoch)
                                worker.submit(frame)
                                print("Controller ARMED", flush=True)

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
            if worker.error is not None:
                command = control.engage_emergency_stop(now)
                try:
                    publisher.publish(command)
                finally:
                    raise RuntimeError(f"model inference failed: {worker.error}")

            frame = subscriber.latest()
            camera_age = (
                None if frame is None else max(0.0, now - frame.received_at)
            )
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
                if recorder is not None:
                    recorder.record(
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

            if control.armed:
                watchdog_command = control.watchdog(monotonic())
                if watchdog_command.reason != command.reason or (
                    watchdog_command.stopped and not command.stopped
                ):
                    command = watchdog_command
                    publisher.publish(command)
                    if not control.armed:
                        epoch += 1
                        worker.set_state(enabled=False, epoch=epoch)
                        print(
                            f"Controller stopped and disarmed: {command.reason}",
                            flush=True,
                        )

            if args.status_period > 0.0 and now >= next_status_at:
                camera_text = "none" if camera_age is None else f"{camera_age:.3f}s"
                print(
                    f"status armed={int(control.armed)} "
                    f"estop={int(control.emergency_stop_latched)} "
                    f"camera_age={camera_text} reason={command.reason} "
                    f"v={command.linear_velocity:+.3f} "
                    f"omega={command.angular_velocity:+.3f} "
                    f"samples={0 if recorder is None else recorder.sample_count}",
                    flush=True,
                )
                next_status_at = now + args.status_period

            draw_status(
                screen,
                fonts,
                published=last_published,
                current_command=command,
                armed=control.armed,
                emergency_stop=control.emergency_stop_latched,
                camera_age=camera_age,
                wheel_scale=args.wheel_action_scale,
                sample_count=0 if recorder is None else recorder.sample_count,
                policy_control_names=policy_control_names,
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
        if recorder is not None:
            recorder.close()
            print(
                f"Recording saved: {recorder.sample_count} samples in "
                f"{recorder.run_dir}"
            )
        if subscriber is not None:
            subscriber.close()
        if publisher is not None:
            publisher.close()
        if pygame_initialized:
            pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
