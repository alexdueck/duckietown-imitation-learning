"""ROS-independent Duckiebot camera and command transport over rosbridge."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import ipaddress
import json
import re
import socket
from threading import Event, Lock, Thread
from time import monotonic, time
from typing import Any, Callable


@dataclass(frozen=True)
class RosbridgeCameraFrame:
    payload: bytes
    suffix: str
    message_format: str
    seq: int
    stamp: float
    received_at: float
    receive_id: int


def normalize_robot_name(value: str | None) -> str:
    if value is None or not value.strip():
        raise ValueError(
            "robot_name is required unless the VEHICLE_NAME environment variable is set"
        )
    robot_name = value.strip()
    if robot_name.endswith(".local"):
        robot_name = robot_name[: -len(".local")]
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", robot_name):
        raise ValueError(f"Invalid Duckiebot hostname: {value!r}")
    return robot_name


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


def compressed_format(message_format: str, payload: bytes) -> str:
    format_lower = message_format.lower()
    if "jpeg" in format_lower or "jpg" in format_lower:
        return "jpeg"
    if "png" in format_lower:
        return "png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    raise ValueError(
        f"Unsupported compressed camera format {message_format!r}; expected JPEG or PNG"
    )


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


class RosbridgeCameraSubscriber:
    def __init__(
        self,
        url: str,
        topic: str,
        *,
        frame_callback: Callable[[RosbridgeCameraFrame], None] | None = None,
        client_id: str = "duckiebot-control",
    ) -> None:
        websocket = import_websocket()
        self._websocket = websocket
        self._url = url
        self._topic = topic
        self._subscription_id = f"{client_id}-camera"
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
                    "id": self._subscription_id,
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
            name=f"{client_id}-camera",
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
                    "id": self._subscription_id,
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
    def __init__(
        self,
        url: str,
        topic: str,
        *,
        client_id: str = "duckiebot-control",
    ) -> None:
        websocket = import_websocket()
        self._websocket = websocket
        self._url = url
        self._topic = topic
        self._advertisement_id = f"{client_id}-command"
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
                    "id": self._advertisement_id,
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
            name=f"{client_id}-command-status",
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
                "id": f"{self._advertisement_id}-{self._sequence}",
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
                    "id": self._advertisement_id,
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
