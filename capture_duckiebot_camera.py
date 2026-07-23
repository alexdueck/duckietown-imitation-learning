#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Capture one compressed camera frame from a physical Duckiebot.

Run this script in a ROS environment connected to the robot, for example in
the container opened by ``dts start_gui_tools ROBOT_NAME``.  The script is
read-only with respect to the robot: it subscribes to the camera topic and
does not publish control commands.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cli_completion import parse_args_with_completion


DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save one frame from a physical Duckiebot's compressed ROS camera "
            "topic, together with diagnostic metadata."
        )
    )
    parser.add_argument(
        "robot_name",
        nargs="?",
        default=os.environ.get("VEHICLE_NAME"),
        help=(
            "Duckiebot hostname without '.local'. Defaults to VEHICLE_NAME when "
            "that environment variable is set."
        ),
    )
    parser.add_argument(
        "--topic",
        default=None,
        help=(
            "Override the camera topic. The default is "
            "/ROBOT_NAME/camera_node/image/compressed."
        ),
    )
    parser.add_argument(
        "--robot-ip",
        default=None,
        help=(
            "Explicit robot IP used to resolve ROBOT_NAME.local inside the "
            "container. Defaults to the numeric host in ROS_MASTER_URI."
        ),
    )
    parser.add_argument(
        "--no-hosts-fix",
        action="store_true",
        help=(
            "Do not update /etc/hosts when ROBOT_NAME.local is unresolved or "
            "points to an address different from ROS_MASTER_URI."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the original compressed camera frame (.jpg or .png).",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Metadata JSON destination. Defaults to OUTPUT with suffix .json.",
    )
    parser.add_argument(
        "--policy-input-output",
        type=Path,
        default=None,
        help=(
            "Optionally save an RGB crop resized to IMAGE_SIZE as a visual "
            "preview of the image presented to a policy."
        ),
    )
    parser.add_argument(
        "--crop-y-start",
        type=int,
        default=0,
        help="First retained image row for --policy-input-output (default: 0).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=(
            "Square width and height for --policy-input-output "
            f"(default: {DEFAULT_IMAGE_SIZE})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for a camera message "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parse_args_with_completion(parser)


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


def camera_topic(robot_name: str, override: str | None) -> str:
    if override is None:
        return f"/{robot_name}/camera_node/image/compressed"
    topic = override.strip()
    if not topic:
        raise ValueError("--topic must not be empty")
    return topic if topic.startswith("/") else f"/{topic}"


def metadata_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def numeric_robot_ip(
    explicit_ip: str | None,
    ros_master_uri: str | None,
) -> str:
    candidate = explicit_ip
    source = "--robot-ip"
    if candidate is None:
        source = "ROS_MASTER_URI"
        if not ros_master_uri:
            raise ValueError(
                "ROS_MASTER_URI is not set; provide the current address with --robot-ip"
            )
        candidate = urlparse(ros_master_uri).hostname
    if not candidate:
        raise ValueError(f"Could not extract a robot address from {source}")
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError as error:
        raise ValueError(
            f"{source} does not contain a numeric robot IP: {candidate!r}; "
            "start GUI tools with --ip or pass --robot-ip"
        ) from error


def _remove_aliases_from_hosts_line(line: str, aliases: set[str]) -> str | None:
    content, separator, comment = line.partition("#")
    fields = content.split()
    if len(fields) < 2:
        return line
    remaining_aliases = [alias for alias in fields[1:] if alias not in aliases]
    if len(remaining_aliases) == len(fields) - 1:
        return line
    if not remaining_aliases:
        return None
    rebuilt = f"{fields[0]} {' '.join(remaining_aliases)}"
    if separator:
        rebuilt += f"  # {comment.strip()}"
    return rebuilt


def _hosts_alias_addresses(
    lines: list[str],
    aliases: set[str],
) -> dict[str, set[str]]:
    addresses = {alias: set() for alias in aliases}
    for line in lines:
        content = line.partition("#")[0]
        fields = content.split()
        if len(fields) < 2:
            continue
        for alias in aliases.intersection(fields[1:]):
            addresses[alias].add(fields[0])
    return addresses


def update_hosts_mapping(
    robot_name: str,
    robot_ip: str,
    hosts_path: Path = Path("/etc/hosts"),
) -> bool:
    """Map the robot's ROS-advertised hostname to its current IP.

    Returns ``True`` when the hosts file changed. Docker supplies /etc/hosts as
    a special mounted file, so it must be rewritten in place rather than via an
    atomic rename.
    """

    aliases = {robot_name, f"{robot_name}.local"}
    try:
        original_lines = hosts_path.read_text().splitlines()
    except OSError as error:
        raise RuntimeError(f"Could not read {hosts_path}: {error}") from error

    hosts_addresses = _hosts_alias_addresses(original_lines, aliases)
    if all(addresses == {robot_ip} for addresses in hosts_addresses.values()):
        return False

    updated_lines: list[str] = []
    for line in original_lines:
        updated_line = _remove_aliases_from_hosts_line(line, aliases)
        if updated_line is not None:
            updated_lines.append(updated_line)
    updated_lines.append(f"{robot_ip} {robot_name}.local {robot_name}")

    try:
        with hosts_path.open("w") as file:
            file.write("\n".join(updated_lines) + "\n")
    except OSError as error:
        raise RuntimeError(
            f"Could not update {hosts_path} with {robot_name}.local -> {robot_ip}. "
            "Run the GUI-tools container as root or use --no-hosts-fix after "
            "configuring name resolution yourself."
        ) from error
    return True


def configure_robot_hostname(
    robot_name: str,
    *,
    explicit_ip: str | None,
    enabled: bool,
    ros_master_uri: str | None = None,
    hosts_path: Path = Path("/etc/hosts"),
) -> tuple[str | None, bool]:
    if not enabled:
        return None, False
    robot_ip = numeric_robot_ip(
        explicit_ip=explicit_ip,
        ros_master_uri=(
            os.environ.get("ROS_MASTER_URI")
            if ros_master_uri is None
            else ros_master_uri
        ),
    )
    changed = update_hosts_mapping(robot_name, robot_ip, hosts_path=hosts_path)
    return robot_ip, changed


def ensure_output_paths_available(paths: list[Path], overwrite: bool) -> None:
    duplicate_paths = {
        path for path in paths if sum(candidate == path for candidate in paths) > 1
    }
    if duplicate_paths:
        formatted = ", ".join(str(path) for path in sorted(duplicate_paths))
        raise ValueError(f"Output paths must be distinct: {formatted}")

    existing_paths = [path for path in paths if path.exists()]
    if existing_paths and not overwrite:
        formatted = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Output already exists: {formatted}. Use --overwrite to replace it."
        )


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


def validate_output_suffix(path: Path, image_format: str) -> None:
    suffix = path.suffix.lower()
    accepted_suffixes = {"jpeg": {".jpg", ".jpeg"}, "png": {".png"}}[image_format]
    if suffix not in accepted_suffixes:
        accepted = " or ".join(sorted(accepted_suffixes))
        raise ValueError(
            f"{path} has suffix {suffix!r}, but the camera sent {image_format}; "
            f"use {accepted}"
        )


def decode_bgr(payload: bytes):
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "OpenCV and NumPy are required to validate the camera frame. Run the "
            "script in the Duckietown GUI tools/ROS container."
        ) from error

    encoded = np.frombuffer(payload, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("OpenCV could not decode the compressed camera frame")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected a three-channel camera image, received shape {image_bgr.shape}"
        )
    return image_bgr


def make_policy_preview(image_bgr, crop_y_start: int, image_size: int):
    import cv2

    if image_size <= 0:
        raise ValueError("--image-size must be positive")
    height = int(image_bgr.shape[0])
    if not 0 <= crop_y_start < height:
        raise ValueError(
            f"--crop-y-start must be in [0, {height - 1}], received {crop_y_start}"
        )

    # OpenCV decodes compressed images as BGR. Convert explicitly so the array
    # follows the RGB contract used by the training pipelines in this repo.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    cropped_rgb = image_rgb[crop_y_start:, :, :]
    return cv2.resize(
        cropped_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_bytes(payload)
    temporary_path.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(path)


def write_rgb_image(path: Path, image_rgb) -> None:
    import cv2

    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(
            f"--policy-input-output must end in .jpg, .jpeg, or .png: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"OpenCV could not write policy preview to {path}")


def ros_stamp_metadata(message) -> dict[str, int | float | str]:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    seconds = int(getattr(stamp, "secs", 0))
    nanoseconds = int(getattr(stamp, "nsecs", 0))
    return {
        "seq": int(getattr(header, "seq", 0)),
        "frame_id": str(getattr(header, "frame_id", "")),
        "stamp_seconds": seconds,
        "stamp_nanoseconds": nanoseconds,
        "stamp": seconds + nanoseconds / 1_000_000_000.0,
    }


def capture_message(topic: str, timeout: float):
    if timeout <= 0:
        raise ValueError("--timeout must be positive")
    try:
        import rospy
        from sensor_msgs.msg import CompressedImage
    except ImportError as error:
        raise RuntimeError(
            "ROS Python packages are unavailable. Start this script inside a "
            "Duckietown ROS environment, for example with dts start_gui_tools."
        ) from error

    rospy.init_node("capture_duckiebot_camera", anonymous=True, disable_signals=True)
    try:
        return rospy.wait_for_message(topic, CompressedImage, timeout=timeout)
    except rospy.ROSException as error:
        raise TimeoutError(
            f"No camera message received from {topic!r} within {timeout:g} seconds"
        ) from error


def save_capture(
    message,
    *,
    robot_name: str,
    topic: str,
    output_path: Path,
    metadata_path: Path,
    policy_input_path: Path | None,
    crop_y_start: int,
    image_size: int,
    robot_ip: str | None = None,
) -> dict[str, Any]:
    payload = bytes(message.data)
    if not payload:
        raise ValueError("Received an empty compressed camera message")

    message_format = str(getattr(message, "format", ""))
    image_format = compressed_format(message_format, payload)
    validate_output_suffix(output_path, image_format)
    image_bgr = decode_bgr(payload)
    height, width, channels = (int(value) for value in image_bgr.shape)

    import cv2

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    channel_means = image_rgb.mean(axis=(0, 1))
    metadata: dict[str, Any] = {
        "robot_name": robot_name,
        "robot_ip": robot_ip,
        "topic": topic,
        "received_at_utc": datetime.now(timezone.utc).isoformat(),
        "message_format": message_format,
        "detected_compressed_format": image_format,
        "compressed_size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "output": str(output_path),
        "decoded_shape_hwc": [height, width, channels],
        "decoder_array_channel_order": "BGR (OpenCV)",
        "policy_array_channel_order": "RGB",
        "rgb_channel_mean_0_255": {
            "red": float(channel_means[0]),
            "green": float(channel_means[1]),
            "blue": float(channel_means[2]),
        },
        "ros_header": ros_stamp_metadata(message),
    }

    if policy_input_path is not None:
        preview_rgb = make_policy_preview(
            image_bgr,
            crop_y_start=crop_y_start,
            image_size=image_size,
        )
        write_rgb_image(policy_input_path, preview_rgb)
        metadata["policy_input_preview"] = {
            "output": str(policy_input_path),
            "crop_y_start": crop_y_start,
            "resize_width": image_size,
            "resize_height": image_size,
            "shape_hwc": [image_size, image_size, 3],
            "array_channel_order_before_encoding": "RGB",
            "normalization_applied": False,
        }

    write_bytes_atomic(output_path, payload)
    write_json_atomic(metadata_path, metadata)
    return metadata


def main() -> int:
    args = parse_args()
    try:
        robot_name = normalize_robot_name(args.robot_name)
        topic = camera_topic(robot_name, args.topic)
        robot_ip, hosts_changed = configure_robot_hostname(
            robot_name,
            explicit_ip=args.robot_ip,
            enabled=not args.no_hosts_fix,
        )
        output_path = args.output.expanduser().resolve()
        metadata_path = (
            args.metadata_output.expanduser().resolve()
            if args.metadata_output is not None
            else metadata_path_for(output_path)
        )
        policy_input_path = (
            args.policy_input_output.expanduser().resolve()
            if args.policy_input_output is not None
            else None
        )
        output_paths = [output_path, metadata_path]
        if policy_input_path is not None:
            output_paths.append(policy_input_path)
        ensure_output_paths_available(output_paths, overwrite=args.overwrite)

        if robot_ip is not None:
            status = "updated" if hosts_changed else "already pinned"
            print(f"Robot hostname: {robot_name}.local -> {robot_ip} ({status})")
        print(f"Waiting for one frame on {topic} ...", flush=True)
        message = capture_message(topic, timeout=args.timeout)
        metadata = save_capture(
            message,
            robot_name=robot_name,
            topic=topic,
            output_path=output_path,
            metadata_path=metadata_path,
            policy_input_path=policy_input_path,
            crop_y_start=args.crop_y_start,
            image_size=args.image_size,
            robot_ip=robot_ip,
        )
    except (FileExistsError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    height, width, _ = metadata["decoded_shape_hwc"]
    print(f"Saved camera frame: {output_path}")
    print(f"Decoded image:      {width}x{height} RGB")
    print(f"Metadata:           {metadata_path}")
    if policy_input_path is not None:
        print(f"Policy preview:     {policy_input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
