#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Drive and record a physical Duckiebot with keyboard or PS4 controller."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from threading import Lock
from time import monotonic, time
from typing import Any

from cli_completion import parse_args_with_completion
from capture_duckiebot_camera import (
    compressed_format,
    configure_robot_hostname,
    normalize_robot_name,
)
from duckiebot_dataset_recorder import PhysicalDatasetRecorder
from duckiebot_hardware_control import PhysicalControlLimits, PhysicalDuckiebotControl
from duckiebot_teleop_input import (
    ActionMixer,
    DriveProfile,
    KeyboardInput,
    PS4Input,
    RemoteInput,
)
from duckietown_action_control import DuckietownActionControl
from duckietown_paths import IMITATION_LEARNING_TRAIN_DATA_DIR


@dataclass(frozen=True)
class CameraFrame:
    payload: bytes
    suffix: str
    seq: int
    stamp: float
    received_at: float


class LatestCamera:
    def __init__(self) -> None:
        self._lock = Lock()
        self._frame: CameraFrame | None = None

    def callback(self, message: Any) -> None:
        payload = bytes(message.data)
        if not payload:
            return
        image_format = compressed_format(str(getattr(message, "format", "")), payload)
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        stamp_value = float(getattr(stamp, "secs", 0)) + float(
            getattr(stamp, "nsecs", 0)
        ) / 1e9
        frame = CameraFrame(
            payload=payload,
            suffix=".jpg" if image_format == "jpeg" else ".png",
            seq=int(getattr(header, "seq", 0)),
            stamp=stamp_value,
            received_at=monotonic(),
        )
        with self._lock:
            self._frame = frame

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._frame


class DirectRosCommandPublisher:
    """Publish through TCPROS; suitable when the robot can reach this host."""

    def __init__(self, rospy_module: Any, message_type: Any, topic: str) -> None:
        self._message_type = message_type
        self._publisher = rospy_module.Publisher(topic, message_type, queue_size=1)

    def publish(self, command: Any) -> None:
        message = self._message_type()
        message.v = command.linear_velocity
        message.omega = command.angular_velocity
        self._publisher.publish(message)

    def close(self) -> None:
        self._publisher.unregister()


class RosbridgeCommandPublisher:
    """Publish over an outbound WebSocket, avoiding Docker's TCPROS callback."""

    def __init__(self, url: str, topic: str) -> None:
        try:
            import websocket
        except ImportError as error:
            raise RuntimeError(
                "websocket-client is required for --command-transport rosbridge"
            ) from error
        self._topic = topic
        self._sequence = 0
        try:
            self._socket = websocket.create_connection(url, timeout=5.0)
            self._send(
                {
                    "op": "advertise",
                    "topic": topic,
                    "type": "duckietown_msgs/Twist2DStamped",
                    "queue_size": 1,
                }
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not connect to ROS bridge at {url}: {error}"
            ) from error

    def publish(self, command: Any) -> None:
        now = time()
        seconds = int(now)
        self._send(
            {
                "op": "publish",
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
                    "v": command.linear_velocity,
                    "omega": command.angular_velocity,
                },
            }
        )
        self._sequence += 1

    def close(self) -> None:
        try:
            self._send({"op": "unadvertise", "topic": self._topic})
        finally:
            self._socket.close()

    def _send(self, payload: dict[str, Any]) -> None:
        self._socket.send(json.dumps(payload, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive a physical Duckiebot and optionally record IL samples."
    )
    parser.add_argument("robot_name", nargs="?", default=os.environ.get("VEHICLE_NAME"))
    parser.add_argument(
        "--input",
        choices=("keyboard", "ps4", "ps4-bridge"),
        default="keyboard",
        help="Use ps4-bridge when Docker Desktop runs on macOS.",
    )
    parser.add_argument("--bridge-host", default="host.docker.internal")
    parser.add_argument("--bridge-port", type=int, default=8765)
    parser.add_argument("--camera-topic", default=None)
    parser.add_argument("--command-topic", default=None)
    parser.add_argument(
        "--command-transport",
        choices=("rosbridge", "ros"),
        default="rosbridge",
        help=(
            "rosbridge works across Docker Desktop; ros uses direct TCPROS "
            "and requires the robot to reach the container."
        ),
    )
    parser.add_argument("--rosbridge-port", type=int, default=9001)
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--no-hosts-fix", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=IMITATION_LEARNING_TRAIN_DATA_DIR)
    parser.add_argument("--control-rate", type=float, default=20.0)
    parser.add_argument(
        "--status-period",
        type=float,
        default=0.0,
        help="Print periodic status every N seconds; disabled by default.",
    )
    parser.add_argument("--max-frame-age", type=float, default=0.50)
    parser.add_argument(
        "--max-linear-velocity",
        type=float,
        default=0.41,
        help="Full-stick forward/reverse speed in m/s (realbot joy_mapper: 0.41).",
    )
    parser.add_argument(
        "--max-angular-velocity",
        type=float,
        default=8.0,
        help="Full-stick turn rate in rad/s (realbot kinematics limit: 8.0).",
    )
    parser.add_argument("--max-linear-acceleration", type=float, default=1.0)
    parser.add_argument("--max-angular-acceleration", type=float, default=20.0)
    parser.add_argument(
        "--rate-limit-analog",
        action="store_true",
        help=(
            "Apply keyboard-style input and command ramps to PS4 input; "
            "analog control is direct by default."
        ),
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Disable reverse driving; reverse is enabled by default for teleop.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.08,
        help="Zero stick axes whose absolute value is below this threshold.",
    )
    parser.add_argument("--throttle-axis", type=int, default=1)
    parser.add_argument("--steering-axis", type=int, default=0)
    parser.add_argument("--arm-button", type=int, default=0, help="PS4 Cross by SDL default.")
    parser.add_argument("--record-button", type=int, default=6, help="PS4 Options by SDL default.")
    parser.add_argument("--emergency-button", type=int, default=1, help="PS4 Circle by SDL default.")
    parser.add_argument("--clear-button", type=int, default=3, help="PS4 Triangle by SDL default.")
    return parse_args_with_completion(parser)


def _effective_wheels(command, limits: PhysicalControlLimits) -> tuple[float, float]:
    linear = command.linear_velocity / limits.max_linear_velocity
    angular = command.angular_velocity / limits.max_angular_velocity
    return linear - angular, linear + angular


def main() -> int:
    args = parse_args()
    if args.control_rate <= 0:
        print("error: --control-rate must be positive", file=sys.stderr)
        return 2
    if args.status_period < 0:
        print("error: --status-period must not be negative", file=sys.stderr)
        return 2
    if not 0 <= args.deadzone < 1:
        print("error: --deadzone must be in [0, 1)", file=sys.stderr)
        return 2
    try:
        import rospy
        from duckietown_msgs.msg import Twist2DStamped
        from sensor_msgs.msg import CompressedImage
    except ImportError as error:
        print(
            "error: ROS Duckietown packages are required; run inside "
            f"dts start_gui_tools: {error}",
            file=sys.stderr,
        )
        return 2

    recorder: PhysicalDatasetRecorder | None = None
    pygame_initialized = False
    control = None
    publisher = None
    input_device = None
    pygame = None
    try:
        robot_name = normalize_robot_name(args.robot_name)
        robot_ip, _ = configure_robot_hostname(
            robot_name,
            explicit_ip=args.robot_ip,
            enabled=not args.no_hosts_fix,
        )
        camera_topic = args.camera_topic or f"/{robot_name}/camera_node/image/compressed"
        command_topic = args.command_topic or f"/{robot_name}/joy_mapper_node/car_cmd"
        smooth_inputs = args.input == "keyboard" or args.rate_limit_analog
        limits = PhysicalControlLimits(
            max_linear_velocity=args.max_linear_velocity,
            max_angular_velocity=args.max_angular_velocity,
            max_linear_acceleration=args.max_linear_acceleration,
            max_angular_acceleration=args.max_angular_acceleration,
            max_frame_age=args.max_frame_age,
            nominal_control_period=1.0 / args.control_rate,
            forward_only=args.forward_only,
            rate_limit_commands=smooth_inputs,
        )
        control = PhysicalDuckiebotControl(
            DuckietownActionControl(mode="wheel"), limits=limits
        )
        profile = DriveProfile(deadzone=args.deadzone)
        mixer = ActionMixer(profile)

        if args.input == "ps4-bridge":
            input_device = RemoteInput(args.bridge_host, args.bridge_port)
            print(f"Connected to PS4 bridge at {args.bridge_host}:{args.bridge_port}")
        else:
            try:
                import pygame as pygame_module
            except ImportError as error:
                raise RuntimeError(
                    "pygame is required for keyboard or direct PS4 input"
                ) from error
            pygame = pygame_module
            pygame.init()
            pygame_initialized = True
            pygame.display.set_mode((720, 180))
            pygame.display.set_caption("Duckiebot teleop — focus here")

        if args.input == "keyboard":
            input_device = KeyboardInput(pygame)
        elif args.input == "ps4":
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("No SDL controller found")
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            input_device = PS4Input(
                pygame,
                joystick,
                throttle_axis=args.throttle_axis,
                steering_axis=args.steering_axis,
                arm_button=args.arm_button,
                recording_button=args.record_button,
                emergency_button=args.emergency_button,
                clear_button=args.clear_button,
            )
            print(f"Controller: {joystick.get_name()}")

        rospy.init_node("physical_duckiebot_teleop", anonymous=True)
        camera = LatestCamera()
        rospy.Subscriber(camera_topic, CompressedImage, camera.callback, queue_size=1)
        if args.command_transport == "rosbridge":
            rosbridge_host = robot_ip or f"{robot_name}.local"
            publisher = RosbridgeCommandPublisher(
                f"ws://{rosbridge_host}:{args.rosbridge_port}",
                command_topic,
            )
        else:
            publisher = DirectRosCommandPublisher(
                rospy, Twist2DStamped, command_topic
            )
        rate = rospy.Rate(args.control_rate)
        print(
            f"Input: {input_device.name}; camera: {camera_topic}; "
            f"command: {command_topic} via {args.command_transport}"
        )
        print(
            f"Limits: v=±{limits.max_linear_velocity:.2f} m/s, "
            f"omega=±{limits.max_angular_velocity:.1f} rad/s, "
            f"deadzone={profile.deadzone:.2f}, "
            f"reverse={'off' if limits.forward_only else 'on'}, "
            f"ramps={'on' if smooth_inputs else 'off'}"
        )
        print("Enter/Cross: arm  R/Options: record  Space/Circle: E-stop  C/Triangle: clear")
        previous_at = monotonic()
        next_status_at = previous_at

        while not rospy.is_shutdown():
            now = monotonic()
            events = pygame.event.get() if pygame is not None else []
            state = input_device.poll(events)
            if state.quit:
                break
            if state.emergency_stop:
                command = control.engage_emergency_stop(now)
                mixer.reset()
                print("EMERGENCY STOP LATCHED", flush=True)
            elif state.clear_emergency_stop:
                command = control.clear_emergency_stop(now)
                mixer.reset()
                print("Emergency stop cleared; controller remains DISARMED", flush=True)
            elif state.arm_toggle:
                command = control.disarm(now) if control.armed else control.arm(now)
                if not control.armed:
                    mixer.reset()
                print(
                    "Controller " + ("ARMED" if control.armed else "DISARMED"),
                    flush=True,
                )

            if state.recording_toggle:
                if recorder is None:
                    candidate_frame = camera.latest()
                    if (
                        candidate_frame is None
                        or now - candidate_frame.received_at > args.max_frame_age
                    ):
                        print(
                            "Recording not started: no fresh camera frame is available.",
                            file=sys.stderr,
                        )
                        rate.sleep()
                        continue
                    recorder = PhysicalDatasetRecorder(
                        args.output_dir,
                        {
                            "env": "physical_duckiebot",
                            "robot_name": robot_name,
                            "camera_topic": camera_topic,
                            "command_topic": command_topic,
                            "source_observation_channel_order": "camera-compressed",
                            "saved_image_channel_order": "camera-encoded",
                            "controller": {
                                "type": input_device.name,
                                "drive_profile": asdict(profile),
                            },
                            "physical_control_limits": asdict(limits),
                            "command_transport": args.command_transport,
                            "sample_period_seconds": 1.0 / args.control_rate,
                        },
                    )
                    print(f"RECORDING -> {recorder.run_dir}")
                else:
                    print(f"Recording stopped: {recorder.sample_count} samples")
                    recorder.close()
                    recorder = None

            dt = now - previous_at
            previous_at = now
            requested_wheels = (
                mixer.update(state, dt, smooth=smooth_inputs)
                if control.armed
                else (0.0, 0.0)
            )
            frame = camera.latest()
            frame_age = None if frame is None else max(0.0, now - frame.received_at)
            command = control.update(
                requested_wheels, timestamp=now, frame_age=frame_age
            )
            publisher.publish(command)

            if args.status_period > 0 and now >= next_status_at:
                camera_status = (
                    "none" if frame_age is None else f"{frame_age:.3f}s"
                )
                print(
                    f"status input=({state.throttle:+.2f},{state.steering:+.2f}) "
                    f"armed={int(control.armed)} recording={int(recorder is not None)} "
                    f"camera_age={camera_status} reason={command.reason} "
                    f"v={command.linear_velocity:+.3f} "
                    f"omega={command.angular_velocity:+.3f}",
                    flush=True,
                )
                next_status_at = now + args.status_period

            if recorder is not None and frame is not None and command.reason == "active":
                left, right = _effective_wheels(command, limits)
                recorder.record(
                    payload=frame.payload,
                    image_suffix=frame.suffix,
                    camera_seq=frame.seq,
                    camera_stamp=frame.stamp,
                    frame_age=frame_age or 0.0,
                    left_action=left,
                    right_action=right,
                    linear_velocity=command.linear_velocity,
                    angular_velocity=command.angular_velocity,
                )
            if pygame is not None:
                pygame.display.set_caption(
                    f"Duckiebot | {'ARMED' if control.armed else command.reason} | "
                    f"{'REC' if recorder else 'not recording'} | "
                    f"v={command.linear_velocity:+.3f} ω={command.angular_velocity:+.3f}"
                )
            rate.sleep()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if recorder is not None:
            recorder.close()
            print(f"Recording saved: {recorder.sample_count} samples in {recorder.run_dir}")
        if control is not None and publisher is not None:
            try:
                command = control.disarm()
                publisher.publish(command)
                publisher.close()
            except Exception:
                pass
        if pygame_initialized:
            pygame.quit()
        if input_device is not None:
            close_input = getattr(input_device, "close", None)
            if close_input is not None:
                close_input()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
