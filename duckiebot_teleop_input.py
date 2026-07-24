#!/usr/bin/env python3
"""Input-device-independent controls for physical Duckiebot teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
from time import monotonic
from typing import Any

@dataclass(frozen=True)
class InputState:
    """One input poll, including edge-triggered operator requests."""

    throttle: float = 0.0
    steering: float = 0.0
    arm_toggle: bool = False
    recording_toggle: bool = False
    emergency_stop: bool = False
    clear_emergency_stop: bool = False
    quit: bool = False


@dataclass(frozen=True)
class DriveProfile:
    forward: float = 1.0
    backward: float = 1.0
    turn: float = 1.0
    throttle_rate: float = 6.0
    steering_rate: float = 6.0
    auto_center_rate: float = 8.0
    deadzone: float = 0.08

    def __post_init__(self) -> None:
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        for name in (
            "forward",
            "backward",
            "turn",
            "throttle_rate",
            "steering_rate",
            "auto_center_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class ActionMixer:
    """Shared smoothing and throttle/steering-to-wheel conversion."""

    def __init__(self, profile: DriveProfile | None = None) -> None:
        self.profile = profile or DriveProfile()
        self.throttle = 0.0
        self.steering = 0.0

    def reset(self) -> None:
        self.throttle = 0.0
        self.steering = 0.0

    def update(
        self,
        state: InputState,
        dt: float,
        *,
        smooth: bool = True,
    ) -> tuple[float, float]:
        dt = max(0.0, min(float(dt), 0.1))
        throttle_input = _deadzone(state.throttle, self.profile.deadzone)
        steering_input = _deadzone(state.steering, self.profile.deadzone)
        throttle_scale = (
            self.profile.forward if throttle_input >= 0.0 else self.profile.backward
        )
        target_throttle = throttle_input * throttle_scale
        target_steering = steering_input * self.profile.turn
        steering_rate = (
            self.profile.steering_rate
            if steering_input != 0.0
            else self.profile.auto_center_rate
        )
        if smooth:
            self.throttle = _move_towards(
                self.throttle,
                target_throttle,
                self.profile.throttle_rate * dt,
            )
            self.steering = _move_towards(
                self.steering,
                target_steering,
                steering_rate * dt,
            )
        else:
            self.throttle = target_throttle
            self.steering = target_steering
        left = self.throttle - self.steering
        right = self.throttle + self.steering
        scale = max(1.0, abs(left), abs(right))
        return left / scale, right / scale


class KeyboardInput:
    """Pygame keyboard adapter."""

    name = "keyboard"

    def __init__(self, pygame_module: Any) -> None:
        self.pg = pygame_module

    def poll(self, events: list[Any]) -> InputState:
        arm = recording = emergency = clear = quit_requested = False
        for event in events:
            if event.type == self.pg.QUIT:
                quit_requested = True
            elif event.type == self.pg.KEYDOWN:
                quit_requested |= event.key == self.pg.K_ESCAPE
                arm |= event.key == self.pg.K_RETURN
                recording |= event.key == self.pg.K_r
                emergency |= event.key == self.pg.K_SPACE
                clear |= event.key == self.pg.K_c

        keys = self.pg.key.get_pressed()
        forward = keys[self.pg.K_w] or keys[self.pg.K_UP]
        backward = keys[self.pg.K_s] or keys[self.pg.K_DOWN]
        left = keys[self.pg.K_a] or keys[self.pg.K_LEFT]
        right = keys[self.pg.K_d] or keys[self.pg.K_RIGHT]
        return InputState(
            throttle=float(forward) - float(backward),
            steering=float(left) - float(right),
            arm_toggle=arm,
            recording_toggle=recording,
            emergency_stop=emergency,
            clear_emergency_stop=clear,
            quit=quit_requested,
        )


class PS4Input:
    """SDL/pygame adapter for a DualShock 4 (axis and button IDs configurable)."""

    name = "ps4"

    def __init__(
        self,
        pygame_module: Any,
        joystick: Any,
        *,
        throttle_axis: int = 1,
        steering_axis: int = 0,
        arm_button: int = 0,
        recording_button: int = 6,
        emergency_button: int = 1,
        clear_button: int = 3,
    ) -> None:
        self.pg = pygame_module
        self.joystick = joystick
        self.throttle_axis = throttle_axis
        self.steering_axis = steering_axis
        self.buttons = {
            arm_button: "arm_toggle",
            recording_button: "recording_toggle",
            emergency_button: "emergency_stop",
            clear_button: "clear_emergency_stop",
        }
        unavailable = [
            button
            for button in self.buttons
            if not 0 <= button < joystick.get_numbuttons()
        ]
        if unavailable:
            raise ValueError(
                f"Controller has {joystick.get_numbuttons()} buttons; "
                f"button(s) {unavailable} are unavailable"
            )
        self._button_pressed = {
            button: bool(joystick.get_button(button)) for button in self.buttons
        }

    def poll(self, events: list[Any]) -> InputState:
        flags = dict.fromkeys(self.buttons.values(), False)
        quit_requested = False
        for event in events:
            if event.type == self.pg.QUIT:
                quit_requested = True
            elif event.type == self.pg.KEYDOWN and event.key == self.pg.K_ESCAPE:
                quit_requested = True

        # Poll button state instead of relying on JOYBUTTONDOWN. On macOS,
        # SDL can expose the axes while omitting raw joystick button events.
        for button, flag in self.buttons.items():
            pressed = bool(self.joystick.get_button(button))
            flags[flag] |= pressed and not self._button_pressed[button]
            self._button_pressed[button] = pressed
        return InputState(
            throttle=-_axis(self.joystick, self.throttle_axis),
            steering=-_axis(self.joystick, self.steering_axis),
            quit=quit_requested,
            **flags,
        )


class SDLControllerInput:
    """Standardized SDL GameController adapter used by the macOS bridge."""

    name = "ps4"

    def __init__(
        self,
        pygame_module: Any,
        controller: Any,
        *,
        throttle_axis: int = 1,
        steering_axis: int = 0,
        arm_button: int = 0,
        recording_button: int = 6,
        emergency_button: int = 1,
        clear_button: int = 3,
    ) -> None:
        self.pg = pygame_module
        self.controller = controller
        self.throttle_axis = throttle_axis
        self.steering_axis = steering_axis
        self.buttons = {
            arm_button: "arm_toggle",
            recording_button: "recording_toggle",
            emergency_button: "emergency_stop",
            clear_button: "clear_emergency_stop",
        }
        for axis in (throttle_axis, steering_axis):
            if not 0 <= axis < self.pg.CONTROLLER_AXIS_MAX:
                raise ValueError(f"SDL controller axis {axis} is unavailable")
        unavailable = [
            button
            for button in self.buttons
            if not 0 <= button < self.pg.CONTROLLER_BUTTON_MAX
        ]
        if unavailable:
            raise ValueError(f"SDL controller button(s) {unavailable} are unavailable")
        self._button_pressed = {
            button: bool(controller.get_button(button)) for button in self.buttons
        }

    def poll(self, events: list[Any]) -> InputState:
        quit_requested = any(
            event.type == self.pg.QUIT
            or (
                event.type == self.pg.KEYDOWN
                and event.key == self.pg.K_ESCAPE
            )
            for event in events
        )
        flags = dict.fromkeys(self.buttons.values(), False)
        for button, flag in self.buttons.items():
            pressed = bool(self.controller.get_button(button))
            flags[flag] = pressed and not self._button_pressed[button]
            self._button_pressed[button] = pressed
        return InputState(
            throttle=-_sdl_axis(self.controller, self.throttle_axis),
            steering=-_sdl_axis(self.controller, self.steering_axis),
            quit=quit_requested,
            **flags,
        )


class RemoteInput:
    """Receive newline-delimited InputState objects from the macOS host."""

    name = "ps4-bridge"

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout: float = 10.0,
        stale_timeout: float = 0.5,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("bridge port must be between 1 and 65535")
        self._socket = socket.create_connection((host, port), timeout=connect_timeout)
        self._socket.setblocking(False)
        self._buffer = b""
        self._latest = InputState()
        self._last_message_at = monotonic()
        self._stale_timeout = stale_timeout

    def poll(self, events: list[Any] | None = None) -> InputState:
        del events
        arm = recording = emergency = clear = quit_requested = False
        while True:
            try:
                chunk = self._socket.recv(65536)
            except BlockingIOError:
                break
            if not chunk:
                raise RuntimeError("PS4 host bridge disconnected")
            self._buffer += chunk

        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if not line:
                continue
            state = input_state_from_json(line)
            self._latest = state
            self._last_message_at = monotonic()
            arm |= state.arm_toggle
            recording |= state.recording_toggle
            emergency |= state.emergency_stop
            clear |= state.clear_emergency_stop
            quit_requested |= state.quit

        if monotonic() - self._last_message_at > self._stale_timeout:
            raise RuntimeError("PS4 host bridge timed out")
        return InputState(
            throttle=self._latest.throttle,
            steering=self._latest.steering,
            arm_toggle=arm,
            recording_toggle=recording,
            emergency_stop=emergency,
            clear_emergency_stop=clear,
            quit=quit_requested,
        )

    def close(self) -> None:
        self._socket.close()


def input_state_from_json(payload: bytes | str) -> InputState:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("bridge message must be a JSON object")
    allowed = set(InputState.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown bridge fields: {', '.join(sorted(unknown))}")
    state = InputState(**data)
    if not math.isfinite(float(state.throttle)) or not math.isfinite(
        float(state.steering)
    ):
        raise ValueError("bridge axes must be finite")
    return InputState(
        throttle=max(-1.0, min(1.0, float(state.throttle))),
        steering=max(-1.0, min(1.0, float(state.steering))),
        arm_toggle=bool(state.arm_toggle),
        recording_toggle=bool(state.recording_toggle),
        emergency_stop=bool(state.emergency_stop),
        clear_emergency_stop=bool(state.clear_emergency_stop),
        quit=bool(state.quit),
    )


def _axis(joystick: Any, index: int) -> float:
    if not 0 <= index < joystick.get_numaxes():
        raise ValueError(
            f"Controller has {joystick.get_numaxes()} axes; axis {index} is unavailable"
        )
    value = float(joystick.get_axis(index))
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0


def _sdl_axis(controller: Any, index: int) -> float:
    value = int(controller.get_axis(index))
    divisor = 32768.0 if value < 0 else 32767.0
    return max(-1.0, min(1.0, value / divisor))


def _deadzone(value: float, deadzone: float) -> float:
    value = max(-1.0, min(1.0, float(value)))
    return 0.0 if abs(value) < deadzone else value


def _move_towards(value: float, target: float, max_delta: float) -> float:
    if value < target:
        return min(value + max_delta, target)
    if value > target:
        return max(value - max_delta, target)
    return value
