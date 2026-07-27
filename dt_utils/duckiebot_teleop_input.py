"""Input-device-independent controls for physical Duckiebot teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
import math
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


class SDLControllerInput:
    """Standardized SDL GameController adapter used by the macOS runtime."""

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
