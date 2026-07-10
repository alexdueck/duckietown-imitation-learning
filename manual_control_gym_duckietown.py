#!/usr/bin/env python3
"""Manual gym-duckietown control with a reward diagnostics sidebar."""

from __future__ import annotations

import argparse
import logging
import sys
import types
from ctypes import POINTER, c_char_p, cast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyglet

from duckietown_rewards import (
    DISPLAY_REWARD_FUNCTIONS,
    compute_reward_breakdowns,
    create_reward_calculators,
    format_wheel_action,
    get_lane_metrics,
    patch_duckietown_world_dynamics,
    reset_reward_calculators,
)


SIDEBAR_WIDTH = 500
BACKGROUND = (18, 22, 26)
SIDEBAR_BG = (27, 31, 36)
TEXT = (238, 241, 245, 255)
MUTED = (166, 174, 184, 255)
ACCENT = (116, 211, 208, 255)
GOOD = (132, 210, 142, 255)
BAD = (238, 118, 118, 255)


@dataclass
class ViewerState:
    action: np.ndarray
    env_reward: float
    reward_breakdowns: dict[str, dict[str, float | dict[str, float]]]
    lane_metrics: dict[str, Any]
    done: bool
    done_reason: str
    step_count: int
    timestamp: float


@dataclass
class ManualActionController:
    throttle: float = 0.0
    steering: float = 0.0

    def reset(self) -> None:
        self.throttle = 0.0
        self.steering = 0.0

    def update(self, pressed_keys: set[int], key_module, args: argparse.Namespace, dt: float) -> np.ndarray:
        dt = max(0.0, min(float(dt), 0.1))
        if key_module.SPACE in pressed_keys:
            self.reset()
            return np.zeros(2, dtype=np.float32)

        forward = key_module.W in pressed_keys or key_module.UP in pressed_keys
        backward = key_module.S in pressed_keys or key_module.DOWN in pressed_keys
        steer_left = key_module.A in pressed_keys or key_module.LEFT in pressed_keys
        steer_right = key_module.D in pressed_keys or key_module.RIGHT in pressed_keys

        target_throttle = 0.0
        if forward and not backward:
            target_throttle = float(args.forward_target)
        elif backward and not forward:
            target_throttle = -float(args.backward_target)

        target_steering = 0.0
        steering_rate = float(args.auto_center_rate)
        if steer_left and not steer_right:
            target_steering = float(args.turn_target)
            steering_rate = float(args.steering_rate)
        elif steer_right and not steer_left:
            target_steering = -float(args.turn_target)
            steering_rate = float(args.steering_rate)

        self.throttle = move_towards(self.throttle, target_throttle, float(args.throttle_rate) * dt)
        self.steering = move_towards(self.steering, target_steering, steering_rate * dt)

        left = self.throttle - self.steering
        right = self.throttle + self.steering
        action = np.array([left, right], dtype=np.float32)
        if key_module.LSHIFT in pressed_keys or key_module.RSHIFT in pressed_keys:
            action *= float(args.boost_multiplier)
        return format_wheel_action(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual gym-duckietown control with reward diagnostics.")
    parser.add_argument("--map-name", default="loop_empty")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100_000_000)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--robot-speed", type=float, default=None)
    parser.add_argument("--accept-start-angle-deg", type=float, default=4.0)
    parser.add_argument("--draw-curve", action="store_true")
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument("--distortion", action="store_true")
    parser.add_argument("--dynamics-rand", action="store_true")
    parser.add_argument("--camera-rand", action="store_true")
    parser.add_argument("--auto-reset", action="store_true", help="Reset immediately after gym-duckietown returns done.")
    parser.add_argument("--forward-target", type=float, default=0.45)
    parser.add_argument("--backward-target", type=float, default=0.30)
    parser.add_argument("--turn-target", type=float, default=0.22)
    parser.add_argument("--throttle-rate", type=float, default=2.0)
    parser.add_argument("--steering-rate", type=float, default=0.75)
    parser.add_argument("--auto-center-rate", type=float, default=0.55)
    parser.add_argument("--boost-multiplier", type=float, default=1.35)
    parser.add_argument("--screenshot-path", type=Path, default=Path("gym_duckietown_manual.png"))
    parser.add_argument("--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def move_towards(value: float, target: float, max_delta: float) -> float:
    if value < target:
        return min(value + max_delta, target)
    if value > target:
        return max(value - max_delta, target)
    return value


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper())
    for logger_name in (
        "gym-duckietown",
        "duckietown_world",
        "geometry",
        "typing",
        "commons",
        "nodes",
        "aido_schemas",
    ):
        logging.getLogger(logger_name).setLevel(level)


def install_windowed_check_hw_stub() -> None:
    """Avoid gym-duckietown's Linux headless check_hw import for visible viewers."""
    module = types.ModuleType("gym_duckietown.check_hw")

    def get_graphics_information() -> dict[str, str]:
        from pyglet import gl

        options = {
            "vendor": gl.GL_VENDOR,
            "renderer": gl.GL_RENDERER,
            "version": gl.GL_VERSION,
            "shading-language-version": gl.GL_SHADING_LANGUAGE_VERSION,
        }
        results = {}
        for name, code in options.items():
            value = gl.glGetString(code)
            if value:
                results[name] = cast(value, c_char_p).value.decode()
            else:
                results[name] = ""
        return results

    module.get_graphics_information = get_graphics_information
    sys.modules["gym_duckietown.check_hw"] = module


def import_simulator():
    pyglet.options["headless"] = False
    install_windowed_check_hw_stub()
    import gym_duckietown  # noqa: F401

    pyglet.options["headless"] = False
    from gym_duckietown.simulator import DEFAULT_ROBOT_SPEED, WINDOW_HEIGHT, WINDOW_WIDTH, Simulator

    patch_duckietown_world_dynamics()
    return Simulator, DEFAULT_ROBOT_SPEED, WINDOW_WIDTH, WINDOW_HEIGHT


def make_env(args: argparse.Namespace):
    Simulator, default_robot_speed, _, _ = import_simulator()
    robot_speed = default_robot_speed if args.robot_speed is None else args.robot_speed
    return Simulator(
        seed=args.seed,
        map_name=args.map_name,
        max_steps=args.max_steps,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_rate=args.frame_rate,
        frame_skip=args.frame_skip,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        robot_speed=robot_speed,
        accept_start_angle_deg=args.accept_start_angle_deg,
        full_transparency=False,
        distortion=args.distortion,
        dynamics_rand=args.dynamics_rand,
        camera_rand=args.camera_rand,
    )


def current_env_reward(env) -> float:
    if hasattr(env, "_compute_done_reward"):
        try:
            return float(env._compute_done_reward().reward)
        except Exception:
            pass
    return float(env.compute_reward(env.cur_pos, env.cur_angle, env.robot_speed))


def done_reason(done: bool, info: dict[str, Any]) -> str:
    simulator_info = info.get("Simulator", {}) if isinstance(info, dict) else {}
    code = simulator_info.get("done_code")
    message = simulator_info.get("msg")
    if code:
        return str(code)
    if message:
        return str(message)
    return "done" if done else "in-progress"


def make_viewer_state(
    env,
    calculators,
    action: np.ndarray,
    env_reward: float,
    done: bool,
    info: dict[str, Any] | None = None,
) -> ViewerState:
    info = {} if info is None else info
    return ViewerState(
        action=format_wheel_action(action),
        env_reward=float(env_reward),
        reward_breakdowns=compute_reward_breakdowns(env, env_reward, calculators),
        lane_metrics=get_lane_metrics(env),
        done=bool(done),
        done_reason=done_reason(done, info),
        step_count=int(getattr(env, "step_count", 0)),
        timestamp=float(getattr(env, "timestamp", 0.0)),
    )


def reset_env(env, calculators) -> ViewerState:
    env.reset()
    reset_reward_calculators(calculators)
    action = np.zeros(2, dtype=np.float32)
    return make_viewer_state(env, calculators, action, current_env_reward(env), False, {})


def draw_rect(x: float, y: float, width: float, height: float, color: tuple[int, int, int]) -> None:
    from pyglet import gl

    pyglet.graphics.draw(
        4,
        gl.GL_QUADS,
        ("v2f", (x, y, x + width, y, x + width, y + height, x, y + height)),
        ("c3B", color * 4),
    )


def prepare_window_2d(window, width: int, height: int) -> None:
    from pyglet import gl

    window.switch_to()
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    gl.glViewport(0, 0, width, height)
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDisable(gl.GL_LIGHTING)
    gl.glDisable(gl.GL_CULL_FACE)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0, width, 0, height, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glColor4ub(255, 255, 255, 255)


def draw_label(
    text: str,
    x: float,
    y: float,
    font_size: int = 14,
    color: tuple[int, int, int, int] = TEXT,
    bold: bool = False,
) -> None:
    pyglet.text.Label(
        text,
        font_name="Arial",
        font_size=font_size,
        bold=bold,
        x=x,
        y=y,
        color=color,
    ).draw()


def draw_rgb(rgb: np.ndarray, x: int, y: int, target_width: int, target_height: int) -> None:
    from pyglet import gl, image

    height, width = rgb.shape[:2]
    frame = np.ascontiguousarray(np.flip(rgb[:, :, :3], axis=0))
    gl.glColor3ub(255, 255, 255)
    image_data = image.ImageData(
        width,
        height,
        "RGB",
        frame.ctypes.data_as(POINTER(gl.GLubyte)),
        pitch=width * 3,
    )
    image_data.blit(x, y, width=target_width, height=target_height)


def fmt(value: Any, precision: int = 4) -> str:
    try:
        return f"{float(value):+.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def sidebar_lines(state: ViewerState, map_name: str) -> list[tuple[str, int, tuple[int, int, int, int], bool]]:
    lane = state.lane_metrics
    lines: list[tuple[str, int, tuple[int, int, int, int], bool]] = [
        ("gym-duckietown rewards", 18, ACCENT, True),
        (f"map {map_name}", 13, MUTED, False),
        (f"step {state.step_count}  t {state.timestamp:.2f}s", 13, MUTED, False),
        ("", 8, MUTED, False),
        (f"left {fmt(state.action[0], 3)}   right {fmt(state.action[1], 3)}", 16, TEXT, True),
        (f"default reward {fmt(state.env_reward, 4)}", 16, TEXT, True),
        (
            f"speed {float(lane['speed']):.4f}   lane_valid {int(bool(lane['lane_valid']))}",
            13,
            MUTED,
            False,
        ),
        (
            f"dot_dir {fmt(lane['dot_dir'], 4)}   dist {fmt(lane['dist'], 4)}",
            15,
            TEXT,
            True,
        ),
        (f"angle {fmt(lane['angle_deg'], 2)} deg", 13, MUTED, False),
        ("", 8, MUTED, False),
    ]

    if state.done:
        lines.append((f"done {state.done_reason}", 15, BAD, True))
        lines.append(("", 8, MUTED, False))

    for name in DISPLAY_REWARD_FUNCTIONS:
        breakdown = state.reward_breakdowns.get(name, {"total": 0.0, "components": {}})
        total = float(breakdown["total"])
        color = GOOD if total >= 0.0 else BAD
        lines.append((f"{name} {fmt(total, 4)}", 15, color, True))
        components = breakdown.get("components", {})
        if isinstance(components, dict) and name != "default":
            for component_name, component_value in components.items():
                lines.append((f"  {component_name} {fmt(component_value, 4)}", 12, MUTED, False))
    return lines


def draw_sidebar(state: ViewerState, map_name: str, x: int, height: int) -> None:
    draw_rect(x, 0, SIDEBAR_WIDTH, height, SIDEBAR_BG)
    cursor_y = height - 30
    for text, font_size, color, bold in sidebar_lines(state, map_name):
        if text:
            draw_label(text, x + 18, cursor_y, font_size=font_size, color=color, bold=bold)
        cursor_y -= max(16, font_size + 8)


def save_screenshot(window, path: Path) -> None:
    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
    buffer.save(str(path.expanduser()))
    print(f"saved screenshot {path.expanduser()}", flush=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    env = make_env(args)
    _, _, image_width, image_height = import_simulator()
    calculators = create_reward_calculators(DISPLAY_REWARD_FUNCTIONS)
    state = reset_env(env, calculators)
    action_controller = ManualActionController()
    paused_due_to_done = False

    from pyglet import window as pyglet_window
    from pyglet.window import key

    window = pyglet_window.Window(
        width=image_width + SIDEBAR_WIDTH,
        height=image_height,
        resizable=False,
        caption="gym-duckietown reward control",
    )
    pressed_keys: set[int] = set()

    print("manual gym-duckietown reward viewer started", flush=True)
    print("WASD drives, arrow keys also work, space stops, backspace or slash resets, enter saves screenshot, escape exits", flush=True)

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal state, paused_due_to_done
        pressed_keys.add(symbol)
        if symbol in (key.BACKSPACE, key.SLASH):
            state = reset_env(env, calculators)
            action_controller.reset()
            paused_due_to_done = False
            print("reset", flush=True)
        elif symbol == key.RETURN:
            save_screenshot(window, args.screenshot_path)
        elif symbol == key.ESCAPE:
            env.close()
            window.close()
            pyglet.app.exit()

    @window.event
    def on_key_release(symbol, modifiers):
        pressed_keys.discard(symbol)

    @window.event
    def on_draw():
        rgb = env.render(mode="rgb_array")
        prepare_window_2d(window, image_width + SIDEBAR_WIDTH, image_height)
        window.clear()
        draw_rect(0, 0, image_width + SIDEBAR_WIDTH, image_height, BACKGROUND)
        draw_rgb(rgb, 0, 0, image_width, image_height)
        draw_sidebar(state, args.map_name, image_width, image_height)

    def update(dt):
        nonlocal state, paused_due_to_done
        if paused_due_to_done:
            return

        action = action_controller.update(pressed_keys, key, args, dt)
        observation, env_reward, done, info = env.step(action)
        del observation
        state = make_viewer_state(env, calculators, action, float(env_reward), bool(done), info)
        if done:
            print(f"done step={state.step_count} reason={state.done_reason}", flush=True)
            if args.auto_reset:
                state = reset_env(env, calculators)
                action_controller.reset()
            else:
                action_controller.reset()
                paused_due_to_done = True

    pyglet.clock.schedule_interval(update, 1.0 / float(args.frame_rate))
    try:
        pyglet.app.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
