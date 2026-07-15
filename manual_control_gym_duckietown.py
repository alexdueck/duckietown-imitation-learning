#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
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

from cli_completion import parse_args_with_completion
from duckietown_paths import EVALUATION_SCREENSHOT_DIR
from gym_duckietown_start_config import TrainingPose, append_training_pose
from duckietown_rewards import (
    DISPLAY_REWARD_FUNCTIONS,
    compute_reward_breakdowns,
    create_reward_calculators,
    format_wheel_action,
    get_lane_metrics,
    gym_duckietown_done_code,
    patch_duckietown_world_dynamics,
    reset_reward_calculators,
)


SIDEBAR_WIDTH = 500
MIN_VIEWER_HEIGHT = 760
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
    env_return: float
    reward_breakdowns: dict[str, dict[str, float | dict[str, float]]]
    reward_returns: dict[str, float]
    lane_metrics: dict[str, Any]
    done: bool
    done_reason: str
    step_count: int
    timestamp: float
    reset_seed: int | None


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
    parser.add_argument(
        "--start-seeds-config",
        type=Path,
        default="configs/gym_duckietown_start_seeds.json",
        help="Existing local start config to which P appends the current training pose.",
    )
    parser.add_argument("--auto-reset", action="store_true", help="Reset immediately after gym-duckietown returns done.")
    parser.add_argument("--forward-target", type=float, default=0.45)
    parser.add_argument("--backward-target", type=float, default=0.30)
    parser.add_argument("--turn-target", type=float, default=0.22)
    parser.add_argument("--throttle-rate", type=float, default=2.0)
    parser.add_argument("--steering-rate", type=float, default=0.75)
    parser.add_argument("--auto-center-rate", type=float, default=0.55)
    parser.add_argument("--boost-multiplier", type=float, default=1.35)
    parser.add_argument(
        "--screenshot-path",
        type=Path,
        default=EVALUATION_SCREENSHOT_DIR / "gym_duckietown_manual.png",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parse_args_with_completion(parser)


def move_towards(value: float, target: float, max_delta: float) -> float:
    if value < target:
        return min(value + max_delta, target)
    if value > target:
        return max(value - max_delta, target)
    return value


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper())
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level)
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)

    for logger_name in (
        "gym-duckietown",
        "duckietown_world",
        "geometry",
        "typing",
        "commons",
        "nodes",
        "aido_schemas",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


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
    return gym_duckietown_done_code(done, info)


def make_viewer_state(
    env,
    calculators,
    action: np.ndarray,
    env_reward: float,
    done: bool,
    info: dict[str, Any] | None = None,
    previous_state: ViewerState | None = None,
    reset_seed: int | None = None,
) -> ViewerState:
    info = {} if info is None else info
    code = done_reason(done, info)
    reward_breakdowns = compute_reward_breakdowns(
        env,
        env_reward,
        calculators,
        done_code=code,
    )
    if previous_state is None:
        env_return = 0.0
        reward_returns = {name: 0.0 for name in reward_breakdowns}
    else:
        env_return = previous_state.env_return + float(env_reward)
        reward_returns = {
            name: previous_state.reward_returns.get(name, 0.0) + float(breakdown["total"])
            for name, breakdown in reward_breakdowns.items()
        }
    return ViewerState(
        action=format_wheel_action(action),
        env_reward=float(env_reward),
        env_return=env_return,
        reward_breakdowns=reward_breakdowns,
        reward_returns=reward_returns,
        lane_metrics=get_lane_metrics(env),
        done=bool(done),
        done_reason=code,
        step_count=int(getattr(env, "step_count", 0)),
        timestamp=float(getattr(env, "timestamp", 0.0)),
        reset_seed=previous_state.reset_seed if previous_state is not None else reset_seed,
    )


def reset_env(env, calculators, seed: int | None = None) -> ViewerState:
    if seed is not None:
        env.seed(seed)
    env.reset()
    reset_reward_calculators(calculators, env)
    action = np.zeros(2, dtype=np.float32)
    return make_viewer_state(
        env,
        calculators,
        action,
        current_env_reward(env),
        False,
        {},
        reset_seed=seed,
    )


def capture_training_pose(env) -> TrainingPose:
    raw_env = getattr(env, "unwrapped", env)
    position = np.asarray(raw_env.cur_pos, dtype=np.float64)
    angle = float(raw_env.cur_angle)
    if not raw_env._valid_pose(position, angle):
        raise ValueError("current pose is not valid")

    tile_x, tile_y = raw_env.get_grid_coords(position)
    tile = raw_env._get_tile(tile_x, tile_y)
    if tile is None or not tile.get("drivable", False):
        raise ValueError("current pose is not on a drivable tile")

    tile_size = float(raw_env.road_tile_size)
    local_position = (
        float(position[0] - tile_x * tile_size),
        0.0,
        float(position[2] - tile_y * tile_size),
    )
    return TrainingPose(
        tile=(tile_x, tile_y),
        position=local_position,
        angle=angle,
    )


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


def append_component_lines(
    lines: list[tuple[str, int, tuple[int, int, int, int], bool]],
    components: dict[str, Any],
    depth: int = 1,
) -> None:
    prefix = "  " * depth
    for component_name, component_value in components.items():
        if isinstance(component_value, dict):
            total = component_value.get("total")
            lines.append(
                (f"{prefix}{component_name} {fmt(total, 4)}", 12, MUTED, True)
            )
            nested = component_value.get("components", {})
            if isinstance(nested, dict):
                append_component_lines(lines, nested, depth + 1)
        else:
            lines.append(
                (f"{prefix}{component_name} {fmt(component_value, 4)}", 11, MUTED, False)
            )


def sidebar_lines(
    state: ViewerState,
    map_name: str,
    seed_input: str | None = None,
    pose_save_status: str | None = None,
) -> list[tuple[str, int, tuple[int, int, int, int], bool]]:
    lane = state.lane_metrics
    seed_label = str(state.reset_seed) if state.reset_seed is not None else "continued RNG"
    lines: list[tuple[str, int, tuple[int, int, int, int], bool]] = [
        ("gym-duckietown rewards", 18, ACCENT, True),
        (f"map {map_name}", 13, MUTED, False),
        (f"reset seed {seed_label}", 13, MUTED, False),
        (f"step {state.step_count}  t {state.timestamp:.2f}s", 13, MUTED, False),
        ("", 8, MUTED, False),
        (f"left {fmt(state.action[0], 3)}   right {fmt(state.action[1], 3)}", 16, TEXT, True),
        (
            f"default reward {fmt(state.env_reward, 4)}  return {fmt(state.env_return, 4)}",
            15,
            TEXT,
            True,
        ),
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

    if seed_input is not None:
        lines.insert(3, (f"new seed > {seed_input}_", 15, ACCENT, True))
    if pose_save_status is not None:
        lines.insert(3, (pose_save_status, 13, ACCENT, True))

    if state.done:
        lines.append((f"done {state.done_reason}", 15, BAD, True))
        lines.append(("", 8, MUTED, False))

    for name in DISPLAY_REWARD_FUNCTIONS:
        breakdown = state.reward_breakdowns.get(name, {"total": 0.0, "components": {}})
        total = float(breakdown["total"])
        color = GOOD if total >= 0.0 else BAD
        reward_return = state.reward_returns.get(name, 0.0)
        lines.append(
            (f"{name} reward {fmt(total, 4)}  return {fmt(reward_return, 4)}", 14, color, True)
        )
        components = breakdown.get("components", {})
        if isinstance(components, dict) and name != "default":
            append_component_lines(lines, components)
    return lines


def draw_sidebar(
    state: ViewerState,
    map_name: str,
    x: int,
    height: int,
    seed_input: str | None = None,
    pose_save_status: str | None = None,
) -> None:
    draw_rect(x, 0, SIDEBAR_WIDTH, height, SIDEBAR_BG)
    cursor_y = height - 30
    for text, font_size, color, bold in sidebar_lines(
        state,
        map_name,
        seed_input,
        pose_save_status,
    ):
        if text:
            draw_label(text, x + 18, cursor_y, font_size=font_size, color=color, bold=bold)
        cursor_y -= max(16, font_size + 8)


def save_screenshot(window, path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
    buffer.save(str(path))
    print(f"saved screenshot {path}", flush=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    env = make_env(args)
    configure_logging(args.log_level)
    _, _, image_width, image_height = import_simulator()
    viewer_height = max(image_height, MIN_VIEWER_HEIGHT)
    image_y = (viewer_height - image_height) // 2
    calculators = create_reward_calculators(DISPLAY_REWARD_FUNCTIONS)
    state = reset_env(env, calculators, seed=args.seed)
    action_controller = ManualActionController()
    paused_due_to_done = False
    seed_input: str | None = None
    pose_save_status: str | None = None

    from pyglet import window as pyglet_window
    from pyglet.window import key

    window = pyglet_window.Window(
        width=image_width + SIDEBAR_WIDTH,
        height=viewer_height,
        resizable=False,
        caption="gym-duckietown reward control",
    )
    pressed_keys: set[int] = set()

    print("manual gym-duckietown reward viewer started", flush=True)
    print(
        "WASD drives, arrow keys also work, R enters a reset seed, P saves a training pose, space stops, "
        "backspace or slash resets, enter saves screenshot, escape exits",
        flush=True,
    )

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal state, paused_due_to_done, seed_input, pose_save_status
        if seed_input is not None:
            if symbol == key.BACKSPACE:
                seed_input = seed_input[:-1]
            elif symbol in (key.ENTER, key.RETURN) and seed_input:
                seed = int(seed_input)
                state = reset_env(env, calculators, seed=seed)
                action_controller.reset()
                pressed_keys.clear()
                paused_due_to_done = False
                seed_input = None
                print(f"reset seed={seed}", flush=True)
            elif symbol == key.ESCAPE:
                seed_input = None
            return

        if symbol == key.R:
            seed_input = ""
            action_controller.reset()
            pressed_keys.clear()
            return

        if symbol == key.P:
            if symbol in pressed_keys:
                return
            pressed_keys.add(symbol)
            if args.start_seeds_config is None:
                pose_save_status = "pose not saved: no config"
                print("Cannot save pose without --start-seeds-config", flush=True)
                return
            try:
                pose = capture_training_pose(env)
                pose_index = append_training_pose(args.start_seeds_config, args.map_name, pose)
            except (OSError, ValueError) as error:
                pose_save_status = "pose save failed; see terminal"
                print(f"Could not save training pose: {error}", flush=True)
            else:
                pose_save_status = f"saved training pose #{pose_index}"
                print(
                    f"saved training_pose={pose_index} tile={pose.tile} "
                    f"position={pose.position} angle={pose.angle:.8f} "
                    f"config={args.start_seeds_config.expanduser()}",
                    flush=True,
                )
            return

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
    def on_text(text):
        nonlocal seed_input
        if seed_input is not None:
            digits = "".join(character for character in text if character.isdigit())
            seed_input = (seed_input + digits)[:20]

    @window.event
    def on_key_release(symbol, modifiers):
        pressed_keys.discard(symbol)

    @window.event
    def on_draw():
        rgb = env.render(mode="rgb_array")
        prepare_window_2d(window, image_width + SIDEBAR_WIDTH, viewer_height)
        window.clear()
        draw_rect(0, 0, image_width + SIDEBAR_WIDTH, viewer_height, BACKGROUND)
        draw_rgb(rgb, 0, image_y, image_width, image_height)
        draw_sidebar(
            state,
            args.map_name,
            image_width,
            viewer_height,
            seed_input,
            pose_save_status,
        )

    def update(dt):
        nonlocal state, paused_due_to_done
        if paused_due_to_done or seed_input is not None:
            return

        action = action_controller.update(pressed_keys, key, args, dt)
        observation, env_reward, done, info = env.step(action)
        del observation
        state = make_viewer_state(
            env,
            calculators,
            action,
            float(env_reward),
            bool(done),
            info,
            previous_state=state,
        )
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
