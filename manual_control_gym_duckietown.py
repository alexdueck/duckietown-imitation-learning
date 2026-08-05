#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Manual gym-duckietown control with a reward diagnostics sidebar."""

from __future__ import annotations

import argparse
import logging
import sys
import types
from ctypes import POINTER, c_char_p, cast
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyglet

from dt_utils.cli_completion import parse_args_with_completion
from dt_utils.duckietown_paths import EVALUATION_SCREENSHOT_DIR
from dt_utils.gym_duckietown_start_config import (
    TrainingPose,
    append_evaluation_pose,
    append_training_pose,
    apply_env_start_pose,
    load_start_config,
    next_pose_name,
)
from dt_utils.duckietown_rewards import (
    REWARD_FUNCTION_CHOICES,
    compute_reward_breakdowns,
    create_reward_calculators,
    format_wheel_action,
    get_lane_metrics,
    gym_duckietown_done_code,
    patch_duckietown_world_dynamics,
    reset_reward_calculators,
)
from dt_utils.velopose_reward import VD2PP_DISTANCE_SQUARED_WEIGHT


SIDEBAR_WIDTH = 500
MIN_VIEWER_HEIGHT = 860
CONTROL_HELP_HEIGHT = 150
MANUAL_REWARD_FUNCTIONS = ("velopose", "posepot")
BACKGROUND = (18, 22, 26)
SIDEBAR_BG = (27, 31, 36)
TEXT = (238, 241, 245, 255)
MUTED = (166, 174, 184, 255)
ACCENT = (116, 211, 208, 255)
GOOD = (132, 210, 142, 255)
BAD = (238, 118, 118, 255)


@dataclass(frozen=True)
class CurrentPose:
    position: tuple[float, float, float]
    tile: tuple[int, int]
    local_position: tuple[float, float, float]
    angle: float


@dataclass(frozen=True)
class SelectableStartPose:
    pose: TrainingPose
    source: str

    @property
    def source_label(self) -> str:
        return "train" if self.source == "training_poses" else "eval"


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
    current_pose: CurrentPose


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


def parse_reward_functions(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("provide at least one reward function")
    unknown = [name for name in names if name not in REWARD_FUNCTION_CHOICES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown reward function(s): {', '.join(unknown)}"
        )
    return names


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
        "--start-poses",
        type=Path,
        default=None,
        help=(
            "Multi-map start config containing training_poses and evaluation_poses. "
            "N/Shift+N browse poses for --map-name and G selects one by name."
        ),
    )
    parser.add_argument(
        "--start-config",
        type=Path,
        default=None,
        help="JSON config created or extended by P (training pose) and Shift+P (evaluation pose).",
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
        "--posepot-gamma",
        type=float,
        default=0.99,
        help="Discount used by the displayed potential-based pose reward.",
    )
    parser.add_argument(
        "--reward-functions",
        type=parse_reward_functions,
        default=MANUAL_REWARD_FUNCTIONS,
        help=(
            "Comma-separated rewards shown in the sidebar; "
            "use vd2pp to inspect the new reward alone."
        ),
    )
    parser.add_argument(
        "--vd2pp-distance-weight",
        type=float,
        default=VD2PP_DISTANCE_SQUARED_WEIGHT,
        help="Beta in vd2pp's direct -beta * scaled_lane_distance^2 term.",
    )
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


def current_pose(env) -> CurrentPose:
    raw_env = getattr(env, "unwrapped", env)
    position = np.asarray(raw_env.cur_pos, dtype=np.float64)
    tile_x, tile_y = raw_env.get_grid_coords(position)
    tile_size = float(raw_env.road_tile_size)
    return CurrentPose(
        position=tuple(float(value) for value in position),
        tile=(int(tile_x), int(tile_y)),
        local_position=(
            float(position[0] - tile_x * tile_size),
            float(position[1]),
            float(position[2] - tile_y * tile_size),
        ),
        angle=float(raw_env.cur_angle),
    )


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
        current_pose=current_pose(env),
    )


def reset_env(
    env,
    calculators,
    seed: int | None = None,
    start_pose: TrainingPose | None = None,
    manual_trim_override: float | None = None,
) -> ViewerState:
    if start_pose is not None:
        apply_env_start_pose(env, start_pose)
    if seed is not None:
        env.seed(seed)
    env.reset()
    if manual_trim_override is not None:
        apply_manual_trim_override(env, manual_trim_override)
    raw_env = getattr(env, "unwrapped", env)
    if start_pose is not None and not raw_env._valid_pose(raw_env.cur_pos, raw_env.cur_angle):
        pose_label = start_pose.name or "unnamed"
        raise ValueError(f"Start pose {pose_label!r} is not valid on this map")
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


def apply_manual_trim_override(env, trim: float) -> None:
    """Rebuild the current reset state with an explicitly requested trim."""
    import geometry
    from duckietown_world import get_DB18_uncalibrated

    raw_env = getattr(env, "unwrapped", env)
    dynamics = get_DB18_uncalibrated(delay=0.15, trim=trim)
    configuration = raw_env.cartesian_from_weird(raw_env.cur_pos, raw_env.cur_angle)
    velocity = geometry.se2_from_linear_angular(np.zeros(2), 0)
    raw_env.state = dynamics.initialize(c0=(configuration, velocity), t0=0)


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


def load_selectable_start_poses(path: Path, map_name: str) -> tuple[SelectableStartPose, ...]:
    config = load_start_config(
        path,
        None,
        require_training_starts=False,
        require_evaluation_scenarios=False,
    )
    entries = tuple(
        SelectableStartPose(pose=pose, source=source)
        for source, poses in (
            ("training_poses", config.training_poses),
            ("evaluation_poses", config.evaluation_poses),
        )
        for pose in poses
        if pose.map_name == map_name
    )
    if not entries:
        raise ValueError(f"config contains no start poses for map {map_name!r}")

    return entries


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
    pose_name_input: str | None = None,
    trim_input: str | None = None,
    manual_trim_override: float | None = None,
    pose_save_status: str | None = None,
    selected_start_pose: SelectableStartPose | None = None,
    selected_start_index: int | None = None,
    start_pose_count: int = 0,
) -> list[tuple[str, int, tuple[int, int, int, int], bool]]:
    lane = state.lane_metrics
    pose = state.current_pose
    seed_label = str(state.reset_seed) if state.reset_seed is not None else "continued RNG"
    lines: list[tuple[str, int, tuple[int, int, int, int], bool]] = [
        ("gym-duckietown rewards", 18, ACCENT, True),
        (f"map {map_name}", 13, MUTED, False),
        (f"reset seed {seed_label}", 13, MUTED, False),
        (
            "manual trim "
            + (fmt(manual_trim_override, 6) if manual_trim_override is not None else "default"),
            13,
            ACCENT if manual_trim_override is not None else MUTED,
            manual_trim_override is not None,
        ),
        (
            (
                f"start pose {selected_start_index + 1}/{start_pose_count} "
                f"{selected_start_pose.pose.name or '<unnamed>'} "
                f"[{selected_start_pose.source_label}]"
                if selected_start_pose is not None and selected_start_index is not None
                else "start pose simulator default"
            ),
            13,
            ACCENT,
            True,
        ),
        (f"step {state.step_count}  t {state.timestamp:.2f}s", 13, MUTED, False),
        (
            f"pose world x {pose.position[0]:+.5f}  y {pose.position[1]:+.5f}  "
            f"z {pose.position[2]:+.5f}",
            12,
            MUTED,
            False,
        ),
        (
            f"pose tile {pose.tile}  local x {pose.local_position[0]:+.5f}  "
            f"z {pose.local_position[2]:+.5f}",
            12,
            MUTED,
            False,
        ),
        (
            f"pose angle {pose.angle:+.6f} rad  {np.degrees(pose.angle):+.2f} deg",
            12,
            MUTED,
            False,
        ),
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
    if pose_name_input is not None:
        lines.insert(3, (f"go to pose > {pose_name_input}_", 15, ACCENT, True))
    if trim_input is not None:
        lines.insert(3, (f"trim > {trim_input}_", 15, ACCENT, True))
    if pose_save_status is not None:
        lines.insert(3, (pose_save_status, 13, ACCENT, True))

    if state.done:
        lines.append((f"done {state.done_reason}", 15, BAD, True))
        lines.append(("", 8, MUTED, False))

    for name, breakdown in state.reward_breakdowns.items():
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
    pose_name_input: str | None = None,
    trim_input: str | None = None,
    manual_trim_override: float | None = None,
    pose_save_status: str | None = None,
    selected_start_pose: SelectableStartPose | None = None,
    selected_start_index: int | None = None,
    start_pose_count: int = 0,
) -> None:
    draw_rect(x, 0, SIDEBAR_WIDTH, height, SIDEBAR_BG)
    cursor_y = height - 30
    for text, font_size, color, bold in sidebar_lines(
        state,
        map_name,
        seed_input,
        pose_name_input,
        trim_input,
        manual_trim_override,
        pose_save_status,
        selected_start_pose,
        selected_start_index,
        start_pose_count,
    ):
        if text:
            draw_label(text, x + 18, cursor_y, font_size=font_size, color=color, bold=bold)
        cursor_y -= max(16, font_size + 8)


def draw_control_help(camera_bottom: int, trim_input: str | None = None) -> None:
    lines = (
        "W/A/S/D or arrows: drive    Space: stop    Shift: boost",
        "P: save train pose    Shift+P: save eval pose",
        "N: next start pose    Shift+N: previous start pose    G: go to pose name",
        "R: reset with seed    T: set trim    Backspace or /: reset current start pose",
        "Enter: screenshot    Esc: cancel input / exit",
    )
    if trim_input is not None:
        lines = (*lines, "trim input: Enter applies, Esc cancels")
    cursor_y = camera_bottom - 24
    for text in lines:
        draw_label(
            text,
            12,
            cursor_y,
            font_size=11,
            color=MUTED,
        )
        cursor_y -= 24


def save_screenshot(window, path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
    buffer.save(str(path))
    print(f"saved screenshot {path}", flush=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        start_poses = (
            load_selectable_start_poses(args.start_poses, args.map_name)
            if args.start_poses is not None
            else ()
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not load --start-poses: {error}") from error
    selected_start_index: int | None = 0 if start_poses else None

    def selected_start_pose() -> SelectableStartPose | None:
        if selected_start_index is None:
            return None
        return start_poses[selected_start_index]

    env = make_env(args)
    configure_logging(args.log_level)
    _, _, image_width, image_height = import_simulator()
    viewer_height = max(image_height + 2 * CONTROL_HELP_HEIGHT, MIN_VIEWER_HEIGHT)
    image_y = (viewer_height - image_height) // 2
    calculators = create_reward_calculators(
        args.reward_functions,
        posepot_gamma=args.posepot_gamma,
        vd2pp_distance_weight=args.vd2pp_distance_weight,
    )
    initial_start = selected_start_pose()
    state = reset_env(
        env,
        calculators,
        seed=args.seed,
        start_pose=initial_start.pose if initial_start is not None else None,
    )
    action_controller = ManualActionController()
    paused_due_to_done = False
    random_start_rng = np.random.default_rng(args.seed)
    random_start_seeds = [int(args.seed)]
    random_start_index = 0
    seed_input: str | None = None
    pose_name_input: str | None = None
    trim_input: str | None = None
    ignore_pose_prompt_text = False
    ignore_trim_prompt_text = False
    manual_trim_override: float | None = None
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
    if initial_start is not None:
        print(
            f"loaded start pose name={initial_start.pose.name or '-'} "
            f"source={initial_start.source_label} tile={initial_start.pose.tile} "
            f"position={initial_start.pose.position} angle={initial_start.pose.angle:.10f} "
            f"file={args.start_poses.expanduser()}",
            flush=True,
        )
    print(
        "WASD or arrows drive; P/Shift+P save train/eval poses; "
        "N/Shift+N select next/previous start; G selects a pose by name; "
        "R enters a reset seed; T sets trim; space stops; backspace or slash resets; "
        "enter saves a screenshot; escape exits",
        flush=True,
    )

    def reset_to_start_pose(index: int, reason: str) -> None:
        nonlocal state, selected_start_index, paused_due_to_done, pose_save_status
        selected_start_index = index % len(start_poses)
        entry = start_poses[selected_start_index]
        state = reset_env(
            env,
            calculators,
            start_pose=entry.pose,
            manual_trim_override=manual_trim_override,
        )
        action_controller.reset()
        pressed_keys.clear()
        paused_due_to_done = False
        pose_save_status = (
            f"{reason}: {entry.pose.name or '<unnamed>'} [{entry.source_label}]"
        )
        print(
            f"selected start_pose={selected_start_index + 1}/{len(start_poses)} "
            f"name={entry.pose.name or '-'} source={entry.source_label} "
            f"map={entry.pose.map_name}",
            flush=True,
        )

    def reset_to_random_start(direction: int, reason: str) -> None:
        nonlocal state, paused_due_to_done, pose_save_status, random_start_index
        if direction < 0 and random_start_index > 0:
            random_start_index -= 1
        elif direction > 0:
            if random_start_index + 1 == len(random_start_seeds):
                while True:
                    seed = int(random_start_rng.integers(0, np.iinfo(np.int32).max))
                    if seed not in random_start_seeds:
                        random_start_seeds.append(seed)
                        break
            random_start_index += 1

        seed = random_start_seeds[random_start_index]
        state = reset_env(
            env,
            calculators,
            seed=seed,
            manual_trim_override=manual_trim_override,
        )
        action_controller.reset()
        pressed_keys.clear()
        paused_due_to_done = False
        pose_save_status = (
            f"{reason}: random seed {seed} "
            f"({random_start_index + 1}/{len(random_start_seeds)})"
        )
        print(
            f"selected random_start={random_start_index + 1}/{len(random_start_seeds)} "
            f"seed={seed}",
            flush=True,
        )

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal state, paused_due_to_done, seed_input, pose_name_input
        nonlocal trim_input, manual_trim_override
        nonlocal ignore_pose_prompt_text, ignore_trim_prompt_text, pose_save_status
        nonlocal random_start_seeds, random_start_index
        if trim_input is not None:
            if symbol == key.BACKSPACE:
                trim_input = trim_input[:-1]
            elif symbol in (key.ENTER, key.RETURN) and trim_input:
                try:
                    manual_trim_override = float(trim_input)
                except ValueError:
                    pose_save_status = "invalid trim; enter a number"
                    print(f"Invalid trim value: {trim_input!r}", flush=True)
                    return
                trim_input = None
                ignore_trim_prompt_text = False
                apply_manual_trim_override(env, manual_trim_override)
                action_controller.reset()
                pressed_keys.clear()
                state = replace(
                    state,
                    action=np.zeros(2, dtype=np.float32),
                    lane_metrics=get_lane_metrics(env),
                    current_pose=current_pose(env),
                )
                pose_save_status = f"manual trim set to {manual_trim_override:+.6f}"
                print(f"manual trim set to {manual_trim_override:+.6f}", flush=True)
            elif symbol == key.ESCAPE:
                trim_input = None
                ignore_trim_prompt_text = False
                print("trim input cancelled", flush=True)
            return

        if pose_name_input is not None:
            if symbol == key.BACKSPACE:
                pose_name_input = pose_name_input[:-1]
            elif symbol in (key.ENTER, key.RETURN) and pose_name_input:
                matches = [
                    index
                    for index, entry in enumerate(start_poses)
                    if entry.pose.name == pose_name_input
                ]
                requested_name = pose_name_input
                pose_name_input = None
                if matches:
                    reset_to_start_pose(matches[0], "selected")
                else:
                    pose_save_status = f"pose not found: {requested_name}"
                    print(f"No start pose named {requested_name!r}", flush=True)
            elif symbol == key.ESCAPE:
                pose_name_input = None
            return

        if seed_input is not None:
            if symbol == key.BACKSPACE:
                seed_input = seed_input[:-1]
            elif symbol in (key.ENTER, key.RETURN) and seed_input:
                seed = int(seed_input)
                state = reset_env(
                    env,
                    calculators,
                    seed=seed,
                    start_pose=(
                        selected_start_pose().pose
                        if selected_start_pose() is not None else None
                    ),
                    manual_trim_override=manual_trim_override,
                )
                action_controller.reset()
                pressed_keys.clear()
                paused_due_to_done = False
                if not start_poses:
                    random_start_seeds = [seed]
                    random_start_index = 0
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

        if symbol == key.T:
            trim_input = ""
            ignore_trim_prompt_text = True
            action_controller.reset()
            pressed_keys.clear()
            return

        if symbol == key.G:
            if not start_poses:
                pose_save_status = "no start poses loaded"
                print("Cannot select a pose without --start-poses", flush=True)
            else:
                pose_name_input = ""
                ignore_pose_prompt_text = True
                action_controller.reset()
                pressed_keys.clear()
            return

        if symbol == key.N:
            if symbol in pressed_keys:
                return
            pressed_keys.add(symbol)
            if not start_poses or selected_start_index is None:
                direction = -1 if modifiers & key.MOD_SHIFT else 1
                reset_to_random_start(direction, "selected")
            else:
                direction = -1 if modifiers & key.MOD_SHIFT else 1
                reset_to_start_pose(selected_start_index + direction, "selected")
                pressed_keys.add(symbol)
            return

        if symbol == key.P:
            if symbol in pressed_keys:
                return
            pressed_keys.add(symbol)
            collection = (
                "evaluation_poses"
                if modifiers & key.MOD_SHIFT
                else "training_poses"
            )
            pose_label = (
                "evaluation" if collection == "evaluation_poses" else "training"
            )
            if args.start_config is None:
                pose_save_status = "pose not saved: no config"
                print("Cannot save pose without --start-config", flush=True)
                return
            try:
                pose = replace(
                    capture_training_pose(env),
                    name=next_pose_name(args.start_config, args.map_name),
                )
                append_function = (
                    append_evaluation_pose
                    if collection == "evaluation_poses"
                    else append_training_pose
                )
                pose_index = append_function(args.start_config, args.map_name, pose)
            except (OSError, ValueError) as error:
                pose_save_status = "pose save failed; see terminal"
                print(f"Could not save {pose_label} pose: {error}", flush=True)
            else:
                pose_save_status = f"saved {pose_label} pose {pose.name}"
                print(
                    f"saved {pose_label}_pose={pose_index} name={pose.name} "
                    f"map={args.map_name} tile={pose.tile} "
                    f"position={pose.position} angle={pose.angle:.8f} "
                    f"config={args.start_config.expanduser()}",
                    flush=True,
                )
            return

        pressed_keys.add(symbol)
        if symbol in (key.BACKSPACE, key.SLASH):
            active_start = selected_start_pose()
            state = reset_env(
                env,
                calculators,
                seed=(state.reset_seed if active_start is None else None),
                start_pose=active_start.pose if active_start is not None else None,
                manual_trim_override=manual_trim_override,
            )
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
        nonlocal seed_input, pose_name_input, trim_input
        nonlocal ignore_pose_prompt_text, ignore_trim_prompt_text
        if seed_input is not None:
            digits = "".join(character for character in text if character.isdigit())
            seed_input = (seed_input + digits)[:20]
        elif pose_name_input is not None:
            if ignore_pose_prompt_text:
                ignore_pose_prompt_text = False
                if text.lower() == "g":
                    return
            printable = "".join(
                character
                for character in text
                if character.isprintable() and character not in "\r\n"
            )
            pose_name_input = (pose_name_input + printable)[:100]
        elif trim_input is not None:
            if ignore_trim_prompt_text:
                ignore_trim_prompt_text = False
                if text.lower() == "t":
                    return
            valid = "".join(
                character for character in text if character in "0123456789+-.eE"
            )
            trim_input = (trim_input + valid)[:32]

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
        draw_control_help(image_y, trim_input)
        active_start = selected_start_pose()
        draw_sidebar(
            state,
            args.map_name,
            image_width,
            viewer_height,
            seed_input=seed_input,
            pose_name_input=pose_name_input,
            trim_input=trim_input,
            manual_trim_override=manual_trim_override,
            pose_save_status=pose_save_status,
            selected_start_pose=active_start,
            selected_start_index=selected_start_index,
            start_pose_count=len(start_poses),
        )

    def update(dt):
        nonlocal state, paused_due_to_done
        if (
            paused_due_to_done
            or seed_input is not None
            or pose_name_input is not None
            or trim_input is not None
        ):
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
                active_start = selected_start_pose()
                if active_start is None:
                    reset_to_random_start(1, "auto reset")
                else:
                    state = reset_env(
                        env,
                        calculators,
                        start_pose=active_start.pose,
                        manual_trim_override=manual_trim_override,
                    )
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
