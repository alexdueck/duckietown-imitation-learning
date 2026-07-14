#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Visually evaluate a trained PPO policy in gym-duckietown on macOS."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyglet
import torch

from cli_completion import parse_args_with_completion
from duckietown_paths import (
    EVALUATION_SCREENSHOT_DIR,
    RL_GYM_DUCKIETOWN_EVALUATION_DIR,
)
from duckietown_rewards import (
    GymDuckietownRewardCalculator,
    REWARD_FUNCTION_CHOICES,
    format_wheel_action,
)
from live_eval_imitation_policy_gym_duckietown import ReturnRecorder, reset_raw, step_raw
from manual_control_gym_duckietown import (
    ACCENT,
    BACKGROUND,
    BAD,
    GOOD,
    MUTED,
    SIDEBAR_BG,
    TEXT,
    configure_logging,
    done_reason,
    draw_label,
    draw_rect,
    draw_rgb,
    import_simulator,
    make_env,
    prepare_window_2d,
    save_screenshot,
)
from rl_models import TanhGaussianPolicy
from train_imitation_learning import resolve_device
from train_rl_ppo_gym_duckietown import make_transform, preprocess


SIDEBAR_WIDTH = 480


@dataclass
class ViewerState:
    mean_action: np.ndarray
    action: np.ndarray
    std: np.ndarray
    env_reward: float
    selected_reward: float
    env_return: float
    selected_return: float
    episode: int
    episode_length: int
    completed_episodes: int
    mean_completed_return: float | None
    done: bool
    done_reason: str
    paused: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained PPO policy visually in gym-duckietown on macOS."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--map-name",
        default=None,
        help="Defaults to the map stored in the checkpoint.",
    )
    parser.add_argument(
        "--reward-function",
        choices=REWARD_FUNCTION_CHOICES,
        default=None,
        help="Defaults to the reward function stored in the checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Simulator episode limit; defaults to the checkpoint's max_episode_steps.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of completed episodes before exiting; 0 runs until Escape.",
    )
    parser.add_argument("--frame-rate", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--robot-speed", type=float, default=None)
    parser.add_argument("--accept-start-angle-deg", type=float, default=None)
    parser.add_argument("--domain-rand", dest="domain_rand", action="store_true")
    parser.add_argument("--no-domain-rand", dest="domain_rand", action="store_false")
    parser.add_argument("--distortion", dest="distortion", action="store_true")
    parser.add_argument("--no-distortion", dest="distortion", action="store_false")
    parser.set_defaults(domain_rand=None, distortion=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--crop-y-start", type=int, default=None)
    parser.add_argument(
        "--source-observation-channel-order",
        choices=("rgb", "bgr"),
        default=None,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the learned Gaussian instead of using tanh(mean).",
    )
    parser.add_argument(
        "--stop-on-done",
        action="store_true",
        help="Pause on an episode end instead of resetting automatically.",
    )
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument(
        "--returns-file",
        type=Path,
        default=None,
        help=(
            "CSV destination; defaults to "
            f"{RL_GYM_DUCKIETOWN_EVALUATION_DIR}/<timestamp>_returns.csv."
        ),
    )
    parser.add_argument(
        "--screenshot-path",
        type=Path,
        default=EVALUATION_SCREENSHOT_DIR / "gym_duckietown_rl_eval.png",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parse_args_with_completion(parser)


def config_value(value, config: dict[str, Any], key: str, default):
    return config.get(key, default) if value is None else value


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TanhGaussianPolicy, dict[str, Any], int]:
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "policy_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not an RL checkpoint: policy_state_dict is missing"
        )
    env_backend = checkpoint.get("env_backend")
    if env_backend not in (None, "gym-duckietown"):
        raise ValueError(
            f"Checkpoint backend is {env_backend!r}, expected 'gym-duckietown'"
        )

    config = checkpoint.get("config", {})
    model_name = config.get("model", "mobilenet_v3_small")
    policy = TanhGaussianPolicy(model_name, pretrained=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.to(device)
    policy.eval()
    return policy, config, int(checkpoint.get("step", 0))


def apply_checkpoint_defaults(args: argparse.Namespace, config: dict[str, Any]) -> None:
    args.map_name = config_value(args.map_name, config, "map_name", "loop_empty")
    args.reward_function = config_value(
        args.reward_function,
        config,
        "reward_function",
        "posangle",
    )
    args.seed = int(config_value(args.seed, config, "seed", 42))
    args.frame_rate = int(config_value(args.frame_rate, config, "frame_rate", 30))
    args.frame_skip = int(config_value(args.frame_skip, config, "frame_skip", 1))
    args.camera_width = int(
        config_value(args.camera_width, config, "camera_width", 640)
    )
    args.camera_height = int(
        config_value(args.camera_height, config, "camera_height", 480)
    )
    args.robot_speed = config_value(args.robot_speed, config, "robot_speed", None)
    args.accept_start_angle_deg = float(
        config_value(
            args.accept_start_angle_deg,
            config,
            "accept_start_angle_deg",
            4.0,
        )
    )
    args.domain_rand = bool(
        config_value(args.domain_rand, config, "domain_rand", False)
    )
    args.distortion = bool(
        config_value(args.distortion, config, "distortion", False)
    )
    args.image_size = int(config_value(args.image_size, config, "image_size", 224))
    args.crop_y_start = int(
        config_value(args.crop_y_start, config, "crop_y_start", 0)
    )
    args.source_observation_channel_order = config_value(
        args.source_observation_channel_order,
        config,
        "source_observation_channel_order",
        "rgb",
    )

    if args.max_steps is None:
        max_episode_steps = int(config.get("max_episode_steps", 0) or 0)
        simulator_max_steps = config.get("simulator_max_steps")
        if max_episode_steps > 0:
            args.max_steps = max_episode_steps
        elif simulator_max_steps is not None and int(simulator_max_steps) > 0:
            args.max_steps = int(simulator_max_steps)
        else:
            args.max_steps = 100_000_000

    # make_env() shares these optional viewer settings with manual control.
    args.dynamics_rand = False
    args.camera_rand = False
    args.draw_curve = False
    args.draw_bbox = False


@torch.no_grad()
def predict_action(
    policy: TanhGaussianPolicy,
    observation: np.ndarray,
    transform,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_tensor = preprocess(
        observation,
        args.crop_y_start,
        args.image_size,
        args.source_observation_channel_order,
        transform,
    ).unsqueeze(0).to(device)
    mean, log_std = policy(obs_tensor)
    distribution = torch.distributions.Normal(mean, log_std.exp())
    raw_action = distribution.sample() if args.stochastic else distribution.mean
    action = torch.tanh(raw_action)
    mean_action = torch.tanh(distribution.mean)
    return (
        mean_action.squeeze(0).cpu().numpy().astype(np.float32),
        format_wheel_action(action.squeeze(0).cpu().numpy()),
        log_std.exp().squeeze(0).cpu().numpy().astype(np.float32),
    )


def fmt(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.{precision}f}"


def draw_sidebar(
    state: ViewerState,
    args: argparse.Namespace,
    checkpoint_step: int,
    x: int,
    height: int,
) -> None:
    draw_rect(x, 0, SIDEBAR_WIDTH, height, SIDEBAR_BG)
    status = "paused" if state.paused else "running"
    mode = "stochastic" if args.stochastic else "deterministic"
    status_color = BAD if state.paused else GOOD
    selected_return_color = GOOD if state.selected_return >= 0.0 else BAD
    env_return_color = GOOD if state.env_return >= 0.0 else BAD
    lines = [
        ("RL policy in gym-duckietown", 18, ACCENT, True),
        (f"map {args.map_name}   {status}", 13, status_color, True),
        (f"checkpoint step {checkpoint_step}   {mode}", 12, MUTED, False),
        (f"episode {state.episode}   step {state.episode_length}", 13, MUTED, False),
        ("", 8, MUTED, False),
        (f"left {fmt(state.action[0], 3)}   right {fmt(state.action[1], 3)}", 16, TEXT, True),
        (f"mean {fmt(state.mean_action[0], 3)}   {fmt(state.mean_action[1], 3)}", 12, MUTED, False),
        (f"std {state.std[0]:.4f}   {state.std[1]:.4f}", 12, MUTED, False),
        ("", 8, MUTED, False),
        (f"{args.reward_function} reward {fmt(state.selected_reward)}", 15, TEXT, True),
        (
            f"{args.reward_function} return {fmt(state.selected_return)}",
            17,
            selected_return_color,
            True,
        ),
        (f"default reward {fmt(state.env_reward)}", 13, MUTED, False),
        (f"default return {fmt(state.env_return)}", 15, env_return_color, True),
        ("", 8, MUTED, False),
        (f"completed episodes {state.completed_episodes}", 13, MUTED, False),
        (f"mean completed return {fmt(state.mean_completed_return)}", 14, TEXT, True),
    ]
    if state.done:
        lines.extend(
            [
                ("", 8, MUTED, False),
                (f"done {state.done_reason}", 14, BAD, True),
            ]
        )

    cursor_y = height - 30
    for text, font_size, color, bold in lines:
        if text:
            draw_label(
                text,
                x + 18,
                cursor_y,
                font_size=font_size,
                color=color,
                bold=bold,
            )
        cursor_y -= max(16, font_size + 8)


def default_returns_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RL_GYM_DUCKIETOWN_EVALUATION_DIR / f"{timestamp}_returns.csv"


def empty_state(
    episode: int,
    completed_returns: list[float],
    paused: bool,
    std: np.ndarray,
) -> ViewerState:
    return ViewerState(
        mean_action=np.zeros(2, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        std=std.copy(),
        env_reward=0.0,
        selected_reward=0.0,
        env_return=0.0,
        selected_return=0.0,
        episode=episode,
        episode_length=0,
        completed_episodes=len(completed_returns),
        mean_completed_return=(
            float(np.mean(completed_returns)) if completed_returns else None
        ),
        done=False,
        done_reason="in-progress",
        paused=paused,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    device = resolve_device(args.device)
    policy, checkpoint_config, checkpoint_step = load_policy(args.checkpoint, device)
    apply_checkpoint_defaults(args, checkpoint_config)

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.episodes < 0:
        raise ValueError("--episodes must be non-negative")
    if args.frame_rate <= 0:
        raise ValueError("--frame-rate must be positive")

    transform = make_transform()
    returns_path = args.returns_file or default_returns_path()
    recorder = ReturnRecorder(returns_path)
    env = make_env(args)
    _, _, image_width, image_height = import_simulator()
    reward_calculator = GymDuckietownRewardCalculator(args.reward_function)
    observation = reset_raw(env)
    reward_calculator.reset(env)

    with torch.no_grad():
        learned_std = policy.log_std.clamp(-5.0, 2.0).exp().cpu().numpy()

    episode = 0
    episode_length = 0
    selected_return = 0.0
    env_return = 0.0
    completed_returns: list[float] = []
    current_episode_recorded = False
    paused = bool(args.start_paused)
    state = empty_state(episode, completed_returns, paused, learned_std)

    from pyglet import window as pyglet_window
    from pyglet.window import key

    window = pyglet_window.Window(
        width=image_width + SIDEBAR_WIDTH,
        height=image_height,
        resizable=False,
        caption="gym-duckietown RL policy evaluation",
    )

    print(f"Checkpoint:       {args.checkpoint.expanduser()}", flush=True)
    print(f"Checkpoint step:  {checkpoint_step}", flush=True)
    print(f"Checkpoint model: {checkpoint_config.get('model', 'mobilenet_v3_small')}", flush=True)
    print(f"Device:           {device}", flush=True)
    print(f"Map:              {args.map_name}", flush=True)
    print(f"Reward function:  {args.reward_function}", flush=True)
    print(f"Policy mode:      {'stochastic' if args.stochastic else 'deterministic'}", flush=True)
    print(
        "Preprocess:       "
        f"channel_order={args.source_observation_channel_order}, "
        f"crop_y_start={args.crop_y_start}, image_size={args.image_size}",
        flush=True,
    )
    print(f"Returns CSV:      {recorder.path}", flush=True)
    print("space pauses, backspace resets, enter saves screenshot, escape exits", flush=True)

    def record_current_episode(status: str, reason: str) -> None:
        nonlocal current_episode_recorded
        if episode_length == 0 or current_episode_recorded:
            return
        recorder.record(
            episode,
            status,
            episode_length,
            args.reward_function,
            selected_return,
            env_return,
            reason,
        )
        current_episode_recorded = True
        print(
            f"episode={episode} status={status} length={episode_length} "
            f"{args.reward_function}_return={selected_return:+.4f} "
            f"default_return={env_return:+.4f} reason={reason}",
            flush=True,
        )

    def start_next_episode() -> None:
        nonlocal observation, episode, episode_length, selected_return, env_return
        nonlocal current_episode_recorded, state
        observation = reset_raw(env)
        reward_calculator.reset(env)
        episode += 1
        episode_length = 0
        selected_return = 0.0
        env_return = 0.0
        current_episode_recorded = False
        state = empty_state(episode, completed_returns, paused, learned_std)

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal paused, state
        if symbol == key.SPACE:
            if state.done:
                start_next_episode()
                paused = False
            else:
                paused = not paused
            state.paused = paused
            print("paused" if paused else "resumed", flush=True)
        elif symbol == key.BACKSPACE:
            record_current_episode("manual_reset", "manual-reset")
            start_next_episode()
            print("reset", flush=True)
        elif symbol == key.RETURN:
            save_screenshot(window, args.screenshot_path)
        elif symbol == key.ESCAPE:
            record_current_episode("interrupted", "escape")
            window.close()
            pyglet.app.exit()

    @window.event
    def on_draw():
        rgb = env.render(mode="rgb_array")
        prepare_window_2d(window, image_width + SIDEBAR_WIDTH, image_height)
        window.clear()
        draw_rect(0, 0, image_width + SIDEBAR_WIDTH, image_height, BACKGROUND)
        draw_rgb(rgb, 0, 0, image_width, image_height)
        draw_sidebar(state, args, checkpoint_step, image_width, image_height)

    def update(dt):
        del dt
        nonlocal observation, episode_length, selected_return, env_return
        nonlocal paused, state
        if paused:
            return

        mean_action, action, std = predict_action(
            policy,
            observation,
            transform,
            args,
            device,
        )
        observation, step_env_reward, done, info = step_raw(env, action)
        step_selected_reward = reward_calculator.compute(env, step_env_reward)
        episode_length += 1
        env_return += step_env_reward
        selected_return += step_selected_reward
        reason = done_reason(done, info)

        state = ViewerState(
            mean_action=mean_action,
            action=action,
            std=std,
            env_reward=step_env_reward,
            selected_reward=step_selected_reward,
            env_return=env_return,
            selected_return=selected_return,
            episode=episode,
            episode_length=episode_length,
            completed_episodes=len(completed_returns),
            mean_completed_return=(
                float(np.mean(completed_returns)) if completed_returns else None
            ),
            done=done,
            done_reason=reason,
            paused=False,
        )

        if args.print_every > 0 and episode_length % args.print_every == 0:
            print(
                f"episode={episode} step={episode_length} "
                f"left={action[0]:+.4f} right={action[1]:+.4f} "
                f"{args.reward_function}_reward={step_selected_reward:+.4f} "
                f"{args.reward_function}_return={selected_return:+.4f} "
                f"default_return={env_return:+.4f}",
                flush=True,
            )

        if not done:
            return

        completed_returns.append(selected_return)
        record_current_episode("completed", reason)
        state.completed_episodes = len(completed_returns)
        state.mean_completed_return = float(np.mean(completed_returns))
        if args.episodes > 0 and len(completed_returns) >= args.episodes:
            pyglet.app.exit()
        elif args.stop_on_done:
            paused = True
            state.paused = True
        else:
            start_next_episode()

    pyglet.clock.schedule_interval(update, 1.0 / float(args.frame_rate))
    try:
        pyglet.app.run()
    except KeyboardInterrupt:
        record_current_episode("interrupted", "keyboard-interrupt")
        print("Interrupted by KeyboardInterrupt.", flush=True)
    finally:
        record_current_episode("interrupted", "shutdown")
        env.close()
        if completed_returns:
            print(
                f"completed_episodes={len(completed_returns)} "
                f"mean_{args.reward_function}_return={float(np.mean(completed_returns)):+.4f}",
                flush=True,
            )
        print(f"Returns saved to {recorder.path}", flush=True)


if __name__ == "__main__":
    main()
