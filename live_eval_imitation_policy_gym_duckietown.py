#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Visually evaluate a Duckiematrix-trained IL policy in gym-duckietown."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyglet
import torch
from PIL import Image
from torchvision import transforms

from cli_completion import parse_args_with_completion
from duckietown_paths import (
    EVALUATION_SCREENSHOT_DIR,
    IL_GYM_DUCKIETOWN_EVALUATION_DIR,
)
from duckietown_rewards import (
    GymDuckietownRewardCalculator,
    REWARD_FUNCTION_CHOICES,
    format_wheel_action,
)
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
from train_imitation_learning import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_COLUMNS,
    build_model,
    resolve_device,
)


SIDEBAR_WIDTH = 460


@dataclass
class EvalState:
    raw_action: np.ndarray
    action: np.ndarray
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


class ReturnRecorder:
    FIELDNAMES = (
        "episode",
        "status",
        "length",
        "selected_reward_function",
        "selected_return",
        "gym_duckietown_return",
        "done_reason",
    )

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=self.FIELDNAMES).writeheader()

    def record(
        self,
        episode: int,
        status: str,
        length: int,
        reward_function: str,
        selected_return: float,
        env_return: float,
        reason: str,
    ) -> None:
        row = {
            "episode": episode,
            "status": status,
            "length": length,
            "selected_reward_function": reward_function,
            "selected_return": selected_return,
            "gym_duckietown_return": env_return,
            "done_reason": reason,
        }
        with self.path.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=self.FIELDNAMES).writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained imitation-learning policy visually in gym-duckietown."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--map-name", default="loop_empty")
    parser.add_argument(
        "--reward-function",
        choices=REWARD_FUNCTION_CHOICES,
        default="posangle",
        help="Reward accumulated as selected_return; the original simulator return is tracked too.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of completed episodes before exiting; 0 runs until Escape.",
    )
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--robot-speed", type=float, default=None)
    parser.add_argument("--accept-start-angle-deg", type=float, default=4.0)
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument("--distortion", action="store_true")
    parser.add_argument("--dynamics-rand", action="store_true")
    parser.add_argument("--camera-rand", action="store_true")
    parser.add_argument("--draw-curve", action="store_true")
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--crop-y-start", type=int, default=0)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Model input size; defaults to the checkpoint config or 224.",
    )
    parser.add_argument(
        "--source-observation-channel-order",
        choices=("rgb", "bgr"),
        default="rgb",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
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
            f"{IL_GYM_DUCKIETOWN_EVALUATION_DIR}/<timestamp>_returns.csv."
        ),
    )
    parser.add_argument(
        "--screenshot-path",
        type=Path,
        default=EVALUATION_SCREENSHOT_DIR / "gym_duckietown_il_eval.png",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parse_args_with_completion(parser)


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    target_columns = tuple(checkpoint.get("target_columns", TARGET_COLUMNS))
    if target_columns != TARGET_COLUMNS:
        raise ValueError(f"Checkpoint predicts {target_columns}, expected {TARGET_COLUMNS}")

    model = build_model(
        model_name=config.get("model", "mobilenet_v3_small"),
        pretrained=False,
        train_backbone=bool(config.get("train_backbone", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def observation_to_rgb(observation: np.ndarray, channel_order: str) -> np.ndarray:
    image = np.ascontiguousarray(observation)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected observation shape (H, W, 3), got {image.shape}")
    if channel_order == "rgb":
        return image
    return np.ascontiguousarray(image[:, :, [2, 1, 0]])


def preprocess_observation(
    observation: np.ndarray,
    crop_y_start: int,
    image_size: int,
    channel_order: str,
    transform: transforms.Compose,
) -> torch.Tensor:
    image = Image.fromarray(observation_to_rgb(observation, channel_order)).convert("RGB")
    width, height = image.size
    crop_y_start = max(0, min(crop_y_start, height - 1))
    image = image.crop((0, crop_y_start, width, height))
    image = image.resize((image_size, image_size), Image.BILINEAR)
    return transform(image)


@torch.no_grad()
def predict_action(
    model: torch.nn.Module,
    observation: np.ndarray,
    transform: transforms.Compose,
    device: torch.device,
    crop_y_start: int,
    image_size: int,
    channel_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = preprocess_observation(
        observation,
        crop_y_start,
        image_size,
        channel_order,
        transform,
    ).unsqueeze(0).to(device)
    raw_action = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
    return raw_action, format_wheel_action(raw_action)


def reset_raw(env) -> np.ndarray:
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_raw(env, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated or truncated), info
    observation, reward, done, info = result
    return observation, float(reward), bool(done), info


def fmt(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.{precision}f}"


def draw_sidebar(state: EvalState, args: argparse.Namespace, x: int, height: int) -> None:
    draw_rect(x, 0, SIDEBAR_WIDTH, height, SIDEBAR_BG)
    status = "paused" if state.paused else "running"
    status_color = BAD if state.paused else GOOD
    selected_return_color = GOOD if state.selected_return >= 0.0 else BAD
    env_return_color = GOOD if state.env_return >= 0.0 else BAD
    lines = [
        ("IL policy in gym-duckietown", 18, ACCENT, True),
        (f"map {args.map_name}   {status}", 13, status_color, True),
        (f"episode {state.episode}   step {state.episode_length}", 13, MUTED, False),
        ("", 8, MUTED, False),
        (f"left {fmt(state.action[0], 3)}   right {fmt(state.action[1], 3)}", 16, TEXT, True),
        (f"raw left {fmt(state.raw_action[0], 3)}   right {fmt(state.raw_action[1], 3)}", 12, MUTED, False),
        ("", 8, MUTED, False),
        (f"{args.reward_function} reward {fmt(state.selected_reward)}", 15, TEXT, True),
        (f"{args.reward_function} return {fmt(state.selected_return)}", 17, selected_return_color, True),
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
            draw_label(text, x + 18, cursor_y, font_size=font_size, color=color, bold=bold)
        cursor_y -= max(16, font_size + 8)


def default_returns_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return IL_GYM_DUCKIETOWN_EVALUATION_DIR / f"{timestamp}_returns.csv"


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.episodes < 0:
        raise ValueError("--episodes must be non-negative")
    if args.frame_rate <= 0:
        raise ValueError("--frame-rate must be positive")

    configure_logging(args.log_level)
    device = resolve_device(args.device)
    model, checkpoint_config = load_policy(args.checkpoint, device)
    image_size = args.image_size or int(checkpoint_config.get("image_size", 224))
    transform = make_transform()
    returns_path = args.returns_file or default_returns_path()
    recorder = ReturnRecorder(returns_path)

    env = make_env(args)
    configure_logging(args.log_level)
    _, _, image_width, image_height = import_simulator()
    reward_calculator = GymDuckietownRewardCalculator(args.reward_function)
    observation = reset_raw(env)
    reward_calculator.reset(env)

    episode = 0
    episode_length = 0
    selected_return = 0.0
    env_return = 0.0
    completed_returns: list[float] = []
    current_episode_recorded = False
    paused = bool(args.start_paused)
    state = EvalState(
        raw_action=np.zeros(2, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        env_reward=0.0,
        selected_reward=0.0,
        env_return=0.0,
        selected_return=0.0,
        episode=episode,
        episode_length=0,
        completed_episodes=0,
        mean_completed_return=None,
        done=False,
        done_reason="in-progress",
        paused=paused,
    )

    from pyglet import window as pyglet_window
    from pyglet.window import key

    window = pyglet_window.Window(
        width=image_width + SIDEBAR_WIDTH,
        height=image_height,
        resizable=False,
        caption="gym-duckietown IL policy evaluation",
    )

    print(f"Checkpoint:      {args.checkpoint.expanduser()}", flush=True)
    print(f"Checkpoint model: {checkpoint_config.get('model', 'mobilenet_v3_small')}", flush=True)
    print(f"Device:          {device}", flush=True)
    print(f"Map:             {args.map_name}", flush=True)
    print(f"Reward function: {args.reward_function}", flush=True)
    print(
        "Preprocess:      "
        f"RGB={args.source_observation_channel_order == 'rgb'}, "
        f"crop_y_start={args.crop_y_start}, image_size={image_size}",
        flush=True,
    )
    print(f"Returns CSV:     {recorder.path}", flush=True)
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
        state = EvalState(
            raw_action=np.zeros(2, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
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
        draw_sidebar(state, args, image_width, image_height)

    def update(dt):
        del dt
        nonlocal observation, episode_length, selected_return, env_return
        nonlocal paused, current_episode_recorded, state
        if paused:
            return

        raw_action, action = predict_action(
            model,
            observation,
            transform,
            device,
            args.crop_y_start,
            image_size,
            args.source_observation_channel_order,
        )
        observation, step_env_reward, done, info = step_raw(env, action)
        step_selected_reward = reward_calculator.compute(env, step_env_reward)
        episode_length += 1
        env_return += step_env_reward
        selected_return += step_selected_reward
        reason = done_reason(done, info)

        state = EvalState(
            raw_action=raw_action,
            action=action,
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
