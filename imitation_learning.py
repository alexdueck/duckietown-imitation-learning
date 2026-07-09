import asyncio
import argparse
import csv
import json
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from datetime import datetime
from time import time, sleep

import numpy as np
import pygame
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from duckietown.sdk.middleware.dtps.base import DTPS
from duckiematrix_telemetry import TELEMETRY_COLUMNS, collect_state_telemetry, format_telemetry_value


# ----------------------------
# Config
# ----------------------------

ENTITY_NAME = "map_0/vehicle_0"
OUT_DIR = Path("~/duckietown/imitation_learning/expert_data").expanduser()
MAX_STEPS = 3000
JPEG_QUALITY = 95
TIMESTAMP_COLUMN = "timestamp in seconds since run start"
NONZERO_ACTION_EPS = 0.1

V_FORWARD = 0.60
V_BACKWARD = -0.40
TURN = 0.35
SAMPLE_PERIOD = 0.1

THROTTLE_RATE = 4.0
STEERING_RATE = 1.6
AUTO_CENTER_RATE = 0.9

SOURCE_OBSERVATION_CHANNEL_ORDER = "bgr"
SAVED_IMAGE_CHANNEL_ORDER = "rgb"

WINDOW_SIZE = (720, 260)
WINDOW_BG = (24, 27, 30)
WINDOW_TEXT = (238, 241, 243)
WINDOW_MUTED = (172, 179, 186)
WINDOW_ACCENT = (90, 200, 190)


def next_run_id(out_dir: Path) -> int:
    run_ids = []

    for path in out_dir.glob("run_*"):
        run_id_part = path.name.removeprefix("run_").split("_", 1)[0]
        if len(run_id_part) == 3 and run_id_part.isdigit():
            run_ids.append(int(run_id_part))

    return max(run_ids, default=0) + 1


def progress_items(items, description: str):
    total = len(items)

    if tqdm is not None:
        yield from tqdm(items, desc=description, total=total)
        return

    if total == 0:
        return

    report_every = max(1, total // 20)

    for index, item in enumerate(items, start=1):
        if index == 1 or index == total or index % report_every == 0:
            print(f"{description}: {index}/{total}")
        yield item


def observation_to_rgb(observation: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(observation)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected observation shape (H, W, 3), got {image.shape}")

    if SOURCE_OBSERVATION_CHANNEL_ORDER == "bgr":
        return np.ascontiguousarray(image[:, :, [2, 1, 0]])

    if SOURCE_OBSERVATION_CHANNEL_ORDER == "rgb":
        return image

    raise ValueError(f"Unknown source channel order {SOURCE_OBSERVATION_CHANNEL_ORDER!r}")


def save_dataset(
    run_dir: Path,
    run_prefix: str,
    observations: list[np.ndarray],
    timestamps: list[float],
    actions: list[np.ndarray],
    step_indices: list[int],
    telemetry_rows: list[dict],
    metadata: dict,
) -> None:
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)

    image_names = [
        f"{run_prefix}_{sample_idx:04d}.jpg"
        for sample_idx in range(len(observations))
    ]

    for observation, image_name in progress_items(
        list(zip(observations, image_names)),
        "Saving JPEGs",
    ):
        image = Image.fromarray(observation_to_rgb(observation))
        image.save(images_dir / image_name, format="JPEG", quality=JPEG_QUALITY)

    actions_file = run_dir / "actions.csv"

    with actions_file.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step_idx",
                TIMESTAMP_COLUMN,
                "image",
                "left_action",
                "right_action",
                *TELEMETRY_COLUMNS,
            ],
        )
        writer.writeheader()

        for step_idx, timestamp, image_name, action, telemetry in zip(
            step_indices,
            timestamps,
            image_names,
            actions,
            telemetry_rows,
        ):
            writer.writerow(
                {
                    "step_idx": step_idx,
                    TIMESTAMP_COLUMN: f"{timestamp:.9f}",
                    "image": image_name,
                    "left_action": f"{float(action[0]):.9f}",
                    "right_action": f"{float(action[1]):.9f}",
                    **{
                        column: format_telemetry_value(telemetry[column])
                        for column in TELEMETRY_COLUMNS
                    },
                }
            )

    metadata.update(
        {
            "num_samples": len(observations),
            "observation_shape": list(observations[0].shape) if observations else None,
            "telemetry_columns": TELEMETRY_COLUMNS,
        }
    )

    with (run_dir / "meta.json").open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def trim_initial_zero_actions(
    observations: list[np.ndarray],
    timestamps: list[float],
    actions: list[np.ndarray],
    step_indices: list[int],
    telemetry_rows: list[dict],
) -> tuple[list[np.ndarray], list[float], list[np.ndarray], list[int], list[dict], int]:
    first_nonzero_idx = None

    for sample_idx, action in enumerate(actions):
        if float(np.max(np.abs(action))) > NONZERO_ACTION_EPS:
            first_nonzero_idx = sample_idx
            break

    if first_nonzero_idx is None:
        return [], [], [], [], [], len(actions)

    return (
        observations[first_nonzero_idx:],
        timestamps[first_nonzero_idx:],
        actions[first_nonzero_idx:],
        step_indices[first_nonzero_idx:],
        telemetry_rows[first_nonzero_idx:],
        first_nonzero_idx,
    )


def shutdown_dtps(timeout: float = 2.0) -> None:
    loop = DTPS._loop

    if loop.is_closed() or not loop.is_running():
        return

    async def shutdown_contexts_and_tasks():
        for context in list(DTPS._contexts.values()):
            try:
                await context.aclose()
            except Exception:
                pass

        current_task = asyncio.current_task()
        pending_tasks = [
            task
            for task in asyncio.all_tasks(loop)
            if task is not current_task and not task.done()
        ]

        for task in pending_tasks:
            task.cancel()

        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    future = asyncio.run_coroutine_threadsafe(shutdown_contexts_and_tasks(), loop)

    try:
        future.result(timeout=timeout)
    except FutureTimeoutError:
        pass
    finally:
        loop.call_soon_threadsafe(loop.stop)

        if DTPS._worker is not None:
            DTPS._worker.join(timeout=timeout)

        DTPS._worker = None
        DTPS._contexts.clear()
        DTPS._connectors.clear()


# ----------------------------
# Keyboard -> action
# ----------------------------

def move_towards(value: float, target: float, max_delta: float) -> float:
    if value < target:
        return min(value + max_delta, target)
    if value > target:
        return max(value - max_delta, target)
    return value


def normalize_action(action: np.ndarray) -> np.ndarray:
    scale = max(1.0, float(np.max(np.abs(action))))
    return action / scale


def render_status_line(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = WINDOW_TEXT,
) -> None:
    screen.blit(font.render(text, True, color), xy)


def render_live_status(
    screen: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    step_idx: int,
    timestamp: float,
    action_controller,
    action: np.ndarray,
    telemetry: dict,
) -> None:
    screen.fill(WINDOW_BG)
    render_status_line(
        screen,
        fonts["large"],
        f"step {step_idx}  t={timestamp:.2f}s",
        (22, 18),
        WINDOW_ACCENT,
    )
    render_status_line(
        screen,
        fonts["large"],
        f"left {action[0]:+.3f}   right {action[1]:+.3f}",
        (22, 58),
    )
    render_status_line(
        screen,
        fonts["normal"],
        f"throttle {action_controller.throttle:+.3f}   steering {action_controller.steering:+.3f}",
        (22, 96),
        WINDOW_MUTED,
    )
    render_status_line(
        screen,
        fonts["large"],
        f"reward {float(telemetry['reward']):+.4f}   speed {float(telemetry['speed']):.4f}",
        (22, 136),
    )
    render_status_line(
        screen,
        fonts["large"],
        f"dot_dir {float(telemetry['lane_dot_dir']):+.4f}   dist {float(telemetry['lane_dist']):+.4f}",
        (22, 176),
    )
    render_status_line(
        screen,
        fonts["normal"],
        f"angle {float(telemetry['lane_angle_deg']):+.2f} deg   "
        f"lane_valid={int(bool(telemetry['lane_position_valid']))}   "
        f"terminated={int(bool(telemetry['terminated']))} truncated={int(bool(telemetry['truncated']))}",
        (22, 216),
        WINDOW_MUTED,
    )
    pygame.display.flip()


class KeyboardActionController:
    def __init__(self):
        self.throttle = 0.0
        self.steering = 0.0
        self.last_update_at = time()

    def update(self) -> np.ndarray:
        now = time()
        dt = max(0.0, min(now - self.last_update_at, 0.1))
        self.last_update_at = now

        keys = pygame.key.get_pressed()

        forward = keys[pygame.K_w] or keys[pygame.K_UP]
        backward = keys[pygame.K_s] or keys[pygame.K_DOWN]
        steer_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        steer_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        target_throttle = 0.0
        if forward and not backward:
            target_throttle = V_FORWARD
        elif backward and not forward:
            target_throttle = V_BACKWARD

        if steer_left and not steer_right:
            target_steering = TURN
            steering_rate = STEERING_RATE
        elif steer_right and not steer_left:
            target_steering = -TURN
            steering_rate = STEERING_RATE
        else:
            target_steering = 0.0
            steering_rate = AUTO_CENTER_RATE

        self.throttle = move_towards(self.throttle, target_throttle, THROTTLE_RATE * dt)
        self.steering = move_towards(self.steering, target_steering, steering_rate * dt)

        left = self.throttle - self.steering
        right = self.throttle + self.steering

        return normalize_action(np.array([left, right], dtype=np.float32))


# ----------------------------
# Main collector loop
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Duckiematrix imitation-learning data or inspect live observations.",
    )
    parser.add_argument(
        "--observe-only",
        "--observation-only",
        action="store_true",
        help="Show live action/reward/lane telemetry without saving any dataset files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    observe_only = bool(args.observe_only)

    run_id = None
    run_dir = None
    run_prefix = None
    metadata = None

    if not observe_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        run_id = next_run_id(OUT_DIR)
        run_started_at = datetime.now()
        run_started_at_timestamp = run_started_at.isoformat(timespec="seconds")
        filename_timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
        run_prefix = f"run_{run_id:03d}"
        run_dir = OUT_DIR / f"{run_prefix}_{filename_timestamp}"
        metadata = {
            "run_id": run_id,
            "created_at": run_started_at_timestamp,
            "sample_period_seconds": SAMPLE_PERIOD,
            "env": "DuckiematrixDB21JEnv",
            "entity_name": ENTITY_NAME,
            "action_format": ["left_wheel", "right_wheel"],
            "action_range": [-1.0, 1.0],
            "image_format": "jpg",
            "jpeg_quality": JPEG_QUALITY,
            "source_observation_channel_order": SOURCE_OBSERVATION_CHANNEL_ORDER,
            "saved_image_channel_order": SAVED_IMAGE_CHANNEL_ORDER,
            "controller": {
                "type": "stateful_keyboard",
                "v_forward": V_FORWARD,
                "v_backward": V_BACKWARD,
                "max_steering": TURN,
                "throttle_rate": THROTTLE_RATE,
                "steering_rate": STEERING_RATE,
                "auto_center_rate": AUTO_CENTER_RATE,
            },
            "notes": [
                "Each CSV row is one training sample.",
                "Samples are aligned as obs_t with action_t; env.step(action_t) is called after the sample is collected in memory.",
                "Reward and telemetry columns describe obs_t and the reward returned by the previous env.step(action_t-1).",
                "JPEG files are written after collection finishes to avoid disk I/O during driving.",
            ],
        }

    observations = []
    timestamps = []
    actions = []
    step_indices = []
    telemetry_rows = []

    env = None
    pygame_initialized = False
    screen = None
    fonts = None
    
    running = True
    t0 = time()
    next_sample_at = time()

    try:
        pygame.init()
        pygame_initialized = True
        screen = pygame.display.set_mode(WINDOW_SIZE)
        if observe_only:
            pygame.display.set_caption("Duckiematrix Observer - Focus here")
        else:
            pygame.display.set_caption("Duckiematrix Expert Collector - Focus here")
        fonts = {
            "large": pygame.font.Font(None, 36),
            "normal": pygame.font.Font(None, 28),
        }

        env = DuckiematrixDB21JEnv(
            entity_name=ENTITY_NAME,
            headless=True,          # Renderer separat benutzen
        )

        obs, info = env.reset()
        action_controller = KeyboardActionController()
        previous_pose_for_current_observation = None
        pending_reward = float("nan")
        pending_terminated = False
        pending_truncated = False
        t0 = time()
        next_sample_at = time()

        for step_idx in range(MAX_STEPS):
            now = time()
            if now < next_sample_at:
                sleep(next_sample_at - now)
            sample_started_at = time()
            next_sample_at = sample_started_at + SAMPLE_PERIOD

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running:
                break

            action = action_controller.update()
            telemetry = collect_state_telemetry(
                env=env,
                previous_pose=previous_pose_for_current_observation,
                reward=pending_reward,
                terminated=pending_terminated,
                truncated=pending_truncated,
            )

            timestamp = time() - t0

            if not observe_only:
                # Dataset-Sample: Bild vor der Aktion + exakt gesendete Expertenaktion
                observations.append(obs.copy())
                timestamps.append(timestamp)
                actions.append(action.copy())
                step_indices.append(step_idx)
                telemetry_rows.append(telemetry)

            print(
                f"{timestamp:+.3f} | step={step_idx} | "
                f"throttle={action_controller.throttle:+.2f} steering={action_controller.steering:+.2f} | "
                f"left={action[0]:+.2f} right={action[1]:+.2f}"
            )

            print(
                f"reward={float(telemetry['reward']):+.4f} speed={float(telemetry['speed']):.4f} "
                f"dot_dir={float(telemetry['lane_dot_dir']):+.4f} "
                f"dist={float(telemetry['lane_dist']):+.4f} "
                f"lane_valid={int(bool(telemetry['lane_position_valid']))}"
            )

            caption_mode = "Observer" if observe_only else "Collector"
            pygame.display.set_caption(
                f"{caption_mode} | step={step_idx} | "
                f"throttle={action_controller.throttle:+.2f} steering={action_controller.steering:+.2f} | "
                f"left={action[0]:+.2f} right={action[1]:+.2f} | "
                f"reward={float(telemetry['reward']):+.3f}"
            )
            render_live_status(
                screen=screen,
                fonts=fonts,
                step_idx=step_idx,
                timestamp=timestamp,
                action_controller=action_controller,
                action=action,
                telemetry=telemetry,
            )

            pose_before_step = getattr(env, "last_pose", None)
            obs, reward, terminated, truncated, info = env.step(action)
            previous_pose_for_current_observation = pose_before_step
            pending_reward = float(reward)
            pending_terminated = bool(terminated)
            pending_truncated = bool(truncated)

            if terminated or truncated:
                if observe_only:
                    print(
                        "Environment ended "
                        f"(terminated={int(bool(terminated))}, truncated={int(bool(truncated))}); resetting."
                    )
                    obs, info = env.reset()
                    previous_pose_for_current_observation = None
                    pending_reward = float("nan")
                    pending_terminated = False
                    pending_truncated = False
                    continue

                break

    except KeyboardInterrupt:
        if observe_only:
            print("Interrupted by KeyboardInterrupt.")
        else:
            print("Interrupted by KeyboardInterrupt; saving collected samples.")

    finally:
        if env is not None:
            env.close()

        shutdown_dtps()

        if pygame_initialized:
            pygame.quit()

        if observe_only:
            print("Observation-only mode: no dataset files were written.")
        else:
            min_samples = min(
                len(observations),
                len(timestamps),
                len(actions),
                len(step_indices),
                len(telemetry_rows),
            )
            observations = observations[:min_samples]
            timestamps = timestamps[:min_samples]
            actions = actions[:min_samples]
            step_indices = step_indices[:min_samples]
            telemetry_rows = telemetry_rows[:min_samples]

            observations, timestamps, actions, step_indices, telemetry_rows, trimmed_initial_samples = trim_initial_zero_actions(
                observations,
                timestamps,
                actions,
                step_indices,
                telemetry_rows,
            )
            num_samples = len(actions)
            metadata["post_processing"] = {
                "trim_initial_zero_actions": True,
                "nonzero_action_eps": NONZERO_ACTION_EPS,
                "num_samples": num_samples,
                "trimmed_initial_samples": trimmed_initial_samples,
            }

            if trimmed_initial_samples:
                print(f"Trimmed {trimmed_initial_samples} initial samples with zero actions.")

            save_dataset(
                run_dir,
                run_prefix,
                observations,
                timestamps,
                actions,
                step_indices,
                telemetry_rows,
                metadata,
            )

            print(f"Saved {len(actions)} samples for run {run_id:03d} to {run_dir}")


if __name__ == "__main__":
    main()
