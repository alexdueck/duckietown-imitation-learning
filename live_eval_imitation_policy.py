#!/usr/bin/env python3
"""Run a trained imitation-learning policy live in gym-duckiematrix."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from io import BytesIO
from pathlib import Path
from time import sleep, time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from duckietown.sdk.middleware.dtps.base import DTPS
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv

from train_imitation_learning import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_COLUMNS,
    build_model,
    resolve_device,
)


ENTITY_NAME = "map_0/vehicle_0"
DEFAULT_CHECKPOINT = Path(
    "checkpoints/imitation_learning/20260707_132144_mobilenet_v3_small/best.pt"
)
SOURCE_OBSERVATION_CHANNEL_ORDER = "bgr"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CROP_Y_START = 200
IMAGE_SIZE = 224
JPEG_QUALITY = 95
SAMPLE_PERIOD = 0.1
MAX_STEPS = 3000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained Duckiebot imitation policy in Duckiematrix."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--entity-name", default=ENTITY_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--sample-period", type=float, default=SAMPLE_PERIOD)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--crop-y-start", type=int, default=CROP_Y_START)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument(
        "--jpeg-roundtrip-stage",
        choices=("before-resize", "after-resize", "none"),
        default="after-resize",
        help=(
            "Where to emulate JPEG compression. Use before-resize for models "
            "trained on raw images/ and after-resize for images_processed/."
        ),
    )
    parser.add_argument(
        "--no-jpeg-roundtrip",
        action="store_true",
        help="Deprecated alias for --jpeg-roundtrip-stage none.",
    )
    parser.add_argument(
        "--no-clip-actions",
        action="store_true",
        help="Send raw model outputs instead of clipping actions to [-1, 1].",
    )
    parser.add_argument(
        "--reset-on-done",
        action="store_true",
        help="Reset the environment and keep running after termination/truncation.",
    )
    return parser.parse_args()


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


def preprocess_observation(
    observation: np.ndarray,
    crop_y_start: int,
    image_size: int,
    jpeg_quality: int,
    jpeg_roundtrip_stage: str,
) -> Image.Image:
    image = Image.fromarray(observation_to_rgb(observation)).convert("RGB")

    if jpeg_roundtrip_stage == "before-resize":
        image = jpeg_roundtrip(image, jpeg_quality)

    width, height = image.size
    crop_y_start = max(0, min(crop_y_start, height - 1))
    image = image.crop((0, crop_y_start, width, height))
    image = image.resize((image_size, image_size), Image.BILINEAR)

    if jpeg_roundtrip_stage == "after-resize":
        image = jpeg_roundtrip(image, jpeg_quality)

    return image


def jpeg_roundtrip(image: Image.Image, jpeg_quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def make_live_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = checkpoint_path.expanduser()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_name = config.get("model", "mobilenet_v3_small")
    train_backbone = bool(config.get("train_backbone", False))
    target_columns = tuple(checkpoint.get("target_columns", TARGET_COLUMNS))

    if target_columns != TARGET_COLUMNS:
        raise ValueError(
            f"Checkpoint predicts {target_columns}, expected {TARGET_COLUMNS}"
        )

    model = build_model(
        model_name=model_name,
        pretrained=False,
        train_backbone=train_backbone,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def predict_action(
    model: torch.nn.Module,
    observation: np.ndarray,
    transform: transforms.Compose,
    device: torch.device,
    crop_y_start: int,
    image_size: int,
    jpeg_quality: int,
    jpeg_roundtrip_stage: str,
    clip_actions: bool,
) -> np.ndarray:
    image = preprocess_observation(
        observation=observation,
        crop_y_start=crop_y_start,
        image_size=image_size,
        jpeg_quality=jpeg_quality,
        jpeg_roundtrip_stage=jpeg_roundtrip_stage,
    )
    image_tensor = transform(image).unsqueeze(0).to(device)
    action = model(image_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)

    if clip_actions:
        action = np.clip(action, -1.0, 1.0)

    return action


def main() -> None:
    args = parse_args()
    jpeg_roundtrip_stage = (
        "none" if args.no_jpeg_roundtrip else args.jpeg_roundtrip_stage
    )
    device = resolve_device(args.device)
    model, config = load_policy(args.checkpoint, device=device)
    transform = make_live_transform()
    training_image_dir = Path(config.get("image_dir", "")).name

    print(f"Checkpoint:       {args.checkpoint.expanduser()}")
    print(f"Checkpoint model: {config.get('model', 'mobilenet_v3_small')}")
    print(f"Training images:  {config.get('image_dir', 'unknown')}")
    print(f"Device:           {device}")
    print(f"Entity:           {args.entity_name}")
    print(f"Camera:           {args.camera_width}x{args.camera_height}")
    print(
        "Preprocess:       "
        f"crop_y_start={args.crop_y_start}, image_size={args.image_size}, "
        f"jpeg_roundtrip_stage={jpeg_roundtrip_stage}, "
        f"jpeg_quality={args.jpeg_quality}"
    )

    if training_image_dir == "images" and (
        args.crop_y_start != 0 or jpeg_roundtrip_stage != "before-resize"
    ):
        print(
            "Warning: checkpoint was trained on raw images/. For a closer live "
            "match, use --crop-y-start 0 --jpeg-roundtrip-stage before-resize.",
            flush=True,
        )
    elif training_image_dir == "images_processed" and (
        args.crop_y_start == 0 or jpeg_roundtrip_stage != "after-resize"
    ):
        print(
            "Warning: checkpoint was trained on images_processed/. Check that "
            "your live crop and JPEG stage match that preprocessing.",
            flush=True,
        )

    env = None
    next_step_at = time()

    try:
        env = DuckiematrixDB21JEnv(
            entity_name=args.entity_name,
            headless=True,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
        )
        observation, info = env.reset()

        for step_idx in range(args.max_steps):
            now = time()
            if now < next_step_at:
                sleep(next_step_at - now)

            step_started_at = time()
            next_step_at = step_started_at + args.sample_period

            action = predict_action(
                model=model,
                observation=observation,
                transform=transform,
                device=device,
                crop_y_start=args.crop_y_start,
                image_size=args.image_size,
                jpeg_quality=args.jpeg_quality,
                jpeg_roundtrip_stage=jpeg_roundtrip_stage,
                clip_actions=not args.no_clip_actions,
            )

            observation, reward, terminated, truncated, info = env.step(action)

            print(
                f"step={step_idx:05d} "
                f"left={action[0]:+.4f} right={action[1]:+.4f} "
                f"reward={float(reward):+.4f} "
                f"terminated={terminated} truncated={truncated}",
                flush=True,
            )

            if terminated or truncated:
                if not args.reset_on_done:
                    break

                print("Episode ended; resetting environment.", flush=True)
                observation, info = env.reset()

    except KeyboardInterrupt:
        print("Interrupted by KeyboardInterrupt.")

    finally:
        if env is not None:
            env.close()

        shutdown_dtps()


if __name__ == "__main__":
    main()
