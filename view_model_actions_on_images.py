#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""View IL or PPO checkpoint actions for a directory of camera images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pygame
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from cli_completion import parse_args_with_completion
from duckietown_action_control import (
    DuckietownActionControl,
    action_control_from_config,
)
from rl_models import TanhGaussianPolicy
from train_imitation_learning import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_COLUMNS,
    build_model,
    resolve_device,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
JPEG_STAGES = ("auto", "none", "before-resize", "after-resize")

WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 820
PANEL_HEIGHT = 250
BACKGROUND = (20, 22, 24)
PANEL = (34, 38, 42)
TEXT = (235, 238, 240)
MUTED = (165, 172, 178)
ACCENT = (79, 189, 186)
BAR_BACKGROUND = (62, 68, 74)
LEFT_COLOR = (92, 184, 92)
RIGHT_COLOR = (235, 137, 88)


@dataclass(frozen=True)
class PreprocessSpec:
    image_size: int
    crop_y_start: int
    jpeg_stage: str
    jpeg_quality: int
    file_channel_order: str
    inference_note: str | None = None


@dataclass
class PolicyBundle:
    checkpoint_type: str
    model: torch.nn.Module
    device: torch.device
    config: dict[str, Any]
    action_control: DuckietownActionControl | None
    preprocess: PreprocessSpec
    transform: transforms.Compose


@dataclass(frozen=True)
class Prediction:
    policy_controls: np.ndarray
    wheel_commands: np.ndarray
    policy_std: np.ndarray | None

    @property
    def normalized_v(self) -> float:
        return float(0.5 * (self.wheel_commands[0] + self.wheel_commands[1]))

    @property
    def normalized_omega(self) -> float:
        return float(0.5 * (self.wheel_commands[1] - self.wheel_commands[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show camera images and the deterministic wheel commands produced "
            "by an IL or PPO checkpoint."
        )
    )
    parser.add_argument("image_dir", type=Path, help="Directory containing camera images.")
    parser.add_argument("checkpoint", type=Path, help="IL or PPO checkpoint (.pt).")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override the checkpoint's model input size.",
    )
    parser.add_argument(
        "--crop-y-start",
        type=int,
        default=None,
        help=(
            "Override the first retained image row. PPO defaults to its checkpoint "
            "value; IL infers 200 for images_processed and otherwise 0."
        ),
    )
    parser.add_argument(
        "--jpeg-stage",
        choices=JPEG_STAGES,
        default="auto",
        help=(
            "Optional JPEG round trip. Auto emulates legacy IL images_processed "
            "and otherwise applies none."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--file-channel-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="Channel order after decoding the files; physical capture JPEGs are RGB.",
    )
    return parse_args_with_completion(parser)


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def find_images(image_dir: Path) -> list[Path]:
    directory = image_dir.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory not found: {directory}")
    images = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=natural_sort_key,
    )
    if not images:
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise FileNotFoundError(
            f"No supported images found in {directory}; expected {supported}"
        )
    return images


def detect_checkpoint_type(checkpoint: dict[str, Any]) -> str:
    if "policy_state_dict" in checkpoint:
        return "RL/PPO"
    if "model_state_dict" in checkpoint:
        return "IL"
    raise ValueError(
        "Unknown checkpoint format: expected policy_state_dict (RL/PPO) or "
        "model_state_dict (IL)"
    )


def resolve_preprocess_spec(
    checkpoint_type: str,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> PreprocessSpec:
    image_size = (
        int(args.image_size)
        if args.image_size is not None
        else int(config.get("image_size", 224))
    )
    if image_size <= 0:
        raise ValueError("--image-size must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    note = None
    training_image_dir = Path(str(config.get("image_dir", ""))).name
    if args.crop_y_start is not None:
        crop_y_start = int(args.crop_y_start)
    elif checkpoint_type == "RL/PPO":
        crop_y_start = int(config.get("crop_y_start", 0))
    elif training_image_dir == "images_processed":
        crop_y_start = 200
        note = "legacy IL preprocessing inferred from image_dir=images_processed"
    else:
        crop_y_start = 0
        if training_image_dir not in ("", "images"):
            note = (
                "IL crop was not stored; using crop_y_start=0 for "
                f"image_dir={training_image_dir}"
            )
    if crop_y_start < 0:
        raise ValueError("--crop-y-start must be non-negative")

    if args.jpeg_stage != "auto":
        jpeg_stage = args.jpeg_stage
    elif checkpoint_type == "IL" and training_image_dir == "images_processed":
        jpeg_stage = "after-resize"
    else:
        jpeg_stage = "none"

    return PreprocessSpec(
        image_size=image_size,
        crop_y_start=crop_y_start,
        jpeg_stage=jpeg_stage,
        jpeg_quality=int(args.jpeg_quality),
        file_channel_order=args.file_channel_order,
        inference_note=note,
    )


def make_transform(checkpoint: dict[str, Any]) -> transforms.Compose:
    mean = tuple(float(value) for value in checkpoint.get("imagenet_mean", IMAGENET_MEAN))
    std = tuple(float(value) for value in checkpoint.get("imagenet_std", IMAGENET_STD))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Checkpoint ImageNet mean/std must contain three values")
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_policy_bundle(args: argparse.Namespace) -> PolicyBundle:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {checkpoint_path}")

    checkpoint_type = detect_checkpoint_type(checkpoint)
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary")
    preprocess_spec = resolve_preprocess_spec(checkpoint_type, config, args)

    if checkpoint_type == "RL/PPO":
        action_control = action_control_from_config(config)
        model = TanhGaussianPolicy(
            config.get("model", "mobilenet_v3_small"),
            action_dim=action_control.policy_action_dim,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["policy_state_dict"])
    else:
        target_columns = tuple(checkpoint.get("target_columns", TARGET_COLUMNS))
        if target_columns != TARGET_COLUMNS:
            raise ValueError(
                f"IL checkpoint predicts {target_columns}, expected {TARGET_COLUMNS}"
            )
        action_control = None
        model = build_model(
            model_name=config.get("model", "mobilenet_v3_small"),
            pretrained=False,
            train_backbone=bool(config.get("train_backbone", False)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    return PolicyBundle(
        checkpoint_type=checkpoint_type,
        model=model,
        device=device,
        config=config,
        action_control=action_control,
        preprocess=preprocess_spec,
        transform=make_transform(checkpoint),
    )


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def load_rgb_image(image_path: Path, file_channel_order: str = "rgb") -> Image.Image:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if file_channel_order == "bgr":
        array = np.asarray(image)
        image = Image.fromarray(np.ascontiguousarray(array[:, :, ::-1]), mode="RGB")
    return image


def preprocess_image(
    image: Image.Image,
    spec: PreprocessSpec,
    transform: transforms.Compose,
) -> torch.Tensor:
    result = image
    if spec.jpeg_stage == "before-resize":
        result = jpeg_roundtrip(result, spec.jpeg_quality)

    width, height = result.size
    crop_y_start = min(spec.crop_y_start, height - 1)
    result = result.crop((0, crop_y_start, width, height))
    result = result.resize((spec.image_size, spec.image_size), Image.BILINEAR)

    if spec.jpeg_stage == "after-resize":
        result = jpeg_roundtrip(result, spec.jpeg_quality)
    return transform(result)


@torch.no_grad()
def predict(bundle: PolicyBundle, image: Image.Image) -> Prediction:
    tensor = preprocess_image(
        image,
        bundle.preprocess,
        bundle.transform,
    ).unsqueeze(0).to(bundle.device)

    if bundle.checkpoint_type == "RL/PPO":
        mean, log_std = bundle.model(tensor)
        policy_controls_tensor = torch.tanh(mean)
        wheel_tensor = bundle.action_control.to_wheels_tensor(policy_controls_tensor)
        controls = policy_controls_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        wheels = wheel_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        std = log_std.exp().squeeze(0).cpu().numpy().astype(np.float32)
    else:
        controls = bundle.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        wheels = np.clip(controls, -1.0, 1.0).astype(np.float32)
        std = None

    if not np.all(np.isfinite(controls)) or not np.all(np.isfinite(wheels)):
        raise ValueError("Model produced non-finite action values")
    return Prediction(controls, wheels, std)


def fit_rect(image_size: tuple[int, int], bounds: pygame.Rect) -> pygame.Rect:
    width, height = image_size
    scale = min(bounds.width / width, bounds.height / height)
    target_width = max(1, int(width * scale))
    target_height = max(1, int(height * scale))
    return pygame.Rect(
        bounds.x + (bounds.width - target_width) // 2,
        bounds.y + (bounds.height - target_height) // 2,
        target_width,
        target_height,
    )


def render_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = TEXT,
) -> None:
    screen.blit(font.render(text, True, color), position)


def draw_signal_bar(
    screen: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    color: tuple[int, int, int],
) -> None:
    pygame.draw.rect(screen, BAR_BACKGROUND, rect, border_radius=4)
    center_x = rect.centerx
    pygame.draw.line(screen, MUTED, (center_x, rect.y), (center_x, rect.bottom), 1)
    clipped = max(-1.0, min(1.0, float(value)))
    half_width = rect.width // 2 - 2
    fill_width = int(abs(clipped) * half_width)
    fill = pygame.Rect(
        center_x if clipped >= 0.0 else center_x - fill_width,
        rect.y + 2,
        fill_width,
        rect.height - 4,
    )
    pygame.draw.rect(screen, color, fill, border_radius=4)


def point_tuple(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_direction_arrow(
    screen: pygame.Surface,
    center: tuple[int, int],
    prediction: Prediction,
) -> None:
    pygame.draw.circle(screen, BAR_BACKGROUND, center, 62)
    pygame.draw.circle(screen, MUTED, center, 62, 1)
    pygame.draw.line(
        screen,
        MUTED,
        (center[0], center[1] + 18),
        (center[0], center[1] - 18),
        3,
    )

    vector = np.array(
        [-prediction.normalized_omega, -prediction.normalized_v],
        dtype=np.float32,
    )
    magnitude = min(1.0, float(np.linalg.norm(vector)))
    if magnitude < 1e-6:
        pygame.draw.circle(screen, ACCENT, center, 5)
        return

    direction = vector / float(np.linalg.norm(vector))
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    origin = np.asarray(center, dtype=np.float32)
    tip = origin + direction * (24.0 + 28.0 * magnitude)
    head_left = tip - direction * 14.0 + perpendicular * 8.0
    head_right = tip - direction * 14.0 - perpendicular * 8.0
    pygame.draw.line(screen, ACCENT, center, point_tuple(tip), 5)
    pygame.draw.polygon(
        screen,
        ACCENT,
        [point_tuple(tip), point_tuple(head_left), point_tuple(head_right)],
    )


def policy_description(bundle: PolicyBundle) -> str:
    if bundle.action_control is None:
        return "direct wheel regression"
    control = bundle.action_control
    description = control.mode
    if control.fixed_throttle is not None:
        description += f", fixed throttle={control.fixed_throttle:.3f}"
    return description


def draw_frame(
    screen: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    image: Image.Image,
    image_path: Path,
    image_index: int,
    image_count: int,
    checkpoint_path: Path,
    bundle: PolicyBundle,
    prediction: Prediction,
) -> None:
    screen.fill(BACKGROUND)
    image_area = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT - PANEL_HEIGHT)
    surface = pygame.image.fromstring(image.tobytes(), image.size, "RGB")
    target = fit_rect(surface.get_size(), image_area.inflate(-24, -24))
    screen.blit(pygame.transform.smoothscale(surface, target.size), target)

    panel = pygame.Rect(0, WINDOW_HEIGHT - PANEL_HEIGHT, WINDOW_WIDTH, PANEL_HEIGHT)
    pygame.draw.rect(screen, PANEL, panel)
    left, right = (float(value) for value in prediction.wheel_commands)

    y = panel.y + 15
    render_text(
        screen,
        fonts["large"],
        f"{image_index + 1}/{image_count}  {image_path.name}",
        (24, y),
        ACCENT,
    )
    render_text(
        screen,
        fonts["small"],
        f"{bundle.checkpoint_type} | {policy_description(bundle)} | {checkpoint_path.name}",
        (520, y + 6),
        MUTED,
    )

    y += 45
    render_text(screen, fonts["normal"], f"left {left:+.3f}", (24, y))
    draw_signal_bar(screen, pygame.Rect(130, y + 3, 240, 20), left, LEFT_COLOR)
    render_text(screen, fonts["normal"], f"right {right:+.3f}", (400, y))
    draw_signal_bar(screen, pygame.Rect(520, y + 3, 240, 20), right, RIGHT_COLOR)
    render_text(screen, fonts["small"], "Bewegung", (815, y + 5), MUTED)
    draw_direction_arrow(screen, (970, y + 48), prediction)

    y += 45
    render_text(
        screen,
        fonts["normal"],
        f"v_norm {prediction.normalized_v:+.3f}    "
        f"omega_norm {prediction.normalized_omega:+.3f}",
        (24, y),
    )
    controls = ", ".join(f"{value:+.3f}" for value in prediction.policy_controls)
    render_text(screen, fonts["small"], f"policy controls [{controls}]", (400, y + 5), MUTED)

    y += 39
    spec = bundle.preprocess
    render_text(
        screen,
        fonts["small"],
        f"Preprocess: RGB, crop y={spec.crop_y_start}, "
        f"resize={spec.image_size}x{spec.image_size}, JPEG={spec.jpeg_stage}",
        (24, y),
        MUTED,
    )
    if prediction.policy_std is not None:
        std = ", ".join(f"{value:.3f}" for value in prediction.policy_std)
        render_text(screen, fonts["small"], f"policy std [{std}]", (690, y), MUTED)

    y += 33
    note = spec.inference_note or "Preprocessing read from checkpoint/defaults"
    render_text(screen, fonts["small"], note, (24, y), MUTED)
    render_text(
        screen,
        fonts["normal"],
        "A: vorheriges Bild    D: naechstes Bild    Esc/Q: beenden",
        (520, y - 4),
        ACCENT,
    )


def run_viewer(
    images: list[Path],
    checkpoint_path: Path,
    bundle: PolicyBundle,
) -> None:
    pygame.init()
    pygame.display.set_caption(
        f"Model Actions on Images - {checkpoint_path.expanduser().name}"
    )
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    fonts = {
        "large": pygame.font.Font(None, 36),
        "normal": pygame.font.Font(None, 28),
        "small": pygame.font.Font(None, 21),
    }
    clock = pygame.time.Clock()
    cache: dict[int, tuple[Image.Image, Prediction]] = {}
    index = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_d and index < len(images) - 1:
                    index += 1
                elif event.key == pygame.K_a and index > 0:
                    index -= 1

        if index not in cache:
            image = load_rgb_image(
                images[index],
                file_channel_order=bundle.preprocess.file_channel_order,
            )
            cache[index] = (image, predict(bundle, image))
        image, prediction = cache[index]
        draw_frame(
            screen,
            fonts,
            image,
            images[index],
            index,
            len(images),
            checkpoint_path,
            bundle,
            prediction,
        )
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main() -> int:
    args = parse_args()
    try:
        images = find_images(args.image_dir)
        bundle = load_policy_bundle(args)
        print(f"Checkpoint type: {bundle.checkpoint_type}")
        print(f"Images:          {len(images)} from {args.image_dir.expanduser().resolve()}")
        print(f"Device:          {bundle.device}")
        print(
            "Preprocess:      "
            f"crop_y_start={bundle.preprocess.crop_y_start}, "
            f"image_size={bundle.preprocess.image_size}, "
            f"jpeg_stage={bundle.preprocess.jpeg_stage}, "
            f"file_channel_order={bundle.preprocess.file_channel_order}"
        )
        if bundle.preprocess.inference_note:
            print(f"Note:            {bundle.preprocess.inference_note}")
        run_viewer(images, args.checkpoint, bundle)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
