import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pygame


DEFAULT_DATA_DIR = Path("~/duckietown/imitation_learning/expert_data").expanduser()
TIMESTAMP_COLUMN = "timestamp in seconds since run start"

BACKGROUND = (20, 22, 24)
PANEL = (34, 38, 42)
TEXT = (235, 238, 240)
MUTED_TEXT = (165, 172, 178)
ACCENT = (79, 189, 186)
BAR_BG = (62, 68, 74)
BAR_LEFT = (92, 184, 92)
BAR_RIGHT = (235, 137, 88)

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 680
PANEL_HEIGHT = 148
HOLD_INTERVAL_MS = 75
DISPLAY_MODES = ("fit", "native")


def newest_run_dir(data_dir: Path) -> Path:
    run_dirs = [
        path
        for path in data_dir.glob("run_*")
        if path.is_dir() and (path / "actions.csv").exists()
    ]

    if not run_dirs:
        raise FileNotFoundError(f"No run directories with actions.csv found in {data_dir}")

    return sorted(run_dirs, key=lambda path: path.stat().st_mtime)[-1]


def resolve_run_dir(path: Path | None) -> Path:
    if path is None:
        return newest_run_dir(DEFAULT_DATA_DIR)

    if (path / "actions.csv").exists():
        return path

    if path.is_dir():
        return newest_run_dir(path)

    raise FileNotFoundError(f"Could not find dataset run directory at {path}")


def load_metadata(run_dir: Path) -> dict[str, Any]:
    meta_file = run_dir / "meta.json"

    if not meta_file.exists():
        return {}

    with meta_file.open() as file:
        return json.load(file)


def load_samples(run_dir: Path) -> list[dict[str, Any]]:
    actions_file = run_dir / "actions.csv"
    samples = []

    with actions_file.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            samples.append(
                {
                    "step_idx": int(row["step_idx"]),
                    "timestamp": float(row[TIMESTAMP_COLUMN]),
                    "image": row["image"],
                    "left_action": float(row["left_action"]),
                    "right_action": float(row["right_action"]),
                }
            )

    if not samples:
        raise ValueError(f"{actions_file} contains no samples")

    return samples


def render_text(surface: pygame.Surface, font: pygame.font.Font, text: str, xy: tuple[int, int],
                color: tuple[int, int, int] = TEXT) -> None:
    surface.blit(font.render(text, True, color), xy)


def scaled_rect(image_size: tuple[int, int], max_rect: pygame.Rect) -> pygame.Rect:
    image_width, image_height = image_size
    scale = min(max_rect.width / image_width, max_rect.height / image_height)
    width = int(image_width * scale)
    height = int(image_height * scale)
    x = max_rect.x + (max_rect.width - width) // 2
    y = max_rect.y + (max_rect.height - height) // 2
    return pygame.Rect(x, y, width, height)


def native_rect(image_size: tuple[int, int], max_rect: pygame.Rect) -> pygame.Rect:
    image_width, image_height = image_size

    if image_width > max_rect.width or image_height > max_rect.height:
        return scaled_rect(image_size, max_rect)

    x = max_rect.x + (max_rect.width - image_width) // 2
    y = max_rect.y + (max_rect.height - image_height) // 2
    return pygame.Rect(x, y, image_width, image_height)


def draw_signal_bar(surface: pygame.Surface, rect: pygame.Rect, value: float,
                    color: tuple[int, int, int]) -> None:
    pygame.draw.rect(surface, BAR_BG, rect, border_radius=4)
    center_x = rect.x + rect.width // 2
    pygame.draw.line(surface, MUTED_TEXT, (center_x, rect.y), (center_x, rect.bottom), 1)

    value = max(-1.0, min(1.0, float(value)))
    if value >= 0:
        fill = pygame.Rect(center_x, rect.y + 2, int((rect.width // 2 - 2) * value), rect.height - 4)
    else:
        fill_width = int((rect.width // 2 - 2) * abs(value))
        fill = pygame.Rect(center_x - fill_width, rect.y + 2, fill_width, rect.height - 4)

    pygame.draw.rect(surface, color, fill, border_radius=4)


def point_tuple(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_direction_arrow(surface: pygame.Surface, center: tuple[int, int], left: float, right: float) -> None:
    cx, cy = center
    radius = 42
    pygame.draw.circle(surface, BAR_BG, center, radius)
    pygame.draw.circle(surface, MUTED_TEXT, center, radius, 1)

    forward = (left + right) / 2.0
    turn_left = right - left
    vector = np.array([-0.85 * turn_left, -forward], dtype=np.float32)
    norm = float(np.linalg.norm(vector))

    if norm < 1e-6:
        pygame.draw.circle(surface, TEXT, center, 5)
        return

    direction = vector / norm
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    start = np.array([cx, cy], dtype=np.float32) - direction * 18
    tip = np.array([cx, cy], dtype=np.float32) + direction * 31
    head_left = tip - direction * 14 + perpendicular * 8
    head_right = tip - direction * 14 - perpendicular * 8

    start_point = point_tuple(start)
    tip_point = point_tuple(tip)
    head_left_point = point_tuple(head_left)
    head_right_point = point_tuple(head_right)

    pygame.draw.line(surface, ACCENT, start_point, tip_point, 5)
    pygame.draw.polygon(surface, ACCENT, [tip_point, head_left_point, head_right_point])


class ImageCache:
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.images: OrderedDict[Path, pygame.Surface] = OrderedDict()

    def get(self, image_path: Path) -> pygame.Surface:
        if image_path in self.images:
            image = self.images.pop(image_path)
            self.images[image_path] = image
            return image

        image = pygame.image.load(str(image_path)).convert()
        self.images[image_path] = image

        if len(self.images) > self.capacity:
            self.images.popitem(last=False)

        return image


def draw_frame(
    screen: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    samples: list[dict[str, Any]],
    metadata: dict[str, Any],
    run_dir: Path,
    image_cache: ImageCache,
    index: int,
    display_mode: str,
) -> None:
    screen.fill(BACKGROUND)

    sample = samples[index]
    image_path = run_dir / "images" / sample["image"]
    frame = image_cache.get(image_path)
    image_area = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT - PANEL_HEIGHT)

    if display_mode == "fit":
        target = scaled_rect(frame.get_size(), image_area.inflate(-24, -24))
    elif display_mode == "native":
        target = native_rect(frame.get_size(), image_area.inflate(-24, -24))
    else:
        raise ValueError(f"Unknown display mode {display_mode!r}")

    if target.size == frame.get_size():
        screen.blit(frame, target)
    else:
        screen.blit(pygame.transform.smoothscale(frame, target.size), target)

    panel = pygame.Rect(0, WINDOW_HEIGHT - PANEL_HEIGHT, WINDOW_WIDTH, PANEL_HEIGHT)
    pygame.draw.rect(screen, PANEL, panel)

    left = sample["left_action"]
    right = sample["right_action"]
    image_width, image_height = frame.get_size()
    run_id = metadata.get("run_id", "n/a")
    created_at = metadata.get("created_at", "n/a")

    y = panel.y + 18
    render_text(screen, fonts["large"], f"{index + 1}/{len(samples)}", (24, y), ACCENT)
    render_text(screen, fonts["normal"], f"step {sample['step_idx']}", (150, y + 5))
    render_text(screen, fonts["normal"], f"timestamp {sample['timestamp']:.3f}s", (270, y + 5))

    y += 42
    render_text(screen, fonts["normal"], f"left {left:+.2f}", (24, y))
    draw_signal_bar(screen, pygame.Rect(112, y + 2, 210, 18), left, BAR_LEFT)
    render_text(screen, fonts["normal"], f"right {right:+.2f}", (350, y))
    draw_signal_bar(screen, pygame.Rect(450, y + 2, 210, 18), right, BAR_RIGHT)
    render_text(screen, fonts["normal"], "drive", (706, y))
    draw_direction_arrow(screen, (830, y + 7), left, right)

    y += 38
    render_text(
        screen,
        fonts["small"],
        f"run {run_id} | created {created_at} | image {image_width}x{image_height} | display {display_mode}",
        (24, y),
        MUTED_TEXT,
    )
    render_text(screen, fonts["small"], f"{run_dir.name} / {sample['image']}", (24, y + 22), MUTED_TEXT)


def key_delta(key: int) -> int:
    if key == pygame.K_w:
        return 10
    if key == pygame.K_s:
        return -10
    if key == pygame.K_d:
        return 1
    if key == pygame.K_a:
        return -1
    return 0


def navigation_delta(keys: pygame.key.ScancodeWrapper) -> int:
    if keys[pygame.K_w]:
        return 10
    if keys[pygame.K_s]:
        return -10
    if keys[pygame.K_d]:
        return 1
    if keys[pygame.K_a]:
        return -1
    return 0


def run_viewer(run_dir: Path, display_mode: str) -> None:
    metadata = load_metadata(run_dir)
    samples = load_samples(run_dir)

    pygame.init()
    pygame.display.set_caption(f"Data Viewer - {run_dir.name}")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    image_cache = ImageCache()
    fonts = {
        "large": pygame.font.Font(None, 36),
        "normal": pygame.font.Font(None, 26),
        "small": pygame.font.Font(None, 22),
    }

    index = 0
    last_move_at = -HOLD_INTERVAL_MS
    running = True

    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d):
                    delta = key_delta(event.key)
                    index = max(0, min(len(samples) - 1, index + delta))
                    last_move_at = now

        keys = pygame.key.get_pressed()
        delta = navigation_delta(keys)

        if delta and now - last_move_at >= HOLD_INTERVAL_MS:
            index = max(0, min(len(samples) - 1, index + delta))
            last_move_at = now

        draw_frame(screen, fonts, samples, metadata, run_dir, image_cache, index, display_mode)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View Duckietown imitation-learning run directories.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help=f"Path to a run directory. Defaults to the newest run in {DEFAULT_DATA_DIR}/.",
    )
    parser.add_argument(
        "--display-mode",
        choices=DISPLAY_MODES,
        default="native",
        help="native shows stored pixel size when it fits; fit scales the image to the viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.path)
    run_viewer(run_dir, args.display_mode)


if __name__ == "__main__":
    main()
