#!/usr/bin/env python3
"""Streaming, simulator-compatible camera/action recording."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

TIMESTAMP_COLUMN = "timestamp in seconds since run start"


def next_run_id(output_dir: Path) -> int:
    run_ids: list[int] = []
    for path in output_dir.glob("run_*"):
        run_id = path.name.removeprefix("run_").split("_", 1)[0]
        if len(run_id) == 3 and run_id.isdigit():
            run_ids.append(int(run_id))
    return max(run_ids, default=0) + 1


ACTION_COLUMNS = (
    "step_idx",
    TIMESTAMP_COLUMN,
    "image",
    "left_action",
    "right_action",
    "camera_seq",
    "camera_stamp",
    "frame_age_seconds",
    "linear_velocity",
    "angular_velocity",
)


class PhysicalDatasetRecorder:
    """Write samples as they arrive, so an interrupted drive remains usable."""

    def __init__(self, output_dir: Path, metadata: dict[str, Any]) -> None:
        output_dir = output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = next_run_id(output_dir)
        started = datetime.now(timezone.utc)
        self.run_prefix = f"run_{run_id:03d}"
        self.run_dir = output_dir / (
            f"{self.run_prefix}_{started.strftime('%Y%m%d_%H%M%S')}"
        )
        self.images_dir = self.run_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=False)
        self._started_at = monotonic()
        self._sample_count = 0
        self._last_camera_key: tuple[int, float] | None = None
        self._metadata = {
            **metadata,
            "run_id": run_id,
            "created_at": started.isoformat(),
            "action_format": ["left_wheel", "right_wheel"],
            "action_range": [-1.0, 1.0],
            "alignment": (
                "Each image is the newest obs_t available at the control tick; "
                "left_action/right_action are the effective wheel-equivalent "
                "values of the Twist2DStamped command sent for that tick."
            ),
        }
        self._csv_file = (self.run_dir / "actions.csv").open("w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=ACTION_COLUMNS)
        self._writer.writeheader()
        self._csv_file.flush()
        self._write_meta(recording=True)

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def record(
        self,
        *,
        payload: bytes,
        image_suffix: str,
        camera_seq: int,
        camera_stamp: float,
        frame_age: float,
        left_action: float,
        right_action: float,
        linear_velocity: float,
        angular_velocity: float,
    ) -> bool:
        camera_key = (int(camera_seq), float(camera_stamp))
        if camera_key == self._last_camera_key:
            return False
        sample_idx = self._sample_count
        image_name = f"{self.run_prefix}_{sample_idx:06d}{image_suffix}"
        image_path = self.images_dir / image_name
        temporary_path = image_path.with_name(f".{image_path.name}.tmp")
        temporary_path.write_bytes(payload)
        temporary_path.replace(image_path)
        self._writer.writerow(
            {
                "step_idx": sample_idx,
                TIMESTAMP_COLUMN: f"{monotonic() - self._started_at:.9f}",
                "image": image_name,
                "left_action": f"{left_action:.9f}",
                "right_action": f"{right_action:.9f}",
                "camera_seq": camera_seq,
                "camera_stamp": f"{camera_stamp:.9f}",
                "frame_age_seconds": f"{frame_age:.9f}",
                "linear_velocity": f"{linear_velocity:.9f}",
                "angular_velocity": f"{angular_velocity:.9f}",
            }
        )
        self._csv_file.flush()
        self._last_camera_key = camera_key
        self._sample_count += 1
        return True

    def close(self) -> None:
        if self._csv_file.closed:
            return
        self._csv_file.flush()
        self._csv_file.close()
        self._write_meta(recording=False)

    def _write_meta(self, *, recording: bool) -> None:
        payload = {
            **self._metadata,
            "num_samples": self._sample_count,
            "recording": recording,
        }
        path = self.run_dir / "meta.json"
        temporary_path = path.with_name(".meta.json.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
        temporary_path.replace(path)
