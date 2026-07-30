"""Tests for curated gym-duckietown start configurations."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dt_utils.gym_duckietown_start_config import (
    TrainingPose,
    append_evaluation_pose,
    append_training_pose,
    choose_training_start,
    load_pose_file,
    load_start_config,
)


class StartConfigTests(unittest.TestCase):
    @staticmethod
    def pose() -> TrainingPose:
        return TrainingPose(
            tile=(3, 5),
            position=(0.51, 0.0, 0.43),
            angle=0.70,
        )

    def test_append_creates_multi_map_pose_config(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "starts.json"

            training_index = append_training_pose(
                path, "loop_empty", self.pose()
            )
            evaluation_index = append_evaluation_pose(
                path, "small_loop", self.pose()
            )

            self.assertEqual(training_index, 1)
            self.assertEqual(evaluation_index, 1)
            payload = json.loads(path.read_text())
            self.assertEqual(
                set(payload),
                {"training_poses", "evaluation_poses"},
            )
            self.assertEqual(
                payload["training_poses"][0]["map_name"],
                "loop_empty",
            )
            self.assertEqual(
                payload["training_poses"][0]["name"],
                "loop_empty_01",
            )
            self.assertEqual(
                payload["evaluation_poses"][0]["map_name"],
                "small_loop",
            )
            self.assertEqual(
                payload["evaluation_poses"][0]["name"],
                "small_loop_01",
            )

    def test_generated_pose_names_are_unique_across_collections(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "starts.json"

            append_training_pose(path, "loop_empty", self.pose())
            append_evaluation_pose(path, "loop_empty", self.pose())
            append_training_pose(path, "loop_empty", self.pose())

            payload = json.loads(path.read_text())
            names = [
                pose["name"]
                for collection in ("training_poses", "evaluation_poses")
                for pose in payload[collection]
            ]
            self.assertEqual(
                names,
                ["loop_empty_01", "loop_empty_03", "loop_empty_02"],
            )

    def test_multi_map_pose_config_loads_and_selects_pose_map(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "starts.json"
            append_training_pose(path, "small_loop", self.pose())
            append_evaluation_pose(path, "loop_empty", self.pose())

            config = load_start_config(
                path,
                ("loop_empty", "small_loop"),
            )
            start = choose_training_start(
                config,
                hard_start_probability=1.0,
                rng=np.random.default_rng(1),
            )

            self.assertEqual(start.kind, "hard_pose")
            self.assertEqual(start.map_name, "small_loop")
            self.assertEqual(
                config.evaluation_poses[0].map_name,
                "loop_empty",
            )

    def test_legacy_single_map_config_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({
                "map_name": "loop_empty",
                "training_seeds": [1],
                "evaluation_seeds": [10042],
                "training_poses": [],
                "evaluation_poses": [],
            }))

            with self.assertRaisesRegex(
                ValueError,
                "map-specific training_poses and evaluation_poses only",
            ):
                load_start_config(path, ("loop_empty", "small_loop"))

    def test_single_pose_file_does_not_require_map_name(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            path.write_text(json.dumps(self.pose().as_json()))

            pose = load_pose_file(path)

            self.assertIsNone(pose.map_name)
            self.assertEqual(pose.tile, (3, 5))
            self.assertEqual(
                pose.position,
                (0.51, 0.0, 0.43),
            )

    def test_pose_map_must_be_configured_in_trainer(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "starts.json"
            append_training_pose(path, "zigzag_dists", self.pose())
            append_evaluation_pose(path, "loop_empty", self.pose())

            with self.assertRaisesRegex(ValueError, "configured maps"):
                load_start_config(path, ("loop_empty", "small_loop"))


if __name__ == "__main__":
    unittest.main()
