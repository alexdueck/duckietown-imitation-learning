"""Tests for live PPO evaluation start selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from live_eval_rl_policy_gym_duckietown import (
    cycle_pose_index,
    draw_distinct_reset_seed,
    load_evaluation_poses,
)


class LiveEvaluationStartTests(unittest.TestCase):
    def test_evaluation_config_loads_every_pose(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "starts.json"
            path.write_text(json.dumps({
                "training_poses": [],
                "evaluation_poses": [
                    {
                        "map_name": "loop_empty",
                        "tile": [1, 2],
                        "position": [0.1, 0.0, 0.2],
                        "angle": 0.3,
                        "name": "first",
                    },
                    {
                        "map_name": "small_loop",
                        "tile": [3, 4],
                        "position": [0.4, 0.0, 0.5],
                        "angle": 0.6,
                        "name": "second",
                    },
                ],
            }))
            args = argparse.Namespace(start_config=path, eval_pose_index=1)

            poses = load_evaluation_poses(args)

        self.assertEqual([pose.name for pose in poses], ["first", "second"])
        self.assertEqual(
            [pose.map_name for pose in poses],
            ["loop_empty", "small_loop"],
        )
        self.assertEqual(args.start_config, path.resolve())

    def test_pose_index_cycles_forward_and_backward(self) -> None:
        self.assertEqual(cycle_pose_index(2, 3, 1), 0)
        self.assertEqual(cycle_pose_index(0, 3, -1), 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            cycle_pose_index(0, 0, 1)

    def test_random_reset_seed_skips_previous_seed(self) -> None:
        class PredictableRng:
            def __init__(self) -> None:
                self.values = iter((17, 17, 23))

            def integers(self, low: int, high: int) -> int:
                del low, high
                return next(self.values)

        rng = PredictableRng()
        self.assertEqual(draw_distinct_reset_seed(rng, None), 17)
        self.assertEqual(draw_distinct_reset_seed(rng, 17), 23)


if __name__ == "__main__":
    unittest.main()
