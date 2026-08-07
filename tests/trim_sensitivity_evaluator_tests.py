"""Focused tests for the trim-sensitivity evaluator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from dt_utils.temporal_rl import FixedHistory
from evaluate_trim_sensitivity import (
    SUMMARY_METRICS,
    aggregate_rows,
    history_inputs,
    load_config,
)


class HistoryAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = FixedHistory(3, 2, 2)
        self.history.reset(torch.zeros(3, 2, 2))
        self.history.append([0.1, 0.2], torch.ones(3, 2, 2))
        self.history.append([0.3, 0.4], torch.full((3, 2, 2), 2.0))

    def test_native_history_is_unchanged(self) -> None:
        expected_observations, expected_actions = self.history.snapshot()

        observations, actions = history_inputs(
            self.history,
            "native",
            torch.device("cpu"),
        )

        self.assertTrue(torch.equal(observations[0], expected_observations))
        self.assertTrue(torch.equal(actions[0], expected_actions))

    def test_zero_action_history_preserves_frames(self) -> None:
        expected_observations, _ = self.history.snapshot()

        observations, actions = history_inputs(
            self.history,
            "zero_action_history",
            torch.device("cpu"),
        )

        self.assertTrue(torch.equal(observations[0], expected_observations))
        self.assertEqual(float(actions.abs().sum()), 0.0)

    def test_current_frame_mode_repeats_latest_frame(self) -> None:
        observations, actions = history_inputs(
            self.history,
            "current_frame_zero_actions",
            torch.device("cpu"),
        )

        self.assertTrue(torch.equal(observations[:, 0], observations[:, -1]))
        self.assertTrue(torch.equal(observations[:, 1], observations[:, -1]))
        self.assertEqual(float(actions.abs().sum()), 0.0)

    def test_reversed_action_history_reverses_only_actions(self) -> None:
        expected_observations, expected_actions = self.history.snapshot()

        observations, actions = history_inputs(
            self.history,
            "reversed_action_history",
            torch.device("cpu"),
        )

        self.assertTrue(torch.equal(observations[0], expected_observations))
        self.assertTrue(torch.equal(actions[0], torch.flip(expected_actions, dims=(0,))))


class AggregationTests(unittest.TestCase):
    @staticmethod
    def episode(trim: float, safe: int, reason: str, offset: float) -> dict:
        row = {
            "checkpoint": "policy",
            "history_mode": "native",
            "trim": trim,
            "safe": safe,
            "done_reason": reason,
        }
        row.update({metric: offset for metric in SUMMARY_METRICS})
        return row

    def test_summary_groups_by_trim_and_averages_metrics(self) -> None:
        rows = [
            self.episode(0.1, 1, "time_limit", 1.0),
            self.episode(0.1, 0, "invalid-pose", 3.0),
            self.episode(-0.1, 1, "time_limit", 5.0),
        ]

        summary = aggregate_rows(rows, ("checkpoint", "history_mode", "trim"))

        self.assertEqual([row["trim"] for row in summary], [-0.1, 0.1])
        positive = summary[1]
        self.assertEqual(positive["episodes"], 2)
        self.assertAlmostEqual(positive["safe_rate"], 0.5)
        self.assertAlmostEqual(positive["invalid_pose_rate"], 0.5)
        self.assertAlmostEqual(positive["mean_selected_return"], 2.0)
        self.assertFalse(math.isnan(positive["mean_mean_abs_lane_distance_m"]))


class ConfigurationTests(unittest.TestCase):
    @staticmethod
    def args(path: Path, **overrides) -> argparse.Namespace:
        values = {
            "config": path,
            "checkpoint": None,
            "trim": None,
            "pose_source": None,
            "pose_name": None,
            "history_mode": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_cli_values_override_json_lists_and_scalars(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {"name": "test", "path": str(Path(__file__).resolve())}
                        ],
                        "trims": [0.0],
                        "start_config": str(Path(__file__).resolve()),
                        "output_dir": str(root / "output"),
                        "max_steps": 500,
                    }
                )
            )

            config = load_config(
                self.args(config_path, trim=[-0.1, 0.1], max_steps=17)
            )

            self.assertEqual(config.trims, (-0.1, 0.1))
            self.assertEqual(config.max_steps, 17)

    def test_random_dynamics_is_rejected_for_explicit_trim_sweep(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {"name": "test", "path": str(Path(__file__).resolve())}
                        ],
                        "trims": [0.0],
                        "start_config": str(Path(__file__).resolve()),
                        "output_dir": str(root / "output"),
                        "dynamics_rand": True,
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "dynamics_rand must be false"):
                load_config(self.args(config_path))


if __name__ == "__main__":
    unittest.main()
