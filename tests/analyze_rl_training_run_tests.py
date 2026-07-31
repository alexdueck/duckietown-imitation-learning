"""Tests for PPO training-report scenario and map grouping."""

from __future__ import annotations

import unittest

from analyze_rl_training_run import (
    evaluation_map_statistics,
    scenario_identity,
    scenario_statistics,
)


def scenario(
    name: str,
    map_name: str,
    *,
    seed: int = 100_001,
    eval_index: int = 1,
    scenario_return: float = 1.0,
    terminated: int = 0,
    done_reason: str | None = None,
) -> dict[str, str]:
    return {
        "scenario_type": "eval_pose",
        "scenario_name": name,
        "scenario_seed": str(seed),
        "map_name": map_name,
        "start_x": "0.1",
        "start_y": "0.0",
        "start_z": "0.2",
        "start_angle": "0.3",
        "eval_index": str(eval_index),
        "train_step": str(eval_index * 100),
        "scenario_return": str(scenario_return),
        "scenario_steps": "150",
        "terminated": str(terminated),
        "done_reason": done_reason or ("invalid-pose" if terminated else "time_limit"),
    }


class ScenarioGroupingTests(unittest.TestCase):
    def test_pose_names_not_synthetic_seeds_define_scenarios(self) -> None:
        rows = [
            scenario("curve_a", "loop_empty", seed=42),
            scenario("curve_b", "loop_empty", seed=42),
        ]

        statistics = scenario_statistics(rows, window=10)

        self.assertEqual(
            [item["label"] for item in statistics],
            ["curve_a", "curve_b"],
        )

    def test_same_pose_name_on_different_maps_remains_distinct(self) -> None:
        rows = [
            scenario("curve", "loop_empty"),
            scenario("curve", "small_loop"),
        ]

        statistics = scenario_statistics(rows, window=10)

        self.assertEqual(len(statistics), 2)
        self.assertEqual(
            {item["map"] for item in statistics},
            {"loop_empty", "small_loop"},
        )

    def test_unnamed_pose_identity_uses_pose_not_seed(self) -> None:
        first = scenario("", "loop_empty", seed=1)
        second = scenario("", "loop_empty", seed=2, eval_index=2)

        self.assertEqual(scenario_identity(first), scenario_identity(second))

    def test_scenario_statistics_track_invalid_pose_explicitly(self) -> None:
        rows = [
            scenario("curve", "loop_empty", eval_index=1),
            scenario("curve", "loop_empty", eval_index=2, terminated=1),
        ]

        statistics = scenario_statistics(rows, window=1)[0]

        self.assertEqual(statistics["invalid_pct"], 50.0)
        self.assertEqual(statistics["first_invalid_pct"], 0.0)
        self.assertEqual(statistics["last_invalid_pct"], 100.0)
        self.assertEqual(statistics["first_complete"], 100.0)
        self.assertEqual(statistics["last_complete"], 0.0)

    def test_map_statistics_average_scenarios_within_each_eval(self) -> None:
        rows = [
            scenario("a", "loop_empty", eval_index=1, scenario_return=2.0),
            scenario("b", "loop_empty", eval_index=1, scenario_return=4.0),
            scenario("a", "loop_empty", eval_index=2, scenario_return=6.0),
            scenario(
                "b",
                "loop_empty",
                eval_index=2,
                scenario_return=8.0,
                terminated=1,
            ),
        ]

        statistics = evaluation_map_statistics(rows, window=1)[0]

        self.assertEqual(statistics["first_return"], 3.0)
        self.assertEqual(statistics["last_return"], 7.0)
        self.assertEqual(statistics["first_complete"], 100.0)
        self.assertEqual(statistics["last_complete"], 50.0)
        self.assertEqual(statistics["first_invalid"], 0.0)
        self.assertEqual(statistics["last_invalid"], 50.0)


if __name__ == "__main__":
    unittest.main()
