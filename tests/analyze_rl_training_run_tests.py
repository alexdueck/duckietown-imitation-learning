"""Tests for PPO training-report scenario and map grouping."""

from __future__ import annotations

import unittest

from analyze_rl_training_run import (
    axis_decimal_places,
    evaluation_map_statistics,
    format_axis_tick,
    scenario_identity,
    scenario_statistics,
    series_filter_controls,
    svg_line_chart,
    training_pose_identity,
    training_pose_statistics,
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


def training_episode(
    name: str,
    map_name: str,
    *,
    episode: int,
    episode_return: float,
    invalid: bool = False,
    start_type: str = "hard_pose",
) -> dict[str, str]:
    return {
        "start_type": start_type,
        "start_name": name,
        "map_name": map_name,
        "episode": str(episode),
        "step": str(episode * 100),
        "episode_return": str(episode_return),
        "episode_return_per_step": str(episode_return / 10.0),
        "episode_length": "10",
        "done_reason": "invalid-pose" if invalid else "time_limit",
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


class TrainingPoseStatisticsTests(unittest.TestCase):
    def test_first_and_last_windows_are_per_pose(self) -> None:
        rows = [
            training_episode("curve", "loop_empty", episode=1, episode_return=1.0),
            training_episode("curve", "loop_empty", episode=2, episode_return=3.0),
            training_episode(
                "curve",
                "loop_empty",
                episode=3,
                episode_return=7.0,
                invalid=True,
            ),
            training_episode(
                "curve",
                "loop_empty",
                episode=4,
                episode_return=9.0,
                invalid=True,
            ),
        ]

        statistics = training_pose_statistics(rows, window=2)[0]

        self.assertEqual(statistics["first"], 2.0)
        self.assertEqual(statistics["last"], 8.0)
        self.assertEqual(statistics["first_invalid_pct"], 0.0)
        self.assertEqual(statistics["last_invalid_pct"], 100.0)
        self.assertEqual(statistics["first_reward_per_step"], 0.2)
        self.assertEqual(statistics["last_reward_per_step"], 0.8)

    def test_random_starts_are_not_treated_as_repeatable_poses(self) -> None:
        row = training_episode(
            "",
            "loop_empty",
            episode=1,
            episode_return=1.0,
            start_type="random",
        )

        self.assertIsNone(training_pose_identity(row))
        self.assertEqual(training_pose_statistics([row], window=50), [])

    def test_same_pose_name_on_different_maps_remains_distinct(self) -> None:
        rows = [
            training_episode("curve", "loop_empty", episode=1, episode_return=1.0),
            training_episode("curve", "small_loop", episode=2, episode_return=2.0),
        ]

        statistics = training_pose_statistics(rows, window=50)

        self.assertEqual(len(statistics), 2)
        self.assertEqual(
            {item["map"] for item in statistics},
            {"loop_empty", "small_loop"},
        )


class InteractiveChartTests(unittest.TestCase):
    def test_axis_precision_follows_tick_spacing(self) -> None:
        self.assertEqual(axis_decimal_places(20.0), 0)
        self.assertEqual(axis_decimal_places(0.2), 1)
        self.assertEqual(axis_decimal_places(0.02), 2)
        self.assertEqual(format_axis_tick(100.39, 20.0), "100")
        self.assertEqual(format_axis_tick(0.1234, 0.01), "0.12")

    def test_chart_is_marked_interactive(self) -> None:
        chart = svg_line_chart(
            (("Pose one", ((1.0, 2.0), (2.0, 3.0)), "#123456"),),
        )

        self.assertIn("class='chart interactive-chart'", chart)
        self.assertIn("tabindex='0'", chart)
        self.assertIn("data-chart-x-min='1'", chart)
        self.assertIn("data-chart-y-min=", chart)
        self.assertIn("points='74.0,", chart)
        self.assertIn(" 956.0,", chart)
        self.assertIn("class='interaction-surface'", chart)

    def test_filter_controls_and_chart_share_group_and_series_keys(self) -> None:
        controls = series_filter_controls(
            "training-poses",
            (("pose-1", "Pose one", "#123456"),),
        )
        chart = svg_line_chart(
            (("Pose one", ((1.0, 2.0), (2.0, 3.0)), "#123456"),),
            filter_group="training-poses",
            series_keys=("pose-1",),
            show_legend=False,
        )

        self.assertIn("data-filter-control-group='training-poses'", controls)
        self.assertIn("data-filter-control-key='pose-1'", controls)
        self.assertIn("data-filter-group='training-poses'", chart)
        self.assertIn("data-filter-key='pose-1'", chart)


if __name__ == "__main__":
    unittest.main()
