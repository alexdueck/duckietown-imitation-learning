"""Tests for adaptive random and curated start sampling."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from dt_utils.adaptive_start_sampling import (
    AdaptiveStartSamplingSettings,
    AdaptiveStartSampler,
    HARD_POSE_DIFFICULTY_STRENGTH,
    HARD_START_PROBABILITY_MIN,
    EmaSuccessRate,
    episode_was_successful,
    training_pose_key,
)
from dt_utils.gym_duckietown_start_config import (
    StartConfig,
    TrainingPose,
    TrainingStart,
)


def pose(name: str, map_name: str = "loop_empty") -> TrainingPose:
    return TrainingPose(
        tile=(1, 2),
        position=(0.1, 0.0, 0.2),
        angle=0.3,
        name=name,
        map_name=map_name,
    )


def config(*poses: TrainingPose) -> StartConfig:
    return StartConfig(
        source_path=Path("starts.json"),
        training_poses=tuple(poses),
        evaluation_poses=(),
    )


class AdaptiveStartSamplingTests(unittest.TestCase):
    def test_success_requires_reaching_an_episode_limit(self) -> None:
        self.assertTrue(episode_was_successful("time_limit"))
        self.assertTrue(episode_was_successful("max-steps-reached"))
        self.assertFalse(episode_was_successful("invalid-pose"))
        self.assertFalse(episode_was_successful("terminated"))

    def test_ema_uses_configured_lambda(self) -> None:
        statistic = EmaSuccessRate(ema_lambda=0.25)
        statistic.update(True)
        self.assertAlmostEqual(
            statistic.value,
            0.5 * (1.0 - 0.25) + 0.25,
        )
        self.assertEqual(statistic.observations, 1)

    def test_equal_probability_bounds_force_only_hard_starts(self) -> None:
        hard_pose = pose("curve")
        sampler = AdaptiveStartSampler(
            config(hard_pose),
            np.random.default_rng(1),
            initial_hard_start_probability=0.5,
            settings=AdaptiveStartSamplingSettings(
                hard_probability_min=1.0,
                hard_probability_max=1.0,
            ),
        )

        self.assertEqual(sampler.hard_start_probability(), 1.0)
        self.assertTrue(all(sampler.choose().kind == "hard_pose" for _ in range(20)))

    def test_initial_probability_is_used_until_both_types_are_observed(self) -> None:
        sampler = AdaptiveStartSampler(
            config(pose("curve")),
            np.random.default_rng(1),
            initial_hard_start_probability=0.6,
        )
        sampler.update(TrainingStart(kind="random", seed=1), success=True)
        self.assertEqual(sampler.hard_start_probability(), 0.6)

    def test_equal_success_rates_use_minimum_hard_probability(self) -> None:
        hard_pose = pose("curve")
        sampler = AdaptiveStartSampler(
            config(hard_pose),
            np.random.default_rng(1),
            initial_hard_start_probability=0.5,
        )
        sampler.update(TrainingStart(kind="random", seed=1), success=False)
        sampler.update(
            TrainingStart(kind="hard_pose", seed=2, pose=hard_pose),
            success=False,
        )
        self.assertAlmostEqual(
            sampler.hard_start_probability(),
            HARD_START_PROBABILITY_MIN,
        )

    def test_harder_curated_starts_increase_hard_probability(self) -> None:
        hard_pose = pose("curve")
        sampler = AdaptiveStartSampler(
            config(hard_pose),
            np.random.default_rng(1),
            initial_hard_start_probability=0.5,
        )
        for _ in range(10):
            sampler.update(TrainingStart(kind="random", seed=1), success=True)
            sampler.update(
                TrainingStart(kind="hard_pose", seed=2, pose=hard_pose),
                success=False,
            )
        self.assertGreater(
            sampler.hard_start_probability(),
            HARD_START_PROBABILITY_MIN,
        )

    def test_difficult_pose_uses_configured_raw_weight_multiplier(self) -> None:
        easy = pose("easy")
        difficult = pose("difficult")
        sampler = AdaptiveStartSampler(
            config(easy, difficult),
            np.random.default_rng(1),
            initial_hard_start_probability=0.5,
        )
        sampler.pose_success[training_pose_key(easy)].value = 1.0
        sampler.pose_success[training_pose_key(difficult)].value = 0.0
        probabilities = sampler.hard_pose_probabilities()
        self.assertAlmostEqual(
            probabilities[training_pose_key(difficult)],
            (1.0 + HARD_POSE_DIFFICULTY_STRENGTH)
            * probabilities[training_pose_key(easy)],
        )

    def test_resume_restores_known_poses_and_initializes_new_poses(self) -> None:
        old_pose = pose("old")
        removed_pose = pose("removed")
        previous = AdaptiveStartSampler(
            config(old_pose, removed_pose),
            np.random.default_rng(7),
            initial_hard_start_probability=0.5,
        )
        previous.update(
            TrainingStart(kind="hard_pose", seed=2, pose=old_pose),
            success=True,
        )
        state = previous.state_dict()

        new_pose = pose("new")
        resumed = AdaptiveStartSampler(
            config(old_pose, new_pose),
            np.random.default_rng(99),
            initial_hard_start_probability=0.5,
        )
        changes = resumed.load_state_dict(state)

        self.assertEqual(changes["restored"], [training_pose_key(old_pose)])
        self.assertEqual(changes["added"], [training_pose_key(new_pose)])
        self.assertEqual(changes["removed"], [training_pose_key(removed_pose)])
        self.assertEqual(
            resumed.pose_success[training_pose_key(old_pose)].observations,
            1,
        )
        self.assertEqual(
            resumed.pose_success[training_pose_key(new_pose)].observations,
            0,
        )

    def test_resume_accepts_changed_sampling_settings(self) -> None:
        curve = pose("curve")
        previous = AdaptiveStartSampler(
            config(curve),
            np.random.default_rng(7),
            initial_hard_start_probability=0.2,
        )
        previous.update(
            TrainingStart(kind="hard_pose", seed=2, pose=curve),
            success=True,
        )
        state = previous.state_dict()
        state["configuration"]["hard_pose_difficulty_strength"] = 123.0

        resumed = AdaptiveStartSampler(
            config(curve),
            np.random.default_rng(99),
            initial_hard_start_probability=0.8,
        )
        changes = resumed.load_state_dict(state)

        self.assertIn(
            "hard_pose_difficulty_strength",
            changes["configuration_changes"],
        )
        self.assertEqual(resumed.initial_hard_start_probability, 0.8)
        self.assertEqual(
            resumed.pose_success[training_pose_key(curve)].observations,
            1,
        )

    def test_state_restores_rng_sequence(self) -> None:
        starts = config(pose("curve"))
        sampler = AdaptiveStartSampler(
            starts,
            np.random.default_rng(3),
            initial_hard_start_probability=1.0,
        )
        state = sampler.state_dict()
        expected = sampler.choose()

        resumed = AdaptiveStartSampler(
            starts,
            np.random.default_rng(999),
            initial_hard_start_probability=1.0,
        )
        resumed.load_state_dict(state)
        actual = resumed.choose()
        self.assertEqual(actual.kind, expected.kind)
        self.assertEqual(actual.seed, expected.seed)
        self.assertEqual(actual.name, expected.name)


if __name__ == "__main__":
    unittest.main()
