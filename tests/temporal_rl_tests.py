"""Tests for fixed observation/action histories used by PPO."""

from __future__ import annotations

import argparse
import json
import unittest
from types import SimpleNamespace

import torch
import numpy as np

from dt_utils.temporal_rl import (
    FixedHistory,
    TEMPORAL_HEAD_MODE_MLP,
    TEMPORAL_HEAD_MODE_RESIDUAL,
    TemporalTanhGaussianPolicy,
    TemporalValueNetwork,
    validate_history_lengths,
)
from train_rl_ppo_gym_duckietown import (
    DEFAULT_MAP_NAMES,
    assign_evaluation_maps,
    choose_episode_map,
    environment_camera_calibration,
    normalize_map_names,
    previous_camera_calibration_history,
    restore_resume_configuration,
)


class FixedHistoryTests(unittest.TestCase):
    def test_initialization_repeats_observation_and_zeros_actions(self) -> None:
        first = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
        history = FixedHistory(5, 4, 2)
        history.reset(first)

        observations, actions = history.snapshot()

        self.assertEqual(tuple(observations.shape), (5, 3, 8, 8))
        self.assertEqual(tuple(actions.shape), (4, 2))
        self.assertTrue(torch.equal(observations[0], observations[-1]))
        self.assertEqual(float(actions.abs().sum()), 0.0)

    def test_append_shifts_both_histories_without_gradients(self) -> None:
        history = FixedHistory(5, 4, 2)
        history.reset(torch.zeros(3, 8, 8))
        action = torch.tensor([0.25, -0.5], requires_grad=True)
        next_observation = torch.ones(3, 8, 8, requires_grad=True)

        history.append(action, next_observation)
        observations, actions = history.snapshot()

        self.assertTrue(torch.equal(observations[-1], torch.ones(3, 8, 8)))
        self.assertTrue(torch.allclose(actions[-1], torch.tensor([0.25, -0.5])))
        self.assertFalse(observations.requires_grad)
        self.assertFalse(actions.requires_grad)

    def test_histories_are_isolated_per_environment(self) -> None:
        first = FixedHistory(5, 4, 2)
        second = FixedHistory(5, 4, 2)
        first.reset(torch.zeros(3, 8, 8))
        second.reset(torch.full((3, 8, 8), 2.0))
        first.append([1.0, -1.0], torch.ones(3, 8, 8))

        first_observations, first_actions = first.snapshot()
        second_observations, second_actions = second.snapshot()

        self.assertEqual(float(first_observations[-1].mean()), 1.0)
        self.assertEqual(float(second_observations[-1].mean()), 2.0)
        self.assertEqual(float(first_actions[-1].abs().sum()), 2.0)
        self.assertEqual(float(second_actions.abs().sum()), 0.0)

    def test_invalid_history_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal"):
            validate_history_lengths(5, 3)


class CameraCalibrationMetadataTests(unittest.TestCase):
    def test_calibration_is_exact_and_json_serializable(self) -> None:
        camera_model = SimpleNamespace(
            camera_matrix=[
                [300.0, 0.0, 310.0],
                [0.0, 301.0, 230.0],
                [0.0, 0.0, 1.0],
            ],
            distortion_coefs=[[-0.2, 0.03, 0.001, -0.001, 0.0]],
            new_camera_matrix=[
                [290.0, 0.0, 305.0],
                [0.0, 291.0, 225.0],
                [0.0, 0.0, 1.0],
            ],
            W=640,
            H=480,
        )
        env = SimpleNamespace(
            unwrapped=SimpleNamespace(
                camera_model=camera_model,
                distortion=True,
                camera_rand=True,
            )
        )

        calibration = environment_camera_calibration(env)

        self.assertTrue(calibration["available"])
        self.assertEqual(calibration["camera_matrix"][0][0], 300.0)
        self.assertEqual(calibration["distortion_coefs"][0][0], -0.2)
        self.assertEqual(calibration["calibration_width"], 640)
        json.dumps(calibration)

    def test_previous_calibration_history_is_copied(self) -> None:
        calibration = {"available": True, "camera_matrix": [[1.0]]}
        metadata = {"environment": {"camera_calibration": calibration}}

        history = previous_camera_calibration_history(metadata)
        history[0]["camera_matrix"][0][0] = 2.0

        self.assertEqual(calibration["camera_matrix"][0][0], 1.0)


class TemporalNetworkTests(unittest.TestCase):
    def test_policy_and_value_forward_shapes(self) -> None:
        policy = TemporalTanhGaussianPolicy(
            "mobilenet_v3_small",
            action_dim=2,
            pretrained=False,
            observation_history_length=5,
            action_history_length=4,
            temporal_hidden_dim=32,
        ).eval()
        value = TemporalValueNetwork(
            "mobilenet_v3_small",
            action_dim=2,
            pretrained=False,
            observation_history_length=5,
            action_history_length=4,
            temporal_hidden_dim=32,
        ).eval()
        self.assertEqual(policy.temporal_head_mode, TEMPORAL_HEAD_MODE_MLP)
        self.assertIsNone(policy.mean)
        self.assertIsNone(value.value)
        self.assertGreater(torch.count_nonzero(policy.temporal_head[-1].weight), 0)
        observations = torch.randn(2, 5, 3, 64, 64)
        actions = torch.randn(2, 4, 2)

        with torch.no_grad():
            mean, log_std = policy(observations, actions)
            values = value(observations, actions)

        self.assertEqual(tuple(mean.shape), (2, 2))
        self.assertEqual(tuple(log_std.shape), (2, 2))
        self.assertEqual(tuple(values.shape), (2,))

    def test_direct_temporal_mlp_uses_history(self) -> None:
        policy = TemporalTanhGaussianPolicy(
            "mobilenet_v3_small", 2, False, 5, 4, 16
        ).eval()
        observations = torch.zeros(2, 5, 3, 64, 64)
        observations[1, :-1] = 1.0
        actions = torch.zeros(2, 4, 2)
        actions[1] = 0.5

        with torch.no_grad():
            mean, _ = policy(observations, actions)

        self.assertFalse(torch.allclose(mean[0], mean[1]))

    def test_residual_mode_starts_as_current_frame_model(self) -> None:
        policy = TemporalTanhGaussianPolicy(
            "mobilenet_v3_small",
            2,
            False,
            5,
            4,
            16,
            temporal_head_mode=TEMPORAL_HEAD_MODE_RESIDUAL,
        ).eval()
        observations = torch.randn(2, 5, 3, 64, 64)
        observations[1, -1] = observations[0, -1]
        actions = torch.randn(2, 4, 2)

        with torch.no_grad():
            mean, _ = policy(observations, actions)

        self.assertIsNotNone(policy.mean)
        self.assertEqual(torch.count_nonzero(policy.temporal_head[-1].weight), 0)
        self.assertTrue(torch.allclose(mean[0], mean[1]))

    def test_incompatible_history_checkpoint_is_rejected(self) -> None:
        five_frame_policy = TemporalTanhGaussianPolicy(
            "mobilenet_v3_small", 2, False, 5, 4, 16
        )
        three_frame_policy = TemporalTanhGaussianPolicy(
            "mobilenet_v3_small", 2, False, 3, 2, 16
        )

        with self.assertRaises(RuntimeError):
            three_frame_policy.load_state_dict(five_frame_policy.state_dict())

    def test_direct_and_residual_checkpoints_are_incompatible(self) -> None:
        direct_policy = TemporalTanhGaussianPolicy(
            "mobilenet_v3_small", 2, False, 5, 4, 16
        )
        residual_policy = TemporalTanhGaussianPolicy(
            "mobilenet_v3_small",
            2,
            False,
            5,
            4,
            16,
            temporal_head_mode=TEMPORAL_HEAD_MODE_RESIDUAL,
        )

        with self.assertRaises(RuntimeError):
            residual_policy.load_state_dict(direct_policy.state_dict())


class ResumeConfigurationTests(unittest.TestCase):
    @staticmethod
    def args() -> argparse.Namespace:
        return argparse.Namespace(
            map_name="loop_empty",
            map_names=DEFAULT_MAP_NAMES,
            model="mobilenet_v3_small",
            action_mode="wheel",
            fixed_throttle=None,
            max_throttle=1.0,
            max_steering=0.5,
            observation_history_length=5,
            action_history_length=4,
            temporal_hidden_dim=256,
            temporal_head_mode=TEMPORAL_HEAD_MODE_MLP,
            domain_rand=True,
            dynamics_rand=True,
            camera_rand=True,
            distortion=True,
            image_size=224,
            crop_y_start=0,
            source_observation_channel_order="rgb",
        )

    def test_legacy_checkpoint_restores_stateless_history(self) -> None:
        args = self.args()
        restore_resume_configuration(args, {"model": "mobilenet_v3_small"}, set())
        self.assertEqual(args.observation_history_length, 1)
        self.assertEqual(args.action_history_length, 0)
        self.assertEqual(args.temporal_head_mode, TEMPORAL_HEAD_MODE_RESIDUAL)
        self.assertFalse(args.domain_rand)
        self.assertEqual(args.map_names, ("loop_empty",))
        self.assertEqual(args.map_name, "loop_empty")

    def test_explicit_incompatible_history_is_rejected(self) -> None:
        args = self.args()
        checkpoint = {
            "observation_history_length": 3,
            "action_history_length": 2,
        }
        with self.assertRaisesRegex(ValueError, "incompatible"):
            restore_resume_configuration(
                args,
                checkpoint,
                {"observation_history_length"},
            )


    def test_explicit_incompatible_temporal_mode_is_rejected(self) -> None:
        args = self.args()
        checkpoint = {
            "observation_history_length": 5,
            "action_history_length": 4,
            "temporal_head_mode": TEMPORAL_HEAD_MODE_RESIDUAL,
        }
        with self.assertRaisesRegex(ValueError, "incompatible"):
            restore_resume_configuration(
                args,
                checkpoint,
                {"temporal_head_mode"},
            )

    def test_explicit_map_names_override_checkpoint_maps(self) -> None:
        args = self.args()
        args.map_names = ("small_loop", "zigzag_dists")
        args.map_name = "small_loop"

        restore_resume_configuration(
            args,
            {"map_names": ("loop_empty",)},
            {"map_names"},
        )

        self.assertEqual(args.map_names, ("small_loop", "zigzag_dists"))


class MultiMapConfigurationTests(unittest.TestCase):
    def test_default_maps_are_unique(self) -> None:
        self.assertEqual(
            DEFAULT_MAP_NAMES,
            ("loop_empty", "small_loop", "zigzag_dists"),
        )
        self.assertEqual(normalize_map_names(DEFAULT_MAP_NAMES), DEFAULT_MAP_NAMES)

    def test_duplicate_maps_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            normalize_map_names(("loop_empty", "loop_empty"))

    def test_map_sampling_is_reproducible_and_can_be_forced(self) -> None:
        first_rng = np.random.default_rng(123)
        second_rng = np.random.default_rng(123)
        first = [choose_episode_map(DEFAULT_MAP_NAMES, first_rng) for _ in range(20)]
        second = [choose_episode_map(DEFAULT_MAP_NAMES, second_rng) for _ in range(20)]

        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)
        self.assertEqual(
            choose_episode_map(DEFAULT_MAP_NAMES, first_rng, "small_loop"),
            "small_loop",
        )

    def test_evaluation_assigns_one_reproducible_map_per_seed(self) -> None:
        evaluation_seeds = (10042, 10043, 10044, 10045)

        first = assign_evaluation_maps(DEFAULT_MAP_NAMES, evaluation_seeds, 42)
        second = assign_evaluation_maps(DEFAULT_MAP_NAMES, evaluation_seeds, 42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(evaluation_seeds))
        self.assertEqual(tuple(seed for seed, _ in first), evaluation_seeds)
        self.assertTrue(
            all(map_name in DEFAULT_MAP_NAMES for _, map_name in first)
        )


if __name__ == "__main__":
    unittest.main()
