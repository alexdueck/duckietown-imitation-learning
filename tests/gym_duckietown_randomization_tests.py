"""Tests for configurable gym-duckietown randomization distributions."""

from __future__ import annotations

import argparse
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from dt_utils.gym_duckietown_randomization import (
    apply_randomization_config,
    parse_randomization_config,
)
import train_rl_ppo_gym_duckietown as trainer


UNIFORM_TRIM = {
    "trim": {
        "type": "uniform",
        "low": -0.1,
        "high": 0.1,
    }
}


class FakeRandomizer:
    def __init__(self) -> None:
        self.randomization_config = {
            "trim": {"type": "normal", "loc": 0.0, "scale": 0.02}
        }
        self.default_config = dict(self.randomization_config)
        self.keys = ["trim"]


class FakeSimulator:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.randomizer = FakeRandomizer()

    @property
    def unwrapped(self):
        return self


class RandomizationConfigTests(unittest.TestCase):
    def test_uniform_trim_config_is_validated(self) -> None:
        self.assertEqual(parse_randomization_config(UNIFORM_TRIM), UNIFORM_TRIM)
        self.assertEqual(
            parse_randomization_config("gym-duckietown defaults"),
            {},
        )

    def test_invalid_distribution_is_rejected(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_randomization_config({"trim": {"type": "triangular"}})

    def test_overrides_are_merged_without_mutating_input(self) -> None:
        env = FakeSimulator()
        effective = apply_randomization_config(env, UNIFORM_TRIM)

        self.assertEqual(env.randomizer.randomization_config["trim"], UNIFORM_TRIM["trim"])
        self.assertEqual(effective["trim"], UNIFORM_TRIM["trim"])
        self.assertIsNot(env.randomizer.randomization_config["trim"], UNIFORM_TRIM["trim"])

    def test_make_env_resets_after_applying_override(self) -> None:
        simulator_module = ModuleType("gym_duckietown.simulator")
        simulator_module.DEFAULT_ROBOT_SPEED = 1.2
        simulator_module.Simulator = FakeSimulator
        args = SimpleNamespace(
            seed=42,
            map_name="loop_empty",
            simulator_max_steps=None,
            max_episode_steps=100,
            robot_speed=None,
            domain_rand=True,
            dynamics_rand=True,
            camera_rand=True,
            frame_rate=30,
            frame_skip=1,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4.0,
            distortion=True,
            randomization_config=UNIFORM_TRIM,
        )

        with patch.dict(sys.modules, {"gym_duckietown.simulator": simulator_module}):
            with patch.object(trainer, "patch_duckietown_world_dynamics"):
                with patch.object(trainer, "reset_raw", return_value=(None, {})) as reset:
                    env = trainer.make_env(args, seed=7)

        self.assertEqual(env.randomizer.randomization_config["trim"], UNIFORM_TRIM["trim"])
        self.assertTrue(env.kwargs["dynamics_rand"])
        reset.assert_called_once_with(env, seed=7)

    def test_resume_prefers_explicit_randomization_config(self) -> None:
        args = trainer.build_arg_parser().parse_args([])
        args.map_names = trainer.DEFAULT_MAP_NAMES
        args.map_name = args.map_names[0]
        args.randomization_config = UNIFORM_TRIM
        checkpoint_config = {
            "randomization_config": {
                "trim": {"type": "normal", "loc": 0.0, "scale": 0.02}
            }
        }

        restored = trainer.restore_resume_configuration(
            args,
            checkpoint_config,
            {"randomization_config"},
        )

        self.assertEqual(args.randomization_config, UNIFORM_TRIM)
        self.assertNotIn("randomization_config", restored)


if __name__ == "__main__":
    unittest.main()
