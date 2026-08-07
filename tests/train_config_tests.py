"""Tests for JSON training configuration and CLI precedence."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dt_utils.train_config import apply_json_defaults
import train_rl_ppo_gym_duckietown as trainer


class TrainConfigTests(unittest.TestCase):
    def test_template_covers_every_configurable_parser_destination(self) -> None:
        parser = trainer.build_arg_parser()
        expected = {
            action.dest
            for action in parser._actions
            if action.dest not in ("help", "train_config")
        }
        template_path = Path("template_configs/train_config.template.json")
        template = json.loads(template_path.read_text())
        self.assertEqual(set(template), expected)

        configured = apply_json_defaults(
            parser,
            template,
            aliases={"map_name": "map_names"},
        )
        self.assertEqual(configured, expected)

        defaults = vars(trainer.build_arg_parser().parse_args([]))
        defaults.pop("train_config")
        defaults["map_names"] = list(trainer.DEFAULT_MAP_NAMES)
        for key, template_value in template.items():
            default_value = defaults[key]
            if key == "randomization_config":
                self.assertEqual(default_value, {})
                self.assertEqual(template_value["trim"]["type"], "uniform")
                continue

            if isinstance(default_value, Path):
                self.assertEqual(Path(template_value).expanduser(), default_value)
            elif isinstance(default_value, tuple):
                self.assertEqual(template_value, list(default_value))
            else:
                self.assertEqual(template_value, default_value)

    def test_explicit_cli_options_override_json_values(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(json.dumps({
                "domain_rand": True,
                "epochs": 8,
                "policy_lr": 3e-6,
                "map_names": ["small_loop", "zigzag_dists"],
            }))
            argv = [
                "train_rl_ppo_gym_duckietown.py",
                "--train-config",
                str(path),
                "--epochs",
                "2",
                "--map-name",
                "loop_empty",
                "--no-domain-rand",
            ]
            with patch.object(sys, "argv", argv):
                args = trainer.parse_args()

        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.policy_lr, 3e-6)
        self.assertEqual(args.map_names, ("loop_empty",))
        self.assertFalse(args.domain_rand)
        self.assertEqual(args.configuration_sources["domain_rand"], "command_line")
        self.assertEqual(args.configuration_sources["epochs"], "command_line")
        self.assertEqual(args.configuration_sources["policy_lr"], "train_config")
        self.assertEqual(args.configuration_sources["map_names"], "command_line")
        self.assertEqual(args.train_config, path.resolve())

    def test_existing_default_config_is_loaded_automatically(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train_config.json"
            path.write_text(json.dumps({"epochs": 7}))
            with patch.object(trainer, "DEFAULT_TRAIN_CONFIG_PATH", path):
                with patch.object(sys, "argv", ["trainer.py"]):
                    args = trainer.parse_args()

        self.assertEqual(args.epochs, 7)
        self.assertEqual(args.train_config, path.resolve())
        self.assertEqual(args.configuration_sources["epochs"], "train_config")

    def test_default_config_is_optional_but_explicit_missing_config_fails(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with patch.object(trainer, "DEFAULT_TRAIN_CONFIG_PATH", missing):
                with patch.object(sys, "argv", ["trainer.py"]):
                    args = trainer.parse_args()
            self.assertIsNone(args.train_config)

            with patch.object(
                sys,
                "argv",
                ["trainer.py", "--train-config", str(missing)],
            ):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        trainer.parse_args()

    def test_invalid_or_unknown_json_option_is_rejected(self) -> None:
        parser = trainer.build_arg_parser()
        with self.assertRaisesRegex(ValueError, "Unknown training config option"):
            apply_json_defaults(parser, {"mystery_option": 1})
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            apply_json_defaults(parser, {"domain_rand": "yes"})


if __name__ == "__main__":
    unittest.main()
