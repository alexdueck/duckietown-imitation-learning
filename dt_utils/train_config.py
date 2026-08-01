"""JSON-backed defaults for command-line training scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def explicit_cli_destinations(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> set[str]:
    """Return parser destinations explicitly selected on the command line."""
    actions_by_option = {
        option: action
        for action in parser._actions
        for option in action.option_strings
    }
    destinations = set()
    for argument in argv:
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        action = actions_by_option.get(option)
        if action is not None:
            destinations.add(action.dest)
    return destinations


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in training config {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"Training config {path} must contain a JSON object")
    return value


def apply_json_defaults(
    parser: argparse.ArgumentParser,
    values: dict[str, Any],
    *,
    aliases: dict[str, str] | None = None,
) -> set[str]:
    """Validate JSON values against argparse actions and install them as defaults."""
    aliases = aliases or {}
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in (argparse.SUPPRESS, "help")
    }
    normalized: dict[str, Any] = {}
    for original_key, raw_value in values.items():
        key = aliases.get(original_key, original_key)
        if key in normalized:
            raise ValueError(
                f"Training config specifies {key!r} more than once through aliases"
            )
        action = actions.get(key)
        if action is None or key == "train_config":
            available = ", ".join(sorted(k for k in actions if k != "train_config"))
            raise ValueError(
                f"Unknown training config option {original_key!r}. "
                f"Supported options: {available}"
            )
        normalized[key] = _coerce_json_value(action, raw_value, original_key)
    parser.set_defaults(**normalized)
    return set(normalized)


def _coerce_json_value(
    action: argparse.Action,
    value: Any,
    key: str,
) -> Any:
    if value is None:
        if action.default is None:
            return None
        raise ValueError(f"Training config option {key!r} may not be null")

    if isinstance(action, argparse.BooleanOptionalAction) or isinstance(
        action,
        (argparse._StoreTrueAction, argparse._StoreFalseAction),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"Training config option {key!r} must be a boolean")
        return value

    if action.dest == "map_names":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Training config option {key!r} must be a list of map names")
        converted: Any = value
    elif action.dest == "eval_seeds":
        if not isinstance(value, list) or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in value
        ):
            raise ValueError(f"Training config option {key!r} must be a list of integers")
        if not value:
            raise ValueError(f"Training config option {key!r} must not be empty")
        if len(set(value)) != len(value):
            raise ValueError(f"Training config option {key!r} must not contain duplicates")
        converted = tuple(value)
    else:
        converted = _apply_action_type(action, value, key)

    if action.choices is not None and converted not in action.choices:
        choices = ", ".join(repr(choice) for choice in action.choices)
        raise ValueError(
            f"Training config option {key!r} must be one of {choices}; got {converted!r}"
        )
    return converted


def _apply_action_type(action: argparse.Action, value: Any, key: str) -> Any:
    if action.type is None:
        return value
    try:
        return action.type(value)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as error:
        raise ValueError(
            f"Invalid value for training config option {key!r}: {value!r} ({error})"
        ) from error
