"""Configuration helpers for gym-duckietown's built-in randomizer."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from typing import Any


LEGACY_DEFAULT_RANDOMIZATION_CONFIG = "gym-duckietown defaults"
SUPPORTED_DISTRIBUTIONS = {
    "int": ({"low", "high"}, {"size"}),
    "normal": ({"loc", "scale"}, {"size"}),
    "uniform": ({"low", "high"}, {"size"}),
}


def parse_randomization_config(value: Any) -> dict[str, dict[str, Any]]:
    """Parse and validate randomizer overrides from JSON config or the CLI."""
    if value is None or value == LEGACY_DEFAULT_RANDOMIZATION_CONFIG:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise argparse.ArgumentTypeError(
                f"randomization config must be a JSON object: {error}"
            ) from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("randomization config must be a JSON object")

    normalized: dict[str, dict[str, Any]] = {}
    for name, definition in value.items():
        if not isinstance(name, str) or not name:
            raise argparse.ArgumentTypeError(
                "randomization parameter names must be non-empty strings"
            )
        if not isinstance(definition, dict):
            raise argparse.ArgumentTypeError(
                f"randomization parameter {name!r} must be a JSON object"
            )
        distribution_type = definition.get("type")
        if distribution_type not in SUPPORTED_DISTRIBUTIONS:
            supported = ", ".join(sorted(SUPPORTED_DISTRIBUTIONS))
            raise argparse.ArgumentTypeError(
                f"randomization parameter {name!r} has unsupported type "
                f"{distribution_type!r}; expected one of {supported}"
            )
        required, optional = SUPPORTED_DISTRIBUTIONS[distribution_type]
        keys = set(definition) - {"type"}
        missing = required - keys
        unexpected = keys - required - optional
        if missing:
            raise argparse.ArgumentTypeError(
                f"randomization parameter {name!r} is missing: "
                + ", ".join(sorted(missing))
            )
        if unexpected:
            raise argparse.ArgumentTypeError(
                f"randomization parameter {name!r} has unknown fields: "
                + ", ".join(sorted(unexpected))
            )
        if "size" in definition:
            _validate_size(definition["size"], name)
        if distribution_type == "normal":
            _validate_positive_numbers(definition["scale"], f"{name}.scale")
        normalized[name] = deepcopy(definition)
    return normalized


def apply_randomization_config(
    env,
    overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge validated overrides into an environment's existing randomizer."""
    raw_env = getattr(env, "unwrapped", env)
    randomizer = getattr(raw_env, "randomizer", None)
    if randomizer is None:
        raise ValueError("Environment does not expose gym-duckietown's randomizer")

    available = set(getattr(randomizer, "randomization_config", {})) | set(
        getattr(randomizer, "default_config", {})
    )
    unknown = set(overrides) - available
    if unknown:
        raise ValueError(
            "Unknown gym-duckietown randomization parameter(s): "
            + ", ".join(sorted(unknown))
        )

    randomizer.randomization_config.update(deepcopy(overrides))
    randomizer.keys = sorted(
        set(randomizer.randomization_config) | set(randomizer.default_config)
    )
    return deepcopy(randomizer.randomization_config)


def _validate_size(value: Any, name: str) -> None:
    values = value if isinstance(value, list) else [value]
    if not values or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in values
    ):
        raise argparse.ArgumentTypeError(
            f"randomization parameter {name!r} size must contain positive integers"
        )


def _validate_positive_numbers(value: Any, name: str) -> None:
    values = value if isinstance(value, list) else [value]
    if not values or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or float(item) <= 0.0
        for item in values
    ):
        raise argparse.ArgumentTypeError(f"{name} must contain positive finite numbers")
