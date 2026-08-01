# Configurations and Paths

## Artifact Root

`dt_utils/duckietown_paths.py` defines:

```text
~/duckietown/
|-- data/
|   |-- imitation_learning/
|   |   |-- train/
|   |   `-- val/
|   |-- physical_control/
|   `-- evaluations/
|       |-- il_gym_duckietown/
|       |-- rl_gym_duckietown/
|       `-- screenshots/
`-- checkpoints/
    |-- imitation_learning/
    |-- rl_ppo_duckiematrix/
    `-- rl_ppo_gym_duckietown/
```

Use CLI output-directory flags to override these locations for a particular
run.

## Training Configuration

The PPO trainer accepts every training option either on the command line or in
a JSON configuration. Create the local default from the complete versioned
template:

```bash
cp configs/train_config.template.json configs/train_config.json
```

`train_rl_ppo_gym_duckietown.py` loads `configs/train_config.json`
automatically when it exists. A different file can be selected with:

```bash
python train_rl_ppo_gym_duckietown.py \
  --train-config configs/my_experiment.json
```

Values are resolved in this order:

```text
explicit command-line option > selected JSON config > built-in default
```

For example, `--epochs 2` overrides `"epochs": 4` in the JSON. Repeated
`--map-name` options replace `map_names` from the JSON rather than extending
it. Boolean values use JSON booleans; command-line `--no-domain-rand` and
similar negative flags override `true` from the file.

The JSON uses argparse destination names in `snake_case`. Repeated maps and
evaluation seeds are represented as JSON lists. `--train-config` itself is the
only bootstrap option not stored inside the selected file, avoiding configs
that recursively select other configs. Unknown options and invalid types are
rejected before the environment is created.

The default file is optional: when it does not exist, built-in defaults are
used. A path supplied explicitly with `--train-config` must exist. The complete
template intentionally contains every supported setting, so it doubles as a
fairly large but honest menu.

## Start Configuration

Create a local file from the versioned template:

```bash
cp configs/gym_duckietown_start_poses.json.template \
   configs/gym_duckietown_start_poses.json
```

Schema:

```json
{
  "training_poses": [
    {
      "name": "optional_name",
      "map_name": "loop_empty",
      "tile": [3, 5],
      "position": [0.51, 0.0, 0.43],
      "angle": 0.70
    }
  ],
  "evaluation_poses": [
    {
      "map_name": "small_loop",
      "tile": [1, 2],
      "position": [0.15, 0.0, 0.09],
      "angle": 1.44
    }
  ]
}
```

Rules:

- Every pose identifies its own `map_name`, which must be configured in the trainer.
- A trainer config needs at least one training pose and one evaluation pose.
- `name` is optional.
- `tile` contains integer map tile coordinates.
- `position` is local to that tile.
- `angle` is simulator yaw in radians.
- Unknown fields are rejected.

When this config is supplied to the trainer, configured evaluation scenarios
come exclusively from `evaluation_poses`; `--eval-seeds` is not used.

## Training Start Sampling

When a start config is present, the trainer adapts two distributions after
every completed episode:

1. Hard versus random starts are weighted from their relative EMA failure
   rates.
2. Hard poses are weighted from their individual EMA failure rates.

Both use `lambda=0.15` by default. `--hard-start-probability` is only the
cold-start probability until hard and random starts have each produced an
outcome. The EMA and adaptive probability bounds are regular training options,
including `--hard-start-probability-min` and
`--hard-start-probability-max`. Setting both bounds to `1.0` forces curated
starts immediately and after sampler statistics are restored from a
checkpoint. A configured pose receives a fresh reset seed for the simulator's
other randomized reset state.

See [Adaptive start sampling](../methodology/adaptive-start-sampling.md) for
the formulas, constants, success definition, logging, and resume behavior.

## Browsing Configured Poses

The manual viewer reads the same multi-map config as the trainer:

```bash
python manual_control_gym_duckietown.py \
  --map-name loop_empty \
  --start-poses configs/gym_duckietown_start_poses.json
```

Only poses matching `--map-name` are available. Press `N` or `Shift+N` to
select the next or previous pose, and press `G` to enter a pose name directly.
The sidebar shows the active pose name and whether it came from
`training_poses` or `evaluation_poses`. Ordinary resets return to the active
pose.

## Capturing a Pose

Start the manual viewer with a new or existing local start config:

```bash
python manual_control_gym_duckietown.py \
  --start-config configs/gym_duckietown_start_poses.json
```

Drive to a valid location. Press `P` to append a training pose or `Shift+P` to
append an evaluation pose. Missing files are created automatically, and each
captured pose records the viewer's current map.

Current lane metrics are useful diagnostics but are not required to reproduce
the pose; they are recomputed from position, angle, and map geometry.

## Git Behavior

The repository ignores:

```text
configs/*.json
```

It tracks:

```text
configs/*.json.template
configs/*.template.json
```

This keeps machine-specific experiment selections local while preserving an
executable schema example.

## Run Configuration

Every PPO run writes `config.json` and stores the same training configuration
inside each checkpoint. It includes model, environment, action, reward,
preprocessing, PPO, evaluation, start, seed, and device settings.

The saved values are fully resolved after applying command-line, JSON, default,
and checkpoint compatibility rules. `configuration_sources` records where each
effective value came from, while `train_config` records the loaded JSON path.

Use the run-local config when reconstructing an experiment. The command in
shell history is helpful; the saved config is less likely to remember only
half the flags.
