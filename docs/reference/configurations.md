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

Both use `lambda=0.15`. `--hard-start-probability` is only the cold-start
probability until hard and random starts have each produced an outcome. A
configured pose receives a fresh reset seed for the simulator's other
randomized reset state.

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
```

This keeps machine-specific experiment selections local while preserving an
executable schema example.

## Run Configuration

Every PPO run writes `config.json` and stores the same training configuration
inside each checkpoint. It includes model, environment, action, reward,
preprocessing, PPO, evaluation, start, seed, and device settings.

Use the run-local config when reconstructing an experiment. The command in
shell history is helpful; the saved config is less likely to remember only
half the flags.
