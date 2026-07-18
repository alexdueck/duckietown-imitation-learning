# Configurations and Paths

## Artifact Root

`duckietown_paths.py` defines:

```text
~/duckietown/
|-- data/
|   |-- imitation_learning/
|   |   |-- train/
|   |   `-- val/
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
cp configs/gym_duckietown_start_seeds.json.template \
   configs/gym_duckietown_start_seeds.json
```

Schema:

```json
{
  "map_name": "loop_empty",
  "training_seeds": [123, 456],
  "evaluation_seeds": [10042, 10043, 10044, 10045],
  "training_poses": [
    {
      "name": "optional_name",
      "tile": [3, 5],
      "position": [0.51, 0.0, 0.43],
      "angle": 0.70
    }
  ],
  "evaluation_poses": []
}
```

Rules:

- `map_name` must match the environment.
- Seeds are non-negative unique integers.
- Training and evaluation seeds may not overlap.
- A trainer config needs at least one training seed/pose and one evaluation
  seed/pose.
- `name` is optional.
- `tile` contains integer map tile coordinates.
- `position` is local to that tile.
- `angle` is simulator yaw in radians.
- Unknown fields are rejected.

When this config is supplied to the trainer, configured evaluation scenarios
replace `--eval-seeds`.

## Training Start Sampling

At every training episode reset:

```text
with probability hard_start_probability:
    choose uniformly from training_seeds + training_poses
otherwise:
    draw a random reset seed not reserved by the config
```

A configured pose receives a fresh non-reserved reset seed for the simulator's
other randomized reset state.

`--hard-start-probability 1.0` always uses curated starts. This is useful for
an intentional overfitting test.

## Single Pose File

`configs/gym_duckietown_pose.json.template` documents the file accepted by
the manual viewer's `--start-pose-file`:

```json
{
  "name": "curve_start",
  "tile": [1, 1],
  "position": [0.149, 0.0, 0.094],
  "angle": 1.442
}
```

This file contains exactly one pose. Ordinary resets return to it.

## Capturing a Pose

Start the manual viewer with an existing local start config:

```bash
python manual_control_gym_duckietown.py \
  --start-seeds-config configs/gym_duckietown_start_seeds.json
```

Drive to a valid location and press `P`. The viewer appends a pose to
`training_poses`.

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
