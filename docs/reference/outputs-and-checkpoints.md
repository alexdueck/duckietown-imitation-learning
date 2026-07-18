# Outputs and Checkpoints

## gym-duckietown PPO Run

Default location:

```text
~/duckietown/checkpoints/rl_ppo_gym_duckietown/
  <timestamp>_ppo_gym_duckietown[_<exp-name>]/
```

A run contains:

| File | Contents |
| --- | --- |
| `config.json` | Complete resolved run configuration and start metadata |
| `history.csv` | Completed training episodes |
| `rollout_history.csv` | PPO rollout, timing, throughput, progress, and loss metrics |
| `ppo_diagnostics.csv` | Probability ratios, KL, exploration, actions, and saturation |
| `reward_components_history.csv` | Flattened reward components for train/eval phases |
| `eval_history.csv` | One aggregate row per evaluation |
| `eval_scenarios.csv` | One row per evaluation scenario |
| `last.pt` | Latest completed PPO rollout |
| `best_return.pt` | Highest mean scenario return so far |
| `best_safe.pt` | Best safety/length/return tuple so far |
| `eval_####_step_##########.pt` | Policy snapshot for every evaluation |

`last.pt` is saved after each PPO update. Interrupting in the middle of a
rollout leaves the most recent completed update.

## Episode History

Important `history.csv` columns:

- `step`: global environment step at episode end
- `episode_return`, `episode_length`, `episode_return_per_step`
- `done_reason`: for example `invalid-pose` or `max-steps-reached`
- `start_type`: random, hard seed, or hard pose
- `start_seed` and optional `start_name`

An episode return and a rollout return are different aggregations. A rollout
can contain several episodes or fragments of two episodes.

## Rollout History

Timing fields:

- `rollout_seconds`: environment data collection
- `preprocess_seconds`
- `policy_value_inference_seconds`
- `env_step_seconds`
- `reward_and_reset_seconds`
- `rollout_overhead_seconds`
- `update_seconds`
- `rollout_update_seconds`

Throughput fields:

- `environment_steps_per_second`: rollout steps divided by rollout time
- `cycle_steps_per_second`: rollout steps divided by rollout plus update time
- `overall_steps_per_second`: all training steps divided by wall time,
  including evaluation and startup
- `progress_percent`, `elapsed_seconds`, and `eta_seconds`

## PPO Diagnostics

The diagnostics separate:

- distribution change: KL, clip fraction, ratio mean/min/max
- exploration parameter: `log_std_*` and `std_*`
- sampled versus deterministic policy controls
- sampled versus deterministic wheel actions
- effective wheel and steering noise
- action/control saturation
- transformed entropy estimate

`std_left` and `std_right` retain historical column names. In
`throttle_steering` mode they refer to policy-control dimensions whose actual
names are stored in `policy_control_0_name` and
`policy_control_1_name`.

## Reward Components

Each row identifies:

- `phase`: `train_rollout`, `eval`, or `eval_scenario`
- train step, rollout, evaluation, scenario, and seed keys
- flattened component path
- component sum
- mean per all steps
- mean when the component was present
- presence count and total step count

Terminal penalties appear only on terminal steps, so
`component_mean_when_present` is often more informative than mean per step.

## Evaluation CSVs

`eval_history.csv` aggregates all scenarios. `eval_return` is the sum across
them; `eval_mean_scenario_return` divides by the scenario count.

`eval_open_episode_return` does not exist in the current fixed-scenario
gym-duckietown format. Each scenario has its own explicit row in
`eval_scenarios.csv`.

Use `eval_scenarios.csv` to identify which seed terminated and where it
started.

## Checkpoint Payload

A PPO checkpoint stores:

- global step
- actor and value state dictionaries
- actor and value optimizer state dictionaries
- full training configuration
- backend identifier
- action-space semantics
- ImageNet normalization constants

Resuming restores both networks and optimizers. The new CLI learning rates
replace optimizer learning rates after loading.

## Imitation Learning Outputs

Default location:

```text
~/duckietown/checkpoints/imitation_learning/<experiment>_<timestamp>/
```

Files:

- `config.json`
- `history.csv`
- `best.pt`
- `last.pt`

The checkpoint stores model and optimizer state, epoch, best validation loss,
target columns, normalization constants, config, and metric history.

## Collected IL Data

Each Duckiematrix collection run stores:

- `images/*.jpg`
- `actions.csv`
- `meta.json`

`actions.csv` includes image name, left/right action, timestamp, reward, and
lane telemetry. The row aligns visible `obs_t`, its reward, and the action
selected from that observation.
