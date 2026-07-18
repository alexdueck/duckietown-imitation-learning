# Evaluation Protocol

## Default Scenarios

Without a start configuration, gym-duckietown PPO evaluates on four fixed
reset seeds:

```text
10042, 10043, 10044, 10045
```

Each scenario runs for at most `--eval-steps` environment steps. Evaluation is
deterministic unless `--eval-stochastic` is passed.

Evaluation uses a separate environment and does not interrupt the current
training episode.

## Seeded Reset Semantics

gym-duckietown's `Simulator` performs a reset inside its constructor. Passing
`seed=10045` to the constructor therefore consumes the first reset generated
by that seed.

To reproduce a scenario, the project explicitly reseeds immediately before the
relevant `env.reset()`. The trainer, manual viewer, and RL live-evaluation
viewer follow this convention.

A seed identifies a reset procedure, not a human-readable pose. Exact poses can
also be stored when seed-level control is insufficient.

## Evaluation Poses

A start configuration can define `evaluation_poses` with:

- tile coordinates
- local position inside that tile
- yaw angle
- optional name

Configured evaluation seeds and poses replace the command-line
`--eval-seeds` list. They are evaluated in a stable order.

Lane values do not need to be stored with a pose. gym-duckietown reconstructs
them from exact simulator position, angle, and map geometry.

## Metrics

For each evaluation, `eval_history.csv` stores:

- total return across all scenarios
- mean scenario return
- reward per environment step
- total executed steps
- number of scenarios and safe scenarios
- scenario-length statistics
- terminated, truncated, and time-limit counts

`eval_scenarios.csv` stores one row per scenario. Aggregate return alone can
hide one failing seed behind three strong seeds, so scenario-level results are
required for diagnosis.

A scenario reaching its evaluation horizon is counted as safe. Safe does not
mean high-quality lane following; a policy can finish all steps with a poor
return.

## Checkpoint Selection

Every evaluation saves:

```text
eval_<index:04d>_step_<step:010d>.pt
```

Two moving best checkpoints are also maintained:

- `best_return.pt`: highest mean scenario return
- `best_safe.pt`: lexicographically best safe-scenario count, mean scenario
  length, then mean scenario return

`last.pt` represents the latest training state, not necessarily the best
policy. PPO performance is not expected to improve monotonically.

## Visual Evaluation

Use the run's checkpoint configuration and one fixed seed:

```bash
RL_CHECKPOINT="$HOME/duckietown/checkpoints/rl_ppo_gym_duckietown/YOUR_RUN/best_return.pt"

python live_eval_rl_policy_gym_duckietown.py \
  --checkpoint "$RL_CHECKPOINT" \
  --seed 10045 \
  --max-steps 250 \
  --stop-on-done
```

The sidebar shows the reward used during training and its components by
default. `--stop-on-done` preserves the final frame and return for inspection.

## Reporting a Result

At minimum report:

- code revision
- run directory and checkpoint filename
- map and backend
- reward function and action mode
- training steps
- evaluation seeds or poses
- steps per scenario
- aggregate and per-scenario return
- safety/termination counts
- whether evaluation was deterministic

One best checkpoint from one run is a useful observation. It is not yet a
confidence interval.
