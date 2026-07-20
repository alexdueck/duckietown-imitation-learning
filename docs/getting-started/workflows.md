# Common Workflows

These commands assume an activated gym-duckietown Python 3.9 environment and a
working simulator smoke test.

## Prepare Artifact Directories

The scripts create run directories as needed, but the intended top-level
layout is:

```bash
mkdir -p ~/duckietown/data
mkdir -p ~/duckietown/checkpoints
```

## Inspect Rewards Manually

```bash
python manual_control_gym_duckietown.py \
  --map-name loop_empty \
  --reward-functions velopose,posepot
```

The sidebar displays reward totals, components, return, lane values, and the
current world pose. Use `R` to enter a seed and reset directly to it.

To inspect `vd2pp`:

```bash
python manual_control_gym_duckietown.py \
  --reward-functions vd2pp \
  --vd2pp-distance-weight 1.0
```

## Create a Local Start Configuration

```bash
cp \
  configs/gym_duckietown_start_seeds.json.template \
  configs/gym_duckietown_start_seeds.json
```

Edit the local JSON, or start the manual viewer with that path and press `P`
to append the current valid pose to `training_poses`:

```bash
python manual_control_gym_duckietown.py \
  --start-seeds-config configs/gym_duckietown_start_seeds.json
```

The local `configs/*.json` files are ignored by Git. The templates remain
versioned.

## Train PPO on macOS

A conservative example using direct velocity-and-pose reward:

```bash
python train_rl_ppo_gym_duckietown.py \
  --map-name loop_empty \
  --reward-function velopose \
  --total-steps 1000000 \
  --rollout-steps 1024 \
  --batch-size 64 \
  --epochs 2 \
  --policy-lr 1e-5 \
  --value-lr 5e-5 \
  --clip-ratio 0.1 \
  --initial-log-std -1.5 \
  --entropy-coef 0.01 \
  --eval-interval-rollouts 5 \
  --eval-steps 250 \
  --action-mode throttle_steering \
  --max-throttle 0.5 \
  --max-steering 0.5 \
  --exp-name velopose_throttle_steering_max_0.5
```

Add `--render-training` when you want a human-view window on macOS. Rendering
the training view is for observation, not for performance.

## Train PPO on Ubuntu over SSH

Wrap the same command in Xvfb:

```bash
xvfb-run -a \
  -s "-screen 0 1280x1024x24 +extension GLX" \
  env LIBGL_ALWAYS_SOFTWARE=1 \
  python train_rl_ppo_gym_duckietown.py \
    --map-name loop_empty \
    --reward-function velopose \
    --total-steps 1000000 \
    --rollout-steps 1024 \
    --batch-size 64 \
    --epochs 2 \
    --policy-lr 1e-5 \
    --value-lr 5e-5 \
    --clip-ratio 0.1 \
    --initial-log-std -1.5 \
    --entropy-coef 0.01 \
    --eval-interval-rollouts 5 \
    --eval-steps 250 \
    --action-mode throttle_steering \
    --max-throttle 0.5 \
    --max-steering 0.5 \
    --exp-name velopose_throttle_steering_max_0.5
```

This is an experimental configuration, not a universal optimum.

## Use Curated Starts

```bash
python train_rl_ppo_gym_duckietown.py \
  --start-seeds-config configs/gym_duckietown_start_seeds.json \
  --hard-start-probability 0.25 \
  --map-name loop_empty \
  --reward-function velopose \
  --total-steps 1000000
```

At each training reset, a configured training seed or pose is selected with
the requested probability. Other resets use random seeds that exclude all
reserved training and evaluation seeds.

When a start config is supplied, its evaluation seeds and poses replace
`--eval-seeds`.

## Overfit One Scenario Deliberately

For an algorithm/debugging test, use one training pose, the same evaluation
pose, and:

```text
--hard-start-probability 1.0
```

This is intentionally not a generalization experiment. Overfitting is the
feature here: it verifies whether the algorithm can solve a known local
control problem.

## Warm-Start from Imitation Learning

```bash
IL_CHECKPOINT="$HOME/duckietown/checkpoints/imitation_learning/YOUR_RUN/best.pt"

python train_rl_ppo_gym_duckietown.py \
  --imitation-checkpoint "$IL_CHECKPOINT" \
  --action-mode wheel \
  --map-name loop_empty \
  --reward-function velopose \
  --total-steps 1000000
```

The current IL checkpoint mapping initializes a wheel-action policy. It is not
available for `throttle_steering`. Cross-simulator warm starts also inherit a
visual domain shift, so a functioning Duckiematrix IL policy is a useful
initialization, not a guarantee.

## Resume PPO

```bash
RL_CHECKPOINT="$HOME/duckietown/checkpoints/rl_ppo_gym_duckietown/YOUR_RUN/last.pt"

python train_rl_ppo_gym_duckietown.py \
  --resume-checkpoint "$RL_CHECKPOINT" \
  --model mobilenet_v3_small \
  --map-name loop_empty \
  --reward-function velopose \
  --action-mode throttle_steering \
  --max-throttle 0.5 \
  --max-steering 0.5
```

The policy, value network, and optimizer states are restored. Learning rates
come from the new command line, allowing deliberate fine-tuning. A new run
directory is created; the checkpoint step is retained. Repeat the checkpoint's
model, action, preprocessing, and environment options explicitly: resume does
not replace the new command-line configuration.

## Evaluate a PPO Policy Visually

```bash
RL_CHECKPOINT="$HOME/duckietown/checkpoints/rl_ppo_gym_duckietown/YOUR_RUN/best_return.pt"

python live_eval_rl_policy_gym_duckietown.py \
  --checkpoint "$RL_CHECKPOINT" \
  --seed 10045 \
  --max-steps 250 \
  --stop-on-done
```

The viewer uses checkpoint configuration by default, including reward,
preprocessing, action mapping, and environment settings. `--stop-on-done`
leaves the final frame and return visible.

Seeded resets explicitly reseed immediately before `env.reset()`. This matters
because gym-duckietown already performs a reset in the `Simulator`
constructor.

## Generate a Training Report

Create a standalone HTML report from the CSV files in a PPO run directory:

```bash
RUN_DIR="$HOME/duckietown/checkpoints/rl_ppo_gym_duckietown/YOUR_RUN"
python analyze_rl_training_run.py "$RUN_DIR"
```

The script writes `training_report.html` into the run directory. It contains
evaluation trends, per-scenario returns, training-start frequencies and
failure rates, PPO diagnostics, reward components, and runtime measurements.
Charts are embedded as SVG, so the report needs neither a server nor an
internet connection. The report opens in the default browser after generation;
pass `--no-open` when running over SSH or on another headless system. Use
`--eval-window`, `--episode-window`, and `--diagnostic-window` to change the
first/last comparison periods.

## Evaluate an IL Policy in gym-duckietown

```bash
IL_CHECKPOINT="$HOME/duckietown/checkpoints/imitation_learning/YOUR_RUN/best.pt"

python live_eval_imitation_policy_gym_duckietown.py \
  --checkpoint "$IL_CHECKPOINT" \
  --map-name loop_empty
```

This is useful for testing sim-to-sim transfer before starting RL.

## Inspect Commands

Every main script supports:

```bash
python train_rl_ppo_gym_duckietown.py --help
```

The `--help` output is the source of truth for current defaults.
