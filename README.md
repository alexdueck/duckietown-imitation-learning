# Duckietown Vision-Based Control Experiments

> **Project status: work in progress.** The results in this repository are
> preliminary research observations, not final benchmarks.

This repository explores vision-based Duckietown control with imitation
learning (IL) and reinforcement learning (RL). A camera image is the model
input; continuous Duckiebot controls are the output.

## Current State

**gym-duckietown is currently the functional RL setup.** It provides simulator
ground truth that supports useful lane-pose and progress rewards. The current
PPO implementation can learn lane following on `loop_empty`, including turns,
under the tested configurations.

The Duckiematrix code remains useful for collecting imitation data, training
IL policies, and testing transfer to another simulator. Its RL path is
experimental: no sufficiently reliable reward function has been established
for gym-duckiematrix yet. In particular, the available lane-distance and pose
signals can assign plausible rewards away from the intended road geometry.
Training there is therefore very good at optimizing the wrong thing, which is
still optimization but not the kind we were hoping for.

| Backend | IL | RL | Current recommendation |
| --- | --- | --- | --- |
| gym-duckietown | Visual evaluation and IL-policy transfer | Functional PPO training | Use for current RL experiments |
| gym-duckiematrix | Data collection, training, and live evaluation | Experimental reward implementations | Use for IL and cross-simulator evaluation |

## What Is Included

- Duckiematrix imitation-data collection and telemetry capture
- MobileNetV3-Small and ResNet-18 imitation training in PyTorch
- PPO trainers for gym-duckietown and gym-duckiematrix
- Squashed-Gaussian continuous policies
- Direct wheel control and throttle/steering control
- Custom `velopose`, `posepot`, and `vd2pp` rewards
- Deterministic evaluation seeds and exact saved start poses
- Per-evaluation checkpoints, safety-aware checkpoint selection, and detailed
  CSV diagnostics
- Standalone HTML reports for completed or running PPO experiments
- Manual reward inspection and visual policy evaluation tools

## Quick Start

Use Python 3.9 for gym-duckietown. Installation differs substantially between
macOS and a display-less Ubuntu host:

- [Install gym-duckietown on macOS](docs/getting-started/gym-duckietown-macos.md)
- [Install gym-duckietown on Ubuntu over SSH](docs/getting-started/gym-duckietown-ubuntu.md)
- [Configure Duckiematrix](docs/getting-started/duckiematrix.md)

After installation, inspect the reward interactively:

```bash
python manual_control_gym_duckietown.py --map-name loop_empty
```

Run a small PPO experiment:

```bash
python train_rl_ppo_gym_duckietown.py \
  --map-name loop_empty \
  --reward-function velopose \
  --action-mode throttle_steering \
  --max-throttle 0.5 \
  --max-steering 0.5 \
  --total-steps 100000
```

See [Common workflows](docs/getting-started/workflows.md) for remote training,
resuming checkpoints, fixed evaluation scenarios, IL warm starts, and visual
evaluation.

## Documentation

Start with the [documentation index](docs/index.md), or go directly to:

- [System overview](docs/methodology/system-overview.md)
- [Observation and model pipeline](docs/methodology/observations-and-models.md)
- [Action mappings](docs/methodology/actions.md)
- [Reward definitions](docs/methodology/rewards.md)
- [PPO implementation](docs/methodology/ppo.md)
- [Evaluation protocol](docs/methodology/evaluation.md)
- [Preliminary results](docs/results/summary.md)
- [Developer architecture](docs/development/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)

## Filesystem Layout

Experiment artifacts live outside the Git repository by default:

```text
~/duckietown/
|-- data/
|   |-- imitation_learning/
|   `-- evaluations/
`-- checkpoints/
    |-- imitation_learning/
    |-- rl_ppo_duckiematrix/
    `-- rl_ppo_gym_duckietown/
```

Local start configurations use `configs/*.json` and are ignored by Git.
Versioned `.json.template` files document their schema.

## Research Disclaimer

Current claims are based on a small number of simulation runs, primarily on
`loop_empty`. They do not yet establish statistical significance,
cross-map generalization, sim-to-sim transfer, or real-world performance.
Configuration, code version, seeds, and per-scenario results should accompany
any reported number.
