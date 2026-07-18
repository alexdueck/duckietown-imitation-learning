# System Overview

## Research Question

The project investigates whether a policy can drive a Duckiebot from camera
images alone, first by imitating demonstrations and then by optimizing a
simulator reward.

The policy does not receive lane position, map coordinates, reward components,
or simulator ground truth as input. Those values are used for rewards,
diagnostics, evaluation, and curated starts.

## Backends

### gym-duckietown

gym-duckietown is the current RL backend. It renders an RGB camera image and
exposes exact simulator pose and lane geometry. This allows the project to
compute signed forward progress, directed-lane heading, lane-center distance,
and explicit invalid-pose termination.

### Duckiematrix

Duckiematrix is the current IL data source and a transfer target. The policy
interacts through gym-duckiematrix while the simulator engine and renderer run
separately.

The Duckiematrix RL trainer is experimental because the available lane and pose
telemetry has not yet yielded a trustworthy reward away from the intended
road. It remains in the repository to support investigation and eventual
backend comparison.

## Main Data Flow

```text
simulator camera observation
        |
        v
channel conversion, crop, resize, ImageNet normalization
        |
        v
CNN policy encoder
        |
        v
policy controls in [-1, 1]
        |
        v
wheel mapping and ratio-preserving scaling
        |
        v
simulator step
        |
        +--> next camera observation
        +--> simulator state used by reward calculator
        +--> termination information
```

Training uses only the image tensor as the policy and value-network input.
Reward ground truth is deliberately privileged training information.

## Imitation Learning

The Duckiematrix collector stores camera images and continuous left/right wheel
actions. The supervised trainer minimizes regression error for those two
commands using MobileNetV3-Small or ResNet-18.

An IL checkpoint can initialize the PPO actor when PPO also uses direct wheel
control. The CNN and final two-output regression layer are transferred into
the actor mean network. PPO then adds and learns a Gaussian log standard
deviation.

## Reinforcement Learning

The RL implementation is on-policy PPO:

1. Sample actions from a squashed Gaussian image policy.
2. Execute the mapped wheel actions in one simulator environment.
3. Store observations, raw Gaussian actions, old log probabilities, rewards,
   done flags, and value predictions.
4. Compute GAE advantages and returns.
5. Reuse that rollout for several shuffled minibatch epochs.
6. Discard the rollout after the PPO update.
7. Evaluate periodically in a separate environment.

Evaluation uses deterministic mean actions unless explicitly configured
otherwise.

## Separation of Concerns

The project keeps backend-specific training and evaluation scripts separate:

- `train_rl_ppo_gym_duckietown.py`
- `train_rl_ppo_duckiematrix.py`

Shared concepts live in focused modules:

- image policies in `rl_models.py`
- gym-duckietown action mapping in `duckietown_action_control.py`
- gym-duckietown rewards in `velopose_reward.py` and
  `duckietown_rewards.py`
- start scenarios in `gym_duckietown_start_config.py`
- artifact paths in `duckietown_paths.py`

This duplication boundary is intentional. The two simulators differ enough in
dependencies, reset semantics, rendering, and reward ground truth that one
trainer with dozens of backend switches would be harder to reason about.
