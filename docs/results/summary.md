# Preliminary Results

> **Status:** Work in progress. These results are preliminary simulation
> observations from a small number of runs. They are not a benchmark.

## Current Strongest Full-Control Run

Run:

```text
20260717_220541_ppo_gym_duckietown_velopose_throttle_steering_max_0.5
```

Main configuration:

| Setting | Value |
| --- | --- |
| Backend | gym-duckietown |
| Map | `loop_empty` |
| Model | MobileNetV3-Small |
| Reward | `velopose` |
| Action mode | learned throttle and steering |
| Maximum throttle | 0.5 |
| Maximum steering | 0.5 |
| Rollout length | 1024 |
| Batch size | 64 |
| PPO epochs | 2 |
| Policy learning rate | `1e-5` |
| Value learning rate | `5e-5` |
| Clip ratio | 0.1 |
| Initial log standard deviation | -1.5 |
| Entropy coefficient | 0.01 |
| Domain randomization | disabled |
| Training starts | random |
| Evaluation | deterministic, four fixed seeds, 250 steps each |

At the CSV snapshot after training step 1,474,560, evaluation 288 achieved:

```text
aggregate return: 857.492
safe scenarios:  4 / 4
executed steps:  1000 / 1000
```

Per-scenario returns:

| Seed | Return | Steps | Outcome |
| --- | ---: | ---: | --- |
| 10042 | 295.72 | 250 | evaluation horizon |
| 10043 | 282.20 | 250 | evaluation horizon |
| 10044 | 281.22 | 250 | evaluation horizon |
| 10045 | -1.66 | 250 | evaluation horizon |

## Interpretation

The first three seeds are solved much more strongly than seed 10045. That seed
starts on the wrong lane. The policy first incurs more than 100 points of
negative return while moving to the desired lane, then accumulates consistently
positive reward and recovers to approximately zero within 250 steps.

This is not an `invalid_pose` failure. It is a safe but expensive correction.
The distinction only became clear after combining per-scenario CSVs with visual
evaluation.

The aggregate maximum was initially observed at evaluation 45 with return
838.35. A later snapshot surpassed it at evaluation 288. PPO performance
between those points was non-monotonic, especially on seed 10045.

## Training Safety

Across the full history, most training `invalid_pose` episodes occurred very
early. In the final analyzed rollout window, the rate had fallen to about nine
invalid-pose terminations per 100,000 environment steps.

Training still samples actions, while evaluation uses deterministic means.
Occasional exploratory departures therefore do not imply that the
deterministic policy fails from the same state.

## Reward Trends

Late training windows showed continued improvement in the training data:

```text
mean Velocity component:             about 0.330 per step
mean Pose component:                 about 0.715 per step
mean ScaledAbsLaneDistance:           about 0.130
mean total training reward:           about 1.043 per step
```

The fixed evaluation set remained more variable because seed 10045 is
qualitatively harder than the other three starts.

## Other Observations

- A fixed-throttle throttle/steering experiment learned the selected curve
  reliably. Reducing the action problem to steering was an effective debugging
  step.
- `posepot` learned the same curated curve quickly, but longer training could
  prefer the wrong lane while receiving a small positive potential term.
- `vd2pp` was introduced to combine potential shaping with a direct squared
  lane-distance cost. Its general performance is not established yet.
- Duckiematrix imitation learning worked well on the simple `loop` map.
  Generalization to the more complex `sandbox` map remains under
  investigation.
- Duckiematrix PPO repeatedly found undesirable local behaviors under the
  available reward signals. Those runs are not evidence against PPO; they are
  evidence that reward geometry matters.

## What Has Not Been Demonstrated

The project has not yet established:

- averages and confidence intervals across independent training seeds
- generalization across gym-duckietown maps
- robustness under domain randomization
- sim-to-sim transfer from gym-duckietown to Duckiematrix
- transfer to a physical Duckiebot
- comparison against standard RL-library implementations
- wall-clock scaling with parallel environments

These are natural next experiments, not fine print.
