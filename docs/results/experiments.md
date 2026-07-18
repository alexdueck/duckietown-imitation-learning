# Experiment Log

This page records the main experimental sequence and the reason each change was
made. It is a research log, not a leaderboard.

Future entries should include:

```text
date
Git commit
run directory
complete command or run config
hypothesis
evaluation scenarios
quantitative result
qualitative observation
conclusion and next action
```

## Duckiematrix Imitation Learning

**Question:** Can a camera-only model imitate manual wheel commands?

**Setup:** Duckiematrix images, continuous left/right targets, MobileNetV3-Small
or ResNet-18.

**Observation:** Imitation learning worked well on the simple `loop` map. The
same approach was not yet robust on `sandbox`.

**Conclusion:** The image-to-wheel formulation is viable, but dataset coverage
and map complexity matter.

## Duckiematrix PPO

**Question:** Can PPO improve a camera policy using gym-duckiematrix rewards?

**Observation:** Policies converged to behaviors such as spinning on grass,
driving backward, or exploiting areas with unexpectedly high lane-related
reward.

Inspection showed lane-distance and pose values that did not consistently
represent the intended road geometry.

**Conclusion:** Further PPO tuning was not justified before fixing the reward
signal. The trainer remains experimental.

## gym-duckietown Reward Port

**Question:** Can reward functions used in previous Duckietown RL work be
reproduced with reliable simulator ground truth?

**Observation:** gym-duckietown provided exact pose and road geometry, making
manual reward inspection coherent. This shifted the active RL backend to
gym-duckietown.

**Conclusion:** Keep Duckiematrix as a transfer target, but train current RL
policies in gym-duckietown.

## Fixed-Throttle Curve Test

**Question:** Is PPO capable of learning the selected visual curve-control
problem when longitudinal control is removed?

**Setup:** One curated difficult pose, fixed throttle, learned steering, short
deterministic evaluation.

**Observation:** The policy learned to drive through the curve and reached
returns near the practical limit imposed by the low fixed speed.

**Conclusion:** PPO and the visual encoder can learn the curve. Joint speed and
steering control was a major source of difficulty.

## posepot Curve Test

**Question:** Can potential-based pose shaping guide the policy without
repeatedly rewarding a stationary good pose?

**Observation:** The policy negotiated the curve by the first evaluation after
five rollouts. Return then improved only slightly.

In a longer, less constrained run, the policy learned to use the wrong lane
while maintaining a positive potential contribution. For a constant negative
potential, `gamma * Phi - Phi` is positive when `gamma < 1`.

**Conclusion:** Potential shaping supplied useful transition guidance but did
not by itself make an undesirable steady state costly.

## vd2pp Reward

**Question:** Can a direct squared lane-distance penalty close the steady-state
loophole while preserving potential guidance?

**Definition:**

```text
velocity + pose_potential - beta * scaled_lane_distance^2
```

**Status:** Implemented and available in training, manual control, and live
evaluation. General results are pending.

## velopose Full-Control Run

**Run:**

```text
20260717_220541_ppo_gym_duckietown_velopose_throttle_steering_max_0.5
```

**Question:** Can direct velocity plus pose quality learn both throttle and
steering from random starts?

**Setup:** MobileNetV3-Small, random starts, `loop_empty`, throttle and steering
both limited to 0.5, four fixed evaluation seeds.

**Observation:** The policy learned the intended lane and curve following. At
evaluation 288 and training step 1,474,560, aggregate return was 857.492 with
all four 250-step scenarios safe.

Seed 10045 remained much weaker because it begins on the wrong lane and must
first pay a substantial pose penalty while changing lanes.

**Conclusion:** This is the strongest current full-control result, but it is
still one run on one map. Independent repeats are required.

## Next Experiments

High-value next steps include:

- repeat the full-control run with multiple training seeds
- oversample seed 10045 or a saved pose near its lane correction
- compare `velopose` and `vd2pp` under identical starts
- evaluate on additional maps without training on them
- test gym-duckietown policies in Duckiematrix
- add parallel environment collection after preserving current diagnostics
