# Reward Functions

## Status

The custom rewards on this page use gym-duckietown simulator ground truth.
They are the current basis for RL experiments.

The similarly named Duckiematrix reward adapters are experimental because the
required geometry has not been reliable enough in that backend.

## Directed Lane Reference

A `DirectedLaneTracker` selects one directed lane and keeps its tangent
orientation consistent over time. Once selected, the lane is not reselected
from the robot's current heading. This prevents a robot facing backward from
silently turning the opposite lane direction into the new target.

Let:

```text
p_t       current world position
p_(t-1)   previous world position
tau       normalized directed lane tangent
r         normalized right vector of the lane
h         normalized robot forward vector
d_lane    signed lateral distance from the lane center
w_lane    lane half width = 0.2 * road_tile_size
d         d_lane / w_lane
```

## Velocity Component

Progress is measured along the directed lane tangent:

```text
progress      = dot(p_t - p_(t-1), tau)
forward_speed = progress / delta_time
velocity      = clip(forward_speed / 0.6 m/s, -1, 1)
```

Reverse progress is negative. Motion perpendicular to the lane does not count
as forward progress.

## Corrective Target Heading

The best heading points toward the lane center when the robot is laterally
displaced:

```text
alpha(d) = -45 degrees * tanh(1.25 * d)
h_target = cos(alpha) * tau + sin(alpha) * r
heading_quality = clip(dot(h, h_target), -1, 1)
```

Near the center, the target approaches the forward lane tangent. Farther away,
the correction saturates at 45 degrees toward the center.

## Pose Quality

```text
scaled_abs_lane_distance = abs(d)
lane_distance_penalty    = -2 * scaled_abs_lane_distance
pose_quality             = heading_quality + lane_distance_penalty
```

The distance penalty is not clipped. Moving farther away continues to reduce
the reward, preserving a direction of improvement even beyond the neighboring
lane.

## velopose

```text
velopose = velocity + pose_quality
```

An `invalid-pose` termination adds `-20` to the transition reward.

This direct state reward makes a good pose valuable at every step, including
while stationary. In practice it has nevertheless produced the strongest
current full-control result.

## posepot

`posepot` keeps velocity but replaces direct pose reward with potential-based
shaping. Let `Phi(s) = pose_quality(s)`. On the transition from the previous
state to the current state:

```text
pose_potential = gamma * Phi(current) - Phi(previous)
posepot        = velocity + pose_potential
```

The default `gamma` is `0.99`. At a terminal transition, the next potential
is defined as zero.

This rewards improvement rather than repeatedly rewarding a state. It also has
an important practical edge case: staying at a constant negative potential
produces

```text
(gamma - 1) * negative_potential > 0
```

per step. A policy on the wrong lane can therefore receive a small positive
shaping reward while maintaining a bad pose. This behavior was observed and
motivated `vd2pp`.

## vd2pp

`vd2pp` adds a direct squared lane-distance cost to `posepot`:

```text
vd2pp = velocity
      + gamma * Phi(current) - Phi(previous)
      - beta * scaled_abs_lane_distance^2
```

The default `beta` is `1.0` and can be changed with
`--vd2pp-distance-weight`.

The direct term removes the constant-bad-pose loophole while retaining
potential-based guidance for pose improvements.

## Other Available Rewards

The gym-duckietown tools also expose rewards adapted from
`kaland313/Duckietown-RL` for comparison:

- `default`: gym-duckietown's native reward
- `default_clipped`: native reward clipped to `[-2, 2]`
- `posangle`: position/target-angle reward plus velocity
- `target_orientation`: wider target-orientation reward plus velocity
- `distance_travelled`: movement measured along local road geometry

The manual viewer displays `velopose` and `posepot` by default. Use
`--reward-functions` to select other breakdowns.

## Reward Reporting

Every custom reward returns a nested component breakdown. Training flattens and
stores components in `reward_components_history.csv` for:

- each training rollout
- each aggregate evaluation
- each individual evaluation scenario

This makes it possible to distinguish a high return caused by velocity from a
high return caused by pose, which is often the difference between a useful
policy and a very confident mistake.
