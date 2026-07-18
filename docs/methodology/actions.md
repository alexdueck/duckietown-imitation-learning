# Action Representation

The environment action is always a pair of continuous wheel commands:

```text
(left_wheel, right_wheel), each in [-1, 1]
```

The PPO policy first produces bounded policy controls:

```text
u = tanh(z), where z is sampled from the actor's Gaussian
```

The meaning of `u` depends on `--action-mode`.

## Direct Wheel Mode

```text
action_mode = wheel
u = (left_wheel, right_wheel)
```

This mode supports forward and reverse motion independently on both wheels. It
is also the mode required by the current imitation-checkpoint warm start,
because the IL model predicts left and right wheel commands directly.

## Throttle and Steering Mode

Without fixed throttle, the policy controls are:

```text
u = (throttle_control, steering_control)
```

The learned throttle is mapped from `[-1, 1]` to a non-negative interval:

```text
throttle = 0.5 * (throttle_control + 1) * max_throttle
steering = steering_control * max_steering
```

Initial wheel commands are then:

```text
left  = throttle - steering
right = throttle + steering
```

Although throttle is non-negative, sufficiently strong steering can make one
wheel negative.

## Fixed Throttle

With `--fixed-throttle T`, the policy has only one output:

```text
steering = tanh(z) * max_steering
left     = T - steering
right    = T + steering
```

This reduces the learning problem to steering. It was useful as a controlled
experiment: the policy learned to negotiate a selected curve when learning
both speed and steering had produced stationary or oscillating behavior.

## Ratio-Preserving Wheel Scaling

The initial wheel pair can exceed the environment range. The implementation
does not clip each wheel independently. Instead it computes:

```text
scale = max(1, abs(left), abs(right))
left_final  = left  / scale
right_final = right / scale
```

This preserves the ratio between wheel commands. Independent clipping would
change curvature whenever only one wheel saturates.

Scaling also means that `max_throttle` and `max_steering` are limits before
the final wheel normalization. If their combination exceeds one, both wheel
commands are reduced together.

## Why Keep Multiple Modes?

Direct wheel control is expressive and matches the IL action head, but PPO must
learn speed and steering through two coupled motors.

Throttle/steering control adds a useful inductive bias. Fixed throttle goes
further and removes longitudinal control entirely. These are different
research conditions and must be recorded with every result.
