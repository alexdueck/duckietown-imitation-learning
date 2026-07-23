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

## Physical Duckiebot Chassis Adapter

Simulator wheel commands are not sent directly to physical motors. The
physical adapter composes the checkpoint's `DuckietownActionControl` with
`PhysicalDuckiebotControl`:

```text
policy controls
    -> checkpoint action mapping
    -> normalized left/right wheels
    -> normalized linear/angular motion
    -> physical v/omega limits
    -> acceleration and coupled-command limits
    -> ChassisCommand
```

For normalized wheel commands `l` and `r`, the chassis coordinates are:

```text
linear_normalized  = (l + r) / 2
angular_normalized = (r - l) / 2
v     = linear_normalized  * max_linear_velocity
omega = angular_normalized * max_angular_velocity
```

Positive `omega` therefore corresponds to the right wheel moving faster than
the left wheel. The adapter also enforces the coupled envelope:

```text
abs(v / v_max) + abs(omega / omega_max) <= 1
```

The initial defaults are deliberately conservative and are not robot
identification results:

| Limit | Default |
| --- | ---: |
| Maximum linear velocity | `0.10 m/s` |
| Maximum angular velocity | `1.50 rad/s` |
| Maximum linear acceleration | `0.25 m/s^2` |
| Maximum angular acceleration | `3.00 rad/s^2` |
| Command timeout | `0.50 s` |
| Maximum frame age | `0.50 s` |
| Nominal control period | `0.10 s` |
| Reverse motion | disabled |

Minimal integration after loading a checkpoint:

```python
from duckiebot_hardware_control import (
    PhysicalControlLimits,
    hardware_control_from_checkpoint_config,
)

hardware = hardware_control_from_checkpoint_config(
    checkpoint["config"],
    PhysicalControlLimits(max_linear_velocity=0.10),
)
hardware.arm()

command = hardware.update(
    deterministic_policy_controls,
    frame_age=measured_frame_age_seconds,
)
# Publish command.linear_velocity and command.angular_velocity only through
# the physical runtime's supported chassis-command topic.
```

The adapter starts disarmed. Movement requires an explicit `arm()`. Invalid
policy controls, stale frames, watchdog timeout, disarming, and emergency stop
all produce an immediate zero command. Input and watchdog faults also disarm
the adapter, so movement cannot resume without an explicit `arm()`. Emergency
stop is latched: clearing it does not arm the controller again.

The module is ROS-independent. A physical runtime must publish accepted
`linear_velocity` and `angular_velocity` values, continuously call the
watchdog, and publish zero whenever the returned command is stopped. A
watchdog that is not scheduled by the runtime cannot stop hardware by itself.
