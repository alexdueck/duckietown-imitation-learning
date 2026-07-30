# Developer Architecture

## Repository Shape

The repository is intentionally script-oriented. User-facing training,
evaluation, and hardware tools remain executable directly from the repository
root. Reusable implementation modules live in the flat `dt_utils/`
package, and test modules live in `tests/`. This keeps the existing
`python SCRIPT.py` and `/workspace/SCRIPT.py` workflows independent of an
editable package installation.

```text
repository root/
|-- *.py                         user-facing entry points
|-- dt_utils/                    reusable importable modules
|-- tests/                       test modules
|-- configs/
|-- docs/
`-- requirements/
```

## Duckiematrix Entry Points

| File | Responsibility |
| --- | --- |
| `imitation_learning.py` | Manual control, observation capture, telemetry, and dataset writing |
| `data_viewer.py` | Browse collected images, actions, reward, and telemetry |
| `train_imitation_learning.py` | Supervised image-to-wheel training |
| `live_eval_imitation_policy.py` | Execute an IL policy in Duckiematrix |
| `train_rl_ppo_duckiematrix.py` | Experimental Duckiematrix PPO trainer |
| `dt_utils/duckiematrix_telemetry.py` | Read pose/lane telemetry from Duckiematrix |
| `dt_utils/rl_rewards.py` | Duckiematrix reward adapters |

## gym-duckietown Entry Points

| File | Responsibility |
| --- | --- |
| `manual_control_gym_duckietown.py` | Manual driving, reward sidebar, seeds, and pose capture |
| `live_eval_imitation_policy_gym_duckietown.py` | Visual IL transfer evaluation |
| `train_rl_ppo_gym_duckietown.py` | Main PPO trainer and deterministic evaluation |
| `live_eval_rl_policy_gym_duckietown.py` | Visual PPO checkpoint evaluation |
| `dt_utils/duckietown_rewards.py` | Reward selection, state tracking, breakdowns, and compatibility patch |
| `dt_utils/velopose_reward.py` | Pure custom reward equations |
| `dt_utils/duckietown_action_control.py` | Policy-control to wheel-action mapping |
| `dt_utils/duckiebot_hardware_control.py` | Fail-closed wheel-action to physical chassis-command mapping |
| `dt_utils/gym_duckietown_start_config.py` | Multi-map pose configuration, validation, and sampling |

## Shared Modules

| File | Responsibility |
| --- | --- |
| `dt_utils/rl_models.py` | CNN actor, Q-network scaffold, tanh log probability, and IL actor loading |
| `dt_utils/duckietown_paths.py` | Artifact locations below `~/duckietown` |
| `dt_utils/cli_completion.py` | Optional argcomplete integration |
| `dt_utils/duckiebot_rosbridge.py` | ROS-independent compressed-camera subscription and chassis-command publication over rosbridge |
| `tests/ppo_control_tests.py` | PPO invariant, Pendulum, and image-control tests |
| `preprocess.py` | Legacy optional offline image preprocessing |

## Physical Duckiebot Entry Points

| File | Responsibility |
| --- | --- |
| `physical_duckiebot_control.py` | Unified host-side manual/model control, GUI, inference, arming, E-stop, rosbridge camera/commands, and recording |
| `dt_utils/duckiebot_teleop_input.py` | Keyboard/SDL device adapters, deadzone, and keyboard-only ramps |
| `dt_utils/duckiebot_dataset_recorder.py` | Stream aligned compressed camera frames and effective wheel-equivalent actions |
| `dt_utils/duckiebot_rosbridge.py` | Shared rosbridge camera subscriber, `Twist2DStamped` publisher, and robot-address resolution |
| `dt_utils/duckiebot_hardware_control.py` | ROS-independent fail-closed geometric conversion from normalized wheels to `v`/`omega` |

## PPO Trainer Structure

The main gym-duckietown trainer currently owns:

1. CLI and run configuration
2. environment construction and compatibility setup
3. deterministic and curated reset handling
4. policy/value initialization and checkpoint loading
5. rollout collection and timing
6. GAE and PPO optimization
7. diagnostics and CSV output
8. fixed-scenario evaluation
9. checkpoint selection

This is a large module, but those responsibilities are kept in explicit
functions. Future extraction should preserve the observable CSV and checkpoint
contracts before pursuing smaller files for their own sake.

## Reward Boundary

`dt_utils/velopose_reward.py` contains equations over NumPy values and
has no direct simulator dependency.

`dt_utils/duckietown_rewards.py` adapts simulator state into those
equations, tracks previous position/potential, recognizes done reasons, and
returns nested breakdowns.

This boundary makes reward mathematics testable independently from OpenGL and
Pyglet.

## Action Boundary

The actor always produces normalized policy controls. Only
`DuckietownActionControl` knows how controls map to wheel commands. The same
mapping is used in rollout collection, deterministic evaluation, diagnostics,
checkpoint metadata, and live evaluation.

Physical deployment adds a second boundary after this mapping.
`PhysicalDuckiebotControl` maps normalized wheels to bounded chassis
velocities, tracks arming and emergency-stop state, limits acceleration, and
fails closed on invalid or stale inputs and watchdog timeout. It deliberately
does not import ROS or publish commands; transport belongs to the physical
runtime.

## Physical Duckiebot Control Architecture

The unified manual/model controller runs in the Mac `gymdt39_venv`. It uses
rosbridge for camera input and command output and needs neither a ROS
installation nor Docker. It does not install or replace code on the robot.

| Component | Location | Ownership |
| --- | --- | --- |
| Keyboard or PS4 controller | macOS host | User hardware |
| `physical_duckiebot_control.py` | macOS host | This repository |
| `dt_utils/duckiebot_rosbridge.py` | macOS host | This repository |
| `rosbridge_websocket` | Physical Duckiebot | ROS/Duckietown |
| `car_cmd_switch_node` | Physical Duckiebot | Duckietown `dt-core` |
| `kinematics_node` | Physical Duckiebot | Duckietown `dt-core` |
| `wheels_driver_node` | Physical Duckiebot | Duckietown hardware stack |

The manual control path is:

```text
keyboard or PS4 controller on macOS
    |
    v
physical_duckiebot_control.py (manual mode)
    |
    | deadzone, arming, emergency stop, wheel mixing, physical limits
    | command rosbridge WebSocket to ROBOT_IP:9001
    v
rosbridge_websocket on the Duckiebot
    |
    | local ROS publication: Twist2DStamped
    v
/ROBOT_NAME/joy_mapper_node/car_cmd
    |
    v
car_cmd_switch_node
    |
    | /ROBOT_NAME/car_cmd_switch_node/cmd
    v
kinematics_node
    |
    | calibrated WheelsCmdStamped
    v
wheels_driver_node
    |
    v
motor hardware
```

The camera and recording path is independent:

```text
camera_node on the Duckiebot
    |
    | CompressedImage over a camera rosbridge WebSocket
    v
RosbridgeCameraSubscriber in dt_utils/duckiebot_rosbridge.py
    |
    | newest unique frame + effective command sent for that control tick
    v
dt_utils/duckiebot_dataset_recorder.py
    |
    +-- images/*
    +-- actions.csv
    +-- meta.json
```

The controller uses separate camera and command WebSockets so a slow
subscriber receive cannot block a command send. Model mode adds inference
between those two transports:

```text
/ROBOT_NAME/camera_node/image/compressed
    |
    | rosbridge WebSocket; newest frame replaces any waiting older frame
    v
physical_duckiebot_control.py on macOS (model mode)
    |
    | checkpoint preprocessing and deterministic IL/PPO inference
    | checkpoint action mapping -> normalized wheels
    | PhysicalDuckiebotControl -> wheel speeds -> geometric v/omega
    v
command rosbridge WebSocket
    |
    v
/ROBOT_NAME/joy_mapper_node/car_cmd
```

Inference normally reacts to every unique camera frame it can process; an
optional maximum inference rate can reduce this frequency. Arming and E-stop
changes increment an inference generation, so a result started in an earlier
generation cannot publish afterward. Successfully published results are paired
with the exact raw frame in `actions.csv` and `images/`.

### Why physical control uses rosbridge

ROS 1 discovery plus TCPROS required ROS Python packages and bidirectional
network reachability. Docker Desktop made this especially awkward because the
robot could not reach container-internal callback addresses. Rosbridge exposes
a fixed WebSocket endpoint instead:

```text
macOS process --> ws://ROBOT_IP:9001
```

Over one outbound connection the runtime subscribes to compressed camera
messages. Over a second it sends rosbridge `advertise` and `publish`
operations for `Twist2DStamped`. The onboard `rosbridge_websocket` process
then participates in ROS locally, where camera, command switch, kinematics,
and wheel-driver nodes can reach it normally. This shared transport is why the
unified controller runs directly on macOS without ROS or Docker.

### Command-source selection

`car_cmd_switch_node` starts with `current_src_name = "joystick"`. Its default
configuration also maps the FSM state `NORMAL_JOYSTICK_CONTROL` to the
`joystick` source:

```text
joystick source --> joy_mapper_node/car_cmd
lane source     --> lane_controller_node/car_cmd
stop source     --> simple_stop_controller_node/car_cmd
```

The current unified runtime does not publish an FSM state or explicitly switch
the robot into joystick mode. It relies on the onboard switch starting in
`joystick`. If an FSM later selects lane following or stop, commands remain
published but are not forwarded to kinematics.

`speed_gain` and `steer_gain` belong to `joy_mapper_node`'s conversion from
`sensor_msgs/Joy` to `Twist2DStamped`. Our runtime already publishes
`Twist2DStamped` on the mapper's output topic, so those gains are reference
values rather than downstream multipliers. `kinematics_node` still applies
`v_max`, `omega_max`, geometry, gain, trim, motor constant, and output limits.

### DTPS alternative used by Duckietown keyboard control

The official command:

```text
dts duckiebot keyboard_control ROBOT_NAME
```

uses a different architecture. Its Duckietown Viewer backend connects to the
robot's fixed DTPS switchboard endpoint:

```text
viewer backend --> http://ROBOT_IP:11911/
```

It reads the active kinematics calibration from the Duckietown KV store,
computes differential PWM in the viewer backend, and publishes
`DifferentialPWM` directly to:

```text
/ROBOT_NAME/actuator/wheels/base/pwm
```

This avoids ROS 1 discovery, rosbridge, `car_cmd_switch_node`, and the ROS
`kinematics_node`. It does not ignore calibration: it reproduces the inverse
kinematics calculation using `baseline`, `radius`, `k`, `gain`, `trim`, and
`limit`.

DTPS is a viable alternative transport for this repository's PS4 control,
especially if matching the current official Duckietown viewer architecture is
more important than retaining the ROS command-selection pipeline. The current
implementation deliberately keeps rosbridge plus `Twist2DStamped` so the
onboard `car_cmd_switch_node` and `kinematics_node` remain authoritative. A
future DTPS backend should be a separate command transport and must preserve
arming, emergency-stop behavior, calibration handling, and the exact action
labels written to imitation-learning datasets.

## Configuration Boundary

CLI arguments are converted to a run-local configuration and stored in both
`config.json` and checkpoints. Live evaluation reads checkpoint configuration
so model preprocessing and action semantics do not need to be re-entered.

Local start configurations are separate experiment inputs. Their source path
and parsed contents are recorded in run metadata.

## Artifact Boundary

Generated data and checkpoints live outside Git under `~/duckietown`.
Repository-local JSON files are configuration inputs; `configs/*.json` is
ignored while templates are versioned.

See [Outputs and checkpoints](../reference/outputs-and-checkpoints.md) for the
persistent contracts.
