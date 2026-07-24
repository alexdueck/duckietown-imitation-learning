# Developer Architecture

## Repository Shape

The repository is intentionally script-oriented. Training and interactive tools
can be run directly, while shared behavior is extracted into modules when it
must remain consistent across tools.

## Duckiematrix Entry Points

| File | Responsibility |
| --- | --- |
| `imitation_learning.py` | Manual control, observation capture, telemetry, and dataset writing |
| `data_viewer.py` | Browse collected images, actions, reward, and telemetry |
| `train_imitation_learning.py` | Supervised image-to-wheel training |
| `live_eval_imitation_policy.py` | Execute an IL policy in Duckiematrix |
| `train_rl_ppo_duckiematrix.py` | Experimental Duckiematrix PPO trainer |
| `duckiematrix_telemetry.py` | Read pose/lane telemetry from Duckiematrix |
| `rl_rewards.py` | Duckiematrix reward adapters |

## gym-duckietown Entry Points

| File | Responsibility |
| --- | --- |
| `manual_control_gym_duckietown.py` | Manual driving, reward sidebar, seeds, and pose capture |
| `live_eval_imitation_policy_gym_duckietown.py` | Visual IL transfer evaluation |
| `train_rl_ppo_gym_duckietown.py` | Main PPO trainer and deterministic evaluation |
| `live_eval_rl_policy_gym_duckietown.py` | Visual PPO checkpoint evaluation |
| `duckietown_rewards.py` | Reward selection, state tracking, breakdowns, and compatibility patch |
| `velopose_reward.py` | Pure custom reward equations |
| `duckietown_action_control.py` | Policy-control to wheel-action mapping |
| `duckiebot_hardware_control.py` | Fail-closed wheel-action to physical chassis-command mapping |
| `gym_duckietown_start_config.py` | Seed/pose configuration, validation, and sampling |

## Shared Modules

| File | Responsibility |
| --- | --- |
| `rl_models.py` | CNN actor, Q-network scaffold, tanh log probability, and IL actor loading |
| `duckietown_paths.py` | Artifact locations below `~/duckietown` |
| `cli_completion.py` | Optional argcomplete integration |
| `ppo_control_tests.py` | PPO invariant, Pendulum, and image-control tests |
| `preprocess.py` | Legacy optional offline image preprocessing |

## Physical Duckiebot Entry Points

| File | Responsibility |
| --- | --- |
| `host_ps4_controller_bridge.py` | Read the macOS SDL GameController and serve normalized input states over TCP |
| `physical_duckiebot_teleop.py` | Arming, emergency stop, input mixing, physical limits, camera subscription, rosbridge command transport, and recording |
| `run_physical_duckiebot_teleop.sh` | Load ROS Noetic and the Duckietown catkin workspace before starting teleop in GUI tools |
| `duckiebot_teleop_input.py` | Device adapters, bridge protocol, deadzone, and keyboard-only ramps |
| `duckiebot_dataset_recorder.py` | Stream aligned compressed camera frames and effective wheel-equivalent actions |
| `duckiebot_hardware_control.py` | ROS-independent fail-closed conversion from normalized wheels to bounded `v`/`omega` |

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

`velopose_reward.py` contains equations over NumPy values and has no direct
simulator dependency.

`duckietown_rewards.py` adapts simulator state into those equations, tracks
previous position/potential, recognizes done reasons, and returns nested
breakdowns.

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

The physical runtime spans the macOS host, a GUI-tools container, and the
onboard Duckietown stack. This repository does not install or replace code on
the robot.

| Component | Location | Ownership |
| --- | --- | --- |
| PS4 controller and `host_ps4_controller_bridge.py` | macOS host | This repository |
| `physical_duckiebot_teleop.py` | GUI-tools container | This repository |
| `rosbridge_websocket` | Physical Duckiebot | ROS/Duckietown |
| `car_cmd_switch_node` | Physical Duckiebot | Duckietown `dt-core` |
| `kinematics_node` | Physical Duckiebot | Duckietown `dt-core` |
| `wheels_driver_node` | Physical Duckiebot | Duckietown hardware stack |

The control path is:

```text
PS4 over Bluetooth
    |
    v
host_ps4_controller_bridge.py on macOS
    |
    | TCP input-state stream to container port 8765
    v
physical_duckiebot_teleop.py in GUI tools
    |
    | deadzone, arming, emergency stop, wheel mixing, physical limits
    | outbound WebSocket connection to ROBOT_IP:9001
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
    | ROS CompressedImage; subscriber connection initiated by GUI tools
    v
LatestCamera in physical_duckiebot_teleop.py
    |
    | newest unique frame + effective command sent for that control tick
    v
duckiebot_dataset_recorder.py
    |
    +-- images/*
    +-- actions.csv
    +-- meta.json
```

### Why commands use rosbridge on Docker Desktop

ROS 1 uses its master for discovery, but topic data is peer-to-peer. After
discovering a publisher, the subscriber connects back to the publisher's
advertised TCPROS address. A `rospy.Publisher` inside Docker Desktop advertised
an address such as:

```text
http://docker-desktop:RANDOM_PORT/
```

The physical robot could register the subscriber through the shared ROS
master, but could not resolve or reach that container-internal callback. This
failure is directional: camera subscription works because the container
connects to a publisher on the robot.

For commands, the container therefore opens an outbound connection to the
fixed rosbridge server on the robot:

```text
GUI-tools container --> ws://ROBOT_IP:9001
```

From the robot's perspective this is an incoming WebSocket connection. Over
the established bidirectional socket, teleop sends rosbridge `advertise` and
`publish` operations. `rosbridge_websocket` then acts as a ROS publisher
onboard the robot, where `car_cmd_switch_node` can reach it locally. Native
Linux deployments with a robot-reachable ROS host may opt into direct TCPROS
with `--command-transport ros`.

### Command-source selection

`car_cmd_switch_node` starts with `current_src_name = "joystick"`. Its default
configuration also maps the FSM state `NORMAL_JOYSTICK_CONTROL` to the
`joystick` source:

```text
joystick source --> joy_mapper_node/car_cmd
lane source     --> lane_controller_node/car_cmd
stop source     --> simple_stop_controller_node/car_cmd
```

The current teleop runtime does not publish an FSM state or explicitly switch
the robot into joystick mode. It relies on the onboard switch starting in
`joystick`. If an FSM later selects lane following or stop, PS4 messages remain
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
