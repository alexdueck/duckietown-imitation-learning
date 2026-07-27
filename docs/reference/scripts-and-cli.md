# Executable Scripts

This is the central user-facing reference for every executable entry point in
the repository root. Tests and internal modules are intentionally excluded.

Run a Python script with `--help` to see its exact current flags and defaults:

```bash
python train_rl_ppo_gym_duckietown.py --help
```

The tables below explain what to run and which parameter groups matter.
Platform setup and complete workflows remain in the linked guides.

## Quick Overview

### Collect and inspect data

| Script | Use it to | Runs in |
| --- | --- | --- |
| [`imitation_learning.py`](#imitation_learningpy) | Drive in Duckiematrix and record image/action data | Duckiematrix environment |
| [`physical_duckiebot_teleop.py`](#physical_duckiebot_teleoppy) | Drive a physical Duckiebot and optionally record image/action data | `dts start_gui_tools` container |
| [`run_physical_duckiebot_teleop.sh`](#run_physical_duckiebot_teleopsh) | Start physical teleoperation with the ROS environment sourced | `dts start_gui_tools` container |
| [`host_ps4_controller_bridge.py`](#host_ps4_controller_bridgepy) | Make a controller connected to macOS available inside Docker Desktop | macOS host |
| [`capture_duckiebot_camera.py`](#capture_duckiebot_camerapy) | Save one physical camera frame and metadata | `dts start_gui_tools` container |
| [`data_viewer.py`](#data_viewerpy) | Inspect recorded frames, actions, rewards, and telemetry | Duckiematrix environment |
| [`preprocess.py`](#preprocesspy) | Create legacy cropped/resized IL images | Duckiematrix environment, or Python with pandas/Pillow |

### Train models

| Script | Use it to | Runs in |
| --- | --- | --- |
| [`train_imitation_learning.py`](#train_imitation_learningpy) | Train an image-to-wheel-action IL model | Python environment with PyTorch |
| [`train_rl_ppo_gym_duckietown.py`](#train_rl_ppo_gym_duckietownpy) | Train the main PPO implementation in gym-duckietown | gym-duckietown environment |
| [`train_rl_ppo_duckiematrix.py`](#train_rl_ppo_duckiematrixpy) | Experiment with PPO in Duckiematrix | Duckiematrix environment |

### Evaluate and run models

| Script | Use it to | Runs in |
| --- | --- | --- |
| [`manual_control_gym_duckietown.py`](#manual_control_gym_duckietownpy) | Drive manually and inspect rewards/start poses | gym-duckietown environment |
| [`live_eval_imitation_policy.py`](#live_eval_imitation_policypy) | Run an IL policy in Duckiematrix | Duckiematrix environment |
| [`live_eval_imitation_policy_gym_duckietown.py`](#live_eval_imitation_policy_gym_duckietownpy) | Run an IL policy in gym-duckietown | gym-duckietown environment |
| [`live_eval_rl_policy_gym_duckietown.py`](#live_eval_rl_policy_gym_duckietownpy) | Run a PPO policy in gym-duckietown | gym-duckietown environment |
| [`view_model_actions_on_images.py`](#view_model_actions_on_imagespy) | Inspect IL/PPO outputs on saved images without driving | Python environment with PyTorch |
| [`physical_duckiebot_model_control.py`](#physical_duckiebot_model_controlpy) | Let an IL/PPO policy drive a physical Duckiebot | macOS gym/PyTorch environment |

### Analyze runs

| Script | Use it to | Runs in |
| --- | --- | --- |
| [`analyze_rl_training_run.py`](#analyze_rl_training_runpy) | Generate an HTML report from PPO CSV logs | Any environment with its analysis dependencies |

## Data Collection and Inspection

### `imitation_learning.py`

Collects a Duckiematrix imitation-learning run consisting of camera images,
effective wheel actions, and available telemetry. The interactive window is
also useful for checking observations and reward signals before recording.

```bash
python imitation_learning.py
python imitation_learning.py --observe-only
```

Parameters:

- `--observe-only` (alias `--observation-only`) displays the live stream and
  telemetry without writing a dataset.

The Duckiematrix entity and connection are configured by the Duckiematrix
setup rather than this small CLI. See
[Duckiematrix setup](../getting-started/duckiematrix.md).

### `physical_duckiebot_teleop.py`

Drives a physical Duckiebot with keyboard, a locally visible PS4 controller,
or the macOS controller bridge. It can record compressed camera frames aligned
with the effective wheel-equivalent actions.

```bash
python physical_duckiebot_teleop.py realbot \
  --input keyboard \
  --command-transport rosbridge \
  --robot-ip 192.168.2.125
```

Important parameter groups:

| Group | Parameters | Meaning |
| --- | --- | --- |
| Robot connection | `robot_name`, `--robot-ip`, `--camera-topic`, `--command-topic` | Select the robot and optionally override ROS topics |
| Transport | `--command-transport`, `--rosbridge-port`, `--no-hosts-fix` | Choose rosbridge or direct ROS/TCPROS |
| Input | `--input`, `--bridge-host`, `--bridge-port` | Choose keyboard, direct PS4, or the macOS bridge |
| Recording | `--output-dir`, `--max-frame-age` | Select dataset location and reject stale camera frames |
| Timing | `--control-rate`, `--status-period` | Set command frequency and console status frequency |
| Motion limits | `--max-linear-velocity`, `--max-angular-velocity`, `--max-linear-acceleration`, `--max-angular-acceleration` | Bound chassis commands and optional ramps |
| Input behavior | `--rate-limit-analog`, `--forward-only`, `--deadzone` | Configure analog ramps, reverse motion, and stick deadzone |
| Controller mapping | `--throttle-axis`, `--steering-axis`, `--arm-button`, `--record-button`, `--emergency-button`, `--clear-button` | Override SDL axis/button indices |

The physical setup guide documents keyboard/controller controls, arming,
recording, and E-stop behavior:
[Physical Duckiebot camera and control](../getting-started/physical-duckiebot-camera.md).

### `run_physical_duckiebot_teleop.sh`

Convenience wrapper for `physical_duckiebot_teleop.py` inside
`dts start_gui_tools`. It sources ROS Noetic and the Duckietown catkin
workspace, then forwards every argument unchanged.

```bash
/workspace/run_physical_duckiebot_teleop.sh realbot --input keyboard
```

It has no parameters of its own. Use:

```bash
/workspace/run_physical_duckiebot_teleop.sh --help
```

to see the parameters forwarded to `physical_duckiebot_teleop.py`.

### `host_ps4_controller_bridge.py`

Reads an SDL-compatible controller connected to macOS and serves normalized
input states over TCP to a teleoperation process in Docker Desktop.

```bash
python host_ps4_controller_bridge.py --list
python host_ps4_controller_bridge.py
```

| Parameters | Meaning |
| --- | --- |
| `--list`, `--controller-index` | Discover controllers or choose one |
| `--port`, `--rate` | Configure the TCP port and controller sampling rate |
| `--throttle-axis`, `--steering-axis` | Override SDL axis indices |
| `--arm-button`, `--record-button`, `--emergency-button`, `--clear-button` | Override SDL button indices |

Run this script on the Mac and start `physical_duckiebot_teleop.py` with
`--input ps4-bridge` in the container.

### `capture_duckiebot_camera.py`

Waits for one compressed ROS camera message and stores the original frame plus
diagnostic JSON metadata. It can also store a crop/resize preview of the image
that a policy would receive.

```bash
python3 /workspace/capture_duckiebot_camera.py realbot \
  --output /workspace/duckiebot_captures/camera_raw.jpg
```

| Parameters | Meaning |
| --- | --- |
| `robot_name`, `--robot-ip`, `--topic` | Select robot, address, and camera topic |
| `--output`, `--metadata-output` | Select image and JSON destinations |
| `--policy-input-output` | Also save the preprocessed policy-input preview |
| `--crop-y-start`, `--image-size` | Configure preview crop and resize |
| `--timeout` | Stop waiting when no frame arrives |
| `--overwrite` | Permit replacement of existing output files |
| `--no-hosts-fix` | Disable the optional container hostname repair |

See [Physical Duckiebot camera and control](../getting-started/physical-duckiebot-camera.md)
for container startup and ROS networking diagnostics.

### `data_viewer.py`

Opens an existing imitation-learning run and browses its images alongside
stored actions, rewards, and lane telemetry.

```bash
python data_viewer.py
python data_viewer.py ~/duckietown/data/imitation_learning/train/RUN_DIR
```

Parameters:

- `path` selects a run; without it, the newest training run is opened.
- `--display-mode fit|native` chooses scaled-to-window or stored pixel size.

### `preprocess.py`

Legacy preprocessing step that creates `images_processed/` inside an
imitation-learning run. Newer inference paths can crop and resize online, so
this script is mainly needed for datasets/checkpoints that use the old
processed-image convention.

```bash
python preprocess.py --input-run ~/duckietown/data/imitation_learning/train/RUN_DIR
```

| Parameters | Meaning |
| --- | --- |
| `--input-run` | Run directory containing the original images |
| `--crop-y-start` | First source-image row retained |
| `--out-width`, `--out-height` | Stored image dimensions |
| `--jpeg-quality` | JPEG encoding quality |
| `--overwrite` | Replace existing processed images |

## Training

### `train_imitation_learning.py`

Trains MobileNetV3-Small or ResNet-18 to regress left/right wheel actions from
camera images. It accepts either one run or a root containing explicit
`train/` and `val/` splits.

```bash
python train_imitation_learning.py \
  --run-dir ~/duckietown/data/imitation_learning \
  --model mobilenet_v3_small \
  --epochs 20
```

| Group | Parameters | Meaning |
| --- | --- | --- |
| Data | `--run-dir`, `--image-dir`, `--val-fraction`, `--limit`, `--skip-missing` | Select samples, image variant, validation split, and smoke-test limits |
| Output | `--output-dir`, `--experiment-name` | Select checkpoint root and run-name prefix |
| Model | `--model`, `--no-pretrained`, `--train-backbone`, `--image-size` | Choose architecture, initialization, fine-tuning scope, and input size |
| Optimization | `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay` | Configure training |
| Runtime | `--num-workers`, `--seed`, `--device` | Configure data loading, reproducibility, and PyTorch device |

Outputs and checkpoint contents are described in
[Outputs and checkpoints](outputs-and-checkpoints.md).

### `train_rl_ppo_gym_duckietown.py`

Main PPO trainer. It trains an image policy in gym-duckietown, periodically
evaluates fixed scenarios, writes diagnostics, and saves resumable
checkpoints.

```bash
python train_rl_ppo_gym_duckietown.py \
  --map-name loop_empty \
  --reward-function velopose \
  --action-mode throttle_steering \
  --max-throttle 0.5 \
  --max-steering 0.5 \
  --total-steps 100000
```

| Group | Parameters | Meaning |
| --- | --- | --- |
| Run/output | `--output-dir`, `--exp-name`, `--seed`, `--device`, `--log-level` | Name and locate the run; configure reproducibility/runtime |
| Environment | `--map-name`, `--frame-skip`, `--frame-rate`, `--robot-speed`, `--simulator-max-steps`, `--camera-width`, `--camera-height` | Configure gym-duckietown |
| Visual variation | `--domain-rand`, `--distortion`, `--source-observation-channel-order` | Configure observation domain and channel interpretation |
| Reward | `--reward-function`, `--vd2pp-distance-weight` | Select the optimized reward and its optional distance term |
| Model/input | `--model`, `--image-size`, `--crop-y-start` | Select encoder and policy input transform |
| Action mapping | `--action-mode`, `--fixed-throttle`, `--max-throttle`, `--max-steering` | Map policy outputs to wheel commands |
| Initialization/resume | `--imitation-checkpoint`, `--resume-checkpoint`, `--initial-log-std`, `--min-log-std`, `--max-log-std` | Warm-start or resume policy/value state and bound exploration |
| PPO | `--total-steps`, `--rollout-steps`, `--epochs`, `--batch-size`, `--gamma`, `--gae-lambda`, `--clip-ratio`, `--policy-lr`, `--value-lr`, `--entropy-coef`, `--value-coef`, `--max-grad-norm` | Configure rollout collection and updates |
| Starts | `--max-episode-steps`, `--reset-random-warmup-steps`, `--reset-random-warmup-retries`, `--reset-random-action-scale`, `--start-seeds-config`, `--hard-start-probability`, `--accept-start-angle-deg` | Configure episode boundaries and start-state distribution |
| Evaluation | `--eval-interval-rollouts`, `--eval-steps`, `--eval-seeds`, `--eval-stochastic` | Configure periodic policy evaluation |
| Diagnostics | `--render-training`, `--debug-initial-action` | Render training or print the first deterministic action |

See [Common workflows](../getting-started/workflows.md) for recommended
training, resuming, fixed-start, and IL-warm-start commands. See
[PPO](../methodology/ppo.md) for algorithm semantics.

### `train_rl_ppo_duckiematrix.py`

Experimental PPO trainer for Duckiematrix. Its reward adapters have not
produced the same reliable lane-following setup as gym-duckietown, so use it
for research experiments rather than as the current default RL path.

```bash
python train_rl_ppo_duckiematrix.py \
  --entity-name map_0/vehicle_0 \
  --reward-function velopose \
  --total-steps 100000
```

| Group | Parameters | Meaning |
| --- | --- | --- |
| Run/environment | `--output-dir`, `--entity-name`, `--map-name`, `--seed`, `--device` | Select Duckiematrix entity and run metadata |
| Reward/model | `--reward-function`, `--model`, `--image-size`, `--crop-y-start`, `--camera-width`, `--camera-height` | Configure reward and observation model |
| Initialization | `--imitation-checkpoint`, `--resume-checkpoint`, `--initial-log-std`, `--min-log-std`, `--max-log-std` | Warm-start, resume, and configure exploration |
| PPO | `--total-steps`, `--rollout-steps`, `--epochs`, `--batch-size`, `--gamma`, `--gae-lambda`, `--clip-ratio`, `--policy-lr`, `--value-lr`, `--entropy-coef`, `--value-coef`, `--max-grad-norm` | Configure rollout collection and optimization |
| Resets | `--max-episode-steps`, `--reset-random-warmup-steps`, `--reset-random-warmup-retries`, `--reset-random-action-scale` | Configure episode boundaries and random warmup |
| Evaluation | `--eval-interval-rollouts`, `--eval-steps`, `--eval-stochastic` | Configure periodic evaluation |
| Diagnostics | `--debug-initial-action` | Print the first deterministic action |

## Simulation Evaluation

### `manual_control_gym_duckietown.py`

Interactive gym-duckietown driver for checking maps, rewards, action behavior,
and deterministic start scenarios before training.

```bash
python manual_control_gym_duckietown.py \
  --map-name loop_empty \
  --reward-functions velopose,posepot,vd2pp
```

| Group | Parameters | Meaning |
| --- | --- | --- |
| Environment | `--map-name`, `--seed`, `--max-steps`, `--frame-rate`, `--frame-skip`, `--camera-width`, `--camera-height`, `--robot-speed`, `--accept-start-angle-deg` | Configure simulator execution |
| Rendering/randomization | `--draw-curve`, `--draw-bbox`, `--domain-rand`, `--distortion`, `--dynamics-rand`, `--camera-rand` | Enable diagnostics and domain variation |
| Starts | `--start-pose-file`, `--start-seeds-config`, `--auto-reset` | Fix a pose, collect poses, or reset automatically |
| Keyboard response | `--forward-target`, `--backward-target`, `--turn-target`, `--throttle-rate`, `--steering-rate`, `--auto-center-rate`, `--boost-multiplier` | Tune manual control ramps and limits |
| Rewards | `--reward-functions`, `--posepot-gamma`, `--vd2pp-distance-weight` | Select displayed reward breakdowns |
| Output/logging | `--screenshot-path`, `--log-level` | Configure screenshots and logs |

The [common workflows](../getting-started/workflows.md) page documents keys
for resetting, entering seeds, and saving poses.

### `live_eval_imitation_policy.py`

Runs a trained IL checkpoint in Duckiematrix and sends its wheel predictions
to the simulated vehicle.

```bash
python live_eval_imitation_policy.py --checkpoint /path/to/best.pt
```

| Group | Parameters | Meaning |
| --- | --- | --- |
| Model/runtime | `--checkpoint`, `--device`, `--entity-name`, `--max-steps`, `--sample-period` | Select model, vehicle, device, duration, and control period |
| Camera/input | `--camera-width`, `--camera-height`, `--crop-y-start`, `--image-size` | Reproduce the training observation transform |
| JPEG compatibility | `--jpeg-quality`, `--jpeg-roundtrip-stage`, `--no-jpeg-roundtrip` | Emulate legacy dataset compression when required |
| Action/episodes | `--no-clip-actions`, `--reset-on-done` | Disable action clipping or continue across episodes |

### `live_eval_imitation_policy_gym_duckietown.py`

Visually evaluates a Duckiematrix-trained IL checkpoint in
gym-duckietown. This is also a sim-to-sim domain-transfer check.

```bash
python live_eval_imitation_policy_gym_duckietown.py \
  --checkpoint /path/to/best.pt \
  --map-name loop_empty
```

| Group | Parameters | Meaning |
| --- | --- | --- |
| Model/input | `--checkpoint`, `--device`, `--crop-y-start`, `--image-size`, `--source-observation-channel-order` | Load the policy and reproduce its input transform |
| Environment | `--map-name`, `--seed`, `--max-steps`, `--episodes`, `--frame-rate`, `--frame-skip`, `--camera-width`, `--camera-height`, `--robot-speed`, `--accept-start-angle-deg` | Configure evaluation scenarios |
| Randomization/rendering | `--domain-rand`, `--distortion`, `--dynamics-rand`, `--camera-rand`, `--draw-curve`, `--draw-bbox` | Configure domain variation and overlays |
| Reward | `--reward-function`, `--posepot-gamma`, `--vd2pp-distance-weight` | Select the reported return |
| Viewer behavior | `--stop-on-done`, `--start-paused`, `--print-every` | Configure reset/pause and console output |
| Output | `--returns-file`, `--screenshot-path`, `--log-level` | Store returns/screenshots and configure logs |

### `live_eval_rl_policy_gym_duckietown.py`

Visually evaluates a PPO checkpoint. Checkpoint configuration supplies most
defaults, so explicit overrides are mainly useful for controlled comparisons.

```bash
python live_eval_rl_policy_gym_duckietown.py \
  --checkpoint /path/to/best_return.pt \
  --stop-on-done
```

| Group | Parameters | Meaning |
| --- | --- | --- |
| Checkpoint/model | `--checkpoint`, `--device`, `--image-size`, `--crop-y-start`, `--source-observation-channel-order` | Load and optionally override policy input settings |
| Scenario | `--map-name`, `--seed`, `--start-seeds-config`, `--eval-pose-index`, `--max-steps`, `--episodes` | Choose evaluation map/start and horizon |
| Environment | `--frame-rate`, `--frame-skip`, `--camera-width`, `--camera-height`, `--robot-speed`, `--accept-start-angle-deg` | Override checkpoint environment settings |
| Variation | `--domain-rand`, `--no-domain-rand`, `--distortion`, `--no-distortion` | Override checkpoint randomization |
| Reward | `--reward-function`, `--vd2pp-distance-weight` | Override the reported reward |
| Policy/viewer | `--stochastic`, `--stop-on-done`, `--start-paused`, `--print-every` | Sample actions or configure viewer behavior |
| Output | `--returns-file`, `--screenshot-path`, `--log-level` | Store results/screenshots and configure logs |

## Offline and Physical Model Evaluation

### `view_model_actions_on_images.py`

Loads either a supported IL or PPO checkpoint, automatically detects its type,
and displays deterministic wheel commands for each image in a directory. The
panel shows raw model controls, gym wheel actions, and a direction arrow.

```bash
python view_model_actions_on_images.py duckiebot_captures /path/to/checkpoint.pt
```

| Parameters | Meaning |
| --- | --- |
| `image_dir`, `checkpoint` | Select images and model |
| `--device` | Select PyTorch device |
| `--image-size`, `--crop-y-start` | Override checkpoint/input preprocessing |
| `--jpeg-stage`, `--jpeg-quality` | Configure an optional compatibility JPEG round trip |
| `--file-channel-order` | Interpret decoded files as RGB or BGR |

Use `D` and `A` to move to the next and previous image.

### `physical_duckiebot_model_control.py`

Runs a supported IL or PPO checkpoint against the physical camera stream over
rosbridge and publishes bounded `Twist2DStamped` commands. The window displays
the inference image, model controls, published wheel-equivalent actions, and a
direction arrow. By default it records each processed frame with its action.

```bash
python physical_duckiebot_model_control.py \
  realbot /path/to/checkpoint.pt \
  --robot-ip 192.168.2.125
```

| Group | Parameters | Meaning |
| --- | --- | --- |
| Robot connection | `robot_name`, `--robot-ip`, `--rosbridge-port`, `--camera-topic`, `--command-topic` | Connect to rosbridge and choose topics |
| Model/input | `checkpoint`, `--device`, `--image-size`, `--crop-y-start`, `--jpeg-stage`, `--jpeg-quality`, `--file-channel-order` | Load the policy and reproduce preprocessing |
| Wheel range | `--wheel-action-scale` | Apply one common scale to both normalized wheel actions |
| Chassis limits | `--max-linear-velocity`, `--max-angular-velocity` | Bound the published `v` and `omega` |
| Optional slew limits | `--rate-limit-commands`, `--max-linear-acceleration`, `--max-angular-acceleration` | Enable and configure command-rate limiting |
| Stream/watchdog | `--command-timeout`, `--max-frame-age`, `--max-inference-rate` | Stop on stale commands/frames and optionally cap inference frequency |
| Recording/status | `--output-dir`, `--no-recording`, `--status-period` | Configure aligned recording and console status |

The keyboard E-stop/arming behavior and first hardware procedure are described
in [Physical Duckiebot camera and control](../getting-started/physical-duckiebot-camera.md).

## Run Analysis

### `analyze_rl_training_run.py`

Reads the CSV logs in one PPO run directory and creates a self-contained HTML
report with embedded SVG charts. By default it writes
`training_report.html` into the run directory and opens it.

```bash
python analyze_rl_training_run.py \
  ~/duckietown/checkpoints/rl_ppo_gym_duckietown/RUN_DIR
```

| Parameters | Meaning |
| --- | --- |
| `run_dir` | PPO run directory containing CSV logs |
| `--output` | Override the report path |
| `--eval-window` | Number of first/last evaluations compared |
| `--episode-window` | Number of first/last training episodes compared |
| `--diagnostic-window` | Number of first/last PPO updates/rollouts compared |
| `--rolling-window` | Smoothing window for chart lines |
| `--no-open` | Generate without opening a browser |

## Common CLI Conventions

### Device Selection

Deep-learning entry points generally accept:

```text
--device auto|cpu|cuda|mps
```

`auto` prefers CUDA, then MPS, then CPU. Rendering and neural-network devices
are separate: Xvfb/Mesa can render observations while PyTorch uses CUDA.

### CLI Completion

Scripts marked with `PYTHON_ARGCOMPLETE_OK` support optional `argcomplete`
integration. Completion requires shell activation and may be slow when a
script imports simulator or neural-network libraries. It is not a runtime
requirement; `--help` is always available.

### Logging

gym-duckietown entry points generally accept:

```text
--log-level DEBUG|INFO|WARNING|ERROR
```

This also updates loggers in the Duckietown dependency stack.

### Defaults versus Experiment Commands

Commands in [Common workflows](../getting-started/workflows.md) document
configurations used in experiments and may deliberately override parser
defaults. A parser default is not automatically a research recommendation.
