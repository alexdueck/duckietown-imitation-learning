# Physical Duckiebot Camera Input Check

`capture_duckiebot_camera.py` reads one frame from the physical Duckiebot's
read-only ROS camera topic. It saves the original compressed message without
re-encoding it and writes a JSON file with its shape, ROS timestamp, checksum,
and RGB channel means.

The default topic is:

```text
/ROBOT_NAME/camera_node/image/compressed
```

The script does not publish to any robot topic and cannot command the wheels.

On macOS, Docker Desktop usually cannot resolve the Duckiebot's `.local`
hostname through mDNS. When GUI tools were started with `--ip`, the script
automatically reads the current numeric robot address from `ROS_MASTER_URI`
and pins both robot hostnames to that address in `/etc/hosts` inside the
ephemeral container before subscribing. The explicit mapping is installed even
if Docker DNS happens to resolve the name, because mixed or slow DNS/mDNS
resolution can otherwise make direct ROS node connections unreliable.
No hard-coded robot IP is required. Use `--no-hosts-fix` to disable this or
`--robot-ip ADDRESS` to override the detected address.

## Capture a raw frame

First confirm that the robot is visible:

```bash
dts fleet discover
```

From the repository root, create a local output directory and open the
Duckietown GUI-tools container with the repository mounted into it:

```bash
mkdir -p duckiebot_captures
dts start_gui_tools \
  --ip \
  --mount "$(pwd):/workspace" \
  ROBOT_NAME
```

The prompt is now inside the ROS-enabled container. Run:

```bash
python3 /workspace/capture_duckiebot_camera.py \
  ROBOT_NAME \
  --output /workspace/duckiebot_captures/camera_raw.jpg
```

The host-side `duckiebot_captures/` directory will contain:

```text
camera_raw.jpg
camera_raw.json
```

The JPEG contains the exact compressed bytes published by the camera. The JSON
records the decoded dimensions and enough metadata to identify the frame.

## Save a policy-input preview

For an imitation-learning model trained with the Duckiematrix preprocessing in
this repository, retain the image below row 200 and resize it to 224 x 224:

```bash
python3 /workspace/capture_duckiebot_camera.py \
  ROBOT_NAME \
  --output /workspace/duckiebot_captures/camera_il_raw.jpg \
  --policy-input-output /workspace/duckiebot_captures/camera_il_input.png \
  --crop-y-start 200 \
  --image-size 224
```

For a gym-duckietown PPO checkpoint trained with the current defaults, use
`--crop-y-start 0`. Always prefer the actual preprocessing values stored with
or documented for the selected checkpoint over these examples.

The preview converts OpenCV's decoded BGR array to RGB explicitly, then crops
and resizes it. ImageNet tensor normalization is deliberately not applied, so
the resulting file remains visually inspectable.

## What to inspect

Check the raw frame and policy preview for:

- expected resolution, normally 640 x 480 for the current experiments;
- natural colors, especially red and blue objects;
- sharp focus and acceptable motion blur;
- the same road region and horizon position seen during training;
- visible lane markings after the configured crop;
- strong shadows, clipping, reflections, or fisheye distortion not represented
  during training.

Use the JSON `rgb_channel_mean_0_255` values as a diagnostic rather than a
pass/fail criterion. Channel means depend strongly on the scene. A visual red
object appearing blue is a more reliable indication of a channel-order error.

## Control the Duckiebot and collect data

`physical_duckiebot_control.py` combines manual teleoperation and physical
model control. It runs directly in the macOS `gymdt39_venv`; do not start GUI
tools. Both the compressed camera subscription and `Twist2DStamped` command
publication use outbound connections to the robot's rosbridge WebSocket. The
program always starts in manual mode and disarmed. It needs neither ROS Python
packages nor `ROS_MASTER_URI`.

See
[Physical Duckiebot Control Architecture](../development/architecture.md#physical-duckiebot-control-architecture)
for the complete host/robot data flow and the DTPS alternative used by
`dts duckiebot keyboard_control`.

```bash
source ~/virtualenvs/gymdt39_venv/bin/activate
python -m pip install -r requirements/gym-duckietown.txt
python physical_duckiebot_control.py \
  ROBOT_NAME \
  --input keyboard
```

Pass `--robot-ip ADDRESS` or set `ROBOT_IP` when
`ROBOT_NAME.local` does not resolve reliably. The default topics are:

```text
/ROBOT_NAME/camera_node/image/compressed
/ROBOT_NAME/joy_mapper_node/car_cmd
```

Keyboard controls are W/S or Up/Down for throttle, A/D or Left/Right for
steering, Enter to arm/disarm, R to start/stop a recording, Space for the
latched emergency stop, C to clear it, M to switch between manual and model
mode, I to switch the manual input device, and Escape to exit. Mode or input
changes disarm the controller and publish zero.

### PS4 controller on macOS

The Mac process can read an SDL-compatible controller directly:

```bash
source ~/virtualenvs/gymdt39_venv/bin/activate
python physical_duckiebot_control.py \
  ROBOT_NAME \
  --input ps4
```

Use `--controller-index` when more than one SDL controller is connected.
The default SDL mapping uses the left stick, Cross to arm/disarm, Options to
start/stop recording, Circle for emergency stop, and Triangle to clear the
stop. SDL mappings can vary by OS or USB/Bluetooth mode; override them with
`--throttle-axis`, `--steering-axis`, `--arm-button`, `--record-button`,
`--emergency-button`, and `--clear-button`. Press I in the focused GUI to
switch between keyboard and PS4 input. If no controller is available, the
switch is rejected and the current input remains active.

The conservative physical defaults limit full input to 0.10 m/s and
1.50 rad/s. Reverse is enabled, and absolute stick values below 0.08 are
treated as exactly zero. Use `--forward-only`, `--deadzone`,
`--max-linear-velocity`, or `--max-angular-velocity` to override these
defaults. Actual parsed limits and the drive profile are saved in every
recording's `meta.json`.

PS4 input is applied directly without throttle, steering, or command ramps;
the analog stick itself supplies the continuous target. Keyboard control keeps
the ramps because its keys are binary. Use `--rate-limit-analog` only when
analog slew limiting is explicitly desired.

Each press of the record control creates a new
`~/duckietown/data/imitation_learning/train/run_*` directory. It contains raw
camera frames and an `actions.csv` accepted by `train_imitation_learning.py`.
Each frame is written once and paired with the effective left/right
wheel-equivalent values of the command actually published at that control
tick. Files are streamed and flushed during driving, so completed samples
remain usable after an interruption. Recording will not start until a fresh
camera frame is available.

Keep the robot raised off the ground for the first mapping test. Confirm
steering direction and the emergency stop before driving on the floor.

## Inspect checkpoint actions on captured images

The offline inspector detects IL checkpoints through `model_state_dict` and
PPO checkpoints through `policy_state_dict`. It shows the deterministic wheel
commands that would be sent to gym-duckietown, together with a qualitative
direction arrow:

```bash
python view_model_actions_on_images.py \
  duckiebot_captures \
  ~/duckietown/checkpoints/PATH/TO/best.pt
```

Use `D` for the next image, `A` for the previous image, and `Escape` or `Q` to
exit. PPO preprocessing and action mapping are read from the checkpoint. Older
IL checkpoints trained on `images_processed` do not store the legacy crop
explicitly; the inspector infers `crop_y_start=200` and a JPEG round trip for
those checkpoints. Override the inference with `--crop-y-start` or
`--jpeg-stage` when the training preprocessing differed.

## Switch to model control

Pass an IL or PPO checkpoint when starting the same controller. The checkpoint
type is detected automatically:

```bash
source ~/virtualenvs/gymdt39_venv/bin/activate
python -m pip install -r requirements/gym-duckietown.txt
python physical_duckiebot_control.py \
  ROBOT_NAME \
  --checkpoint ~/duckietown/checkpoints/PATH/TO/best.pt
```

The process still starts in manual mode and disarmed. Press M to select model
mode, then Enter to arm it. M returns to manual mode. The GUI always displays
the active mode and selected manual input device. Mode or input changes
publish zero and require explicit re-arming.

The script uses `--robot-ip` or the `ROBOT_IP` environment variable when
provided; otherwise it resolves `ROBOT_NAME.local` on the Mac. The default
topics are:

```text
/ROBOT_NAME/camera_node/image/compressed
/ROBOT_NAME/joy_mapper_node/car_cmd
```

This direct connection does not use `ROS_MASTER_URI` or a GUI-tools
`/etc/hosts` entry.

The window shows the live camera image and current mode, input, arming,
E-stop, recording, wheel-action, `v`/`omega`, and direction-arrow state. In
model mode it additionally separates raw policy controls, gym wheel actions,
scaled requested wheels, and the effective wheel-equivalent action represented
by the published command.

Default limits in both modes are
`max_linear_velocity=0.10 m/s` and
`max_angular_velocity=1.50 rad/s`; reverse motion is not blocked. Model
inference is deterministic. The checkpoint-specific action mapping first
produces normalized left/right wheel commands. `--wheel-action-scale` then
multiplies both wheels by the same value, preserving their ratio; its default
is `1.0`, for an allowed post-scale range of `[-1, 1]`. A camera frame older
than 0.5 s or, normally, 0.5 s without a new command stops and disarms the
controller; adjust these thresholds with `--max-frame-age` and
`--command-timeout`. For a deliberately low `--max-inference-rate`, the
command timeout automatically grows to 1.25 times the selected period unless
you set it explicitly.

Every new unique frame is submitted by default, with no fixed frequency cap.
If inference is slower than the camera, only the newest waiting frame is kept
rather than building a stale queue. Use `--max-inference-rate 10` to impose a
lower rate. Independent `v`/`omega` slew limiting is off by default because it
can temporarily alter the wheel ratio; enable it with
`--rate-limit-commands` and configure the acceleration limits when desired.
For example:

```bash
python physical_duckiebot_control.py ROBOT_NAME \
  --checkpoint CHECKPOINT \
  --wheel-action-scale 0.5 \
  --max-linear-velocity 0.05 \
  --max-angular-velocity 1.0 \
  --max-inference-rate 10
```

Processed frames and their actions are recorded automatically below
`~/duckietown/data/physical_control/run_*` while model mode is selected. Each
run contains:

```text
images/       raw compressed camera payloads
actions.csv   policy controls, model/scaled wheels, published v/omega, timing
meta.json     checkpoint, preprocessing, topics, limits, and sample count
```

While armed, an image is recorded only with the command produced from that
image. Press R/Options to toggle recording. Pass `--no-recording` to disable
collection, `--model-output-dir PATH` to select another model-action root, or
`--manual-output-dir PATH` to select another manual IL-data root.

## Troubleshooting

If `capture_duckiebot_camera.py` reports that ROS Python packages are
unavailable, it was started on the host rather than in the ROS-enabled
GUI-tools container.

If no message arrives, inspect the active topic and publication rate inside the
container:

```bash
rostopic list | grep camera
rostopic hz /ROBOT_NAME/camera_node/image/compressed
```

Pass a different discovered topic with `--topic`. If an output file already
exists, choose a new filename or explicitly pass `--overwrite`.
