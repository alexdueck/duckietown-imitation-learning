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

## Drive with keyboard or PS4 controller and collect IL data

Start GUI tools with the repository mounted as above. The teleop program uses
the same camera topic, publishes safe `Twist2DStamped` commands through the
robot's rosbridge WebSocket, and starts disarmed. The outbound WebSocket is
the default because a physical robot cannot connect back to Docker Desktop's
dynamically advertised TCPROS port.

See
[Physical Duckiebot Control Architecture](../development/architecture.md#physical-duckiebot-control-architecture)
for the complete host/container/robot data flow and the DTPS alternative used
by `dts duckiebot keyboard_control`.

```bash
python3 /workspace/physical_duckiebot_teleop.py ROBOT_NAME --input keyboard
```

When starting a detached GUI-tools container with `docker exec`, use the
repository wrapper so the ROS Noetic and Duckietown catkin environments are
loaded first:

```bash
docker exec -it CONTAINER_NAME \
  bash /workspace/run_physical_duckiebot_teleop.sh \
  ROBOT_NAME \
  --input keyboard
```

On macOS, `dts start_gui_tools` requires XQuartz's `xhost` command. If it
fails with `FileNotFoundError: ... 'xhost'`, install XQuartz, log out and back
in, and start XQuartz once:

```bash
brew install --cask xquartz
open -a XQuartz
command -v xhost
```

The last command should print `/opt/X11/bin/xhost`. If it does not, use
`export PATH="/opt/X11/bin:$PATH"` in that terminal before invoking `dts`.
The Avahi warning printed immediately before this error is unrelated; `--ip`
avoids relying on `.local` name resolution.

Keyboard controls are W/S or Up/Down for throttle, A/D or Left/Right for
steering, Enter to arm/disarm, R to start/stop a recording, Space for the
latched emergency stop, C to clear it, and Escape to exit.

### PS4 controller on macOS

Docker Desktop does not expose a Bluetooth controller from macOS directly to
the Linux container. Run the host bridge in a normal Mac terminal first:

```bash
cd ~/git/duckietown-imitation-learning
source ~/virtualenvs/gymdt39_venv/bin/activate
python host_ps4_controller_bridge.py
```

It opens a small status window and waits on TCP port 8765. Leave it running.
Start GUI tools from a second terminal and run the ROS side inside the
container:

```bash
dts start_gui_tools \
  --ip \
  --mount "$(pwd):/workspace" \
  ROBOT_NAME

docker exec -it duckiebot-camera-tools \
  bash /workspace/run_physical_duckiebot_teleop.sh \
  ROBOT_NAME \
  --input ps4-bridge \
  --output-dir /workspace/duckiebot_recordings
```

Docker resolves `host.docker.internal` to the Mac automatically. The ROS side
stops safely if the bridge disconnects or stops sending data. The default SDL
mapping uses the left stick, Cross to arm/disarm, Options to start/stop
recording, Circle for emergency stop, and Triangle to clear the stop. SDL
mappings can vary by OS or USB/Bluetooth mode; pass the following options to
the **host bridge**:
`--throttle-axis`, `--steering-axis`, `--arm-button`, `--record-button`,
`--emergency-button`, and `--clear-button` to override them.

The physical teleop defaults follow `realbot`'s installed joystick and
kinematics parameters: full stick is limited to 0.41 m/s and 8.0 rad/s,
reverse is enabled, and absolute stick values below 0.08 are treated as
exactly zero. Use `--forward-only`, `--deadzone`,
`--max-linear-velocity`, or `--max-angular-velocity` to override these
defaults. Actual parsed limits and the drive profile are saved in every
recording's `meta.json`.

PS4 input is applied directly without throttle, steering, or command ramps;
the analog stick itself supplies the continuous target. Keyboard control keeps
the ramps because its keys are binary. Use `--rate-limit-analog` only when
analog slew limiting is explicitly desired.

To verify detection without starting ROS:

```bash
~/virtualenvs/gymdt39_venv/bin/python host_ps4_controller_bridge.py --list
```

Linux hosts that pass a controller device directly into the container can
still use `--input ps4`.

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

## Troubleshooting

If the script reports that ROS Python packages are unavailable, it was started
on the host rather than in the ROS-enabled GUI-tools container.

If no message arrives, inspect the active topic and publication rate inside the
container:

```bash
rostopic list | grep camera
rostopic hz /ROBOT_NAME/camera_node/image/compressed
```

Pass a different discovered topic with `--topic`. If an output file already
exists, choose a new filename or explicitly pass `--overwrite`.
