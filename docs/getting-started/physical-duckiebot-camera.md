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
