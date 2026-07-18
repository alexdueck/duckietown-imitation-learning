# gym-duckietown on Ubuntu 24.04 over SSH

This setup targets a remote Ubuntu 24.04 machine without an attached display.
Training uses PyTorch on CUDA when available, while gym-duckietown renders its
camera observations through an Xvfb OpenGL display.

"Headless training" still needs rendering: the observation is produced by an
OpenGL camera framebuffer. No person needs to see the window, but OpenGL needs
somewhere to believe that a window exists.

## Install Python 3.9

Ubuntu 24.04 does not provide Python 3.9 in its standard repository. One option
is the Deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev
```

Install the rendering and build dependencies:

```bash
sudo apt install -y \
  build-essential \
  xvfb \
  xauth \
  mesa-utils \
  libgl1-mesa-dri \
  libglx-mesa0 \
  libglu1-mesa \
  libosmesa6
```

## Create the Environment

The documented local layout keeps the virtual environment inside the project:

```bash
cd ~/git/duckietown-imitation-learning
python3.9 -m venv venv39_gymdt
source venv39_gymdt/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/gym-duckietown.txt
```

## Install gym-duckietown

```bash
cd ~/git
git clone https://github.com/duckietown/gym-duckietown.git
cd gym-duckietown
python -m pip install -e . --no-deps
```

`--no-deps` avoids gym-duckietown's obsolete NumPy/OpenCV constraints. The
compatible replacements are installed from this project's requirements file.

If Pyglet 1.5.15 produces headless-import problems on the host, the tested
fallback is:

```bash
python -m pip install --upgrade "pyglet==1.5.27"
```

## Make Pyglet Respect Xvfb

The upstream gym-duckietown package selects Pyglet headless mode on every
non-macOS system. That bypasses Xvfb and attempts EGL instead. In the local
gym-duckietown clone, change `src/gym_duckietown/__init__.py` so headless mode
is selected only when no X display exists:

```python
import os
import platform

import pyglet

pyglet.options["headless"] = (
    platform.system() != "Darwin"
    and not bool(os.environ.get("DISPLAY"))
)
```

Set this option before importing simulator modules that create Pyglet windows.
See [Compatibility patches](../development/compatibility-patches.md) for why
this is needed.

## Verify Xvfb and OpenGL

```bash
xvfb-run -a \
  -s "-screen 0 1280x1024x24 +extension GLX" \
  env LIBGL_ALWAYS_SOFTWARE=1 \
  glxinfo -B
```

A working software setup reports a Mesa renderer such as `llvmpipe`.

Then test gym-duckietown:

```bash
xvfb-run -a \
  -s "-screen 0 1280x1024x24 +extension GLX" \
  env LIBGL_ALWAYS_SOFTWARE=1 \
  python -c "from gym_duckietown.simulator import Simulator; env=Simulator(map_name='loop_empty', domain_rand=False); obs=env.reset(); print(obs.shape); env.close()"
```

Expected output includes:

```text
(480, 640, 3)
```

## Start Training

```bash
cd ~/git/duckietown-imitation-learning
source venv39_gymdt/bin/activate

xvfb-run -a \
  -s "-screen 0 1280x1024x24 +extension GLX" \
  env LIBGL_ALWAYS_SOFTWARE=1 \
  python train_rl_ppo_gym_duckietown.py \
    --map-name loop_empty \
    --reward-function velopose \
    --action-mode throttle_steering \
    --max-throttle 0.5 \
    --max-steering 0.5 \
    --total-steps 1000000
```

`--device cuda` is normally unnecessary because `--device auto` selects
CUDA when PyTorch can see it:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

`LIBGL_ALWAYS_SOFTWARE=1` affects OpenGL rendering, not the PyTorch CUDA
device. The simulator can use Mesa while the policy trains on the NVIDIA GPU.

## Performance Note

The current trainer uses one environment. On the measured Ubuntu setup,
environment interaction was the larger cost, while CNN inference and PPO
updates also remained significant. Reducing the camera from 640x480 to 320x240
only produced a modest improvement because the network input is resized to
224x224 and rendering has substantial fixed overhead.

Continue with [Common workflows](workflows.md).
