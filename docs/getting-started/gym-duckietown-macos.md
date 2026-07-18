# gym-duckietown on macOS

This is the recommended setup for interactive development, manual control, and
visual policy evaluation. The tested machine is Apple Silicon.

## Prerequisites

- macOS with Command Line Tools installed
- Homebrew
- Python 3.9
- A local clone of this repository
- A local clone of gym-duckietown

Python 3.9 is intentional. gym-duckietown 6.2.0 has legacy dependency metadata
that does not fit comfortably with Python 3.12.

## Create the Environment

```bash
brew install python@3.9

mkdir -p ~/virtualenvs
cd ~/git/duckietown-imitation-learning
/opt/homebrew/bin/python3.9 -m venv ~/virtualenvs/gymdt39_venv
source ~/virtualenvs/gymdt39_venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/gym-duckietown.txt
```

If Homebrew exposes Python 3.9 at a different path, use the path below the
prefix printed by:

```bash
brew --prefix python@3.9
```

## Install gym-duckietown

Clone the simulator next to this repository:

```bash
cd ~/git
git clone https://github.com/duckietown/gym-duckietown.git
cd gym-duckietown
python -m pip install -e . --no-deps
```

`--no-deps` is deliberate. The package metadata requests old NumPy and OpenCV
versions that pip may try to build from source on modern Apple Silicon. The
repository's requirements file supplies newer compatible versions explicitly.

The installed distribution is named `duckietown-gym-daffy`, while the Python
import is `gym_duckietown`. Therefore use:

```bash
python -m pip show duckietown-gym-daffy
python -c "import gym_duckietown; print(gym_duckietown.__version__)"
```

`pip show gym-duckietown` reporting "not found" does not mean the import is
missing; package naming has simply taken the scenic route.

## Smoke Test

```bash
python -c "from gym_duckietown.simulator import Simulator; env=Simulator(map_name='loop_empty', domain_rand=False); obs=env.reset(); print(obs.shape); env.close()"
```

Expected observation shape:

```text
(480, 640, 3)
```

## Open the Manual Viewer

```bash
cd ~/git/duckietown-imitation-learning
source ~/virtualenvs/gymdt39_venv/bin/activate
python manual_control_gym_duckietown.py --map-name loop_empty
```

Controls:

- `W` and `S`: forward and backward throttle
- `A` and `D`: steer
- `Space`: stop
- `R`: enter a reset seed
- `P`: append the current pose to the configured training starts
- `Backspace` or `/`: reset
- `Enter`: save a screenshot
- `Escape`: exit

## Devices

All deep-learning scripts default to `--device auto`:

- CUDA is selected when available.
- Apple MPS is selected on supported Macs.
- Otherwise CPU is used.

Pass `--device mps` only when you want to require MPS explicitly. Simulator
rendering still uses OpenGL; the neural-network device does not render the
camera image.

## Next Step

Continue with [Common workflows](workflows.md).
