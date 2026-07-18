# Troubleshooting

## `pip show gym-duckietown` Says Not Found

The distribution is named `duckietown-gym-daffy`; the import package is
`gym_duckietown`.

```bash
python -m pip show duckietown-gym-daffy
python -c "from gym_duckietown.simulator import Simulator; print('import ok')"
```

## pip Tries to Build Ancient NumPy or OpenCV

Install this repository's requirements, then install the cloned simulator
without dependencies:

```bash
python -m pip install -r requirements/gym-duckietown.txt
cd ~/git/gym-duckietown
python -m pip install -e . --no-deps
```

Do not let gym-duckietown's legacy dependency metadata choose NumPy/OpenCV on a
modern Apple Silicon or Ubuntu setup.

## `NoSuchConfigException: No standard config is available`

On a remote Ubuntu host, run under Xvfb and ensure the local gym-duckietown
clone selects Pyglet headless mode only when `DISPLAY` is absent.

```bash
xvfb-run -a \
  -s "-screen 0 1280x1024x24 +extension GLX" \
  env LIBGL_ALWAYS_SOFTWARE=1 \
  python train_rl_ppo_gym_duckietown.py --help
```

See [Ubuntu installation](getting-started/gym-duckietown-ubuntu.md).

## EGL Import Error or Segmentation Fault

The tested Ubuntu configuration uses Xvfb plus GLX, not direct Pyglet EGL.

Confirm:

```bash
xvfb-run -a \
  -s "-screen 0 1280x1024x24 +extension GLX" \
  env LIBGL_ALWAYS_SOFTWARE=1 \
  glxinfo -B
```

If gym-duckietown still attempts EGL, its Pyglet headless option was probably
set before `DISPLAY` was considered.

## The Window Is Black

Check the terminal for repeated invalid spawn attempts. Also verify:

- gym-duckietown can complete the smoke test
- Pyglet is not in headless mode for a visible/Xvfb window
- the OpenGL context reports a renderer
- the local hardware-check compatibility path is active

A brief black frame during environment construction is different from a
permanently black rendered observation.

## The Image Becomes Dark after Reset

gym-duckietown reset changes shared OpenGL lighting state. Use the current
trainer's `--render-training` path, which restores the relevant state before
human rendering. Make sure the local script and gym-duckietown clone are up to
date.

## Same Seed, Different Start Position

`Simulator(seed=N)` resets once inside its constructor. A second unseeded
`env.reset()` advances to the next RNG state.

The current trainer, manual viewer, and RL live evaluator reseed immediately
before the relevant reset. If another tool differs, compare the exact order:

```python
env.seed(seed)
observation = env.reset()
```

Also ensure map, `accept_start_angle_deg`, domain randomization, and local
start-pose fields match.

## Live Evaluation Does Not Match Training

Check:

- correct checkpoint
- checkpoint action mode and control limits
- deterministic versus `--stochastic`
- seed or evaluation pose
- image channel order
- crop and image size
- camera dimensions and frame rate
- reward function from checkpoint
- whether `--stop-on-done` is needed to inspect the final state

The RL viewer takes these settings from checkpoint config unless overridden.

## Training Has Occasional `invalid_pose` but Evaluation Is Safe

Training samples from the Gaussian policy. Default evaluation uses
`tanh(mean)`. Inspect `action_noise_steering_std`, learned standard
deviation, and per-seed evaluation before concluding that the deterministic
policy regressed.

Early training terminations can dominate lifetime counts. Compare rates in
recent step windows.

## Return Is High but Behavior Is Wrong

Inspect `reward_components_history.csv` and the manual reward viewer.

Questions to ask:

- Is velocity dominating pose?
- Is the policy stationary while receiving state reward?
- Is a potential term positive at a constant bad pose?
- Does aggregate evaluation hide one bad seed?
- Is lane geometry selecting the intended directed lane?

A reward is a specification, not a moral compass.

## Eval Return Is Not Monotonic

PPO updates the policy after every rollout, so each evaluation observes a
different policy. Fixed seeds remove reset noise; they do not make training
monotonic.

Use per-evaluation checkpoints, `best_return.pt`, and `best_safe.pt`. Compare
KL and clip fraction before reducing learning rate.

## Many Identical Python Entries in htop

htop may display user-space threads as separate rows. PyTorch, OpenMP, BLAS,
image libraries, and rendering can create many threads with the same command
line.

Press `H` in htop to hide userland threads. Confirm process relationships and
memory before assuming multiple trainers were launched.

## CUDA NVML Internal Assert during Parallel Training

Two independent CNN PPO processes can exceed allocator or driver limits even
when each individual batch size works. Two batches of 64 are not equivalent to
one batch of 128: both processes hold complete models, optimizers, rollout
tensors, CUDA contexts, and caches.

Current practical workaround:

- train sequentially
- reduce batch and rollout memory
- avoid overlapping diagnostics over full rollout tensors
- inspect `nvidia-smi`
- restart the failed process after releasing GPU memory

## NaNs in the Policy

First verify that the input tensor and loaded parameters are finite. For an IL
warm start, compare the IL model action and PPO actor mean on the same
preprocessed image.

Also inspect:

- channel order
- normalization
- learning rate and gradient norm
- log standard-deviation bounds
- MPS/CUDA-specific behavior
- checkpoint architecture

## Tab Completion Is Slow or Missing

argcomplete requires shell integration and may invoke Python for each
completion. Calling scripts as `python train_rl_ppo_gym_duckietown.py` is not handled equally by
all shell setups.

Use `python train_rl_ppo_gym_duckietown.py --help` when completion is inconvenient. The training
runtime does not depend on argcomplete.

## Artifacts Appear in an Old Directory

Current defaults use:

```text
~/duckietown/data/
~/duckietown/checkpoints/
```

An explicit old `--output-dir`, checkpoint-embedded path shown as metadata, or
stale shell command can still reference repository-local directories.
