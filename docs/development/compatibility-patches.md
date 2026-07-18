# Compatibility Patches

gym-duckietown 6.2.0 is an older simulator running alongside newer Python
packages and operating systems. This project keeps compatibility work explicit
instead of pretending that dependency resolution was uneventful.

## Dependency Metadata

The `duckietown-gym-daffy` package requests old NumPy and OpenCV versions.
Those versions may have no wheel for a modern platform and can trigger large,
failing source builds.

The documented installation therefore uses:

```bash
python -m pip install -r requirements/gym-duckietown.txt
cd ~/git/gym-duckietown
python -m pip install -e . --no-deps
```

The project requirements select tested replacement versions.

## DynamicModel.integrate Runtime Patch

Newer `duckietown-world` versions can return array-shaped linear/angular
values where gym-duckietown's older dynamics code expects scalars.

`patch_duckietown_world_dynamics()` replaces
`pwm_dynamics.DynamicModel.integrate` at runtime with a scalar-compatible
implementation. It:

- converts previous velocity and acceleration values to explicit scalars
- integrates longitudinal and angular velocity
- delegates SE(2) kinematics to `GenericKinematicsSE2`
- reconstructs wheel-axis rotation
- returns a new `DynamicModel`

The patch marks the replacement function and is idempotent. Yes, Python allows
methods to be replaced at runtime. This is powerful, practical, and a good
reason to document exactly what happened.

The upstream package on disk is not modified by this runtime patch.

## Pyglet Headless Selection

Pyglet headless mode uses EGL. On the tested Ubuntu host, direct EGL attempts
failed despite a render device being present.

Xvfb with Mesa GLX worked, but upstream gym-duckietown selected headless mode
on Linux even when `DISPLAY` was provided. The local gym-duckietown clone was
changed to:

```python
pyglet.options["headless"] = (
    platform.system() != "Darwin"
    and not bool(os.environ.get("DISPLAY"))
)
```

Under Xvfb, `DISPLAY` is set and Pyglet uses the X11 window backend. Without a
display it can still select its EGL headless path.

This is a change to the local gym-duckietown clone, not a runtime patch in this
repository.

## Hardware Check Stub

The windowed tools install a small `gym_duckietown.check_hw` compatibility
module before importing the simulator. It reports OpenGL strings from the
active Pyglet context without relying on incompatible upstream hardware-check
behavior.

## Seeded Resets

The simulator calls `reset()` in its constructor. Constructing
`Simulator(seed=10045)` therefore consumes one seeded start before application
code calls reset.

The shared trainer reset helper tries the modern `reset(seed=...)` API and
falls back to:

```python
env.seed(seed)
observation = env.reset()
```

The RL live evaluator uses the same helper. This is required for visual
evaluation to reproduce trainer seed 10045 rather than the next RNG state.

## Rendering State after Reset

A simulator reset enables OpenGL lighting. The optional training viewer resets
relevant OpenGL state before human rendering so that a reset does not leave the
displayed image unexpectedly dark. The camera observation and human view use
different framebuffers, but both share OpenGL state.

## Patch Policy

Compatibility patches should be:

- as small as possible
- idempotent
- invoked before affected behavior
- documented with upstream and local version context
- covered by a focused smoke test
- removed when the supported dependency no longer needs them
