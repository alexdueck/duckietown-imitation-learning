# Testing

The project currently combines lightweight automated checks with simulator
smoke tests and visual validation.

## Syntax Check

Run after changing Python files:

```bash
python -m py_compile *.py
```

This catches syntax and import-independent compilation errors.

## PPO Invariant Test

```bash
python ppo_control_tests.py --test invariant
```

This verifies:

- transformed log probability is reproducible from stored raw actions
- PPO Dropout is disabled
- BatchNorm remains frozen in evaluation mode
- strongly biased means still produce saturated bounded actions

## Image-Control Test

```bash
python ppo_control_tests.py --test image
```

This constructs a simple image-based continuous-control problem and checks that
the PPO implementation learns the encoded target action.

Use `--image-rollouts` to change test length.

## Pendulum Test

```bash
python ppo_control_tests.py --test pendulum
```

This checks the same PPO update logic on Gym's `Pendulum-v1`. The default
300,000 steps are intentionally a learning test rather than a quick unit test.

Run all checks with:

```bash
python ppo_control_tests.py --test all
```

## gym-duckietown Smoke Test

macOS:

```bash
python -c "from gym_duckietown.simulator import Simulator; env=Simulator(map_name='loop_empty', domain_rand=False); obs=env.reset(); print(obs.shape); env.close()"
```

Ubuntu:

```bash
xvfb-run -a \
  -s "-screen 0 1280x1024x24 +extension GLX" \
  env LIBGL_ALWAYS_SOFTWARE=1 \
  python -c "from gym_duckietown.simulator import Simulator; env=Simulator(map_name='loop_empty', domain_rand=False); obs=env.reset(); print(obs.shape); env.close()"
```

## Behavioral Checks

Before trusting a reward change, inspect it in
`manual_control_gym_duckietown.py` at:

- lane center and correct heading
- lateral offsets on both sides
- corrective headings toward the center
- opposite heading
- forward, stationary, and reverse motion
- invalid pose and reset
- a curve and a straight section

Before trusting a reset change, compare trainer, manual viewer, and live
evaluator from the same seed and exact pose.

## Documentation Validation

Documentation changes should check:

- all relative links resolve
- commands use current filenames
- example flags exist in `--help`
- formulas match implementation constants
- results include run identity and evaluation horizon
- local paths use `~/duckietown`, not old repository-local artifact paths

## Missing Coverage

There is not yet a comprehensive automated suite for:

- reward equations over synthetic geometry
- action mappings and scaling boundaries
- start-config parsing and sampling
- checkpoint backward compatibility
- seeded reset equivalence
- CSV schemas

Those are high-value additions before larger architectural refactors.
