# Scripts and CLI Reference

Run any entry point with `--help` for current options and defaults:

```bash
python train_rl_ppo_gym_duckietown.py --help
```

This page describes responsibilities and common use, not every flag.

## Imitation Learning and Duckiematrix

| Script | Purpose |
| --- | --- |
| `imitation_learning.py` | Drive through gym-duckiematrix, display telemetry/rewards, and collect image/action datasets |
| `data_viewer.py` | Inspect collected images and the associated action, reward, and lane values |
| `preprocess.py` | Legacy optional crop/resize/JPEG preprocessing |
| `train_imitation_learning.py` | Train MobileNetV3-Small or ResNet-18 wheel-action regression |
| `live_eval_imitation_policy.py` | Execute an IL checkpoint in gym-duckiematrix |
| `train_rl_ppo_duckiematrix.py` | Experimental PPO training with Duckiematrix reward adapters |

## gym-duckietown

| Script | Purpose |
| --- | --- |
| `manual_control_gym_duckietown.py` | Manual driving, reward breakdowns, seeded resets, and pose capture |
| `live_eval_imitation_policy_gym_duckietown.py` | Visually evaluate a Duckiematrix-trained IL policy in gym-duckietown |
| `train_rl_ppo_gym_duckietown.py` | Main gym-duckietown PPO trainer |
| `live_eval_rl_policy_gym_duckietown.py` | Visually evaluate an RL checkpoint using its stored configuration |
| `analyze_rl_training_run.py` | Generate a standalone HTML report from a PPO run's CSV files |
| `ppo_control_tests.py` | PPO invariant, Pendulum, and synthetic image-control tests |

## Physical Duckiebot

| Script | Purpose |
| --- | --- |
| `capture_duckiebot_camera.py` | Capture one physical Duckiebot camera frame and optional policy-input preview through ROS |

See [Physical camera input check](../getting-started/physical-duckiebot-camera.md)
for the ROS-container invocation and the input-interface checklist.

## Supporting Modules

| Module | Purpose |
| --- | --- |
| `duckietown_paths.py` | Shared data/checkpoint paths |
| `rl_models.py` | CNN policies, tanh log probability, and IL actor loading |
| `duckietown_action_control.py` | Policy-control to wheel mapping |
| `velopose_reward.py` | Custom reward mathematics |
| `duckietown_rewards.py` | Simulator reward adapter and compatibility patch |
| `gym_duckietown_start_config.py` | Seed and pose configuration |
| `duckiematrix_telemetry.py` | Duckiematrix state telemetry |
| `rl_rewards.py` | Duckiematrix reward adapters |
| `cli_completion.py` | Optional argcomplete hook |

## CLI Completion

Scripts marked with `PYTHON_ARGCOMPLETE_OK` call the optional
`argcomplete` integration. Completion still requires shell activation; merely
installing it does not make `python train_rl_ppo_gym_duckietown.py --<Tab>` work in every
shell configuration.

Because completion launches Python to inspect the parser, it can feel slow on
a remote machine or for scripts with expensive imports. Completion is a
convenience, not a runtime requirement. `--help` is always available.

## Device Selection

Deep-learning entry points generally accept:

```text
--device auto|cpu|cuda|mps
```

`auto` prefers CUDA, then MPS, then CPU. Rendering and neural-network devices
are separate: Xvfb/Mesa can render observations while PyTorch uses CUDA.

## Logging

gym-duckietown entry points accept:

```text
--log-level DEBUG|INFO|WARNING|ERROR
```

The default is `INFO`. This setting also updates loggers used by the
Duckietown dependency stack.

## Stable Examples versus Defaults

Commands in [Common workflows](../getting-started/workflows.md) record
configurations used in current experiments. They may override parser defaults.
Do not infer a research recommendation from a default alone; defaults are
often selected to make a script broadly runnable.
