# Developer Architecture

## Repository Shape

The repository is intentionally script-oriented. Training and interactive tools
can be run directly, while shared behavior is extracted into modules when it
must remain consistent across tools.

## Duckiematrix Entry Points

| File | Responsibility |
| --- | --- |
| `imitation_learning.py` | Manual control, observation capture, telemetry, and dataset writing |
| `data_viewer.py` | Browse collected images, actions, reward, and telemetry |
| `train_imitation_learning.py` | Supervised image-to-wheel training |
| `live_eval_imitation_policy.py` | Execute an IL policy in Duckiematrix |
| `train_rl_ppo_duckiematrix.py` | Experimental Duckiematrix PPO trainer |
| `duckiematrix_telemetry.py` | Read pose/lane telemetry from Duckiematrix |
| `rl_rewards.py` | Duckiematrix reward adapters |

## gym-duckietown Entry Points

| File | Responsibility |
| --- | --- |
| `manual_control_gym_duckietown.py` | Manual driving, reward sidebar, seeds, and pose capture |
| `live_eval_imitation_policy_gym_duckietown.py` | Visual IL transfer evaluation |
| `train_rl_ppo_gym_duckietown.py` | Main PPO trainer and deterministic evaluation |
| `live_eval_rl_policy_gym_duckietown.py` | Visual PPO checkpoint evaluation |
| `duckietown_rewards.py` | Reward selection, state tracking, breakdowns, and compatibility patch |
| `velopose_reward.py` | Pure custom reward equations |
| `duckietown_action_control.py` | Policy-control to wheel-action mapping |
| `duckiebot_hardware_control.py` | Fail-closed wheel-action to physical chassis-command mapping |
| `gym_duckietown_start_config.py` | Seed/pose configuration, validation, and sampling |

## Shared Modules

| File | Responsibility |
| --- | --- |
| `rl_models.py` | CNN actor, Q-network scaffold, tanh log probability, and IL actor loading |
| `duckietown_paths.py` | Artifact locations below `~/duckietown` |
| `cli_completion.py` | Optional argcomplete integration |
| `ppo_control_tests.py` | PPO invariant, Pendulum, and image-control tests |
| `preprocess.py` | Legacy optional offline image preprocessing |

## PPO Trainer Structure

The main gym-duckietown trainer currently owns:

1. CLI and run configuration
2. environment construction and compatibility setup
3. deterministic and curated reset handling
4. policy/value initialization and checkpoint loading
5. rollout collection and timing
6. GAE and PPO optimization
7. diagnostics and CSV output
8. fixed-scenario evaluation
9. checkpoint selection

This is a large module, but those responsibilities are kept in explicit
functions. Future extraction should preserve the observable CSV and checkpoint
contracts before pursuing smaller files for their own sake.

## Reward Boundary

`velopose_reward.py` contains equations over NumPy values and has no direct
simulator dependency.

`duckietown_rewards.py` adapts simulator state into those equations, tracks
previous position/potential, recognizes done reasons, and returns nested
breakdowns.

This boundary makes reward mathematics testable independently from OpenGL and
Pyglet.

## Action Boundary

The actor always produces normalized policy controls. Only
`DuckietownActionControl` knows how controls map to wheel commands. The same
mapping is used in rollout collection, deterministic evaluation, diagnostics,
checkpoint metadata, and live evaluation.

Physical deployment adds a second boundary after this mapping.
`PhysicalDuckiebotControl` maps normalized wheels to bounded chassis
velocities, tracks arming and emergency-stop state, limits acceleration, and
fails closed on invalid or stale inputs and watchdog timeout. It deliberately
does not import ROS or publish commands; transport belongs to the physical
runtime.

## Configuration Boundary

CLI arguments are converted to a run-local configuration and stored in both
`config.json` and checkpoints. Live evaluation reads checkpoint configuration
so model preprocessing and action semantics do not need to be re-entered.

Local start configurations are separate experiment inputs. Their source path
and parsed contents are recorded in run metadata.

## Artifact Boundary

Generated data and checkpoints live outside Git under `~/duckietown`.
Repository-local JSON files are configuration inputs; `configs/*.json` is
ignored while templates are versioned.

See [Outputs and checkpoints](../reference/outputs-and-checkpoints.md) for the
persistent contracts.
