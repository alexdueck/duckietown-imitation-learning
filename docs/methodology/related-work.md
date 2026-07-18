# Related Work and Provenance

## PPO and GAE

The trainer implements Proximal Policy Optimization directly in PyTorch rather
than wrapping an RL library.

Core algorithm references:

- Schulman et al., [Proximal Policy Optimization
  Algorithms](https://arxiv.org/abs/1707.06347)
- Schulman et al., [High-Dimensional Continuous Control Using Generalized
  Advantage Estimation](https://arxiv.org/abs/1506.02438)

The local implementation adds project-specific image encoders, tanh-squashed
continuous controls, action mappings, diagnostics, fixed Duckietown scenarios,
and checkpoint conventions.

## Vision-Based Duckietown RL

The closest direct reference is:

- Kalapos et al., [Vision-based reinforcement learning for lane-tracking
  control](https://arxiv.org/abs/2012.07461)
- Associated [Duckietown-RL
  implementation](https://github.com/kaland313/Duckietown-RL)

That work studied PPO-based visual lane tracking in gym-duckietown and compared
reward formulations and network choices. The `posangle`,
`target_orientation`, and `distance_travelled` comparison rewards in this
repository were adapted from its reward-wrapper implementation.

The custom `velopose`, `posepot`, and `vd2pp` rewards are local
experiments developed after manually inspecting failure modes in both
simulators.

## Simulators

gym-duckietown supplies rendered observations and simulator geometry for the
current RL setup:

- [Duckietown gym-duckietown](https://github.com/duckietown/gym-duckietown)

Duckiematrix is used as the original IL environment and as a future
sim-to-sim transfer target. The two backends are not assumed to have identical
camera rendering, action dynamics, pose APIs, or reward reliability.

## Implementation Position

This repository is not intended as a faithful reimplementation of one paper.
It combines established PPO/GAE mathematics with:

- a project-specific squashed-Gaussian PyTorch implementation
- direct wheel and throttle/steering controls
- IL checkpoint transfer
- curated seed and pose curricula
- new reward shaping experiments
- unusually detailed evaluation and action diagnostics

When reporting results, distinguish inherited algorithmic ideas, adapted
comparison rewards, and newly introduced project behavior.
