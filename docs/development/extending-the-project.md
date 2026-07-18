# Extending the Project

## Add a gym-duckietown Reward

1. Implement the numerical equation in `velopose_reward.py` when possible.
2. Return a nested structure with `total` and `components`.
3. Add the public name to `REWARD_FUNCTION_CHOICES` in
   `duckietown_rewards.py`.
4. Add state initialization and reset behavior to
   `GymDuckietownRewardCalculator`.
5. Handle invalid and terminal transitions explicitly.
6. Add the reward to manual/live display choices where appropriate.
7. Verify that training stores all components in
   `reward_components_history.csv`.
8. Add equation and semantics to the reward documentation.
9. Test a stationary state, forward motion, reverse motion, lane departure,
   wrong heading, reset, and terminal transition.

Avoid deriving reward from rendered pixels when exact simulator geometry is
available. The policy must infer from pixels; the reward does not need to make
the same task harder.

## Add an Action Mode

1. Add the mode to `ACTION_MODE_CHOICES`.
2. Define its policy output dimension and control names.
3. Implement identical NumPy and PyTorch wheel mappings.
4. Preserve the final wheel range and document any scaling.
5. Store all semantics in checkpoint `action_space` metadata.
6. Ensure the live evaluator reconstructs the mapping from checkpoint config.
7. Add mapping and saturation tests.

Never reinterpret an existing checkpoint's outputs silently. A two-dimensional
tensor has no opinion about whether it means two wheels or throttle plus
steering.

## Add a Start Strategy

Start selection belongs in `gym_duckietown_start_config.py`. Keep these
properties:

- evaluation scenarios are deterministic
- training and evaluation seed sets do not overlap
- local poses validate against the selected map
- random training seeds exclude reserved seeds
- run metadata records the actual parsed starts

## Add Diagnostics

Prefer a dedicated CSV with stable columns over adding increasingly long
terminal lines. Terminal output should answer "is it alive?"; CSV output should
support analysis.

When adding fields:

- document units and aggregation
- distinguish per-step, per-rollout, per-episode, and per-scenario values
- avoid changing the meaning of an existing column
- include enough keys to join against step, rollout, evaluation, and seed

## Add a Backend

Create backend-specific entry points when environment construction, reset
semantics, or dependencies differ materially. Reuse shared actor and PPO math,
but do not force unrelated simulators through one conditional-heavy CLI.

A backend should define:

- observation channel order and preprocessing
- action contract
- reset and done semantics
- reward ground truth
- evaluation scenarios
- resource cleanup
- installation requirements

## Maintain Checkpoint Compatibility

New configuration fields should have sensible defaults when absent from older
checkpoints. Validate architecture and action semantics before loading weights.

Checkpoint migrations should be explicit. A loud error is preferable to a
quietly wrong steering convention.

## Documentation Checklist

A behavioral change is complete when:

- CLI help describes it
- run configuration records it
- checkpoint metadata preserves it
- live evaluation reproduces it
- CSV diagnostics expose it when relevant
- tests cover its central invariant
- methodology and reference documentation are updated
