# Adaptive Start Sampling

Curated starts are useful only while they are harder than ordinary simulator
starts. The trainer therefore adapts both the probability of selecting a hard
start and the distribution over individual hard poses. Evaluation outcomes do
not influence this process.

The implementation and validated per-run settings live in
`dt_utils/adaptive_start_sampling.py`. Defaults are exposed through both the
training CLI and `configs/train_config.template.json`.

## Episode Success

A completed training episode is a binary success when its `done_reason` is:

- `time_limit`, from the trainer's `--max-episode-steps`; or
- `max-steps-reached`, from the simulator limit.

`invalid-pose`, generic termination, and other truncations are failures. This
definition measures whether the policy stayed valid for the complete episode,
not whether its return crossed an arbitrary threshold.

## Exponential Success Estimates

Random starts, all hard starts, and every individual hard pose maintain a
separate exponentially smoothed success rate:

```text
S_new = (1 - lambda) * S_old + lambda * success
```

The defaults are:

```text
lambda = 0.15
initial success estimate = 0.5
```

`lambda=0.15` reacts roughly like a window over the latest 12 observations of
that category or pose. Counts are tracked as diagnostics, but the EMA update
does not require an episode buffer. Configure it with
`start_success_ema_lambda` in JSON or `--start-success-ema-lambda` on the CLI.

## Hard Versus Random Starts

Let:

```text
e_random = 1 - S_random
e_hard   = 1 - S_hard
```

The relative excess failure of hard starts is:

```text
d = max(0, (e_hard - e_random) / (e_hard + e_random + epsilon))
```

The next hard-start probability is:

```text
p_hard = p_min + (p_max - p_min) * d
p_min  = 0.20
p_max  = 0.80
```

If hard and random starts perform equally, only the minimum 20% hard coverage
remains. Hard starts receive more samples only when their failure rate exceeds
the random failure rate. At least 20% random coverage is retained even for a
large difficulty gap under the defaults. The bounds are configurable through
`hard_start_probability_min` and `hard_start_probability_max`. Equal bounds
turn adaptation into a fixed probability; setting both to `1.0` selects only
curated starts.

Until both start types have at least one observed outcome, the trainer uses
`--hard-start-probability` as the cold-start probability. Its default remains
0.5. Equal minimum and maximum bounds define a fixed probability immediately,
including during this cold-start phase.

## Sampling Individual Hard Poses

For hard pose `i`:

```text
d_i = 1 - S_i
w_i = 1 + difficulty_strength * d_i
difficulty_strength = 5.0
P(i | hard) = w_i / sum_j(w_j)
```

A completely unsuccessful pose therefore receives six times the raw weight of a
completely successful pose. The normalized maximum automatically depends on
the number of configured poses; there is no fixed 25% cap. Every pose retains
nonzero probability, so a learned pose can reveal later regressions.
`hard_pose_difficulty_strength` controls this weighting per run.

Hard poses are identified by `map_name:name`. Unnamed poses use a deterministic
fallback based on map, tile, local position, and angle. Unique names are
recommended because they make logs and configuration changes much easier to
interpret.

## Logging

`start_sampling_history.csv` contains one row per completed training episode:

- completed start type, map, and pose name;
- binary success and termination reason;
- hard and random EMA success rates and observation counts;
- hard-start probability used for the next selection;
- selected pose EMA and conditional sampling probability; and
- JSON snapshots of every hard-pose EMA and probability.

This file is deliberately separate from PPO diagnostics. Start sampling is a
curriculum decision, not a policy-gradient statistic.

## Checkpoints and Resume

Every PPO checkpoint stores `adaptive_start_sampler_state`, including:

- both group EMAs and counts;
- all per-pose EMAs and counts;
- the cold-start probability;
- the sampler constants and schema version; and
- the NumPy random-generator state used for start selection.

On resume, known poses recover their statistics, newly configured poses start
at the neutral 0.5 estimate, and removed poses are reported and ignored. A
legacy checkpoint without sampler state begins with neutral estimates.

Sampler constants and `--hard-start-probability` belong to the new run and may
change when training resumes. The saved EMA statistics and sampler RNG state are
still restored; changed settings are reported in the startup log. Model and
history settings remain subject to their separate compatibility checks.
