# PPO Implementation

## Policy Distribution

The actor predicts a mean vector and a learned state-independent log standard
deviation:

```text
z ~ Normal(mu(observation), exp(log_std))
policy_control = tanh(z)
```

The Gaussian sample before `tanh`, called `raw_action` in the code, is stored
in the rollout. PPO recomputes its transformed log probability with the
Jacobian correction for `tanh`.

Storing only the bounded action would require an unstable inverse transform
near `-1` and `1`.

## Why log_std?

The optimizer can update an unconstrained real parameter while
`exp(log_std)` guarantees a positive standard deviation. Multiplicative
changes in standard deviation also become additive parameter changes.

A fresh PPO run selects `log_std=-0.5` before applying bounds. With the
default `--max-log-std=-1.0`, its effective initial value is therefore
`-1.0`. An IL warm start defaults to `-2.0`. Explicit values are clamped
between `--min-log-std` and `--max-log-std`; resumed checkpoints load their
learned value and are then clamped to the new bounds.

## Rollout Collection

For each environment step the trainer stores:

- preprocessed observation
- sampled bounded policy control
- raw Gaussian action
- mapped wheel action
- old transformed log probability
- reward and done flag
- value prediction

Rollout boundaries and episode boundaries are independent. An episode may end
inside a rollout, and an unfinished episode may continue into the next
rollout. The value estimate bootstraps GAE at a non-terminal rollout boundary.

`--max-episode-steps` is a separate training time limit. It resets and logs an
episode but does not force a PPO update.

## GAE

Advantages are computed backward using:

```text
delta_t = r_t + gamma * V(s_(t+1)) * (1 - done_t) - V(s_t)
A_t     = delta_t + gamma * lambda * (1 - done_t) * A_(t+1)
return_t = A_t + V(s_t)
```

Advantages are normalized over the collected rollout before optimization.

## PPO Update

For every rollout:

1. Shuffle all rollout indices.
2. Split them into minibatches.
3. Repeat for `--epochs` passes.
4. Optimize the clipped policy objective.
5. Optimize value mean-squared error.
6. Clip actor and value gradient norms.
7. Clamp learned `log_std`.

For a 1024-step rollout, batch size 64, and two epochs, PPO performs:

```text
(1024 / 64) * 2 = 32 policy updates and 32 value updates
```

Older rollouts are discarded. PPO is on-policy; there is no replay buffer and
no offline dataset in the current trainer.

## Clipped Objective

For the stored raw action:

```text
ratio = pi_new(action | observation) / pi_old(action | observation)

L_policy = -mean(
    min(
        ratio * advantage,
        clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
    )
)
```

`epsilon` is `--clip-ratio`. It clips the probability-ratio contribution,
not the continuous action itself.

There is currently no target-KL early stopping. Approximate KL and clip
fraction are recorded for diagnosis.

## Entropy

The update uses the analytic entropy of the unsquashed Normal distribution:

```text
loss = policy_loss - entropy_coef * normal_entropy
```

The diagnostics additionally report `squashed_entropy_estimate = -old_logp`,
which better reflects the transformed policy samples. Negative differential
entropy is valid for concentrated continuous distributions.

## Evaluation

Training actions are sampled. Evaluation actions are deterministic by default:

```text
action_eval = tanh(mu(observation))
```

This distinction explains why occasional `invalid_pose` episodes can remain
in training while fixed-seed evaluation is safe.

## Diagnostics

`ppo_diagnostics.csv` records:

- pre-update log-probability invariants
- approximate KL and clip fraction
- ratio extrema
- learned log standard deviations
- sampled and deterministic control statistics
- effective wheel-action noise
- steering noise and saturation fractions
- squashed entropy estimate

These values should be checked before attributing a regression to learning
rate, exploration, or PPO clipping.
