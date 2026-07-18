# Duckiematrix Setup

Duckiematrix is currently the data-collection and imitation-learning backend.
The repository assumes an existing working installation of:

- Duckiematrix engine
- Duckiematrix renderer
- gym-duckiematrix
- Duckietown SDK and DTPS connectivity

Those components are not installed by this repository.

## Python Dependencies

Activate the Python environment used by the local Duckiematrix installation,
then install the common project dependencies:

```bash
cd ~/git/duckietown-imitation-learning
python -m pip install -r requirements/duckiematrix.txt
```

Verify that the expected entity is available. Most scripts default to:

```text
map_0/vehicle_0
```

## Current Scope

The Duckiematrix tools support:

- manual imitation-data collection
- image and action storage
- reward and lane telemetry inspection
- imitation training
- live IL-policy evaluation
- experimental PPO training

The PPO path should not currently be treated as a validated RL setup. The
available Duckiematrix pose/lane telemetry has produced inconsistent
lane-distance behavior and high rewards in geometrically undesirable areas.
Until that signal is corrected, gym-duckietown is the recommended RL backend.

## Collect Data

```bash
python imitation_learning.py
```

Use `--observe-only` to inspect observations, actions, reward variants, and
telemetry without writing a dataset:

```bash
python imitation_learning.py --observe-only
```

Collected runs are written below:

```text
~/duckietown/data/imitation_learning/
```

Each run contains JPEG images, `actions.csv`, and `meta.json`. The reward and
lane values in a row describe the visible observation `obs_t`; the action in
the same row is `action_t`, computed or entered from that observation.

## Train Imitation Learning

The trainer expects `images_processed` by default. To train directly on the
stored images, select the image directory explicitly:

```bash
python train_imitation_learning.py \
  --run-dir ~/duckietown/data/imitation_learning/train \
  --image-dir images \
  --experiment-name duckiematrix_il
```

Use a split root containing `train/` and `val/` run directories for
run-wise validation. A single run uses a random frame split.

## Evaluate Imitation Learning

```bash
IL_CHECKPOINT="$HOME/duckietown/checkpoints/imitation_learning/YOUR_RUN/best.pt"

python live_eval_imitation_policy.py \
  --checkpoint "$IL_CHECKPOINT"
```

The earlier `preprocess.py` workflow remains available but is optional. It is
not part of the current minimal image pipeline.

## Experimental PPO

```bash
python train_rl_ppo_duckiematrix.py --help
```

The script is retained for reward investigation and future comparison. A run
completing successfully does not validate the reward: an optimizer will accept
almost any objective without filing a complaint.
