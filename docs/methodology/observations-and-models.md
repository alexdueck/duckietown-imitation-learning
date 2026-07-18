# Observations and Models

## Policy Observation

The policy receives exactly one current camera image. There is no frame stack,
recurrent state, optical flow, map input, or lane telemetry.

gym-duckietown normally returns a `uint8` RGB array with shape:

```text
(height, width, 3) = (480, 640, 3)
```

Duckiematrix observations are treated as BGR by the current collection and
live-evaluation scripts. Saved imitation images are converted to RGB.

The channel order is explicit in the gym-duckietown trainer through
`--source-observation-channel-order`. Its default is `rgb`.

## RL Preprocessing

For each gym-duckietown observation:

1. Validate an `(H, W, 3)` array and convert to `uint8` if necessary.
2. Convert BGR to RGB only when configured.
3. Crop all rows above `crop_y_start`.
4. Resize the remaining image to `image_size x image_size` with bilinear
   interpolation.
5. Convert pixels to a PyTorch tensor in `[0, 1]`.
6. Normalize with ImageNet statistics.

The defaults are:

```text
crop_y_start = 0
image_size = 224
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

There is no JPEG round trip in the PPO image pipeline.

## Imitation Preprocessing

The imitation trainer loads RGB images through Pillow, resizes them to the
configured square image size, converts them to tensors, and applies the same
ImageNet normalization.

The trainer looks for `images_processed` by default. Passing
`--image-dir images` selects the original collected JPEGs and avoids the
legacy offline preprocessing step.

The preprocessing used to train an IL checkpoint must match live evaluation
and any PPO warm start. A model can be mathematically correct and visually
confused at the same time.

## Encoders

Supported encoders are:

- `mobilenet_v3_small`
- `resnet18`

For PPO, actor and value network use separate CNN encoders. They do not share
weights.

A fresh PPO actor and value network use ImageNet initialization. An IL warm
start replaces the actor parameters with the matching IL checkpoint. A resumed
PPO run restores both networks from the RL checkpoint.

## Actor

The PPO actor contains:

- CNN encoder
- linear mean head
- learned state-independent `log_std` vector

For an observation (o_t):

```text
mu_t = actor_mean(o_t)
z_t  ~ Normal(mu_t, exp(log_std))
u_t  = tanh(z_t)
```

The `tanh` output `u_t` is a policy control, not necessarily the final wheel
command. The selected action mode maps it to the environment action.

## Value Network

The value network predicts one scalar (V(o_t)) from the same preprocessed
image format. It has its own encoder and linear value head.

## Deterministic Network Behavior During PPO

MobileNet contains BatchNorm and Dropout modules. PPO log-probability ratios
require the same observation and parameters to produce the same distribution
before an update.

The implementation therefore:

- sets all Dropout probabilities in the PPO encoder to zero
- keeps actor and value modules in evaluation mode during PPO
- freezes BatchNorm running statistics
- still computes gradients through convolutional, normalization-affine, and
  linear parameters

Evaluation mode here controls module behavior; it does not disable gradients.
