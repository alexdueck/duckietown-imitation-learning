# Documentation

This documentation is organized by task. Researchers and developers share much
of the same system context, so duplicating two almost-identical manuals would
mostly create two places for facts to become outdated.

## I Want to Run an Experiment

1. Install the selected backend:
   - [gym-duckietown on macOS](getting-started/gym-duckietown-macos.md)
   - [gym-duckietown on Ubuntu over SSH](getting-started/gym-duckietown-ubuntu.md)
   - [Duckiematrix](getting-started/duckiematrix.md)
2. Follow [Common workflows](getting-started/workflows.md).
3. Read the [evaluation protocol](methodology/evaluation.md) before comparing
   returns.
4. Use [Troubleshooting](troubleshooting.md) when OpenGL remembers that it has
   opinions.

## I Want to Understand the Research

- [System overview](methodology/system-overview.md)
- [Observations and models](methodology/observations-and-models.md)
- [Actions](methodology/actions.md)
- [Rewards](methodology/rewards.md)
- [PPO](methodology/ppo.md)
- [Evaluation](methodology/evaluation.md)
- [Related work and provenance](methodology/related-work.md)
- [Preliminary result summary](results/summary.md)
- [Experiment log](results/experiments.md)

## I Want to Change the Code

- [Architecture](development/architecture.md)
- [Extending the project](development/extending-the-project.md)
- [Compatibility patches](development/compatibility-patches.md)
- [Testing](development/testing.md)
- [Scripts and CLI](reference/scripts-and-cli.md)
- [Configurations](reference/configurations.md)
- [Outputs and checkpoints](reference/outputs-and-checkpoints.md)

## Documentation Rules

The pages intentionally separate three kinds of information.

### Procedures

Installation and workflow pages contain commands that should be directly
executable. They describe how to reproduce an operation.

### Method Specifications

Methodology pages describe what the code computes: observation transforms,
action mappings, reward equations, PPO behavior, and evaluation semantics.
These pages should change when behavior changes.

### Empirical Results

Result pages describe what happened in dated runs. Every quantitative result
should identify the run, code revision when available, configuration, seeds,
and evaluation horizon. Results are evidence, not defaults.

## Source of Truth

- Python `--help` output is authoritative for current CLI flags and defaults.
- Methodology documentation is authoritative for intended behavior.
- The implementation and tests decide actual behavior when documentation and
  code disagree.
- Run-local `config.json` and CSV files define a particular experiment.

If those four disagree, congratulations: you have found a documentation bug or
a software bug. Either one is worth fixing.
