# Managed experiments

The experiment layer treats a complete training history—not an individual ZIP
file—as the unit of work. It currently supports the graph policy and its
capacity-matched MLP baseline.

## Start

```bash
python -m nervenet.cli.experiment start \
  --name graph-v0-seed0 \
  --policy-type graph \
  --modules 3 \
  --timesteps 1000000 \
  --checkpoint-every 100000 \
  --evaluation-episodes 20 \
  --seed 0 \
  --control-cost-weight 0.05 \
  --target-kl 0.03
```

Names use lowercase letters, numbers, underscores, and hyphens. Starting
refuses to reuse an existing directory, preventing an old run from being
silently replaced.

The configuration is immutable after creation. A changed morphology, seed,
reward, architecture, or PPO configuration is a new experiment and therefore
needs a new name.

`--target-kl` enables Stable-Baselines3's PPO early-stop guard for an update
whose approximate KL divergence grows too large. Omitting it preserves the
unlimited-update behavior used by the earlier manual runs. Because it changes
optimization, policies trained with and without this setting belong to
different experiment series and should not be continued across that boundary.

## Continue

```bash
python -m nervenet.cli.experiment continue \
  --name graph-v0-seed0 \
  --timesteps 500000
```

`--timesteps` is the requested number of additional environment steps. The
command finds the latest recorded checkpoint, verifies its policy and timestep
count, restores the policy, critic, optimizer, and counter, and starts a new
logging session. It never asks the user to repeat the original configuration.

PPO updates only after collecting a complete rollout. With the current
2,048-step rollout size, a 100,000-step boundary is completed at step 100,352.
Checkpoint filenames always record this truthful counter. Intermediate
checkpoints are taken after the completed PPO update, rather than saving a
partially consumed rollout. The final update is always checkpointed even when
it does not cross the next configured boundary.

## Evaluate and view

```bash
python -m nervenet.cli.experiment evaluate --name graph-v0-seed0
mjpython -m nervenet.cli.experiment view --name graph-v0-seed0
```

Both commands select the latest checkpoint by default. Pass an exact stored
timestep with `--step`, for example:

```bash
python -m nervenet.cli.experiment evaluate \
  --name graph-v0-seed0 \
  --step 200704
```

Automatic and manual evaluations use the configured number of deterministic
episodes and the fixed seed sequence `0..N-1`. They record signed lateral
displacement, absolute lateral displacement, mean absolute lateral speed, and
final heading error. Measuring all four avoids mistaking left/right cancellation
for straight locomotion.

## Files

```text
experiments/graph-v0-seed0/
├── experiment.json
├── checkpoints/
│   ├── step_000100352.zip
│   ├── step_000200704.zip
│   └── ...
└── logs/
    ├── session_001/
    │   ├── progress.csv
    │   └── progress.json
    └── session_002/
        ├── progress.csv
        └── progress.json
```

`experiment.json` is written atomically after every important state change. It
contains:

- the environment, policy, architecture, PPO, checkpoint, and evaluation
  configuration;
- the Git commit, dirty-worktree state, Python version, and important package
  versions at creation and at every continuation session;
- requested, starting, and ending timesteps for every session;
- the complete SB3 training history from each session;
- every checkpoint path and its nearby training metrics;
- aggregate evaluation metrics and every underlying episode result;
- manually requested evaluations and the checkpoint and software revision used;
- current status, latest checkpoint, and timestamps.

The CSV and newline-delimited JSON logs remain available for convenient
analysis without loading the larger manifest. The `experiments/` directory is
ignored by Git because checkpoints and run data are generated artifacts.

If training is interrupted, completed checkpoints remain valid. The manifest
marks the session as interrupted when Python receives the interrupt normally;
a later `continue` resumes from the latest completed checkpoint.
