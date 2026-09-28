# NerveNet Reimplementation

An independent, from-scratch reimplementation and study of the method described in *NerveNet: Learning Structured Policy with Graph Neural Networks*.

The project uses modern Python and MuJoCo. It does not reuse the authors' implementation.

## Status

MuJoCo 3.14.0 has been verified on macOS. The current model is a modular crawler
generated through MuJoCo's `MjSpec` API. It has a functional head marker,
mirrored actuated legs, passive hinges connecting neighboring torsos, and a
free-moving root. A Gymnasium environment defines the action and observation
spaces, deterministic reset, physics stepping at a 50 Hz control rate, a
forward-velocity reward, reproducible reset randomization, and a configurable
episode time limit.

Preview the passive spine of a two-module crawler:

```bash
mjpython -m nervenet.cli.view_crawler --modules 2 --motion spine
```

Preview the coordinated leg motion:

```bash
mjpython -m nervenet.cli.view_crawler --modules 2 --motion gait
```

Run physics and optionally apply a constant command to one motor:

```bash
mjpython -m nervenet.cli.view_crawler --modules 2 --motion physics \
  --actuator module_1_left_hip_motor --control 1.0
```

Change `--modules` to construct another morphology from the same builder. The
`spine` and `gait` motions are kinematic previews; `physics` advances MuJoCo.

Run the model tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Run one reproducible random-policy episode as an environment sanity check:

```bash
.venv/bin/python -m nervenet.cli.rollout_random --modules 2 --seed 0
```

Train and save the two-module flat PPO baseline:

```bash
.venv/bin/python -m nervenet.cli.train_flat \
  --timesteps 100000 \
  --seed 0 \
  --output artifacts/flat_policy
```

Compare the frozen flat policy with seeded random actions:

```bash
.venv/bin/python -m nervenet.cli.compare_policies \
  --model artifacts/flat_policy.zip \
  --episodes 5
```
