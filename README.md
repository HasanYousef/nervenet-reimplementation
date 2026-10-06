# NerveNet Reimplementation

An independent, from-scratch reimplementation and study of the method described in *NerveNet: Learning Structured Policy with Graph Neural Networks*.

The project uses modern Python and MuJoCo. It does not reuse the authors' implementation.

## Status

MuJoCo 3.14.0 has been verified on macOS. The current model is a modular crawler
generated through MuJoCo's `MjSpec` API. It has a functional head marker,
mirrored actuated legs, passive hinges connecting neighboring torsos, and a
free-moving root. Its initial height is derived from the leg geometry so the
feet begin just above the ground. A Gymnasium environment defines the action and observation
spaces, deterministic reset, physics stepping at a 50 Hz control rate, a
forward-velocity reward with a normalized motor-effort cost, reproducible reset
randomization, and a configurable episode time limit.

The structured-policy foundation now derives a body graph from the compiled
MuJoCo model and reads current observations locally for each node. The root
torso receives its orientation and motion state, while hip, knee, and passive
spine bodies receive their own joint angle and angular velocity.
One shared input encoder converts every padded local observation into a
fixed-width hidden representation without depending on the number of nodes.
The graph actor then performs recurrent neighbor message passing with a shared
GRU and decodes motor-owning node states into action means in MuJoCo actuator
order. The same actor instance supports crawler morphologies with different
module and actuator counts. A Gymnasium observation wrapper exposes the same
environment state as padded per-node matrices for graph-policy training while
leaving the flat PPO baseline environment unchanged. A custom
Stable-Baselines3 policy now connects the graph actor and flat critic to PPO,
and the graph training path is covered by a short end-to-end optimization test.

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

Train and save the three-module flat PPO baseline:

```bash
.venv/bin/python -m nervenet.cli.train_flat \
  --modules 3 \
  --timesteps 100000 \
  --seed 0 \
  --control-cost-weight 0.05
```

Train and save the three-module graph PPO policy:

```bash
.venv/bin/python -m nervenet.cli.train_graph \
  --modules 3 \
  --timesteps 100000 \
  --seed 0 \
  --control-cost-weight 0.05
```

Compare the frozen flat policy with seeded random actions:

```bash
.venv/bin/python -m nervenet.cli.compare_policies \
  --modules 3 \
  --episodes 5 \
  --control-cost-weight 0.05
```

The comparison reports forward and lateral displacement, motor-command
magnitude and saturation, command changes, and actuated joint speed so reward
exploitation is visible rather than hidden behind a single return value.
Use the same `--control-cost-weight` for training and comparison when reporting
returns from an experiment.

Play the trained policy in the passive MuJoCo viewer:

```bash
mjpython -m nervenet.cli.view_policy --modules 3
```
