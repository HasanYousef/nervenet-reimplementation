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
MuJoCo model and reads current observations for each node. The root torso
receives its orientation and motion state, while hip, knee, and passive spine
bodies receive their own joint angle and angular velocity. Every node also
receives the external torque and force acting on its associated body. Graph
nodes are classified as root, joint-owning, or jointless body nodes separately
from whether they own an actuator.
One shared input encoder converts every padded local observation into a
fixed-width hidden representation without depending on the number of nodes.
During every propagation round, a shared two-layer `tanh` MLP computes outgoing
messages and incoming messages are averaged. Root, joint-owning, and jointless
body nodes then use separate GRU update networks, with all nodes of the same
type sharing parameters.
The graph actor then uses a shared two-layer MLP to decode motor-owning node
states into action means in MuJoCo actuator order. The same actor instance
supports crawler morphologies with different module and actuator counts. A Gymnasium observation wrapper exposes the same
environment state as padded per-node matrices for graph-policy training and a
matched MLP baseline while leaving the original flat PPO baseline unchanged. A custom
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

## Managed experiments

Use the experiment CLI for research runs. One name owns the immutable
configuration, checkpoints, deterministic evaluations, raw SB3 logs, training
sessions, and software-version metadata:

```bash
.venv/bin/python -m nervenet.cli.experiment start \
  --name graph-v0-seed0 \
  --policy-type graph \
  --modules 3 \
  --timesteps 1000000 \
  --checkpoint-every 100000 \
  --evaluation-episodes 20 \
  --seed 0 \
  --control-cost-weight 0.05
```

Continue the same experiment from its latest checkpoint:

```bash
.venv/bin/python -m nervenet.cli.experiment continue \
  --name graph-v0-seed0 \
  --timesteps 500000
```

Evaluate or view its latest checkpoint without repeating its configuration:

```bash
.venv/bin/python -m nervenet.cli.experiment evaluate \
  --name graph-v0-seed0

mjpython -m nervenet.cli.experiment view \
  --name graph-v0-seed0
```

Experiments are stored under `experiments/<name>/`. Checkpoints use their real
completed PPO timestep, such as `step_000100352.zip`. `experiment.json` embeds
the complete configuration, session histories, checkpoint metrics, aggregate
and per-episode evaluations, Git revisions, and dependency versions. Raw SB3
CSV and JSON logs are also retained separately for every continuation session.
See [docs/experiments.md](docs/experiments.md) for the lifecycle and schema.

The lower-level commands below remain useful for short debugging runs and
manual checkpoint management.

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

Train the three-module matched MLP baseline. It receives the same graph
observations as the graph policy, flattens them, and has approximately the same
active actor parameter count while using the exact same critic and PPO output
path:

```bash
.venv/bin/python -m nervenet.cli.train_matched \
  --modules 3 \
  --timesteps 100000 \
  --seed 0 \
  --control-cost-weight 0.05
```

Continue either policy from a checkpoint. Here `--timesteps 100000` means
100,000 additional environment steps. A separate output is required so the
source checkpoint remains unchanged:

```bash
.venv/bin/python -m nervenet.cli.train_graph \
  --modules 3 \
  --timesteps 100000 \
  --control-cost-weight 0.05 \
  --resume artifacts/graph_controlled_100k_seed0.zip \
  --output artifacts/graph_controlled_200k_seed0

.venv/bin/python -m nervenet.cli.train_matched \
  --modules 3 \
  --timesteps 100000 \
  --control-cost-weight 0.05 \
  --resume artifacts/matched_controlled_100k_seed0.zip \
  --output artifacts/matched_controlled_200k_seed0
```

Stable-Baselines3 may finish the current PPO rollout, so the stored timestep
counter can be slightly above the requested round number.

Evaluate the flat policy over reproducible episode seeds:

```bash
.venv/bin/python -m nervenet.cli.evaluate_policy \
  --policy-type flat \
  --modules 3 \
  --episodes 5 \
  --control-cost-weight 0.05
```

Evaluate the graph policy with the same metrics and episode seeds:

```bash
.venv/bin/python -m nervenet.cli.evaluate_policy \
  --policy-type graph \
  --modules 3 \
  --episodes 20 \
  --control-cost-weight 0.05
```

Evaluate the matched MLP policy independently:

```bash
.venv/bin/python -m nervenet.cli.evaluate_policy \
  --policy-type matched \
  --modules 3 \
  --episodes 20 \
  --control-cost-weight 0.05
```

The evaluation command supports `--policy-type flat`, `--policy-type matched`,
and `--policy-type graph`. It reports forward and lateral displacement, motor-command
magnitude and saturation, command changes, and actuated joint speed so reward
exploitation is visible rather than hidden behind a single return value.
Use the same `--control-cost-weight` for training and evaluation when reporting
returns from an experiment.

Play the trained flat policy in the passive MuJoCo viewer:

```bash
mjpython -m nervenet.cli.view_policy --modules 3 --policy-type flat
```

Play the trained graph policy with the same viewer:

```bash
mjpython -m nervenet.cli.view_policy --modules 3 --policy-type graph
```

The same viewer accepts `--policy-type matched` for the matched MLP baseline.
