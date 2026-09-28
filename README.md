# NerveNet Reimplementation

An independent, from-scratch reimplementation and study of the method described in *NerveNet: Learning Structured Policy with Graph Neural Networks*.

The project uses modern Python and MuJoCo. It does not reuse the authors' implementation.

## Status

MuJoCo 3.14.0 has been verified on macOS. The current model is a modular crawler
generated through MuJoCo's `MjSpec` API. It has a functional head marker,
mirrored actuated legs, passive hinges connecting neighboring torsos, and a
free-moving root. A Gymnasium environment skeleton defines the action and
observation spaces, deterministic reset, and physics stepping. Reward and
episode time-limit logic are not implemented yet.

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
