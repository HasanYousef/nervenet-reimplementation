# NerveNet Reimplementation

An independent, from-scratch reimplementation and study of the method described in *NerveNet: Learning Structured Policy with Graph Neural Networks*.

The project uses modern Python and MuJoCo. It does not reuse the authors' implementation.

## Status

MuJoCo 3.14.0 has been verified on macOS. The current model is a modular crawler
generated through MuJoCo's `MjSpec` API. It has a functional head marker,
mirrored two-segment legs, and passive hinges connecting neighboring torsos.

Preview the passive spine of a two-module crawler:

```bash
mjpython -m nervenet.cli.view_crawler --modules 2 --motion spine
```

Preview the coordinated leg motion:

```bash
mjpython -m nervenet.cli.view_crawler --modules 2 --motion gait
```

Change `--modules` to construct another morphology from the same builder. These
commands are kinematic previews; they do not run physics yet.

Run the model tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
