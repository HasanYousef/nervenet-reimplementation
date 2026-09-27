# NerveNet Reimplementation

An independent, from-scratch reimplementation and study of the method described in *NerveNet: Learning Structured Policy with Graph Neural Networks*.

The project uses modern Python and MuJoCo. It does not reuse the authors' implementation.

## Status

MuJoCo 3.14.0 has been verified on macOS. The current learning model introduces
MJCF bodies, geoms, a hinge joint, and a motor actuator.

Open the model in MuJoCo's native viewer:

```bash
python -m mujoco.viewer --mjcf="$PWD/models/test.xml"
```
