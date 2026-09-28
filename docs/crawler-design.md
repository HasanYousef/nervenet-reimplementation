# Modular paired-leg crawler

## Purpose

Build a MuJoCo morphology family that is simple to train, easy to measure, and
structurally variable enough to test a NerveNet-style graph policy.

The same repeated module can create crawlers of different lengths. A shared
policy can eventually be trained on several module counts and evaluated on an
unseen body size without changing its learned parameters.

## Design

- The first task is forward locomotion along world `+X`.
- Each module has a sphere torso and two-segment left/right legs.
- Each leg has a Z-axis hip hinge and a local Y-axis knee hinge.
- Hip limits are `-40..40` degrees; knee limits are `-35..50` degrees.
- Each torso has an explicit mass of `2.0 kg`; upper and lower leg segments are
  `0.3 kg` and `0.2 kg` respectively, for `3.0 kg` per module.
- Neighboring torsos connect through passive Z-axis spine hinges limited to
  `-20..20` degrees.
- Each module contributes four torque actuators and policy actions.
- Primitive geoms are used instead of decorative meshes.

```text
left lower -- knee -- left upper -- hip -- sphere torso
                                             |
right lower -- knee -- right upper -- hip ---+

head -- passive spine hinge -- module 2 -- passive spine hinge -- ...
```

Module 1 is the head. Additional modules extend behind it toward world `-X`.
A visible, non-colliding `head_collection_site` beyond its `+X` surface defines
the future collection point and forward direction.

The morphology is constructed programmatically using MuJoCo's `MjSpec` API.
Handwritten MJCF is not a source of truth for repeated bodies.

## Current boundary

The model builder, kinematic previews, free-moving root, leg motors, and physics
preview exist. The Gymnasium environment defines module-dependent action and
observation spaces, deterministic reset, action validation, physics stepping at
a 50 Hz control rate, a forward-velocity reward, reproducible joint-state reset
noise, and a configurable episode time limit.

## Next steps

1. Validate the environment with Gymnasium's checker.
2. Train a flat-policy baseline before implementing graph message passing.
