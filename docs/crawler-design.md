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
- Neighboring torsos connect through passive Z-axis spine hinges limited to
  `-20..20` degrees.
- Each module contributes four future actuators and policy actions.
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

The model builder and kinematic previews exist. The root body remains fixed,
the leg joints do not yet have actuators, and the previews do not run physics.

## Next steps

1. Make the crawler dynamic and validate its initial ground contact.
2. Add leg actuators while keeping spine joints passive.
3. Define observations, actions, resets, reward, and evaluation.
4. Train a flat-policy baseline before implementing graph message passing.
