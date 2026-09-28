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

The initial torso height is derived from the lower-leg geometry and a small
ground clearance. Episodes therefore begin at contact height instead of with a
large fall that a policy could exploit for forward displacement.

Each normalized motor command in `[-1, 1]` maps through an actuator gear of
`20`, providing enough torque authority for the 9 kg three-module crawler while
keeping the policy-facing action space independent of the physical scale.

The experiment tooling defaults to a three-module crawler and can train a
conventional flat PPO policy, run seeded
episodes through a policy-independent evaluator, and compare the frozen policy
with a reproducible random-action baseline. Generated models live under the
ignored `artifacts/` directory.

## Next steps

1. Establish a meaningful flat-policy learning curve across multiple seeds.
2. Add deterministic playback for trained policies.
3. Define the morphology graph before implementing graph message passing.
