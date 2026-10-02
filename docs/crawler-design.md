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
`12`, a physically plausible initial maximum for the 9 kg three-module crawler.
The policy-facing action space remains independent of this physical scale.

The reward subtracts `0.05 * mean(action^2)` from forward velocity. Averaging
keeps the effort cost comparable across morphologies with different actuator
counts; it is an optimization proxy for motor effort, not a literal energy
model.

The experiment tooling defaults to a three-module crawler and can train a
conventional flat PPO policy, run seeded episodes through a policy-independent
evaluator, and compare the frozen policy with a reproducible random-action
baseline. Generated models live under the ignored `artifacts/` directory.

Evaluation also records lateral displacement, motor-command magnitude,
saturation and change, and actuated joint speed. These diagnostics exposed a
high-return policy that drifted sideways while rapidly switching near-maximum
motor commands.

Training and comparison both expose the control-cost weight explicitly. An
experiment must use the same value for both commands when comparing returns.
The tracked policy viewer uses a non-colliding one-meter grid over a dark
40-by-40-meter floor; stronger lines mark five-meter intervals.

## Body graph

The compiled MuJoCo model is converted into a body-based graph for the
structured policy.

- Every robot body except the MuJoCo world becomes one node.
- Parent body IDs define the graph hierarchy.
- A joint belongs to the child body that it moves.
- An actuator belongs to the body containing its target joint.
- Passive bodies remain graph nodes even when they own no actuator.

For one crawler module, the graph contains one torso and four leg-segment
nodes. Every additional module adds one torso and four leg-segment nodes.

## Local observations

Changing simulation state is kept separate from the static body graph. At each
step, every body node receives the values owned by its joint:

- The root torso receives height, quaternion rotation, three linear velocities,
  and three angular velocities. Global X and Y position are excluded, leaving
  11 values.
- A hip, knee, or passive spine body receives its hinge angle and angular
  velocity, leaving two values.

Graph observations remain separated by node and preserve graph order. A
three-module crawler has one 11-value root observation and fourteen 2-value
hinge observations, containing the same 39 state values as the flat policy's
observation in a different organization.

Before entering a neural network, shorter observations are padded with zeros to
the root observation width. The resulting matrix has one row per graph node and
11 columns. A two-module crawler therefore produces a `10 x 11` input matrix;
padding changes the layout but does not add simulated state.

## Shared input encoder

The first learned policy component applies one shared PyTorch linear layer and
`tanh` activation to every padded node row. With the current defaults, each
11-value observation becomes a 64-value hidden state. The same 768 trainable
parameters process every body node, so increasing the module count changes the
number of rows but does not change the encoder's parameter count.

The encoder does not exchange information between nodes or produce motor
commands. It only creates the initial hidden representation used by later
message-passing stages.

## Next steps

1. Derive directed message-passing connections from the body hierarchy.
2. Implement graph message passing while keeping PPO and the environment fixed.
3. Compare flat and graph policies across multiple training seeds.
