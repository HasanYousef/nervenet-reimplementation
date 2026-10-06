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

Each parent-child body connection is also converted into two directed message
routes: parent to child and child to parent. Routes use graph-node indexes, so
they address the corresponding rows in the policy's hidden-state matrix rather
than MuJoCo body IDs. This only defines where information may travel; message
aggregation and node-state updates remain separate policy components.

## Local observations

Changing simulation state is kept separate from the static body graph. At each
step, every graph node combines the values owned by its joint with the external
torque and force acting on its associated MuJoCo body:

- The root torso receives height, quaternion rotation, three linear velocities,
  three angular velocities, three external torque values, and three external
  force values. Global X and Y position are excluded, leaving 17 values.
- A hip, knee, or passive spine body receives its hinge angle and angular
  velocity followed by three external torque values and three external force
  values, leaving eight values.

Graph observations remain separated by node and preserve graph order. A
three-module crawler has one 17-value root observation and fourteen 8-value
hinge observations. Unlike the flat baseline observation, this structured
observation includes body-specific force information.

MuJoCo stores each body wrench in `cfrc_ext` using torque-before-force order and
does not populate that array for this model automatically. Graph observation
collection therefore runs MuJoCo's post-constraint force calculation before
reading the body values. These quantities use MuJoCo's center-of-mass-based
`c` frame, which is oriented like the world frame.

Before entering a neural network, shorter observations are padded with zeros to
the root observation width. The resulting matrix has one row per graph node and
17 columns. A two-module crawler therefore produces a `10 x 17` input matrix;
padding changes the layout but does not add simulated state.

## Shared input encoder

The first learned policy component applies one shared PyTorch linear layer and
`tanh` activation to every padded node row. With the current defaults, each
17-value observation becomes a 64-value hidden state. The same 1,152 trainable
parameters process every body node, so increasing the module count changes the
number of rows but does not change the encoder's parameter count.

The encoder does not exchange information between nodes or produce motor
commands. It only creates the initial hidden representation used by later
message-passing stages.

## Message passing

One message-passing round first applies a shared two-layer `tanh` MLP to every
node's hidden state. The resulting outgoing messages are sent along the graph's
directed routes and averaged at each receiver. Nodes with no incoming route
receive a zero message. A shared GRU cell then combines each node's aggregated
neighbor message with its current 64-value hidden state. The output preserves
the `node_count x 64` shape, and gradients flow through message computation,
aggregation, and the state update.

Aggregation contains no learned parameters. The message MLP and GRU parameters
are shared by every node, so the same message-passing layer handles crawler
graphs with different module counts. This matches the paper's reported
experimental choice of an MLP message function, average aggregation, and a GRU
update.

A graph processor recurrently applies that same layer for a configurable
number of propagation steps. Each round moves information across one more
physical connection without adding another set of trainable parameters. The
default is four rounds, which sits within the paper's tested range of three to
six. The default hidden width of 64 and default propagation depth are defined
once in the policy configuration and reused by every policy layer, the trainer,
and the CLI.

## Actuator output

The body graph records which node owns each MuJoCo actuator. A deterministic
mapping selects motor-owning hidden-state rows in actuator-ID order, skipping
passive torso and spine nodes. One shared linear decoder maps every selected
64-value node state to a scalar action mean. It does not squash or sample the
action; Gaussian sampling and action-space handling belong to PPO integration.

## Graph actor

The graph actor composes the shared input encoder, recurrent graph processor,
and shared actuator decoder into one differentiable network. It accepts padded
node observations plus the static routes and actuator-node mapping, and returns
one action mean per MuJoCo actuator. The same actor instance has been verified
on one- and three-module crawlers without changing its parameters.

The actor also preserves arbitrary leading batch dimensions. PPO can therefore
process many collected simulation snapshots together as
`batch_size x node_count x feature_count` tensors while keeping message
aggregation isolated within each snapshot.

The base crawler environment continues to return its original flat observation
for baseline compatibility. A separate Gymnasium observation wrapper derives
the static body graph and replaces each returned flat vector with a padded
`node_count x 17` matrix read from the same MuJoCo state. Physics, actions,
rewards, termination, and reset behavior remain owned by the base environment.

## Value network

PPO's critic is a separate flat MLP. It flattens the padded graph observation,
passes it through two 64-value `tanh` hidden layers, and predicts one scalar
state value per observation. This follows the NerveNet-MLP separation: the
actor uses the physical graph while the critic evaluates the complete state
without message passing.

## Stable-Baselines3 adapter

Stable-Baselines3 normally flattens observations before its policy network. A
pass-through feature extractor preserves the graph tensor instead. A combined
actor-critic extractor then sends that tensor to the graph actor and flat value
network, returning one action mean per actuator and one state value per batch
item. These adapters contain no additional policy logic beyond satisfying the
Stable-Baselines3 interface.

The custom actor-critic policy uses those outputs directly rather than adding
another actor or critic head. It combines the actor's action means with one
learned Gaussian log-standard-deviation per actuator, giving PPO a stochastic
continuous-action distribution for exploration. The policy validates that its
actuator-node mapping matches the environment's action dimension and builds one
optimizer over the graph actor, critic, and exploration parameters.

## Next steps

Graph-policy training is now connected to PPO and covered by a short
end-to-end optimization test. The training CLI saves the resulting PPO model
and the shared policy viewer reconstructs either the flat or graph observation
environment before loading the corresponding model.
The deterministic episode evaluator also accepts either observation interface,
while reading displacement and joint diagnostics from the same underlying
MuJoCo environment. This allows random, flat, and graph results to use the same
episode seeds and metrics.

1. Continue auditing node types, local observations, and propagation depth
   against the paper before spending more compute.
2. Compare flat and graph policies across multiple training seeds.
