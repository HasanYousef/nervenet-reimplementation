from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class MorphologyNode:
    """A movable MuJoCo body and the controls physically attached to it."""

    index: int
    body_id: int
    name: str
    parent_index: int | None
    joint_ids: tuple[int, ...]
    actuator_ids: tuple[int, ...]


@dataclass(frozen=True)
class MorphologyEdge:
    """A physical parent-child connection between two body nodes."""

    parent: int
    child: int


@dataclass(frozen=True)
class MorphologyGraph:
    nodes: tuple[MorphologyNode, ...]
    edges: tuple[MorphologyEdge, ...]

    @property
    def action_size(self) -> int:
        return sum(len(node.actuator_ids) for node in self.nodes)


def _ids_grouped_by_body(values: list[int], body_ids: list[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for value, body_id in zip(values, body_ids, strict=True):
        grouped.setdefault(body_id, []).append(value)
    return grouped


def build_morphology_graph(model: mujoco.MjModel) -> MorphologyGraph:
    """Extract the robot's body graph and local control ownership from MuJoCo."""

    body_ids = list(range(1, model.nbody))  # Body 0 is MuJoCo's static world.
    node_index_by_body_id = {
        body_id: node_index for node_index, body_id in enumerate(body_ids)
    }

    joint_ids = list(range(model.njnt))
    joints_by_body = _ids_grouped_by_body(joint_ids, model.jnt_bodyid.tolist())

    actuator_ids = list(range(model.nu))
    actuator_body_ids: list[int] = []
    for actuator_id in actuator_ids:
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(
                "Morphology graphs currently require joint-transmission actuators"
            )
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_body_ids.append(int(model.jnt_bodyid[joint_id]))
    actuators_by_body = _ids_grouped_by_body(actuator_ids, actuator_body_ids)

    nodes: list[MorphologyNode] = []
    edges: list[MorphologyEdge] = []

    for node_index, body_id in enumerate(body_ids):
        parent_body_id = int(model.body_parentid[body_id])
        parent_index = node_index_by_body_id.get(parent_body_id)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None:
            raise ValueError(f"Body {body_id} must have a name")

        nodes.append(
            MorphologyNode(
                index=node_index,
                body_id=body_id,
                name=name,
                parent_index=parent_index,
                joint_ids=tuple(joints_by_body.get(body_id, ())),
                actuator_ids=tuple(actuators_by_body.get(body_id, ())),
            )
        )
        if parent_index is not None:
            edges.append(MorphologyEdge(parent=parent_index, child=node_index))

    graph = MorphologyGraph(nodes=tuple(nodes), edges=tuple(edges))
    if graph.action_size != model.nu:
        raise ValueError("Every actuator must belong to exactly one graph node")
    return graph
