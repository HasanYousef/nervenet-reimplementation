from dataclasses import dataclass
from enum import Enum

import mujoco


class BodyNodeType(Enum):
    ROOT = "root"
    JOINT = "joint"
    BODY = "body"


@dataclass(frozen=True)
class BodyNode:
    body_id: int
    name: str
    node_type: BodyNodeType
    parent_body_id: int | None
    joint_ids: tuple[int, ...]
    actuator_ids: tuple[int, ...]


@dataclass(frozen=True)
class BodyGraph:
    nodes: tuple[BodyNode, ...]


def get_body_node_type(
    model: mujoco.MjModel,
    joint_ids: tuple[int, ...],
) -> BodyNodeType:
    for joint_id in joint_ids:
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return BodyNodeType.ROOT

    if joint_ids:
        return BodyNodeType.JOINT

    return BodyNodeType.BODY


def build_body_graph(model: mujoco.MjModel) -> BodyGraph:
    nodes = []

    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        parent_body_id = int(model.body_parentid[body_id])

        joint_ids = tuple(
            joint_id
            for joint_id in range(model.njnt)
            if int(model.jnt_bodyid[joint_id]) == body_id
        )

        actuator_ids = tuple(
            actuator_id
            for actuator_id in range(model.nu)
            if int(model.actuator_trnid[actuator_id, 0]) in joint_ids
        )

        nodes.append(
            BodyNode(
                body_id=body_id,
                name=name,
                node_type=get_body_node_type(model, joint_ids),
                parent_body_id=(None if parent_body_id == 0 else parent_body_id),
                joint_ids=joint_ids,
                actuator_ids=actuator_ids,
            )
        )

    return BodyGraph(nodes=tuple(nodes))


def get_message_routes(
    graph: BodyGraph,
) -> list[tuple[int, int]]:
    node_index_by_body_id = {
        node.body_id: node_index for node_index, node in enumerate(graph.nodes)
    }

    routes = []

    for child_index, node in enumerate(graph.nodes):
        if node.parent_body_id is None:
            continue

        parent_index = node_index_by_body_id[node.parent_body_id]

        routes.append((parent_index, child_index))
        routes.append((child_index, parent_index))

    return routes


def get_actuator_node_indices(
    graph: BodyGraph,
) -> tuple[int, ...]:
    actuator_nodes = []

    for node_index, node in enumerate(graph.nodes):
        for actuator_id in node.actuator_ids:
            actuator_nodes.append((actuator_id, node_index))

    actuator_nodes.sort()

    return tuple(node_index for actuator_id, node_index in actuator_nodes)
