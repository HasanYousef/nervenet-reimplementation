from dataclasses import dataclass
import mujoco


@dataclass(frozen=True)
class BodyNode:
    body_id: int
    name: str
    parent_body_id: int | None
    joint_ids: tuple[int, ...]
    actuator_ids: tuple[int, ...]


@dataclass(frozen=True)
class BodyGraph:
    nodes: tuple[BodyNode, ...]


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
                parent_body_id=(None if parent_body_id == 0 else parent_body_id),
                joint_ids=joint_ids,
                actuator_ids=actuator_ids,
            )
        )

    return BodyGraph(nodes=tuple(nodes))
