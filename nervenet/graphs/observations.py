import mujoco
import numpy as np

from nervenet.graphs.body_graph import BodyGraph, BodyNode


ROOT_JOINT_OBSERVATION_SIZE = 11
BODY_EXTERNAL_FORCE_SIZE = 6
GRAPH_OBSERVATION_WIDTH = (
    ROOT_JOINT_OBSERVATION_SIZE + BODY_EXTERNAL_FORCE_SIZE
)


def get_hinge_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_id: int,
) -> np.ndarray:
    position_index = int(model.jnt_qposadr[joint_id])
    velocity_index = int(model.jnt_dofadr[joint_id])

    return np.array(
        [
            data.qpos[position_index],
            data.qvel[velocity_index],
        ],
        dtype=np.float64,
    )


def get_root_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_id: int,
) -> np.ndarray:
    position_index = int(model.jnt_qposadr[joint_id])
    velocity_index = int(model.jnt_dofadr[joint_id])

    position_and_rotation = data.qpos[position_index + 2 : position_index + 7]
    velocity = data.qvel[velocity_index : velocity_index + 6]

    return np.concatenate([position_and_rotation, velocity]).astype(
        np.float64, copy=True
    )


def get_body_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    node: BodyNode,
) -> np.ndarray:
    observations = []

    for joint_id in node.joint_ids:
        joint_type = model.jnt_type[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            observation = get_root_observation(
                model,
                data,
                joint_id,
            )
        elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            observation = get_hinge_observation(
                model,
                data,
                joint_id,
            )
        else:
            raise NotImplementedError(f"Unsupported joint type: {joint_type}")

        observations.append(observation)

    observations.append(
        np.array(
            data.cfrc_ext[node.body_id],
            dtype=np.float64,
            copy=True,
        )
    )

    return np.concatenate(observations)


def get_graph_observations(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    graph: BodyGraph,
) -> list[np.ndarray]:
    # MuJoCo does not populate cfrc_ext unless post-constraint forces are
    # requested by a sensor or computed explicitly.
    mujoco.mj_rnePostConstraint(model, data)

    return [get_body_observation(model, data, node) for node in graph.nodes]


def pad_graph_observations(
    observations: list[np.ndarray],
) -> np.ndarray:
    width = max(observation.size for observation in observations)

    padded = np.zeros(
        (len(observations), width),
        dtype=np.float64,
    )

    for node_index, observation in enumerate(observations):
        padded[node_index, : observation.size] = observation

    return padded
