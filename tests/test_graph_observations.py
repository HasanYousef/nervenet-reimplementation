import unittest

import mujoco
import numpy as np

from nervenet.graphs import (
    build_body_graph,
    get_graph_observations,
    get_hinge_observation,
    get_root_observation,
)
from nervenet.graphs.observations import get_body_observation
from nervenet.models import build_crawler_model


class GraphObservationTest(unittest.TestCase):
    def test_hinge_observation_contains_its_angle_and_velocity(self) -> None:
        model = build_crawler_model(1)
        data = mujoco.MjData(model)
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "module_1_left_knee",
        )

        position_index = int(model.jnt_qposadr[joint_id])
        velocity_index = int(model.jnt_dofadr[joint_id])
        data.qpos[position_index] = 0.4
        data.qvel[velocity_index] = -1.2

        observation = get_hinge_observation(model, data, joint_id)

        np.testing.assert_array_equal(observation, [0.4, -1.2])
        self.assertEqual(observation.dtype, np.float64)

    def test_root_observation_excludes_global_horizontal_position(self) -> None:
        model = build_crawler_model(1)
        data = mujoco.MjData(model)
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "root",
        )

        data.qpos[:7] = [10.0, 20.0, 0.5, 1.0, 0.0, 0.0, 0.0]
        data.qvel[:6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        observation = get_root_observation(model, data, joint_id)

        np.testing.assert_array_equal(
            observation,
            [0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        self.assertEqual(observation.dtype, np.float64)

    def test_body_observation_dispatches_by_joint_type(self) -> None:
        model = build_crawler_model(1)
        data = mujoco.MjData(model)
        graph = build_body_graph(model)
        nodes = {node.name: node for node in graph.nodes}

        knee = nodes["module_1_left_lower_leg"]
        knee_joint_id = knee.joint_ids[0]
        data.qpos[model.jnt_qposadr[knee_joint_id]] = 0.25
        data.qvel[model.jnt_dofadr[knee_joint_id]] = -0.75

        root_observation = get_body_observation(
            model,
            data,
            nodes["module_1_torso"],
        )
        knee_observation = get_body_observation(model, data, knee)

        self.assertEqual(root_observation.shape, (11,))
        np.testing.assert_array_equal(knee_observation, [0.25, -0.75])

    def test_one_module_graph_observation_shapes(self) -> None:
        model = build_crawler_model(1)
        data = mujoco.MjData(model)
        graph = build_body_graph(model)

        observations = get_graph_observations(model, data, graph)

        self.assertEqual(
            [observation.shape for observation in observations],
            [(11,), (2,), (2,), (2,), (2,)],
        )
        self.assertEqual(sum(observation.size for observation in observations), 19)

    def test_three_module_graph_observations_preserve_all_state_values(self) -> None:
        model = build_crawler_model(3)
        data = mujoco.MjData(model)
        graph = build_body_graph(model)

        observations = get_graph_observations(model, data, graph)

        self.assertEqual(len(observations), 15)
        self.assertEqual(observations[0].shape, (11,))
        self.assertTrue(
            all(observation.shape == (2,) for observation in observations[1:])
        )
        self.assertEqual(sum(observation.size for observation in observations), 39)


if __name__ == "__main__":
    unittest.main()
