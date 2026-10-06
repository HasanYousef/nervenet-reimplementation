import unittest

import mujoco
import numpy as np

from nervenet.graphs import (
    build_body_graph,
    get_graph_observations,
    get_hinge_observation,
    get_root_observation,
)
from nervenet.graphs.observations import (
    get_body_observation,
    pad_graph_observations,
)
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
        data.cfrc_ext[knee.body_id] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        root_observation = get_body_observation(
            model,
            data,
            nodes["module_1_torso"],
        )
        knee_observation = get_body_observation(model, data, knee)

        self.assertEqual(root_observation.shape, (17,))
        np.testing.assert_array_equal(
            knee_observation,
            [0.25, -0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )

    def test_one_module_graph_observation_shapes(self) -> None:
        model = build_crawler_model(1)
        data = mujoco.MjData(model)
        graph = build_body_graph(model)

        observations = get_graph_observations(model, data, graph)

        self.assertEqual(
            [observation.shape for observation in observations],
            [(17,), (8,), (8,), (8,), (8,)],
        )
        self.assertEqual(sum(observation.size for observation in observations), 49)

    def test_three_module_graph_observations_preserve_all_state_values(self) -> None:
        model = build_crawler_model(3)
        data = mujoco.MjData(model)
        graph = build_body_graph(model)

        observations = get_graph_observations(model, data, graph)

        self.assertEqual(len(observations), 15)
        self.assertEqual(observations[0].shape, (17,))
        self.assertTrue(
            all(observation.shape == (8,) for observation in observations[1:])
        )
        self.assertEqual(sum(observation.size for observation in observations), 129)

    def test_padding_preserves_values_and_fills_unused_columns_with_zero(self) -> None:
        observations = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0]),
        ]

        padded = pad_graph_observations(observations)

        np.testing.assert_array_equal(
            padded,
            [
                [1.0, 2.0, 3.0],
                [4.0, 0.0, 0.0],
            ],
        )

    def test_two_module_observations_pad_to_ten_by_seventeen(self) -> None:
        model = build_crawler_model(2)
        data = mujoco.MjData(model)
        graph = build_body_graph(model)
        observations = get_graph_observations(model, data, graph)

        padded = pad_graph_observations(observations)

        self.assertEqual(padded.shape, (10, 17))
        np.testing.assert_array_equal(padded[1, 8:], np.zeros(9))


if __name__ == "__main__":
    unittest.main()
