import unittest

import mujoco

from nervenet.models.crawler import build_crawler_model


def names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> set[str]:
    return {
        name
        for object_id in range(count)
        if (name := mujoco.mj_id2name(model, object_type, object_id)) is not None
    }


class CrawlerModelTest(unittest.TestCase):
    def test_one_module_has_four_leg_joints_and_no_spine(self) -> None:
        model = build_crawler_model(1)
        joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        site_names = names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)

        self.assertEqual(model.njnt, 4)
        self.assertNotIn("spine_1_to_2", joint_names)
        self.assertIn("module_1_left_hip", joint_names)
        self.assertIn("module_1_right_knee", joint_names)
        self.assertIn("head_collection_site", site_names)

    def test_two_modules_add_four_leg_joints_and_one_spine(self) -> None:
        model = build_crawler_model(2)
        joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        geom_names = names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)

        self.assertEqual(model.njnt, 9)
        self.assertIn("spine_1_to_2", joint_names)
        self.assertIn("module_2_left_hip", joint_names)
        self.assertIn("module_2_right_knee", joint_names)
        self.assertIn("module_1_torso_geom", geom_names)
        self.assertIn("module_2_torso_geom", geom_names)

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        head_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "module_1_torso"
        )
        second_torso_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "module_2_torso"
        )
        collection_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "head_collection_site"
        )

        self.assertLess(data.xpos[second_torso_id, 0], data.xpos[head_id, 0])
        self.assertGreater(data.site_xpos[collection_site_id, 0], data.xpos[head_id, 0])

    def test_module_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            build_crawler_model(0)


if __name__ == "__main__":
    unittest.main()
