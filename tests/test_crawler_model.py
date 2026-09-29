import math
import unittest

import mujoco

from nervenet.models.crawler import (
    ACTUATOR_GEAR,
    GROUND_CLEARANCE,
    GROUND_HALF_SIZE,
    LOWER_LEG_RGBA,
    UPPER_LEG_RGBA,
    build_crawler_model,
)


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

        self.assertEqual(model.njnt, 5)
        self.assertEqual(model.nu, 4)
        self.assertIn("root", joint_names)
        self.assertNotIn("spine_1_to_2", joint_names)
        self.assertIn("module_1_left_hip", joint_names)
        self.assertIn("module_1_right_knee", joint_names)
        self.assertIn("head_collection_site", site_names)

        hip_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "module_1_left_hip",
        )
        self.assertAlmostEqual(
            model.jnt_range[hip_id, 0],
            math.radians(-40.0),
        )
        self.assertAlmostEqual(
            model.jnt_range[hip_id, 1],
            math.radians(40.0),
        )

    def test_two_modules_add_four_leg_joints_and_one_spine(self) -> None:
        model = build_crawler_model(2)
        joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        geom_names = names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)

        self.assertEqual(model.njnt, 10)
        self.assertEqual(model.nu, 8)
        self.assertAlmostEqual(model.body_mass.sum(), 6.0)
        self.assertIn("spine_1_to_2", joint_names)
        self.assertIn("module_2_left_hip", joint_names)
        self.assertIn("module_2_right_knee", joint_names)
        self.assertIn("module_1_torso_geom", geom_names)
        self.assertIn("module_2_torso_geom", geom_names)

        ground_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "ground",
        )
        self.assertEqual(model.geom_size[ground_id, 0], GROUND_HALF_SIZE)
        self.assertEqual(model.geom_size[ground_id, 1], GROUND_HALF_SIZE)

        grid_line_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "ground_grid_x_positive_1",
        )
        self.assertEqual(model.geom_contype[grid_line_id], 0)
        self.assertEqual(model.geom_conaffinity[grid_line_id], 0)

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "module_1_torso")
        second_torso_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "module_2_torso"
        )
        collection_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "head_collection_site"
        )

        self.assertLess(data.xpos[second_torso_id, 0], data.xpos[head_id, 0])
        self.assertGreater(data.site_xpos[collection_site_id, 0], data.xpos[head_id, 0])

    def test_three_module_feet_start_just_above_ground(self) -> None:
        model = build_crawler_model(3)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        self.assertEqual(model.njnt, 15)
        self.assertEqual(model.nu, 12)
        self.assertAlmostEqual(model.body_mass.sum(), 9.0)
        self.assertTrue((model.actuator_gear[:, 0] == ACTUATOR_GEAR).all())

        upper_leg_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "module_1_left_upper_leg_geom",
        )
        lower_leg_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "module_1_left_lower_leg_geom",
        )
        for actual, expected in zip(
            model.geom_rgba[upper_leg_id],
            UPPER_LEG_RGBA,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            model.geom_rgba[lower_leg_id],
            LOWER_LEG_RGBA,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

        for module_index in range(1, 4):
            for side in ("left", "right"):
                geom_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"module_{module_index}_{side}_lower_leg_geom",
                )
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                radius = model.geom_size[geom_id, 0]
                half_length = model.geom_size[geom_id, 1]
                lowest_point = (
                    data.geom_xpos[geom_id, 2]
                    - abs(rotation[2, 2]) * half_length
                    - radius
                )

                self.assertAlmostEqual(lowest_point, GROUND_CLEARANCE)

    def test_module_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            build_crawler_model(0)


if __name__ == "__main__":
    unittest.main()
