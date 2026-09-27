import argparse
import math
import time

import mujoco
import mujoco.viewer

from nervenet.models.crawler import TORSO_SPACING, build_crawler_model


CYCLE_SECONDS = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and preview a procedurally generated crawler."
    )
    parser.add_argument(
        "--modules",
        type=int,
        default=2,
        help="Number of torso modules to generate (default: 2)",
    )
    parser.add_argument(
        "--motion",
        choices=("static", "spine", "gait"),
        default="spine",
        help="Kinematic preview to display (default: spine)",
    )
    return parser.parse_args()


def joint_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if result < 0:
        raise ValueError(f"Joint not found: {name}")
    return result


def set_normalized_joint_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint: int,
    position: float,
) -> None:
    minimum, maximum = model.jnt_range[joint]
    midpoint = (minimum + maximum) / 2.0
    amplitude = (maximum - minimum) / 2.0
    data.qpos[model.jnt_qposadr[joint]] = midpoint + amplitude * position


def preview_spine(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    module_count: int,
    phase: float,
) -> None:
    for module_index in range(2, module_count + 1):
        spine = joint_id(model, f"spine_{module_index - 1}_to_{module_index}")
        set_normalized_joint_position(model, data, spine, math.sin(phase))


def preview_gait(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    module_count: int,
    phase: float,
) -> None:
    for module_index in range(1, module_count + 1):
        for side in ("left", "right"):
            hip = joint_id(model, f"module_{module_index}_{side}_hip")
            knee = joint_id(model, f"module_{module_index}_{side}_knee")

            set_normalized_joint_position(model, data, hip, -math.cos(phase))
            knee_position = math.sin(phase) if side == "left" else -math.sin(phase)
            set_normalized_joint_position(model, data, knee, knee_position)


def main() -> None:
    args = parse_args()
    model = build_crawler_model(args.modules)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [
            -TORSO_SPACING * (args.modules - 1) / 2.0,
            0.0,
            0.78,
        ]
        viewer.cam.distance = max(1.8, 0.65 * args.modules)
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0

        started_at = time.monotonic()
        while viewer.is_running():
            phase = 2.0 * math.pi * (time.monotonic() - started_at) / CYCLE_SECONDS

            if args.motion == "spine":
                preview_spine(model, data, args.modules, phase)
            elif args.motion == "gait":
                preview_gait(model, data, args.modules, phase)

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
