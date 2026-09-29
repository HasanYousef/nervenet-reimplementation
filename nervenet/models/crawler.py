import math

import mujoco

TORSO_RADIUS = 0.18
TORSO_SPACING = 2.0 * TORSO_RADIUS

HIP_OFFSET = 0.16
UPPER_LEG_ENDPOINT = [0.20, 0.0, 0.04]
LOWER_LEG_ENDPOINT = [0.28, 0.0, -0.20]
LEG_RADIUS = 0.04
GROUND_CLEARANCE = 0.01
TORSO_HEIGHT = (
    LEG_RADIUS - UPPER_LEG_ENDPOINT[2] - LOWER_LEG_ENDPOINT[2] + GROUND_CLEARANCE
)

TORSO_MASS = 2.0
UPPER_LEG_MASS = 0.3
LOWER_LEG_MASS = 0.2
ACTUATOR_GEAR = 12.0

GROUND_HALF_SIZE = 20
GRID_SPACING = 1
GRID_LINE_HALF_WIDTH = 0.006
GRID_LINE_HALF_HEIGHT = 0.001
GROUND_RGBA = [0.16, 0.18, 0.21, 1.0]
GRID_RGBA = [0.32, 0.35, 0.39, 1.0]
MAJOR_GRID_RGBA = [0.50, 0.54, 0.59, 1.0]

HEAD_TORSO_RGBA = [0.16, 0.38, 0.64, 1.0]
TORSO_RGBA = [0.38, 0.42, 0.48, 1.0]
UPPER_LEG_RGBA = [0.56, 0.59, 0.64, 1.0]
LOWER_LEG_RGBA = [0.46, 0.49, 0.54, 1.0]
COLLECTION_SITE_RGBA = [0.16, 0.70, 0.36, 0.75]


def _add_motor(spec: mujoco.MjSpec, joint_name: str) -> None:
    spec.add_actuator(
        name=f"{joint_name}_motor",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=joint_name,
        ctrllimited=True,
        ctrlrange=[-1.0, 1.0],
        gear=[ACTUATOR_GEAR, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def _add_leg(
    spec: mujoco.MjSpec,
    torso: mujoco.MjsBody,
    module_index: int,
    side: str,
) -> None:
    side_sign = 1.0 if side == "left" else -1.0
    prefix = f"module_{module_index}_{side}"

    upper_leg = torso.add_body(
        name=f"{prefix}_upper_leg",
        pos=[0.0, side_sign * HIP_OFFSET, 0.0],
        euler=[0.0, 0.0, side_sign * math.pi / 2.0],
    )
    hip_name = f"{prefix}_hip"
    upper_leg.add_joint(
        name=hip_name,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 0.0, 1.0],
        range=[math.radians(-40.0), math.radians(40.0)],
        limited=True,
        damping=0.5,
    )
    _add_motor(spec, hip_name)
    upper_leg.add_geom(
        name=f"{prefix}_upper_leg_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.0, 0.0, 0.0, *UPPER_LEG_ENDPOINT],
        size=[LEG_RADIUS, 0.0, 0.0],
        mass=UPPER_LEG_MASS,
        rgba=UPPER_LEG_RGBA,
    )

    lower_leg = upper_leg.add_body(
        name=f"{prefix}_lower_leg",
        pos=UPPER_LEG_ENDPOINT,
    )
    knee_name = f"{prefix}_knee"
    lower_leg.add_joint(
        name=knee_name,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 1.0, 0.0],
        range=[math.radians(-35.0), math.radians(50.0)],
        limited=True,
        damping=0.5,
    )
    _add_motor(spec, knee_name)
    lower_leg.add_geom(
        name=f"{prefix}_lower_leg_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.0, 0.0, 0.0, *LOWER_LEG_ENDPOINT],
        size=[LEG_RADIUS, 0.0, 0.0],
        mass=LOWER_LEG_MASS,
        rgba=LOWER_LEG_RGBA,
    )


def _add_module_geometry(
    spec: mujoco.MjSpec,
    torso: mujoco.MjsBody,
    module_index: int,
    *,
    is_head: bool = False,
) -> None:
    torso.add_geom(
        name=f"module_{module_index}_torso_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[TORSO_RADIUS, 0.0, 0.0],
        mass=TORSO_MASS,
        rgba=HEAD_TORSO_RGBA if is_head else TORSO_RGBA,
    )
    if is_head:
        torso.add_site(
            name="head_collection_site",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[TORSO_RADIUS + 0.03, 0.0, 0.0],
            size=[0.04, 0.0, 0.0],
            rgba=COLLECTION_SITE_RGBA,
        )
    _add_leg(spec, torso, module_index, "left")
    _add_leg(spec, torso, module_index, "right")


def _grid_coordinate_name(coordinate: int) -> str:
    if coordinate < 0:
        return f"negative_{abs(coordinate)}"
    if coordinate > 0:
        return f"positive_{coordinate}"
    return "zero"


def _add_ground_grid(spec: mujoco.MjSpec) -> None:
    for coordinate in range(
        -GROUND_HALF_SIZE,
        GROUND_HALF_SIZE + GRID_SPACING,
        GRID_SPACING,
    ):
        coordinate_name = _grid_coordinate_name(coordinate)
        color = MAJOR_GRID_RGBA if coordinate % 5 == 0 else GRID_RGBA

        spec.worldbody.add_geom(
            name=f"ground_grid_x_{coordinate_name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[coordinate, 0.0, GRID_LINE_HALF_HEIGHT],
            size=[
                GRID_LINE_HALF_WIDTH,
                GROUND_HALF_SIZE,
                GRID_LINE_HALF_HEIGHT,
            ],
            rgba=color,
            contype=0,
            conaffinity=0,
        )
        spec.worldbody.add_geom(
            name=f"ground_grid_y_{coordinate_name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, coordinate, GRID_LINE_HALF_HEIGHT],
            size=[
                GROUND_HALF_SIZE,
                GRID_LINE_HALF_WIDTH,
                GRID_LINE_HALF_HEIGHT,
            ],
            rgba=color,
            contype=0,
            conaffinity=0,
        )


def build_crawler_spec(module_count: int) -> mujoco.MjSpec:
    if module_count < 1:
        raise ValueError("module_count must be at least 1")

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.modelname = f"crawler_{module_count}_modules"

    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[GROUND_HALF_SIZE, GROUND_HALF_SIZE, 0.1],
        rgba=GROUND_RGBA,
    )
    _add_ground_grid(spec)

    torso = spec.worldbody.add_body(
        name="module_1_torso",
        pos=[0.0, 0.0, TORSO_HEIGHT],
    )
    torso.add_freejoint(name="root")
    _add_module_geometry(spec, torso, 1, is_head=True)

    for module_index in range(2, module_count + 1):
        next_torso = torso.add_body(
            name=f"module_{module_index}_torso",
            pos=[-TORSO_SPACING, 0.0, 0.0],
        )
        next_torso.add_joint(
            name=f"spine_{module_index - 1}_to_{module_index}",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=[TORSO_RADIUS, 0.0, 0.0],
            axis=[0.0, 0.0, 1.0],
            range=[math.radians(-20.0), math.radians(20.0)],
            limited=True,
            damping=1.0,
        )
        _add_module_geometry(spec, next_torso, module_index)
        torso = next_torso

    return spec


def build_crawler_model(module_count: int) -> mujoco.MjModel:
    return build_crawler_spec(module_count).compile()
