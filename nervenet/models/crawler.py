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
        rgba=[0.20, 0.45, 0.80, 1.0] if is_head else [0.35, 0.38, 0.43, 1.0],
    )
    if is_head:
        torso.add_site(
            name="head_collection_site",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[TORSO_RADIUS + 0.03, 0.0, 0.0],
            size=[0.04, 0.0, 0.0],
            rgba=[0.20, 0.85, 0.35, 0.65],
        )
    _add_leg(spec, torso, module_index, "left")
    _add_leg(spec, torso, module_index, "right")


def build_crawler_spec(module_count: int) -> mujoco.MjSpec:
    if module_count < 1:
        raise ValueError("module_count must be at least 1")

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.modelname = f"crawler_{module_count}_modules"

    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.1],
    )

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
