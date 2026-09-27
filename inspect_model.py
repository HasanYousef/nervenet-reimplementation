import time
from pathlib import Path

import mujoco
import mujoco.viewer

model_path = Path(__file__).parent / "models" / "test.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))

data = mujoco.MjData(model)

duration = 10.0

steps = int(duration / model.opt.timestep)

print_every = int(0.1 / model.opt.timestep)

target_position = 1.0
kp = 100.0
kd = 10.0

viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.lookat[:] = [0.3, 0.0, 0.8]
viewer.cam.distance = 3.0

for step in range(steps):
    position = data.qpos[0]
    velocity = data.qvel[0]

    control = kp * (target_position - position) - kd * velocity

    data.ctrl[0] = max(-100.0, min(100.0, control))

    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(model.opt.timestep)

    if (step + 1) % print_every == 0:
        print(
            f"time={data.time:.1f}s, "
            f"position={data.qpos[0]:.3f} rad, "
            f"velocity={data.qvel[0]:.3f} rad/s"
        )
