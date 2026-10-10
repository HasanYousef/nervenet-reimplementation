"""Render a saved comparison policy as a short MuJoCo GIF.

On macOS, run this module with ``mjpython`` so MuJoCo can create its OpenGL
context on the main thread. The GIF is documentation media, not an evaluation:
reported metrics still come from the held-out multi-episode comparison.
"""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nervenet.models.crawler import (
    GRID_LINE_HALF_HEIGHT,
    GRID_LINE_HALF_WIDTH,
    GRID_RGBA,
    GROUND_HALF_SIZE,
    MAJOR_GRID_RGBA,
    TORSO_HEIGHT,
    TORSO_SPACING,
)
from nervenet.policy_loading import load_crawler_policy


def _selected_checkpoint(
    report_dir: Path,
    runs_dir: Path,
    policy_type: str,
    training_seed: int,
) -> Path:
    with (report_dir / "results.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if (
                row["policy_type"] == policy_type
                and int(row["training_seed"]) == training_seed
            ):
                step = int(row["selected_checkpoint_timesteps"])
                checkpoint = (
                    runs_dir
                    / f"{policy_type}-seed{training_seed}"
                    / "checkpoints"
                    / f"step_{step:09d}.zip"
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                return checkpoint

    raise ValueError(
        f"No {policy_type} seed {training_seed} result in {report_dir}"
    )


def _annotate(
    pixels: np.ndarray,
    policy_type: str,
    training_seed: int,
    elapsed: float,
    distance: float,
) -> Image.Image:
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    title = f"{policy_type.upper()}  |  training seed {training_seed}"
    progress = f"{elapsed:4.1f} s    forward {distance:+.1f} m"
    draw.rectangle((0, 0, image.width, 36), fill=(16, 22, 29))
    draw.text((12, 9), title, font=font, fill=(238, 243, 247))
    progress_width = draw.textlength(progress, font=font)
    draw.text(
        (image.width - progress_width - 12, 9),
        progress,
        font=font,
        fill=(238, 243, 247),
    )
    return image


def _extend_visual_grid(scene: mujoco.MjvScene) -> None:
    """Add non-physical grid lines beyond the model's forward visual edge."""
    identity = np.eye(3, dtype=np.float64).reshape(9)
    grid_end = 2 * GROUND_HALF_SIZE

    def add_box(
        position: tuple[float, float, float],
        size: tuple[float, float, float],
        color: list[float],
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo scene has no room for the extended grid")
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(size, dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            identity,
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1

    for coordinate in range(GROUND_HALF_SIZE + 1, grid_end + 1):
        color = MAJOR_GRID_RGBA if coordinate % 5 == 0 else GRID_RGBA
        add_box(
            (coordinate, 0.0, GRID_LINE_HALF_HEIGHT),
            (GRID_LINE_HALF_WIDTH, GROUND_HALF_SIZE, GRID_LINE_HALF_HEIGHT),
            color,
        )

    for coordinate in range(-GROUND_HALF_SIZE, GROUND_HALF_SIZE + 1):
        color = MAJOR_GRID_RGBA if coordinate % 5 == 0 else GRID_RGBA
        add_box(
            (1.5 * GROUND_HALF_SIZE, coordinate, GRID_LINE_HALF_HEIGHT),
            (0.5 * GROUND_HALF_SIZE, GRID_LINE_HALF_WIDTH, GRID_LINE_HALF_HEIGHT),
            color,
        )


def render_gif(
    checkpoint: Path,
    policy_type: str,
    training_seed: int,
    episode_seed: int,
    module_count: int,
    control_cost_weight: float,
    output: Path,
    fps: int = 8,
    seconds: float = 10.0,
) -> None:
    if fps <= 0 or seconds <= 0:
        raise ValueError("fps and seconds must be positive")

    policy, env = load_crawler_policy(
        model_path=checkpoint,
        policy_type=policy_type,
        module_count=module_count,
        control_cost_weight=control_cost_weight,
    )
    observation, _ = env.reset(seed=episode_seed)
    crawler = env.unwrapped
    initial_x = float(crawler.data.qpos[0])
    mujoco.mj_forward(crawler.model, crawler.data)

    # Brighter lighting is applied only to this rendered copy of the model.
    crawler.model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    crawler.model.vis.headlight.diffuse[:] = (0.7, 0.7, 0.7)
    ground_id = mujoco.mj_name2id(
        crawler.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "ground",
    )
    crawler.model.geom_size[ground_id, :2] = (2 * GROUND_HALF_SIZE,) * 2

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = max(3.0, 1.0 * module_count)
    camera.azimuth = 120.0
    camera.elevation = -25.0

    frames: list[Image.Image] = []
    total_steps = min(
        crawler.max_episode_steps,
        round(seconds / crawler.control_timestep),
    )
    if total_steps < 1:
        raise ValueError("seconds must include at least one control step")
    if fps > 1.0 / crawler.control_timestep:
        raise ValueError("fps cannot exceed the control-step frequency")

    frame_count = min(
        total_steps + 1,
        max(2, round(total_steps * crawler.control_timestep * fps)),
    )
    frame_steps = {
        round(index * total_steps / (frame_count - 1))
        for index in range(frame_count)
    }

    with mujoco.Renderer(crawler.model, height=315, width=560) as renderer:
        for step in range(total_steps + 1):
            if step in frame_steps:
                head = crawler.data.xpos[1]
                camera.lookat[:] = (
                    head[0] - TORSO_SPACING * (module_count - 1) / 2.0,
                    head[1],
                    TORSO_HEIGHT,
                )
                renderer.update_scene(crawler.data, camera=camera)
                _extend_visual_grid(renderer.scene)
                frames.append(
                    _annotate(
                        renderer.render(),
                        policy_type,
                        training_seed,
                        crawler.data.time,
                        float(crawler.data.qpos[0]) - initial_x,
                    )
                )

            if step == total_steps:
                break

            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            if (terminated or truncated) and step + 1 < total_steps:
                break

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=[
            round((index + 1) * 100 / fps) * 10
            - round(index * 100 / fps) * 10
            for index in range(len(frames))
        ],
        loop=0,
        optimize=True,
    )
    env.close()
    print(f"Saved {len(frames)} frames to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="crawler-1m-v1")
    parser.add_argument("--report", default="target_001300000")
    parser.add_argument("--policy-type", choices=("matched", "graph"), required=True)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--episode-seed", type=int)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    args = parser.parse_args()

    report_dir = args.results_dir / args.name / "reports" / args.report
    runs_dir = args.experiments_dir / args.name / "runs"
    with (report_dir / "summary.json").open() as file:
        summary = json.load(file)
    config = summary["config"]
    checkpoint = _selected_checkpoint(
        report_dir,
        runs_dir,
        args.policy_type,
        args.training_seed,
    )
    output = (
        args.results_dir
        / args.name
        / "media"
        / f"{args.policy_type}-seed{args.training_seed}.gif"
    )
    render_gif(
        checkpoint=checkpoint,
        policy_type=args.policy_type,
        training_seed=args.training_seed,
        episode_seed=(
            summary["test_seed_start"]
            if args.episode_seed is None
            else args.episode_seed
        ),
        module_count=config["module_count"],
        control_cost_weight=config["control_cost_weight"],
        output=output,
        fps=args.fps,
        seconds=args.seconds,
    )


if __name__ == "__main__":
    main()
