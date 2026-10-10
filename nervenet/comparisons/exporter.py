from csv import DictWriter
from datetime import datetime
import json
from pathlib import Path
import shutil
from statistics import median
from typing import Any

from nervenet.comparisons.manifest import ComparisonStore
from nervenet.experiments import ExperimentStore


def export_comparison_results(
    comparisons_root: Path,
    name: str,
    output_root: Path = Path("results"),
) -> Path:
    store = ComparisonStore(comparisons_root, name)
    if not store.exists():
        raise FileNotFoundError(f"comparison does not exist: {name}")

    archives = sorted(
        path
        for path in store.reports_directory.glob("target_*")
        if path.is_dir()
    )
    if not archives:
        raise ValueError("generate a comparison report before exporting")

    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(store.manifest_path, destination / "comparison.json")

    exported_reports = destination / "reports"
    exported_reports.mkdir(exist_ok=True)
    summaries = []
    for archive in archives:
        target = exported_reports / archive.name
        shutil.copytree(archive, target, dirs_exist_ok=True)
        for csv_path in target.glob("*.csv"):
            csv_path.write_text(csv_path.read_text())
        summary = json.loads((archive / "summary.json").read_text())
        summary.setdefault(
            "target_timesteps",
            int(archive.name.removeprefix("target_")),
        )
        summary.setdefault(
            "test_seed_start",
            summary["config"]["test_seed_start"],
        )
        summaries.append(summary)

    training_sessions = _collect_training_sessions(store)
    with (destination / "training_sessions.csv").open(
        "w",
        newline="",
    ) as file:
        writer = DictWriter(
            file,
            fieldnames=list(training_sessions[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(training_sessions)

    (destination / "README.md").write_text(
        _render_results_index(name, summaries, training_sessions)
    )
    return destination


def _collect_training_sessions(
    store: ComparisonStore,
) -> list[dict[str, Any]]:
    rows = []
    config = store.config()
    for policy_type in config.policy_types:
        for seed in config.training_seeds:
            run_store = ExperimentStore(
                store.runs_directory,
                store.run_name(policy_type, seed),
            )
            manifest = run_store.load()
            for session in manifest["sessions"]:
                started_at = session["started_at"]
                ended_at = session.get("ended_at")
                elapsed_seconds = None
                if ended_at is not None:
                    elapsed_seconds = (
                        datetime.fromisoformat(ended_at)
                        - datetime.fromisoformat(started_at)
                    ).total_seconds()
                completed_timesteps = (
                    session["ending_timesteps"]
                    - session["starting_timesteps"]
                )
                steps_per_second = (
                    completed_timesteps / elapsed_seconds
                    if elapsed_seconds
                    else None
                )
                software = session.get("software", {})
                rows.append(
                    {
                        "policy_type": policy_type,
                        "training_seed": seed,
                        "session_index": session["index"],
                        "status": session["status"],
                        "starting_timesteps": session["starting_timesteps"],
                        "ending_timesteps": session["ending_timesteps"],
                        "completed_timesteps": completed_timesteps,
                        "elapsed_seconds": (
                            f"{elapsed_seconds:.3f}"
                            if elapsed_seconds is not None
                            else ""
                        ),
                        "steps_per_second": (
                            f"{steps_per_second:.3f}"
                            if steps_per_second is not None
                            else ""
                        ),
                        "started_at": started_at,
                        "ended_at": ended_at or "",
                        "git_commit": software.get("git_commit", ""),
                        "git_dirty": software.get("git_dirty", ""),
                    }
                )
    return rows


def _render_results_index(
    name: str,
    summaries: list[dict[str, Any]],
    training_sessions: list[dict[str, Any]],
) -> str:
    latest = max(
        summaries,
        key=lambda value: value["target_timesteps"],
    )
    config = latest["config"]
    lines = [
        f"# {name} results",
        "",
        (
            "This directory is a lightweight, version-controlled export "
            "of the comparison results. Model checkpoints and training "
            "logs are intentionally excluded."
        ),
        "",
        "## Protocol",
        "",
        f"- Robot: {config['module_count']}-module crawler.",
        (
            f"- Training seeds: "
            f"{', '.join(str(seed) for seed in config['training_seeds'])}."
        ),
        (
            f"- Validation: {config['validation_episode_count']} fixed "
            f"episodes every {config['checkpoint_interval']:,} timesteps."
        ),
        (
            f"- Final evaluation: {config['test_episode_count']} held-out "
            "episodes per training seed and budget."
        ),
        (
            f"- Graph policy: hidden size {config['graph_hidden_size']}, "
            f"{config['message_passing_steps']} message-passing rounds."
        ),
        (
            f"- Matched MLP actor hidden size: "
            f"{config['matched_actor_hidden_size']}."
        ),
        "",
        "## Held-out results",
        "",
        (
            "Values are mean ± standard deviation across training seeds. "
            "Each training budget uses a fresh held-out episode seed range, "
            "so cross-budget differences are not paired measurements."
        ),
        "",
        "| Budget | Test seeds | Policy | Reward | Forward distance "
        "| Absolute lateral distance | Action saturation | Joint speed |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    for summary in sorted(
        summaries,
        key=lambda value: value["target_timesteps"],
    ):
        budget = summary["target_timesteps"]
        test_seed_start = summary["test_seed_start"]
        test_count = summary["config"]["test_episode_count"]
        test_seed_end = test_seed_start + test_count - 1
        for policy in summary["config"]["policy_types"]:
            metrics = summary["aggregate"][policy]
            reward = metrics["mean_total_reward"]
            distance = metrics["mean_forward_distance"]
            lateral = metrics["mean_absolute_lateral_distance"]
            saturation = metrics["action_saturation_fraction"]
            speed = metrics["mean_actuated_joint_speed"]
            lines.append(
                f"| {budget:,} | {test_seed_start:,}–{test_seed_end:,} "
                f"| {policy.title()} "
                f"| {reward['mean']:.2f} ± "
                f"{reward['standard_deviation']:.2f} "
                f"| {distance['mean']:.2f} ± "
                f"{distance['standard_deviation']:.2f} m "
                f"| {lateral['mean']:.2f} ± "
                f"{lateral['standard_deviation']:.2f} m "
                f"| {saturation['mean']:.1%} ± "
                f"{saturation['standard_deviation']:.1%} "
                f"| {speed['mean']:.2f} ± "
                f"{speed['standard_deviation']:.2f} rad/s |"
            )

    latest_metrics = latest["aggregate"]
    if "matched" in latest_metrics and "graph" in latest_metrics:
        matched_distance = latest_metrics["matched"][
            "mean_forward_distance"
        ]["mean"]
        graph_distance = latest_metrics["graph"][
            "mean_forward_distance"
        ]["mean"]
        improvement = (graph_distance / matched_distance - 1.0) * 100.0
        lines.extend(
            [
                "",
                "## Latest-budget observation",
                "",
                (
                    f"At {latest['target_timesteps']:,} timesteps, the "
                    f"Graph policy travelled {improvement:.1f}% farther "
                    "than the capacity-matched MLP on their held-out "
                    "evaluations. This is a descriptive result for this "
                    "crawler and protocol, not a general significance claim."
                ),
            ]
        )

    run_times: dict[tuple[str, int], float] = {}
    for session in training_sessions:
        key = (session["policy_type"], session["training_seed"])
        elapsed = session["elapsed_seconds"]
        if elapsed:
            run_times[key] = run_times.get(key, 0.0) + float(elapsed)
    lines.extend(
        [
            "",
            "## Recorded training time",
            "",
            (
                "These are wall-clock session times from this machine, not "
                "hardware-independent benchmarks. The median is used because "
                "pauses or system sleep can create large outliers."
            ),
            "",
            "| Policy | Median per seed | Range |",
            "|---|---:|---:|",
        ]
    )
    for policy in config["policy_types"]:
        durations = sorted(
            seconds
            for (run_policy, _), seconds in run_times.items()
            if run_policy == policy
        )
        if not durations:
            continue
        median_seconds = median(durations)
        lines.append(
            f"| {policy.title()} | {median_seconds / 60:.1f} min "
            f"| {durations[0] / 60:.1f}–{durations[-1] / 60:.1f} min |"
        )

    lines.extend(
        [
            "",
            "## Included artifacts",
            "",
            (
                "Each `reports/target_*` directory contains the Markdown "
                "report, aggregate and per-run CSV data, complete JSON results "
                "including per-episode evaluations, and learning-curve figures."
            ),
            "",
            (
                "`training_sessions.csv` contains the compact timing, timestep, "
                "interruption, and Git provenance record for every training "
                "session."
            ),
            "",
            (
                "The cumulative validation curves use the same fixed validation "
                "episodes throughout training and are the best source for "
                "judging learning progress across budgets."
            ),
            "",
        ]
    )
    return "\n".join(lines)
