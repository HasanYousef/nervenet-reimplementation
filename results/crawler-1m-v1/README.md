# crawler-1m-v1 results

This directory is a lightweight, version-controlled export of the comparison results. Model checkpoints and training logs are intentionally excluded.

## Protocol

- Robot: 3-module crawler.
- Training seeds: 0, 1, 2, 3, 4.
- Validation: 10 fixed episodes every 50,000 timesteps.
- Final evaluation: 50 held-out episodes per training seed and budget.
- Graph policy: hidden size 64, 4 message-passing rounds.
- Matched MLP actor hidden size: 151.

## Held-out results

Values are mean ± standard deviation across training seeds. Each training budget uses a fresh held-out episode seed range, so cross-budget differences are not paired measurements.

| Budget | Test seeds | Policy | Reward | Forward distance | Absolute lateral distance | Action saturation | Joint speed |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1,000,000 | 20,000–20,049 | Matched | 760.84 ± 91.13 | 15.58 ± 1.83 m | 2.60 ± 0.15 m | 62.2% ± 2.4% | 8.32 ± 0.29 rad/s |
| 1,000,000 | 20,000–20,049 | Graph | 1160.12 ± 88.65 | 23.50 ± 1.79 m | 2.43 ± 1.28 m | 46.0% ± 7.2% | 8.23 ± 0.56 rad/s |
| 1,300,000 | 30,000–30,049 | Matched | 783.15 ± 77.35 | 16.04 ± 1.57 m | 3.09 ± 1.12 m | 65.1% ± 6.2% | 8.55 ± 0.62 rad/s |
| 1,300,000 | 30,000–30,049 | Graph | 1213.79 ± 78.31 | 24.59 ± 1.58 m | 1.96 ± 1.17 m | 49.4% ± 4.6% | 8.54 ± 0.21 rad/s |

## Latest-budget observation

At 1,300,000 timesteps, the Graph policy travelled 53.3% farther than the capacity-matched MLP on their held-out evaluations. This is a descriptive result for this crawler and protocol, not a general significance claim.

## Recorded training time

These are wall-clock session times from this machine, not hardware-independent benchmarks. The median is used because pauses or system sleep can create large outliers.

| Policy | Median per seed | Range |
|---|---:|---:|
| Matched | 8.6 min | 8.6–8.9 min |
| Graph | 83.3 min | 81.8–381.0 min |

## Included artifacts

Each `reports/target_*` directory contains the Markdown report, aggregate and per-run CSV data, complete JSON results including per-episode evaluations, and learning-curve figures.

`training_sessions.csv` contains the compact timing, timestep, interruption, and Git provenance record for every training session.

The cumulative validation curves use the same fixed validation episodes throughout training and are the best source for judging learning progress across budgets.

The two GIFs in `media/` are offscreen MuJoCo renders of the validation-selected
seed-0 checkpoints from the 1,300,000-step report. Both use held-out episode
seed 30,000, deterministic actions, a 10-second simulation, and the same camera
and lighting. The recording extends the visible grid beyond the model's normal
edge; this changes no physics or reported measurement. They illustrate one
episode each; the tables above summarize 50 test episodes for each of five
training seeds. Regenerating the GIFs requires
the corresponding local checkpoints in the ignored `experiments/` directory.
