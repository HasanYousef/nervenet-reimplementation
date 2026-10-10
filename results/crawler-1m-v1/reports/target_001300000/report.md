# crawler-1m-v1

## Experiment

Matched MLP and graph PPO policies were trained for 1,300,000 timesteps on a 3-module crawler. Results use 5 independent training seeds.

Checkpoints were validated every 50,000 timesteps on 10 fixed episodes. The best checkpoint per run was selected by validation reward, then evaluated on 50 held-out episodes starting at seed 30,000.

## Learning curves

![Validation reward](figures/reward.png)

![Forward distance](figures/distance.png)

## Held-out results

Values are mean ± standard deviation across training seeds.

| Policy | Reward | Forward distance | Absolute lateral distance | Action saturation | Joint speed |
|---|---:|---:|---:|---:|---:|
| Matched | 783.15 ± 77.35 | 16.04 ± 1.57 m | 3.09 ± 1.12 m | 65.1% ± 6.2% | 8.55 ± 0.62 rad/s |
| Graph | 1213.79 ± 78.31 | 24.59 ± 1.58 m | 1.96 ± 1.17 m | 49.4% ± 4.6% | 8.54 ± 0.21 rad/s |

## Selected checkpoints

| Policy | Training seed | Timestep | Validation reward |
|---|---:|---:|---:|
| Matched | 0 | 550,912 | 736.76 |
| Matched | 1 | 1,050,624 | 886.30 |
| Matched | 2 | 1,150,976 | 820.41 |
| Matched | 3 | 651,264 | 706.85 |
| Matched | 4 | 901,120 | 738.34 |
| Graph | 0 | 1,200,128 | 1285.91 |
| Graph | 1 | 950,272 | 1258.44 |
| Graph | 2 | 1,150,976 | 1299.37 |
| Graph | 3 | 1,251,328 | 1112.24 |
| Graph | 4 | 700,416 | 1166.56 |

## Interpretation

This report records descriptive results for this controlled comparison. Conclusions should be limited to the tested crawler, reward, optimizer configuration, and training budget.

Raw per-run results are in `results.csv`; aggregated values and the complete configuration are in `summary.json`.
