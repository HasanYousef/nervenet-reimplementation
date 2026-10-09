# Controlled policy comparisons

The comparison layer runs repeated experiments and produces a small,
human-readable research report. It keeps orchestration and reporting separate
from the environment and policy implementations.

## Protocol

The default comparison is intentionally fixed:

- graph PPO versus the capacity-matched MLP PPO baseline;
- five independent training seeds (`0..4`);
- 1,000,000 environment timesteps per run;
- checkpoints every 50,000 requested timesteps;
- ten fixed validation episodes per checkpoint, starting at seed 10,000;
- selection of one checkpoint per run by mean validation reward;
- fifty held-out test episodes, starting at seed 20,000.

Validation and test seeds never overlap. The test episodes are not used to
choose a checkpoint. This avoids reporting the same episodes that made a
checkpoint look best.

## Create and run

Create the immutable comparison configuration:

```bash
python -m nervenet.cli.comparison create --name crawler-1m-v1
```

Run all ten experiments sequentially:

```bash
python -m nervenet.cli.comparison run --name crawler-1m-v1
```

The command is safe to interrupt after a checkpoint. Running it again skips
finished work and continues incomplete runs from their latest checkpoints.
For deliberate one-run-at-a-time execution, select a policy and seed:

```bash
python -m nervenet.cli.comparison run \
  --name crawler-1m-v1 \
  --policy-type graph \
  --seed 2
```

Inspect progress without starting training:

```bash
python -m nervenet.cli.comparison status --name crawler-1m-v1
```

## Generate the report

After all ten runs finish:

```bash
python -m nervenet.cli.comparison report --name crawler-1m-v1
```

The report command loads the validation-selected checkpoint for every run,
evaluates it on the held-out test seeds, aggregates results across training
seeds, and writes:

```text
experiments/crawler-1m-v1/
├── comparison.json
├── runs/
│   ├── matched-seed0/
│   ├── graph-seed0/
│   └── ...
├── results.csv
├── learning_curves.csv
├── summary.json
├── report.md
└── figures/
    ├── reward.png
    └── distance.png
```

`report.md` contains the protocol, learning curves, held-out summary table,
selected checkpoints, and a deliberately limited interpretation. `results.csv`
has one row per training run. `summary.json` preserves the complete comparison
configuration, aggregate statistics, validation evaluations, and every
held-out episode result.

The shaded regions in the learning curves show one standard deviation across
training seeds. They describe variation between runs; they are not confidence
intervals or formal significance tests.
