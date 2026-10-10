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

## Extend a completed comparison

Only extend a comparison after generating its report. For example, to continue
every 1,000,000-timestep run for another 300,000 timesteps:

```bash
python -m nervenet.cli.comparison extend \
  --name crawler-1m-v1 \
  --timesteps 300000
```

The extension is recorded before training begins, so the command is safe to
interrupt. Resume it with the normal `run` command. Each run continues from its
latest checkpoint; it does not restart and it does not continue from the
validation-selected checkpoint.

Extending preserves the previous report in a target-specific directory,
advances the requested training target, and assigns a fresh held-out test seed
range. This prevents the already-inspected 1,000,000-timestep test episodes
from being reused as unseen evidence for the extended policies. Generate a new
report after all extended runs finish.

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
├── reports/
│   ├── target_001000000/
│   └── target_001300000/
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
held-out episode result. The files at the comparison root are always the latest
report, while `reports/target_*/` keeps immutable budget-specific snapshots.

`comparison.json` records the current target, current held-out seed range, and
the full budget history. The original comparison configuration is retained as
the initial protocol rather than silently rewritten.

## Export lightweight public results

Reports are stored inside the ignored `experiments/` directory alongside large
checkpoints and logs. Export only the small, useful research artifacts into a
version-controlled package with:

```bash
python -m nervenet.cli.comparison export --name crawler-1m-v1
```

The default destination is `results/<name>/`. It contains every archived
budget report, CSV dataset, complete JSON evaluation, figure, comparison
manifest, compact per-session timing/provenance CSV, and a generated overview.
It deliberately excludes model weights and verbose training logs.

The shaded regions in the learning curves show one standard deviation across
training seeds. They describe variation between runs; they are not confidence
intervals or formal significance tests.
