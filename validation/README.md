# Collapse validation

`collapse_multiseed_validation.ipynb` retrains Task 2 DAN/DANN and Task 3
DAN-DG at their original main settings with seeds 6305 and 6306. The original
seed-6304 source split remains fixed so the experiment measures training-seed
sensitivity rather than split sensitivity.

Open the notebook from the repository root and run it top to bottom on a CUDA
machine. The equivalent command is:

```bash
venv/bin/python validation/collapse_multiseed.py
```

To verify paths, split integrity, and dataset sizes without training:

```bash
venv/bin/python validation/collapse_multiseed.py --check-only
```

Completed checkpoints and curves are reused, allowing an interrupted run to
resume. Pass `--force` only when the existing validation checkpoints should be
replaced. CPU execution is rejected by default because six ResNet-18 runs would
be very slow; `--allow-cpu` overrides that guard deliberately.

Expected outputs under `validation/results/` are:

- `additional_seed_results.csv` for seeds 6305 and 6306;
- `combined_seed_results.csv` including the original seed 6304;
- `multiseed_summary.csv` with means, standard deviations, collapse counts, and
  extreme-loss counts;
- `multiseed_accuracy.png` comparing source and Sketch accuracy by seed;
- one training-curve CSV and checkpoint per method/seed; and
- `protocol.json`, which records the thresholds and frozen experimental design.

Checkpoints are ignored by Git; tables, curves, figures, and protocol records are
intended to be committed.
