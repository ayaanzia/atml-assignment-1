# ATML Programming Assignment 1

This repository contains the four experimental pipelines and the machine-readable
results used for the assignment report. Raw datasets, feature caches, and model
checkpoints are intentionally excluded from Git; CSV/JSON results and report
figures are retained.

## Environment

The recorded environment is in `requirements.txt`. The repository-local `venv`
can be used directly when present. Commands below assume they are run from the
repository root.

## Full experiment notebooks

- `task1/task1.ipynb`: STL-10 inductive-bias and representation experiments.
- `task2/task2.ipynb`: PACS unsupervised domain adaptation.
- `task3/task3.ipynb`: PACS domain generalization.
- `task4/task4.ipynb`: CIFAR-10/CIFAR-100 open-set recognition.

These notebooks include training and can be computationally expensive. Each task
directory contains data-layout and execution details.

## Cached post-processing (no retraining)

```bash
venv/bin/python task1/postprocess_cached_results.py
venv/bin/python task2/reproduce_result_figures.py
venv/bin/python task3/reproduce_result_figures.py
venv/bin/python task4/postprocess_unknown_classes.py
venv/bin/python validate_saved_artifacts.py
```

The Task 1 command rebuilds all clean/transformed t-SNE panels with legends,
including cue conflict, and exports the deterministic cue-conflict manifest.
The Task 4 command exports per-unknown-class acceptance and absorption analyses.
The Task 2/3 commands regenerate every CSV-backed report figure. The final command
checks saved checkpoint finiteness, hashes, and cross-file metric consistency; it
does not establish training-run reproducibility.

Two-seed retraining validation for the collapsed configurations is isolated in
`validation/collapse_multiseed_validation.ipynb`. It requires CUDA; see
`validation/README.md` for the resumable command-line equivalent.

