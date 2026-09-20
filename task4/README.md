# Task 4 — Open-Set Recognition

The primary deliverable is `task4.ipynb`. It implements the fixed CIFAR-10/CIFAR-100 open-set protocol in `plan_task4.md`, while this package holds the reusable model, data, score, training, extraction, and evaluation code used by the notebook.

## Reproduction

Run the notebook from the repository root. Its sections must be run in order: CIFAR-100 unknowns are deliberately not constructed until all three checkpoints and all score definitions are fixed. The default run trains Vanilla and GCSC for 100 epochs and fine-tunes PROSER for 50 epochs, so a CUDA GPU is strongly recommended.

Command-line equivalents for training are:

```bash
python -m task4.train vanilla
python -m task4.train gcsc
python -m task4.train proser
```

Dependencies: Python 3.10+, PyTorch, torchvision, NumPy, pandas, scikit-learn, matplotlib, PyYAML, tqdm, and Jupyter. Every loader uses `num_workers=0`; training and extraction use the current `torch.amp` API and channels-last tensors. Checkpoints are selected only by CIFAR-10 validation accuracy.

Generated artifacts belong in `results/checkpoints/` and `results/cache/`; tables, the required figure, and final-only failure cases are written directly under `results/`. No placeholder result numbers are committed.

PROSER follows Zhou, Ye, and Zhan, *Learning Placeholders for Open-Set Recognition*, CVPR 2021. Five dummy classifiers are reduced to the strongest dummy response as specified by the paper, and the placeholder detector is its reference Delta-P score at temperature 1024.
