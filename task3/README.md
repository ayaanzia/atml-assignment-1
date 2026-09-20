# Task 3 — PACS domain generalization

Open task3.ipynb with the repository's venv Python environment and run it
from top to bottom. The notebook loads Task 2's unchanged source-only
checkpoint, trains DAN-DG and SAM, runs the fixed DAN-DG alignment-weight
study, and writes all requested artifacts below results/.

The implementation expects Task 2's existing split manifest and checkpoints
under ../task2/results/. PACS is discovered under ../task2/data/ by default;
set PACS_ROOT to use another extracted PACS location.

The target domain is isolated by construction: source setup resolves only
Photo, Art Painting, and Cartoon. The held-out directory is first resolved in
the notebook's explicitly marked final-evaluation section, after models and
the controlled-study grid are frozen.

Shared code factored from Task 2 is in utils/pacs_shared.py. Every DataLoader
uses zero workers, ResNet inputs use channels-last layout, neural-network
forward passes use the current torch.amp API, BatchNorm running statistics
stay frozen during training, and figures have printed descriptions rather
than embedded titles.

Expected runtime packages are:

    torch torchvision scikit-learn pandas matplotlib pillow tqdm jupyter

The full notebook is computationally substantial: it trains the main DAN-DG
and SAM models plus two additional DAN-DG settings, each with source-only
early stopping.
