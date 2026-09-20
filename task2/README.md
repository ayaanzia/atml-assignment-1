# Task 2 — PACS unsupervised domain adaptation

Open `task2.ipynb` with the Python environment at `venv/` and run it from top
to bottom. The notebook implements Source-only ERM, DAN, DANN, CDAN, common
evaluation, and the controlled DAN alignment-weight study.

## Data layout

Extract PACS locally (raw data is ignored by Git), then either place it under
`task2/data/` or set `PACS_ROOT`. The loader accepts the common layouts below:

```text
PACS_ROOT/
  photo/
  art_painting/       # `art painting` and `art-painting` are also accepted
  cartoon/
  sketch/
```

The four domain directories may also be nested below `PACS/`, `pacs/`, or
`kfold/`. The official archive's double-nested `pacs_data/pacs_data/` layout
is detected directly. The archive also includes `dct2_images/dct2_images/`;
that is a derived image set and is deliberately not selected by default.
Each domain must use an ImageFolder-style class-directory layout.

Install the runtime packages if they are not already available:

```text
torch torchvision scikit-learn pandas matplotlib pillow tqdm jupyter
```

All generated artifacts are written to `results/`. See `results/README.md`
for the stable Source-only checkpoint path shared with Task 3.

CUDA training prefers bfloat16 on supported GPUs because adversarial DANN
training can overflow fp16's limited exponent range. On older GPUs the notebook
falls back to fp16 plus gradient scaling, and explicit finite-value checks stop
training before an invalid checkpoint can be selected.
