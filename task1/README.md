# Task 1 — Inductive Biases and Feature Representations

## Contents
- `task1.ipynb` — main deliverable notebook (run top-to-bottom).
- `postprocess_cached_results.py` — cache-only figure/manifest regeneration;
  this adds the cue-conflict representation panels without editing the notebook.
- `analysis/cue_conflict_calibration.py` — stable cue IDs, stratified manual
  review sampling, resumable ipywidgets review, and threshold-sweep artifacts.
- `utils/` — reusable modules imported by the notebook:
  - `config.py` — seeds (SEED=6304), device/precision, paths, `make_rng`.
  - `backbones.py` — frozen ResNet-50 / ViT-B/16 / OpenCLIP ViT-B/32 wrappers.
  - `transforms.py` — grayscale, LAB class-swap color transfer, translation
    (reflection-pad + shifted crop), patch-shuffle.
  - `adain.py` — AdaIN style-transfer model (VGG encoder + decoder).
  - `data.py` — STL-10 loading, stratified split, eval-subset selection,
    feature caching, linear-head training.
  - `metrics.py` — accuracy/F1/confidence, consistency, representation
    stability (I_T), cue-conflict rejection rule, t-SNE/UMAP projection.
- `results/` — all output artifacts land here (JSON configs, CSV metric
  tables, `cache/` for extracted features, `figures/` for plots).
- `models/` — put a pretrained AdaIN checkpoint here before running Step 3
  (see below).

## Before running
This notebook needs a real environment, which the sandbox it was authored
in does not have:
1. **A CUDA GPU.** The code path still runs on CPU but will be slow and the
   mixed-precision/`channels_last` optimizations become no-ops.
2. **Network access** to download pretrained weights: torchvision's
   ResNet-50/ViT-B-16 ImageNet weights, OpenCLIP's `ViT-B-32` `openai`
   weights (via `open_clip_torch`, from Hugging Face / OpenAI's CDN), and
   the STL-10 tarball (`ai.stanford.edu`).
3. **An AdaIN checkpoint** for Step 3 (cue-conflict generation): download a
   public pretrained encoder/decoder pair (e.g. the widely used
   `naoto0804/pytorch-AdaIN` repo's `vgg_normalised.pth` + `decoder.pth`)
   and place them at `models/vgg_normalised.pth` / `models/decoder.pth`, or
   edit the `VGG_WEIGHTS_PATH` / `DECODER_WEIGHTS_PATH` variables in the
   Step 3 cell. `AdaINStyleTransfer` deliberately raises an error rather
   than stylizing with an untrained decoder.
4. `pip install torch torchvision open_clip_torch scikit-image scikit-learn
   pandas matplotlib ipywidgets umap-learn` (umap-learn only needed if you switch the
   projection method from the default t-SNE to UMAP).

## Recorded execution status

The main notebook has been executed and the repository contains its CSV/JSON
results, feature caches, and figures. The cache-only supplement can be run with:

```bash
venv/bin/python task1/postprocess_cached_results.py
```

It performs no backbone inference, AdaIN inference, or training. It regenerates
the plots from the accepted-row manifest and stable per-image feature caches.

## Cue-conflict threshold calibration

The notebook generates the SSIM histogram before manual review, samples about
10% from each pair/direction bucket with stable content-derived IDs, and writes
each pass/fail click immediately to `results/cue_conflict_manual_ratings.jsonl`.
Restarting the review cell skips IDs already present in that file. After all
ratings are complete, the threshold sweep writes
`results/cue_conflict_threshold_sweep.csv` and
`results/figures/cue_conflict_threshold_sweep.png`; fail is the positive class,
and tied kappa values select the stricter threshold. Record the histogram-informed
sweep range before rating, then update `SSIM_MIN_THRESHOLD`, raise the candidate
buffer to 30 per bucket, and rerun the notebook from a fresh kernel.

The completed 20-image review retained the preregistered 0.20--0.50 sweep:
the original 220-image distribution had minimum 0.211, 5th percentile 0.269,
median 0.417, and 95th percentile 0.577, so the range covered the lower tail
through above the median where rejection decisions were informative. The
selected threshold is **0.30** (75.0% agreement, Cohen's $\kappa=0.419$,
fail precision 1.000, fail recall 0.375, fail F1 0.545); the stricter-threshold
tie-break was not invoked. With the final 30-per-bucket buffer, regeneration
accepted 263/300 images and rejected 37; individual buckets retain 21--30
accepted images, so the total and balance requirements remain satisfied.

## CLIP-linear confidence caveat

The raw mean maximum softmax confidence for `clip_vit_b_32_linear` is not
directly comparable to the other decision rules. Its inputs are L2-normalized
CLIP features, and the weight-decay-regularized linear head has no logit scale
analogous to CLIP's learned `logit_scale.exp()` used by zero-shot inference.
Consequently, small logit magnitudes can yield a flatter softmax despite high
accuracy. The required raw confidence is retained without temperature scaling
and should not be interpreted as a calibrated cross-model ranking.
