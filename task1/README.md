# Task 1 — Inductive Biases and Feature Representations

## Contents
- `task1.ipynb` — main deliverable notebook (run top-to-bottom).
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
   pandas matplotlib umap-learn` (umap-learn only needed if you switch the
   projection method from the default t-SNE to UMAP).

## What was and wasn't verified here
The authoring sandbox had no GPU and no egress to the hosts above (only
package registries), so the notebook could not be executed end-to-end here.
Instead, every non-trivial piece of indexing/math logic was extracted and
smoke-tested standalone with plain NumPy:
- the color-swap derangement (no fixed points, valid permutation),
- per-image deterministic patch permutations (reproducible, non-identity),
- the class-balanced eval-subset sampler (correct total/per-class counts),
- the LAB Reinhard-style color-transfer math (verified it shifts a
  synthetic class's color statistics toward a donor class's),
- the SSIM-based cue-conflict rejection rule (verified it accepts an
  identical image, and rejects both a degenerate flat image and a heavily
  corrupted one),
- the reflection-pad + shifted-crop translation directions (verified with a
  single-marker-pixel test that "up"/"down"/"left"/"right" move the visible
  content the way their names claim — an earlier version had this
  backwards and was caught and fixed by this test),
- the patch-shuffle reassembly indexing (verified identity-permutation
  round-trips exactly, and a simple 2-patch swap lands where expected).

What was **not** executed: any actual model forward pass, linear-head
training, AdaIN stylization, or STL-10 I/O, since none of the required
weights/data/GPU were reachable. Treat this as carefully logic-checked but
unexecuted code — run it in a proper environment and inspect the outputs
under `results/` before treating any numbers as final.
