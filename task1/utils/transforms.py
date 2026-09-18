"""
Shared intervention pipeline. All interventions operate on [3, 224, 224] (or
batched [N, 3, 224, 224]) float tensors in [0, 1] RGB — i.e. the common,
backbone-agnostic representation described in Sec. 4 of the spec. Backbone
normalization is applied later, inside each backbone wrapper.

Interventions implemented here:
  - to_grayscale3            (Sec. 6a)
  - class_swap_color_transfer (Sec. 6b, LAB Reinhard-style transfer)
  - translate_reflect        (Sec. 8)
  - patch_shuffle            (Sec. 9)

Design note (what each transform changes vs. preserves — see notebook
markdown for the graded discussion, this is the implementation only):
  - Grayscale removes ALL chrominance (a,b in LAB / hue+saturation in
    RGB) while keeping luminance and hence exact geometry/edges/texture.
  - Class-swapped LAB transfer is a per-pixel *affine* recoloring
    (channel-wise scale+shift in LAB). It changes only the global color
    statistics (each channel's mean/std) toward a donor class's typical
    palette; it cannot alter shape, texture, or geometry because the
    transform is spatially uniform (same affine map applied identically at
    every pixel).
  - Translation (reflection-pad + shifted crop) preserves object identity,
    shape, texture, and color exactly; it only changes retinal position.
  - Patch-shuffle preserves every local patch's texture/color exactly but
    destroys global shape/geometry by permuting patch locations.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage import color as skcolor

from .config import IMG_SIZE, STL10_CLASSES, make_rng


# ---------------------------------------------------------------------------
# 6a. Grayscale
# ---------------------------------------------------------------------------
_RGB_TO_GRAY = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def to_grayscale3(images_0_1: torch.Tensor) -> torch.Tensor:
    """Convert to luminance, replicate to 3 channels. Preserves geometry."""
    w = _RGB_TO_GRAY.to(images_0_1.device, images_0_1.dtype)
    gray = (images_0_1 * w).sum(dim=1, keepdim=True)  # [N,1,H,W]
    return gray.repeat(1, 3, 1, 1).clamp(0, 1)


# ---------------------------------------------------------------------------
# 6b. Class-swapped LAB color-statistics transfer
# ---------------------------------------------------------------------------
def compute_class_lab_stats(
    images_by_class: Dict[int, torch.Tensor],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    images_by_class: {class_idx: Tensor[N_c, 3, H, W] in [0,1]} drawn from the
    STL-10 *training* split (the 80% partition used for linear heads — never
    the eval subset, per the guardrails in Sec. 13).

    Returns {class_idx: (mean_lab[3], std_lab[3])}, each averaged over all
    pixels of all training images of that class.
    """
    stats = {}
    for c, imgs in images_by_class.items():
        imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()  # [N,H,W,3] in [0,1]
        lab = skcolor.rgb2lab(imgs_np)  # per-image conversion, vectorized over batch dim
        lab_pixels = lab.reshape(-1, 3)
        mean = lab_pixels.mean(axis=0)
        std = lab_pixels.std(axis=0) + 1e-6
        stats[c] = (mean, std)
    return stats


def make_class_derangement(num_classes: int, seed_name: str = "color_swap_pairing") -> Dict[int, int]:
    """
    Fixed, seeded derangement (permutation with no fixed points) of class
    indices -> donor class indices. Verified to have zero fixed points.
    """
    rng = make_rng(seed_name)
    idx = np.arange(num_classes)
    while True:
        perm = rng.permutation(idx)
        if np.all(perm != idx):
            break
    return {int(i): int(perm[i]) for i in idx}


def class_swap_color_transfer(
    images_0_1: torch.Tensor,
    class_idx: int,
    mapping: Dict[int, int],
    class_lab_stats: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor:
    """
    Reinhard-style linear LAB color-statistics transfer:
        pixel' = (pixel - mean_c) / std_c * std_donor + mean_donor
    applied per-channel in LAB space, then clipped and converted back to RGB.

    images_0_1: Tensor[N, 3, H, W] in [0,1], all belonging to `class_idx`.
    """
    donor = mapping[class_idx]
    mean_c, std_c = class_lab_stats[class_idx]
    mean_d, std_d = class_lab_stats[donor]

    imgs_np = images_0_1.permute(0, 2, 3, 1).cpu().numpy()  # [N,H,W,3]
    lab = skcolor.rgb2lab(imgs_np)  # [N,H,W,3]

    lab_t = (lab - mean_c.reshape(1, 1, 1, 3)) / std_c.reshape(1, 1, 1, 3)
    lab_t = lab_t * std_d.reshape(1, 1, 1, 3) + mean_d.reshape(1, 1, 1, 3)

    # Clip to valid LAB ranges before converting back.
    lab_t[..., 0] = np.clip(lab_t[..., 0], 0, 100)
    lab_t[..., 1] = np.clip(lab_t[..., 1], -128, 127)
    lab_t[..., 2] = np.clip(lab_t[..., 2], -128, 127)

    rgb_t = skcolor.lab2rgb(lab_t)
    rgb_t = np.clip(rgb_t, 0, 1).astype(np.float32)
    out = torch.from_numpy(rgb_t).permute(0, 3, 1, 2).contiguous()
    return out.to(images_0_1.device)


# ---------------------------------------------------------------------------
# Step 4: Translation via reflection-pad + shifted crop
# ---------------------------------------------------------------------------
DIRECTIONS = ("up", "down", "left", "right")


def translate_reflect(images_0_1: torch.Tensor, delta: int, direction: str) -> torch.Tensor:
    """
    Reflection-pad by `delta` on all sides, then crop a 224x224 window shifted
    by `delta` pixels in `direction` (relative to the centered/original crop),
    so the object appears translated by exactly `delta` px while the output
    stays 224x224 and every pixel is real (reflected) image content — no
    black borders.
    """
    if delta == 0:
        return images_0_1.clone()
    assert direction in DIRECTIONS
    n, c, h, w = images_0_1.shape
    assert h == IMG_SIZE and w == IMG_SIZE
    padded = F.pad(images_0_1, (delta, delta, delta, delta), mode="reflect")
    # Padded size = (H + 2*delta, W + 2*delta). The centered crop (no shift)
    # would start at (delta, delta). Shifting the crop window by `delta` in
    # the given direction moves the *content* by `delta` px in that
    # direction within the output frame.
    # Crop top-left coordinates within the padded image for each direction.
    # output(i,j) = padded(top+i, left+j) = original(top+i-delta, left+j-delta).
    # So a LARGER top pulls content from further down in the original image
    # up into the top of the output frame -> the visible object appears to
    # have moved UP; conversely top=0 makes the object appear to move DOWN.
    # (Verified empirically with a single-marker-pixel test.)
    if direction == "up":
        top, left = 2 * delta, delta
    elif direction == "down":
        top, left = 0, delta
    elif direction == "left":
        top, left = delta, 2 * delta
    elif direction == "right":
        top, left = delta, 0
    else:
        raise ValueError(direction)
    out = padded[:, :, top:top + h, left:left + w]
    return out.contiguous()


# ---------------------------------------------------------------------------
# Step 5: Patch shuffle
# ---------------------------------------------------------------------------
def make_patch_permutation(image_id: str, grid: int = 4) -> np.ndarray:
    """
    Deterministic, non-identity permutation of the grid*grid patches for a
    given image_id (so the SAME permutation is reused across all backbones
    for that image, per the spec).
    """
    rng = make_rng(f"patch_permutation:{image_id}")
    n = grid * grid
    idx = np.arange(n)
    while True:
        perm = rng.permutation(idx)
        if not np.array_equal(perm, idx):
            break
    return perm


def patch_shuffle(image_0_1: torch.Tensor, perm: np.ndarray, grid: int = 4) -> torch.Tensor:
    """
    image_0_1: Tensor[3, H, W]. Splits into grid x grid patches (row-major
    order) and rearranges them according to `perm` (perm[i] = source patch
    index placed at destination position i).
    """
    c, h, w = image_0_1.shape
    ph, pw = h // grid, w // grid
    patches = (
        image_0_1
        .unfold(1, ph, ph)  # [C, grid, W, ph]
        .unfold(2, pw, pw)  # [C, grid, grid, ph, pw]
        .contiguous()
        .view(c, grid * grid, ph, pw)
        .permute(1, 0, 2, 3)  # [grid*grid, C, ph, pw]
    )
    shuffled = patches[perm]  # reorder patches
    # Reassemble.
    out = shuffled.view(grid, grid, c, ph, pw).permute(2, 0, 3, 1, 4).contiguous()
    out = out.view(c, grid * ph, grid * pw)
    return out


def patch_shuffle_batch(images_0_1: torch.Tensor, image_ids, grid: int = 4) -> torch.Tensor:
    outs = []
    for img, iid in zip(images_0_1, image_ids):
        perm = make_patch_permutation(iid, grid=grid)
        outs.append(patch_shuffle(img, perm, grid=grid))
    return torch.stack(outs, dim=0)
