"""
Evaluation metrics and the (model-prediction-independent) cue-conflict
rejection rule.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score

from .config import make_rng


# ---------------------------------------------------------------------------
# Basic classification metrics
# ---------------------------------------------------------------------------
def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return float((preds == labels).mean())


def macro_f1(preds: Sequence[int], labels: Sequence[int]) -> float:
    return float(f1_score(labels, preds, average="macro"))


def mean_max_softmax_confidence(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    conf = probs.max(dim=-1).values
    return float(conf.mean().item())


def prediction_consistency(preds_a: Sequence[int], preds_b: Sequence[int]) -> float:
    """Fraction of images whose prediction is unchanged between condition A
    (typically clean) and condition B (typically a transformed version)."""
    a = np.asarray(preds_a)
    b = np.asarray(preds_b)
    assert len(a) == len(b)
    return float((a == b).mean())


# ---------------------------------------------------------------------------
# Representation stability I_T (Sec. 10)
# ---------------------------------------------------------------------------
def representation_stability(feat_clean: torch.Tensor, feat_transformed: torch.Tensor) -> float:
    """
    Mean cosine similarity between paired clean/transformed representations,
    I_T = (1/N) * sum_i cos(f(x_i), f(T(x_i))).
    feat_clean, feat_transformed: [N, D] tensors, row i corresponds to the
    same underlying image in both.
    """
    a = F.normalize(feat_clean.float(), dim=-1)
    b = F.normalize(feat_transformed.float(), dim=-1)
    cos = (a * b).sum(dim=-1)
    return float(cos.mean().item())


# ---------------------------------------------------------------------------
# Step 3: cue-conflict rejection rule
#
# Defined BEFORE any model is run on the stylized images, and implemented as
# a pure image-quality check that never looks at any model's prediction.
#
# Rule: reject a stylization if its structural similarity (SSIM, computed on
# grayscale) to the ORIGINAL CONTENT image falls below a fixed threshold,
# OR if the stylized image is degenerate (near-uniform / near-zero variance,
# e.g. a decoder failure producing a flat color field).
# ---------------------------------------------------------------------------
# Selected by the preregistered human review in notebook Section 8b. The full
# sweep and its stricter-threshold tie rule are persisted under results/.
SSIM_MIN_THRESHOLD = 0.30
DEGENERATE_STD_THRESHOLD = 0.02  # per-channel pixel std below this -> flat/degenerate output


def is_valid_cue_conflict(content_0_1: np.ndarray, stylized_0_1: np.ndarray) -> Dict:
    """
    content_0_1, stylized_0_1: numpy arrays [H, W, 3] in [0, 1], same size.
    Returns a dict with the boolean decision and the underlying measurements
    (for logging), using ONLY pixel statistics — no model predictions.
    """
    content_gray = content_0_1.mean(axis=-1)
    stylized_gray = stylized_0_1.mean(axis=-1)
    score = ssim(content_gray, stylized_gray, data_range=1.0)
    px_std = float(stylized_0_1.std())
    degenerate = px_std < DEGENERATE_STD_THRESHOLD
    accepted = bool((score >= SSIM_MIN_THRESHOLD) and not degenerate)
    return {
        "ssim_to_content": float(score),
        "pixel_std": px_std,
        "degenerate": degenerate,
        "accepted": accepted,
    }


# ---------------------------------------------------------------------------
# Shape-bias classification for cue-conflict predictions
# ---------------------------------------------------------------------------
def classify_shape_texture(pred_class: int, shape_class: int, texture_class: int) -> str:
    if pred_class == shape_class:
        return "shape"
    if pred_class == texture_class:
        return "texture"
    return "other"


def shape_bias_summary(decisions: Sequence[str]) -> Dict:
    decisions = list(decisions)
    n_total = len(decisions)
    n_shape = decisions.count("shape")
    n_texture = decisions.count("texture")
    denom = n_shape + n_texture
    shape_bias_pct = 100.0 * n_shape / denom if denom > 0 else float("nan")
    coverage_pct = 100.0 * denom / n_total if n_total > 0 else float("nan")
    return {
        "n_total": n_total,
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_total - denom,
        "shape_bias_pct": shape_bias_pct,
        "coverage_pct": coverage_pct,
    }


# ---------------------------------------------------------------------------
# Step 6: 2D projection (t-SNE or UMAP) for visualization
# ---------------------------------------------------------------------------
def fit_2d_projection(features: np.ndarray, method: str = "tsne", seed_name: str = "tsne", **kwargs) -> np.ndarray:
    """
    Fits ONE 2D projection on the combined clean+transformed feature matrix
    (caller concatenates before calling this). method in {'tsne', 'umap'}.
    """
    rng = make_rng(seed_name)
    seed_int = int(rng.integers(0, 2**31 - 1))
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = kwargs.get("perplexity", 30)
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            random_state=seed_int,
            metric="cosine",
        )
    elif method == "umap":
        import umap  # requires `pip install umap-learn`
        n_neighbors = kwargs.get("n_neighbors", 15)
        min_dist = kwargs.get("min_dist", 0.1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed_int,
            metric="cosine",
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(features)
