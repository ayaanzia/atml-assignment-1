"""
Global configuration for Task 1: Inductive Biases and Feature Representations.

Centralizes: seeds, device selection, directory layout, and small helpers for
mixed-precision autocast contexts. Every stochastic operation in the pipeline
(splits, subset selection, permutations, projections) must draw from either
the global seed below or from `make_rng` with a documented, fixed sub-seed so
runs are reproducible end-to-end.
"""
from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Master seed. Every other seed used anywhere in the notebook is derived from
# this constant so a single value controls full reproducibility.
# ---------------------------------------------------------------------------
SEED = 6304

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "data"

for _d in (RESULTS_DIR, CACHE_DIR, FIGURES_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device / precision
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
# bf16 is preferred when supported (Ampere+) since it needs no GradScaler and
# is numerically safer; fall back to fp16 with GradScaler otherwise.
AMP_DTYPE = torch.bfloat16 if (USE_CUDA and torch.cuda.is_bf16_supported()) else torch.float16

IMG_SIZE = 224
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

if not USE_CUDA:
    print(
        "[WARN] CUDA is not available in this environment. This pipeline is "
        "designed and required (per spec) to run with a CUDA GPU for mixed "
        "precision + reasonable runtime. Falling back to CPU will be slow "
        "and autocast/channels_last settings below will be no-ops or "
        "CPU-appropriate fallbacks."
    )


def seed_everything(seed: int = SEED) -> None:
    """Seed python/numpy/torch (CPU + CUDA) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark=True is a required perf optimization (fixed 224x224
    # inputs make this safe); it does introduce a small amount of
    # nondeterminism in kernel selection, which is acceptable here since we
    # never rely on bitwise-exact reproducibility, only on fixed *logical*
    # seeds for splits/subsets/permutations (handled via `make_rng`).
    torch.backends.cudnn.benchmark = True


def make_rng(sub_seed_name: str, base_seed: int = SEED) -> np.random.Generator:
    """
    Return an independent, deterministic numpy Generator for a named
    stochastic operation (e.g. 'eval_subset', 'color_swap_pairing',
    'patch_permutation:<image_id>', 'tsne'). Deriving a distinct sub-seed per
    named operation (instead of reusing one global np.random stream) means
    the order in which cells run does not change any individual operation's
    outcome.
    """
    # Stable string -> int hash (Python's built-in hash() is salted per
    # process, so we use a fixed, simple deterministic hash instead).
    h = 0
    for ch in sub_seed_name:
        h = (h * 131 + ord(ch)) % (2**32 - 1)
    derived = (base_seed * 1_000_003 + h) % (2**32 - 1)
    return np.random.default_rng(derived)


@contextmanager
def autocast_ctx():
    """Autocast context used for all backbone forward passes / head training."""
    if USE_CUDA:
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            yield
    else:
        # No autocast benefit on CPU; yield a null context.
        yield


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def cache_path(name: str) -> Path:
    return CACHE_DIR / name


# NUM_WORKERS = min(8, os.cpu_count() or 4)
NUM_WORKERS = 0