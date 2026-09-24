"""Rebuild Task 1 representation figures from cached features only.

This script deliberately does not instantiate a backbone, train a linear head, or
run AdaIN. Cue-conflict rows are read from the manifest written by the notebook;
they cannot be reconstructed from positional indices after calibrated rejection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import torch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
FIGURES = RESULTS / "figures"
SEED = 6304
CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
BACKBONES = ("resnet50", "vit_b_16", "clip_vit_b_32")
CACHE_NAMES = {"clip_vit_b_32": "clip_vit_b_32_quickgelu"}
CONDITIONS = ("grayscale", "cue_conflict", "translate_d32_up", "patch_shuffle")
CUE_PAIRS = (("cat", "truck"), ("bird", "ship"), ("dog", "car"),
             ("horse", "airplane"), ("monkey", "deer"))


def named_rng(name: str) -> np.random.Generator:
    stable_hash = 0
    for character in name:
        stable_hash = (stable_hash * 131 + ord(character)) % (2**32 - 1)
    derived = (SEED * 1_000_003 + stable_hash) % (2**32 - 1)
    return np.random.default_rng(derived)


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def load_aggregate(backbone: str, condition: str) -> dict:
    backbone = CACHE_NAMES.get(backbone, backbone)
    paths = sorted(CACHE.glob(f"feat__{backbone}__{condition}__*.pt"))
    paths = [path for path in paths if "__image_" not in path.name]
    if len(paths) != 1:
        raise RuntimeError(f"Expected one aggregate cache for {backbone}/{condition}, found {paths}")
    return torch.load(paths[0], map_location="cpu", weights_only=False)


def load_for_ids(backbone: str, condition: str, image_ids: list[str]) -> dict:
    """Load an ordered bundle from either legacy aggregate or stable per-image caches."""
    backbone = CACHE_NAMES.get(backbone, backbone)
    aggregate_paths = sorted(CACHE.glob(f"feat__{backbone}__{condition}__*.pt"))
    aggregate_paths = [path for path in aggregate_paths if "__image_" not in path.name]
    for path in aggregate_paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if set(image_ids).issubset(bundle["image_ids"]):
            return {"features": take_rows(bundle, image_ids), "image_ids": image_ids}

    per_image = []
    for image_id in image_ids:
        digest = hashlib.sha1(image_id.encode()).hexdigest()[:16]
        path = CACHE / f"feat__{backbone}__{condition}__image_{digest}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing stable feature cache for {backbone}/{condition}/{image_id}")
        per_image.append(torch.load(path, map_location="cpu", weights_only=False)["features"])
    return {"features": torch.cat(per_image), "image_ids": image_ids}


def take_rows(bundle: dict, image_ids: list[str]) -> torch.Tensor:
    lookup = {image_id: row for row, image_id in enumerate(bundle["image_ids"])}
    missing = [image_id for image_id in image_ids if image_id not in lookup]
    if missing:
        raise KeyError(f"Cache is missing {len(missing)} requested IDs; first: {missing[:3]}")
    return bundle["features"][[lookup[image_id] for image_id in image_ids]]


def load_cue_manifest() -> pd.DataFrame:
    path = RESULTS / "cue_conflict_manifest.csv"
    if not path.exists():
        raise FileNotFoundError("Run the cue-conflict notebook cells to write the accepted-row manifest")
    manifest = pd.read_csv(path)
    required = {"cue_id", "content_id", "content_class_idx", "style_id", "direction"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Cue-conflict manifest is missing columns: {sorted(missing)}")
    if not manifest["cue_id"].is_unique:
        raise ValueError("Approved cue-conflict manifest contains duplicate IDs")
    return manifest


def project(features: np.ndarray, perplexity: float) -> np.ndarray:
    random_state = int(named_rng("tsne").integers(0, 2**31 - 1))
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=random_state, metric="cosine").fit_transform(features)


def plot_projection(backbone: str, condition: str, clean: torch.Tensor,
                    transformed: torch.Tensor, labels: list[int], perplexity: float) -> Path:
    combined = torch.cat((clean, transformed), dim=0).float().numpy()
    coordinates = project(combined, perplexity)
    split = len(clean)
    clean_xy, transformed_xy = coordinates[:split], coordinates[split:]
    labels_array = np.asarray(labels)
    colors = plt.get_cmap("tab10")

    figure, axis = plt.subplots(figsize=(7.4, 5.4))
    for class_index, class_name in enumerate(CLASSES):
        mask = labels_array == class_index
        axis.scatter(clean_xy[mask, 0], clean_xy[mask, 1], color=colors(class_index),
                     marker="o", s=17, alpha=0.65, linewidths=0)
        axis.scatter(transformed_xy[mask, 0], transformed_xy[mask, 1],
                     color=colors(class_index), marker="x", s=19, alpha=0.8,
                     linewidths=0.8)
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.set_title(f"{backbone}: clean content vs {condition}", fontsize=10)

    class_handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                            markerfacecolor=colors(i), markeredgecolor="none", label=name)
                     for i, name in enumerate(CLASSES)]
    source_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="black", label="clean/content"),
        Line2D([0], [0], marker="x", linestyle="none", color="black", label=condition),
    ]
    source_legend = axis.legend(handles=source_handles, loc="upper right", frameon=False, fontsize=8)
    axis.add_artist(source_legend)
    axis.legend(handles=class_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                frameon=False, fontsize=8, title="Content class", title_fontsize=8)
    figure.tight_layout()
    output = FIGURES / f"tsne_{backbone}_{condition}.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = load_cue_manifest()
    subset = read_json(RESULTS / "eval_subset_ids.json")
    eval_ids = list(subset["image_ids"])
    eval_labels = [int(label) for label in subset["labels"]]

    outputs = []
    for backbone in BACKBONES:
        clean_bundle = load_aggregate(backbone, "clean_eval")
        for condition in CONDITIONS:
            if condition == "cue_conflict":
                content_ids = manifest["content_id"].tolist()
                clean = take_rows(clean_bundle, content_ids)
                transformed = load_for_ids(backbone, condition, manifest["cue_id"].tolist())["features"]
                labels = manifest["content_class_idx"].astype(int).tolist()
            else:
                clean = take_rows(clean_bundle, eval_ids)
                transformed = load_for_ids(backbone, condition, eval_ids)["features"]
                labels = eval_labels
            outputs.append(str(plot_projection(backbone, condition, clean, transformed,
                                               labels, args.perplexity).relative_to(ROOT)))

    metadata = {
        "seed": SEED,
        "projection": "t-SNE",
        "metric": "cosine",
        "perplexity": args.perplexity,
        "cue_manifest_rows": len(manifest),
        "model_training_or_inference_performed": False,
        "outputs": outputs,
    }
    with (RESULTS / "cached_postprocessing_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote {len(outputs)} figures and {len(manifest)} cue-conflict manifest rows")


if __name__ == "__main__":
    main()
