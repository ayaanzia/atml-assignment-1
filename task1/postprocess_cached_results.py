"""Rebuild Task 1 representation figures from cached features only.

This script deliberately does not instantiate a backbone, train a linear head, or
run AdaIN.  It also reconstructs the cue-conflict source-pair manifest from the
fixed seed.  That reconstruction is exact for the committed run because its
rejection log records zero rejected candidates in every bucket.
"""
from __future__ import annotations

import argparse
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
    paths = sorted(CACHE.glob(f"feat__{backbone}__{condition}__*.pt"))
    paths = [path for path in paths if "__image_" not in path.name]
    if len(paths) != 1:
        raise RuntimeError(f"Expected one aggregate cache for {backbone}/{condition}, found {paths}")
    return torch.load(paths[0], map_location="cpu", weights_only=False)


def take_rows(bundle: dict, image_ids: list[str]) -> torch.Tensor:
    lookup = {image_id: row for row, image_id in enumerate(bundle["image_ids"])}
    missing = [image_id for image_id in image_ids if image_id not in lookup]
    if missing:
        raise KeyError(f"Cache is missing {len(missing)} requested IDs; first: {missing[:3]}")
    return bundle["features"][[lookup[image_id] for image_id in image_ids]]


def reconstruct_cue_manifest() -> pd.DataFrame:
    subset = read_json(RESULTS / "eval_subset_ids.json")
    rejection_log = read_json(RESULTS / "cue_conflict_rejection_log.json")
    if any(entry["rejected"] != 0 for entry in rejection_log.values()):
        raise RuntimeError("Cannot reconstruct accepted IDs without rerunning AdaIN when a bucket has rejections")

    pools: dict[int, list[str]] = {index: [] for index in range(len(CLASSES))}
    for image_id, label in zip(subset["image_ids"], subset["labels"]):
        pools[int(label)].append(image_id)

    class_to_idx = {name: index for index, name in enumerate(CLASSES)}
    rng = named_rng("cue_conflict_sampling")
    rows = []
    cue_index = 0
    for class_a, class_b in CUE_PAIRS:
        a_idx, b_idx = class_to_idx[class_a], class_to_idx[class_b]
        directions = (("A_shape_B_texture", a_idx, b_idx),
                      ("B_shape_A_texture", b_idx, a_idx))
        for direction, content_idx, texture_idx in directions:
            bucket = f"{class_a}-{class_b}:{direction}"
            count = int(rejection_log[bucket]["target"])
            content_rows = rng.integers(0, len(pools[content_idx]), size=count)
            texture_rows = rng.integers(0, len(pools[texture_idx]), size=count)
            for content_row, texture_row in zip(content_rows, texture_rows):
                rows.append({
                    "cue_id": f"cueconflict-{cue_index}",
                    "pair": f"{class_a}-{class_b}",
                    "direction": direction,
                    "content_class": CLASSES[content_idx],
                    "content_class_idx": content_idx,
                    "texture_class": CLASSES[texture_idx],
                    "texture_class_idx": texture_idx,
                    "content_id": pools[content_idx][int(content_row)],
                    "style_id": pools[texture_idx][int(texture_row)],
                })
                cue_index += 1

    manifest = pd.DataFrame(rows)
    cue_bundle = load_aggregate(BACKBONES[0], "cue_conflict")
    if manifest["cue_id"].tolist() != list(cue_bundle["image_ids"]):
        raise RuntimeError("Reconstructed cue order does not match the feature cache")
    if manifest["content_class_idx"].tolist() != [int(x) for x in cue_bundle["labels"]]:
        raise RuntimeError("Reconstructed cue labels do not match the feature cache")
    manifest.to_csv(RESULTS / "cue_conflict_manifest.csv", index=False)
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
    manifest = reconstruct_cue_manifest()
    subset = read_json(RESULTS / "eval_subset_ids.json")
    eval_ids = list(subset["image_ids"])
    eval_labels = [int(label) for label in subset["labels"]]

    outputs = []
    for backbone in BACKBONES:
        clean_bundle = load_aggregate(backbone, "clean_eval")
        for condition in CONDITIONS:
            transformed_bundle = load_aggregate(backbone, condition)
            if condition == "cue_conflict":
                content_ids = manifest["content_id"].tolist()
                clean = take_rows(clean_bundle, content_ids)
                transformed = take_rows(transformed_bundle, manifest["cue_id"].tolist())
                labels = manifest["content_class_idx"].astype(int).tolist()
            else:
                clean = take_rows(clean_bundle, eval_ids)
                transformed = take_rows(transformed_bundle, eval_ids)
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
