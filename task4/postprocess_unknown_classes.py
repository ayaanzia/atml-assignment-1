"""Produce class-level open-set diagnostics from cached vanilla outputs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR100


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
KNOWN_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_cache(split: str) -> dict:
    return torch.load(CACHE / f"vanilla_{split}.pt", map_location="cpu", weights_only=False)


def mls_unknownness(logits: torch.Tensor) -> torch.Tensor:
    return -logits[:, :10].float().amax(dim=1)


def main() -> None:
    classes = CIFAR100(ROOT / "data" / "cifar", train=False, download=False).classes
    score_table = pd.read_csv(RESULTS / "vanilla_score_comparison.csv")
    threshold = float(score_table.loc[score_table["score"] == "MLS", "threshold"].iloc[0])
    class_rows = []
    absorption_rows = []
    split_manifest = {"seed": 6304, "splits": {}}

    for split in ("train_unaug", "val", "test", "near", "far"):
        cache = load_cache(split)
        split_manifest["splits"][split] = {
            "dataset": "CIFAR-100" if split in {"near", "far"} else "CIFAR-10",
            "source_indices": [int(value) for value in cache["indices"].tolist()],
        }
    with (RESULTS / "split_indices_seed6304.json").open("w") as handle:
        json.dump(split_manifest, handle, indent=2)

    for group in ("near", "far"):
        cache = load_cache(group)
        scores = mls_unknownness(cache["logits"])
        predictions = cache["logits"][:, :10].argmax(dim=1)
        labels = cache["labels"].long()
        for label in sorted(labels.unique().tolist()):
            mask = labels == label
            accepted = mask & (scores <= threshold)
            accepted_predictions = predictions[accepted]
            counts = torch.bincount(accepted_predictions, minlength=10)
            most_common = int(counts.argmax()) if int(counts.sum()) else None
            total = int(mask.sum())
            accepted_count = int(accepted.sum())
            class_rows.append({
                "group": group,
                "unknown_class": classes[label],
                "unknown_class_idx": label,
                "n": total,
                "accepted_count": accepted_count,
                "accepted_rate": accepted_count / total,
                "mean_mls_unknownness": float(scores[mask].mean()),
                "most_common_absorbing_class": KNOWN_CLASSES[most_common] if most_common is not None else "",
                "most_common_absorbing_count": int(counts[most_common]) if most_common is not None else 0,
                "most_common_share_of_accepted": (int(counts[most_common]) / accepted_count
                                                   if accepted_count else 0.0),
            })
            for known_index, count in enumerate(counts.tolist()):
                absorption_rows.append({
                    "group": group,
                    "unknown_class": classes[label],
                    "predicted_cifar10_class": KNOWN_CLASSES[known_index],
                    "accepted_count": int(count),
                    "share_of_accepted_for_unknown_class": count / accepted_count if accepted_count else 0.0,
                })

    class_frame = pd.DataFrame(class_rows).sort_values(["group", "accepted_rate"], ascending=[True, False])
    absorption_frame = pd.DataFrame(absorption_rows)
    class_frame.to_csv(RESULTS / "unknown_class_acceptance.csv", index=False)
    absorption_frame.to_csv(RESULTS / "accepted_unknown_to_known_confusion.csv", index=False)

    expected_acceptance = {
        "near": 1.0 - float(score_table.loc[score_table["score"] == "MLS", "near_rejection"].iloc[0]),
        "far": 1.0 - float(score_table.loc[score_table["score"] == "MLS", "far_rejection"].iloc[0]),
    }
    for group, expected in expected_acceptance.items():
        selected = class_frame[class_frame.group == group]
        observed = float(selected.accepted_count.sum() / selected.n.sum())
        if not math.isclose(observed, expected, abs_tol=1e-12):
            raise RuntimeError(f"{group} class aggregation {observed} does not match summary table {expected}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, group in zip(axes, ("near", "far")):
        selected = class_frame[class_frame["group"] == group].sort_values("accepted_rate")
        axis.barh(selected["unknown_class"], selected["accepted_rate"], color="#4C78A8")
        axis.set_xlim(0, 1)
        axis.set_xlabel("Fraction incorrectly accepted")
        axis.set_ylabel(f"{group.capitalize()} unknown class")
    print("Figure: Per-class acceptance rates for near and far CIFAR-100 unknowns under vanilla MLS")
    figure.savefig(RESULTS / "unknown_class_acceptance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    ordered = class_frame.sort_values(["group", "accepted_rate"], ascending=[True, False])
    matrix = np.zeros((len(ordered), len(KNOWN_CLASSES)), dtype=float)
    row_labels = []
    for row_index, row in enumerate(ordered.itertuples(index=False)):
        row_labels.append(f"{row.group}: {row.unknown_class}")
        selected = absorption_frame[(absorption_frame.group == row.group) &
                                    (absorption_frame.unknown_class == row.unknown_class)]
        lookup = dict(zip(selected.predicted_cifar10_class, selected.share_of_accepted_for_unknown_class))
        matrix[row_index] = [lookup[name] for name in KNOWN_CLASSES]
    figure, axis = plt.subplots(figsize=(10, 7))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(KNOWN_CLASSES)), KNOWN_CLASSES, rotation=45, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel("Predicted CIFAR-10 class among accepted unknowns")
    axis.set_ylabel("CIFAR-100 unknown class")
    figure.colorbar(image, ax=axis, label="Share of accepted samples")
    figure.tight_layout()
    print("Figure: CIFAR-10 labels absorbing incorrectly accepted CIFAR-100 classes")
    figure.savefig(RESULTS / "accepted_unknown_to_known_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    leaders = {}
    for group in ("near", "far"):
        row = class_frame[class_frame.group == group].iloc[0]
        leaders[group] = {
            "most_accepted_unknown_class": row.unknown_class,
            "accepted_rate": float(row.accepted_rate),
            "main_absorbing_cifar10_class": row.most_common_absorbing_class,
            "absorbing_share_of_accepted": float(row.most_common_share_of_accepted),
        }
    summary = {"score": "MLS", "threshold": threshold, "leaders": leaders,
               "aggregate_acceptance_validated_against_score_table": True,
               "model_training_or_inference_performed": False}
    with (RESULTS / "unknown_class_analysis_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
