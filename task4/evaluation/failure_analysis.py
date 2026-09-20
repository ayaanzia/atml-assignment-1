"""Select confidently accepted unknowns without feeding them back into the experiment."""
from __future__ import annotations

import pandas as pd
import torch

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def accepted_failure_rows(group, cache, scores, threshold, cifar100_classes, count: int = 3):
    accepted = torch.nonzero(scores <= threshold, as_tuple=False).flatten()
    if len(accepted) < count:
        raise RuntimeError(f"Only {len(accepted)} incorrectly accepted {group} examples; need {count}")
    # Lowest unknownness = most confident failure; inspection remains final-only.
    chosen = accepted[torch.argsort(scores[accepted])[:count]]
    rows = []
    for i in chosen.tolist():
        unknown_name = cifar100_classes[int(cache["labels"][i])]
        predicted = CIFAR10_CLASSES[int(cache["logits"][i, :10].argmax())]
        plausible = (
            predicted in {"automobile", "truck"} and unknown_name in {"bus", "pickup_truck", "tractor"}
        ) or (predicted in {"cat", "dog"} and unknown_name in {"wolf", "fox", "leopard"})
        rows.append({
            "group": group, "cache_row": i, "source_index": int(cache["indices"][i]),
            "unknown_class": unknown_name, "predicted_cifar10_class": predicted,
            "mls_unknownness": float(scores[i]), "threshold": float(threshold),
            "note": "semantically plausible" if plausible else "surprising",
        })
    return rows


def write_failure_cases(near_cache, far_cache, near_scores, far_scores, threshold,
                        cifar100_classes, output_path):
    rows = accepted_failure_rows("near", near_cache, near_scores, threshold, cifar100_classes)
    rows += accepted_failure_rows("far", far_cache, far_scores, threshold, cifar100_classes)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame
