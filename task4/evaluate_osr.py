"""Pure-cache OSR evaluation, table generation, figure, and failure analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from task4.evaluation.failure_analysis import write_failure_cases
from task4.evaluation.metrics import evaluate_score
from task4.methods.proser import placeholder_unknownness
from task4.scores import (energy_unknownness, fit_shared_diagonal_gaussian,
                          mahalanobis_unknownness, mls_unknownness, msp_unknownness)

ROOT = Path(__file__).resolve().parent


def load_cache(cache_dir, model, split):
    path = Path(cache_dir) / f"{model}_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the extraction cells first")
    return torch.load(path, map_location="cpu", weights_only=False)


def score_bundle(cache, means=None, variance=None):
    logits = cache["logits"][:, :10]
    result = {
        "MSP": msp_unknownness(logits), "MLS": mls_unknownness(logits),
        "Energy": energy_unknownness(logits),
    }
    if means is not None:
        result["Mahalanobis"] = mahalanobis_unknownness(cache["features"], means, variance)
    return result


def evaluate_all(cache_dir, results_dir, cifar100_classes):
    cache_dir, results_dir = Path(cache_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    vanilla = {s: load_cache(cache_dir, "vanilla", s) for s in ("train_unaug", "val", "test", "near", "far")}
    means, variance = fit_shared_diagonal_gaussian(
        vanilla["train_unaug"]["features"], vanilla["train_unaug"]["labels"])
    scores = {split: score_bundle(data, means, variance) for split, data in vanilla.items() if split != "train_unaug"}
    rows = []
    for name in ("MSP", "MLS", "Energy", "Mahalanobis"):
        rows.append({"score": name, **evaluate_score(scores["val"][name], scores["test"][name],
                                                       scores["near"][name], scores["far"][name])})
    vanilla_table = pd.DataFrame(rows)
    vanilla_table.to_csv(results_dir / "vanilla_score_comparison.csv", index=False)

    trained_rows = []
    for model_name in ("vanilla", "gcsc", "proser"):
        cached = {s: load_cache(cache_dir, model_name, s) for s in ("val", "test", "near", "far")}
        model_scores = {s: mls_unknownness(x["logits"][:, :10]) for s, x in cached.items()}
        metrics = evaluate_score(model_scores["val"], model_scores["test"], model_scores["near"], model_scores["far"])
        csa = float((cached["test"]["logits"][:, :10].argmax(1) == cached["test"]["labels"]).float().mean())
        trained_rows.append({"model": model_name.capitalize(), "score": "MLS", "csa": csa, **metrics})
        if model_name == "proser":
            placeholder_scores = {s: placeholder_unknownness(x["logits"]) for s, x in cached.items()}
            placeholder_metrics = evaluate_score(placeholder_scores["val"], placeholder_scores["test"],
                                                  placeholder_scores["near"], placeholder_scores["far"])
            trained_rows.append({"model": "PROSER", "score": "Placeholder Delta-P", "csa": csa,
                                 **placeholder_metrics})
    trained_table = pd.DataFrame(trained_rows)
    trained_table.to_csv(results_dir / "trained_model_comparison.csv", index=False)

    figure_scores = ("MSP", "MLS", "Mahalanobis")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for ax, name in zip(axes, figure_scores):
        ax.hist(scores["test"][name].numpy(), bins=40, density=True, alpha=.55, label="CIFAR-10 test")
        ax.hist(scores["near"][name].numpy(), bins=40, density=True, alpha=.45, label="Near unknown")
        ax.hist(scores["far"][name].numpy(), bins=40, density=True, alpha=.45, label="Far unknown")
        ax.set_xlabel(f"{name} unknownness")
        ax.set_ylabel("Density")
    axes[-1].legend(frameon=False, fontsize=8)
    print("Figure: Vanilla score distributions for CIFAR-10 known, near unknown, and far unknown samples")
    fig.savefig(results_dir / "score_distribution_or_roc_figure.png", dpi=180, bbox_inches="tight")
    plt.show()

    mls_row = vanilla_table.loc[vanilla_table.score == "MLS"].iloc[0]
    failures = write_failure_cases(vanilla["near"], vanilla["far"], scores["near"]["MLS"],
                                   scores["far"]["MLS"], float(mls_row.threshold),
                                   cifar100_classes, results_dir / "failure_cases.csv")
    return vanilla_table, trained_table, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(ROOT / "results" / "cache"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--data-root", default=str(ROOT / "data" / "cifar"))
    args = parser.parse_args()
    # Loading metadata here is safe: all checkpoints/scores/threshold definitions are already fixed.
    from torchvision.datasets import CIFAR100
    classes = CIFAR100(args.data_root, train=False, download=False).classes
    evaluate_all(args.cache_dir, args.results_dir, classes)


if __name__ == "__main__":
    main()
