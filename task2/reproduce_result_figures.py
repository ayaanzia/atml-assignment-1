"""Reproduce Task 2 CSV-backed report figures without training models."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def training_curves() -> None:
    for csv_path in sorted((RESULTS / "training_curves").glob("*.csv")):
        frame = pd.read_csv(csv_path)
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
        axes[0].plot(frame.epoch, frame.classification_loss, color="#4C78A8")
        axes[0].set(xlabel="Epoch", ylabel="Classification loss")
        axes[1].plot(frame.epoch, frame.adaptation_loss, color="#F58518")
        axes[1].set(xlabel="Epoch", ylabel="Weighted adaptation loss")
        axes[2].plot(frame.epoch, frame.mean_source_val_f1, color="#54A24B")
        axes[2].set(xlabel="Epoch", ylabel="Mean source validation macro-F1", ylim=(0, 1))
        for axis in axes:
            axis.grid(alpha=0.2)
        output = csv_path.with_suffix(".png")
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)


def comparison_figure() -> None:
    frame = pd.read_csv(RESULTS / "main_comparison_table.csv")
    figure, axis = plt.subplots(figsize=(7, 4))
    positions = range(len(frame))
    width = 0.36
    axis.bar([x - width / 2 for x in positions], frame.mean_source_acc, width,
             label="Mean source accuracy", color="#4C78A8")
    axis.bar([x + width / 2 for x in positions], frame.target_acc, width,
             label="Sketch accuracy", color="#F58518")
    axis.set_xticks(list(positions), frame.method.str.replace("_", " "))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS / "main_accuracy_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def controlled_figure() -> None:
    frame = pd.read_csv(RESULTS / "controlled_study_results.csv").sort_values("lambda_mmd")
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    axes[0].semilogx(frame.lambda_mmd, frame.mean_source_acc, marker="o", label="Source")
    axes[0].semilogx(frame.lambda_mmd, frame.target_acc, marker="o", label="Sketch")
    axes[0].set(xlabel="MMD weight (lambda)", ylabel="Accuracy", ylim=(0, 1))
    axes[0].legend(frameon=False)
    axes[1].semilogx(frame.lambda_mmd, frame.domain_separability, marker="o", color="#E45756")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    axes[1].set(xlabel="MMD weight (lambda)", ylabel="Domain separability", ylim=(0, 1))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(RESULTS / "controlled_study.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    training_curves()
    comparison_figure()
    controlled_figure()
    print("Reproduced Task 2 training, main-comparison, and controlled-study figures from CSV files")


if __name__ == "__main__":
    main()
