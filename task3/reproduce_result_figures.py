"""Reproduce Task 3 CSV-backed report figures without training models."""
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
        loss_column = "alignment_loss" if "alignment_loss" in frame else None
        panel_count = 3 if loss_column else 2
        figure, axes = plt.subplots(1, panel_count, figsize=(3.6 * panel_count, 3.1), constrained_layout=True)
        axes[0].plot(frame.epoch, frame.classification_loss, color="#4C78A8")
        axes[0].set(xlabel="Epoch", ylabel="Classification loss")
        next_axis = 1
        if loss_column:
            axes[1].plot(frame.epoch, frame[loss_column], color="#F58518")
            axes[1].set(xlabel="Epoch", ylabel="Weighted alignment loss")
            next_axis = 2
        axes[next_axis].plot(frame.epoch, frame.mean_source_val_f1, color="#54A24B")
        axes[next_axis].set(xlabel="Epoch", ylabel="Mean source validation macro-F1", ylim=(0, 1))
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.savefig(csv_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(figure)


def comparison_figure() -> None:
    frame = pd.read_csv(RESULTS / "main_comparison_table.csv")
    figure, axis = plt.subplots(figsize=(6.5, 4))
    positions = range(len(frame))
    width = 0.36
    axis.bar([x - width / 2 for x in positions], frame.mean_source_acc, width,
             label="Mean source accuracy", color="#4C78A8")
    axis.bar([x + width / 2 for x in positions], frame.sketch_acc, width,
             label="Sketch accuracy", color="#F58518")
    axis.set_xticks(list(positions), frame.method)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS / "main_accuracy_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def controlled_figure() -> None:
    frame = pd.read_csv(RESULTS / "controlled_study_results.csv").sort_values("lambda_dg")
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    axes[0].semilogx(frame.lambda_dg, frame.mean_source_acc, marker="o", label="Source")
    axes[0].semilogx(frame.lambda_dg, frame.sketch_acc, marker="o", label="Sketch")
    axes[0].set(xlabel="DG alignment weight (lambda)", ylabel="Accuracy", ylim=(0, 1))
    axes[0].legend(frameon=False)
    axes[1].semilogx(frame.lambda_dg, frame.source_domain_separability, marker="o", color="#E45756")
    axes[1].axhline(1 / 3, color="black", linestyle="--", linewidth=1, label="Chance")
    axes[1].set(xlabel="DG alignment weight (lambda)", ylabel="Source-domain separability", ylim=(0, 1))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(RESULTS / "controlled_study.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    training_curves()
    comparison_figure()
    controlled_figure()
    print("Reproduced Task 3 training, main-comparison, and controlled-study figures from CSV files")


if __name__ == "__main__":
    main()
