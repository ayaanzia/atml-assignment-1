"""Assemble compact report figures from committed experiment artifacts."""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def copy_artifacts() -> None:
    sources = {
        ROOT / "task1/results/figures/translation_curves.png": "translation_curves.png",
        ROOT / "task1/results/figures/cue_conflict_gallery.png": "cue_conflict_gallery.png",
        ROOT / "task2/results/controlled_study.png": "uda_controlled.png",
        ROOT / "task3/results/controlled_study.png": "dg_controlled.png",
        ROOT / "task4/results/score_distribution_or_roc_figure.png": "osr_scores.png",
        ROOT / "task4/results/unknown_class_acceptance.png": "unknown_acceptance.png",
        ROOT / "validation/results/multiseed_accuracy.png": "multiseed_accuracy.png",
        ROOT / "task2/results/confusion_matrices/source_only.png": "pacs_confusion_erm.png",
        ROOT / "task2/results/confusion_matrices/dan.png": "pacs_confusion_dan.png",
        ROOT / "task2/results/confusion_matrices/dann.png": "pacs_confusion_dann.png",
        ROOT / "task2/results/confusion_matrices/cdan.png": "pacs_confusion_cdan.png",
        ROOT / "task3/results/confusion_matrices/dan_dg.png": "pacs_confusion_dan_dg.png",
        ROOT / "task3/results/confusion_matrices/sam.png": "pacs_confusion_sam.png",
    }
    for source, name in sources.items():
        shutil.copy2(source, OUT / name)


def tsne_grid() -> None:
    backbones = ("resnet50", "vit_b_16", "clip_vit_b_32")
    conditions = ("grayscale", "cue_conflict", "translate_d32_up", "patch_shuffle")
    pretty_backbones = ("ResNet-50", "ViT-B/16", "CLIP ViT-B/32")
    pretty_conditions = ("Grayscale", "Cue conflict", "Translate 32 px", "Patch shuffle")
    figure, axes = plt.subplots(3, 4, figsize=(15, 10.4), constrained_layout=True)
    for row, (backbone, backbone_name) in enumerate(zip(backbones, pretty_backbones)):
        for column, (condition, condition_name) in enumerate(zip(conditions, pretty_conditions)):
            path = ROOT / "task1/results/figures" / f"tsne_{backbone}_{condition}.png"
            axes[row, column].imshow(Image.open(path))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(condition_name, fontsize=12)
            if column == 0:
                axes[row, column].text(-0.03, 0.5, backbone_name, rotation=90,
                                       va="center", ha="right", fontsize=12,
                                       transform=axes[row, column].transAxes)
    figure.savefig(OUT / "task1_tsne_grid.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def pacs_curves() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.3), constrained_layout=True)
    task2_names = {
        "source_only": "Source-only", "dan": "DAN", "dann": "DANN", "cdan": "CDAN"
    }
    for stem, label in task2_names.items():
        frame = pd.read_csv(ROOT / "task2/results/training_curves" / f"{stem}.csv")
        axes[0, 0].plot(frame.epoch, frame.classification_loss, marker="o", ms=2.5, label=label)
        if stem != "source_only":
            axes[0, 1].plot(frame.epoch, frame.adaptation_loss, marker="o", ms=2.5, label=label)
    axes[0, 0].set(xlabel="Epoch", ylabel="Classification loss", title="UDA classification")
    axes[0, 1].set(xlabel="Epoch", ylabel="Alignment/domain loss", title="UDA adaptation")
    axes[0, 0].set_yscale("symlog", linthresh=1)
    axes[0, 1].set_yscale("symlog", linthresh=1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)

    task3_names = {"erm_loaded_unchanged": "ERM", "dan_dg": "DAN-DG", "sam": "SAM"}
    for stem, label in task3_names.items():
        frame = pd.read_csv(ROOT / "task3/results/training_curves" / f"{stem}.csv")
        axes[1, 0].plot(frame.epoch, frame.classification_loss, marker="o", ms=2.5, label=label)
        if "alignment_loss" in frame and not np.allclose(frame.alignment_loss, 0):
            axes[1, 1].plot(frame.epoch, frame.alignment_loss, marker="o", ms=2.5, label=label)
    axes[1, 0].set(xlabel="Epoch", ylabel="Classification loss", title="DG classification")
    axes[1, 1].set(xlabel="Epoch", ylabel="Pairwise MMD", title="DG alignment")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.savefig(OUT / "pacs_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def pacs_per_class() -> None:
    task2 = pd.read_csv(ROOT / "task2/results/per_class_target_accuracy.csv")
    task3 = pd.read_csv(ROOT / "task3/results/per_class_sketch_accuracy.csv")
    classes = task2[task2.method == "source_only"]["class"].tolist()
    positions = np.arange(len(classes))
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    width = 0.25
    for offset, method in enumerate(("dan", "dann", "cdan")):
        frame = task2[task2.method == method].set_index("class").loc[classes]
        axes[0].bar(positions + (offset - 1) * width, frame.delta_vs_source_only,
                    width, label=method.upper())
    for offset, method in enumerate(("DAN-DG", "SAM")):
        frame = task3[task3.method == method].set_index("class").loc[classes]
        axes[1].bar(positions + (offset - 0.5) * width, frame.delta_vs_erm,
                    width, label=method)
    for axis, title in zip(axes, ("UDA: change from Source-only", "DG: change from ERM")):
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, classes, rotation=40, ha="right")
        axis.set_ylabel("Sketch accuracy change")
        axis.set_title(title)
        axis.legend(frameon=False, fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(OUT / "pacs_per_class.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    copy_artifacts()
    tsne_grid()
    pacs_curves()
    pacs_per_class()
    print(f"Prepared report figures in {OUT}")


if __name__ == "__main__":
    main()
