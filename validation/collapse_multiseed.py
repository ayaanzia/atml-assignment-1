"""Two-seed retraining validation for the Task 2/3 collapsed configurations.

The original seed-6304 source split remains fixed.  Validation seeds affect model
initialization, minibatch ordering, stochastic augmentation, and DANN dropout.
Results and checkpoints are isolated under validation/results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
TASK2 = ROOT / "task2"
TASK3 = ROOT / "task3"
sys.path.insert(0, str(TASK3))

from utils import (  # noqa: E402
    NUM_CLASSES,
    SOURCE_DOMAINS,
    ResNet18Adapter,
    freeze_batchnorm_running_stats,
    load_source_protocol,
    make_transforms,
    multi_kernel_mmd,
    resolve_domain_dir,
)


ORIGINAL_SEED = 6304
VALIDATION_SEEDS = (6305, 6306)
SOURCE_BATCH_PER_DOMAIN = 8
TARGET_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
COLLAPSE_SOURCE_ACCURACY = 0.30
COLLAPSE_TARGET_ACCURACY = 0.10
EXTREME_LOSS_THRESHOLD = 1_000.0

OUT = ROOT / "validation" / "results"
CHECKPOINTS = OUT / "checkpoints"
CURVES = OUT / "training_curves"
for directory in (OUT, CHECKPOINTS, CURVES):
    directory.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UnlabeledDataset(Dataset):
    def __init__(self, labeled: Dataset):
        self.labeled = labeled

    def __len__(self):
        return len(self.labeled)

    def __getitem__(self, index):
        image, _ = self.labeled[index]
        return image


class RestartingLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.alpha = alpha
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, inputs):
        return self.network(inputs)


@dataclass
class Protocol:
    source_train: dict
    source_val: dict
    target_adaptation: Dataset
    target_eval: Dataset
    class_to_idx: dict


def prepare_protocol() -> Protocol:
    pacs_root = Path(os.environ.get("PACS_ROOT", TASK2 / "data")).expanduser().resolve()
    split_path = TASK2 / "results" / f"pacs_splits_seed{ORIGINAL_SEED}.json"
    source = load_source_protocol(pacs_root, split_path, ORIGINAL_SEED)
    train_transform, eval_transform = make_transforms()
    sketch_dir = resolve_domain_dir(pacs_root, "sketch")
    target_train = datasets.ImageFolder(sketch_dir, transform=train_transform)
    target_eval = datasets.ImageFolder(sketch_dir, transform=eval_transform)
    if target_eval.class_to_idx != source["class_to_idx"]:
        raise ValueError("Sketch class mapping differs from the fixed source protocol")
    if len(target_eval) != 3929:
        raise ValueError(f"Expected 3929 original Sketch images, found {len(target_eval)}")
    return Protocol(source["train_sets"], source["val_sets"],
                    UnlabeledDataset(target_train), target_eval, source["class_to_idx"])


def make_loader(dataset, device, *, batch_size, shuffle=False, drop_last=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, generator=generator, num_workers=0,
                      pin_memory=device.type == "cuda")


def amp_context(device, amp_dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=amp_dtype)


def make_scaler(device, amp_dtype):
    return torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16)
    )


def train_mode_with_frozen_bn(*modules) -> None:
    for module in modules:
        if module is not None:
            module.train()
            freeze_batchnorm_running_stats(module)


def make_source_streams(protocol: Protocol, seed: int, device):
    streams = {}
    for offset, domain in enumerate(SOURCE_DOMAINS):
        generator = torch.Generator().manual_seed(seed + offset)
        loader = make_loader(protocol.source_train[domain], device,
                             batch_size=SOURCE_BATCH_PER_DOMAIN, shuffle=True,
                             drop_last=True, generator=generator)
        streams[domain] = RestartingLoader(loader)
    steps = max(math.ceil(len(protocol.source_train[d]) / SOURCE_BATCH_PER_DOMAIN)
                for d in SOURCE_DOMAINS)
    return streams, steps


def next_source_batch(streams, device):
    images_by_domain, labels_by_domain = [], []
    for domain in SOURCE_DOMAINS:
        images, labels = streams[domain].next()
        if len(images) != SOURCE_BATCH_PER_DOMAIN:
            raise RuntimeError("Source loader violated the domain-balanced batch size")
        images_by_domain.append(images.to(device, non_blocking=True,
                                          memory_format=torch.channels_last))
        labels_by_domain.append(labels.to(device, non_blocking=True))
    return images_by_domain, labels_by_domain


@torch.inference_mode()
def evaluate(model, dataset, device, amp_dtype):
    model.eval()
    truths, predictions = [], []
    loader = make_loader(dataset, device, batch_size=EVAL_BATCH_SIZE)
    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        with amp_context(device, amp_dtype):
            logits, _ = model(images)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        truths.append(labels.numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    truth, prediction = np.concatenate(truths), np.concatenate(predictions)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def evaluate_sources(model, protocol, device, amp_dtype):
    metrics = {domain: evaluate(model, protocol.source_val[domain], device, amp_dtype)
               for domain in SOURCE_DOMAINS}
    return metrics, float(np.mean([value["macro_f1"] for value in metrics.values()]))


def dann_alpha(progress: float) -> float:
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def save_best(path: Path, model, discriminator, *, method, seed, epoch,
              source_metrics, mean_source_f1, amp_dtype):
    torch.save({
        "model_state": model.state_dict(),
        "discriminator_state": discriminator.state_dict() if discriminator is not None else None,
        "method": method,
        "seed": seed,
        "split_seed": ORIGINAL_SEED,
        "epoch": epoch,
        "source_val": source_metrics,
        "mean_source_val_macro_f1": mean_source_f1,
        "amp_dtype": str(amp_dtype),
    }, path)


def train_uda(method: str, seed: int, protocol: Protocol, device, amp_dtype, force=False):
    if method not in {"dan", "dann"}:
        raise ValueError(method)
    checkpoint_path = CHECKPOINTS / f"{method}_seed{seed}.pt"
    curve_path = CURVES / f"{method}_seed{seed}.csv"
    if checkpoint_path.exists() and curve_path.exists() and not force:
        return checkpoint_path, pd.read_csv(curve_path).to_dict("records")

    seed_everything(seed)
    model = ResNet18Adapter().to(device, memory_format=torch.channels_last)
    discriminator = DomainDiscriminator(model.feature_dim).to(device) if method == "dann" else None
    parameters = list(model.parameters())
    if discriminator is not None:
        parameters.extend(discriminator.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler(device, amp_dtype)
    source_streams, steps = make_source_streams(protocol, seed, device)
    target_loader = make_loader(protocol.target_adaptation, device,
                                batch_size=TARGET_BATCH_SIZE, shuffle=True, drop_last=True,
                                generator=torch.Generator().manual_seed(seed + 100))
    target_stream = RestartingLoader(target_loader)
    best_f1, stale, global_step = -math.inf, 0, 0
    history = []
    total_steps = MAX_EPOCHS * steps

    for epoch in range(1, MAX_EPOCHS + 1):
        train_mode_with_frozen_bn(model, discriminator)
        classification_sum = adaptation_sum = 0.0
        progress_bar = tqdm(range(steps), desc=f"{method} seed {seed} epoch {epoch}", leave=True)
        for batch_index in progress_bar:
            domain_images, domain_labels = next_source_batch(source_streams, device)
            source_images, source_labels = torch.cat(domain_images), torch.cat(domain_labels)
            target_images = target_stream.next().to(
                device, non_blocking=True, memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_dtype):
                all_images = torch.cat((source_images, target_images))
                all_logits, all_features = model(all_images)
                source_count = len(source_images)
                source_logits = all_logits[:source_count]
                classification_loss = F.cross_entropy(source_logits, source_labels)
                if method == "dan":
                    adaptation_loss = multi_kernel_mmd(
                        all_features[:source_count], all_features[source_count:])
                else:
                    alpha = dann_alpha(global_step / max(total_steps - 1, 1))
                    domain_labels = torch.cat((
                        torch.zeros(source_count, dtype=torch.long, device=device),
                        torch.ones(len(target_images), dtype=torch.long, device=device),
                    ))
                    reversed_features = GradientReversal.apply(all_features, alpha)
                    adaptation_loss = F.cross_entropy(discriminator(reversed_features), domain_labels)
                loss = classification_loss + adaptation_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{method} seed {seed}: non-finite loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            classification_sum += float(classification_loss.detach())
            adaptation_sum += float(adaptation_loss.detach())
            global_step += 1
            progress_bar.set_postfix(cls=classification_sum / (batch_index + 1),
                                     adapt=adaptation_sum / (batch_index + 1))

        source_metrics, mean_source_f1 = evaluate_sources(model, protocol, device, amp_dtype)
        history.append({"epoch": epoch,
                        "classification_loss": classification_sum / steps,
                        "adaptation_loss": adaptation_sum / steps,
                        "mean_source_val_f1": mean_source_f1})
        pd.DataFrame(history).to_csv(curve_path, index=False)
        if mean_source_f1 > best_f1:
            best_f1, stale = mean_source_f1, 0
            save_best(checkpoint_path, model, discriminator, method=method, seed=seed,
                      epoch=epoch, source_metrics=source_metrics,
                      mean_source_f1=mean_source_f1, amp_dtype=amp_dtype)
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return checkpoint_path, history


def train_dan_dg(seed: int, protocol: Protocol, device, amp_dtype, force=False):
    method = "dan_dg"
    checkpoint_path = CHECKPOINTS / f"{method}_seed{seed}.pt"
    curve_path = CURVES / f"{method}_seed{seed}.csv"
    if checkpoint_path.exists() and curve_path.exists() and not force:
        return checkpoint_path, pd.read_csv(curve_path).to_dict("records")

    seed_everything(seed)
    model = ResNet18Adapter().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler(device, amp_dtype)
    streams, steps = make_source_streams(protocol, seed, device)
    best_f1, stale, history = -math.inf, 0, []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_mode_with_frozen_bn(model)
        classification_sum = alignment_sum = 0.0
        progress_bar = tqdm(range(steps), desc=f"dan_dg seed {seed} epoch {epoch}", leave=True)
        for batch_index in progress_bar:
            images, labels = next_source_batch(streams, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_dtype):
                logits, features = model(torch.cat(images))
                classification_loss = F.cross_entropy(logits, torch.cat(labels))
                chunks = features.split(SOURCE_BATCH_PER_DOMAIN)
                alignment_loss = (multi_kernel_mmd(chunks[0], chunks[1])
                                  + multi_kernel_mmd(chunks[0], chunks[2])
                                  + multi_kernel_mmd(chunks[1], chunks[2])) / 3.0
                loss = classification_loss + alignment_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"dan_dg seed {seed}: non-finite loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            classification_sum += float(classification_loss.detach())
            alignment_sum += float(alignment_loss.detach())
            progress_bar.set_postfix(cls=classification_sum / (batch_index + 1),
                                     align=alignment_sum / (batch_index + 1))

        source_metrics, mean_source_f1 = evaluate_sources(model, protocol, device, amp_dtype)
        history.append({"epoch": epoch,
                        "classification_loss": classification_sum / steps,
                        "adaptation_loss": alignment_sum / steps,
                        "mean_source_val_f1": mean_source_f1})
        pd.DataFrame(history).to_csv(curve_path, index=False)
        if mean_source_f1 > best_f1:
            best_f1, stale = mean_source_f1, 0
            save_best(checkpoint_path, model, None, method=method, seed=seed,
                      epoch=epoch, source_metrics=source_metrics,
                      mean_source_f1=mean_source_f1, amp_dtype=amp_dtype)
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return checkpoint_path, history


def evaluate_checkpoint(checkpoint_path: Path, history, protocol, device, amp_dtype):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ResNet18Adapter().to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_state"])
    source_metrics = checkpoint["source_val"]
    source_accuracy = float(np.mean([value["accuracy"] for value in source_metrics.values()]))
    source_f1 = float(np.mean([value["macro_f1"] for value in source_metrics.values()]))
    target = evaluate(model, protocol.target_eval, device, amp_dtype)
    maximum_loss = float(pd.DataFrame(history)[["classification_loss", "adaptation_loss"]].max().max())
    return {
        "seed": int(checkpoint["seed"]),
        "method": checkpoint["method"],
        "selected_epoch": int(checkpoint["epoch"]),
        "mean_source_accuracy": source_accuracy,
        "mean_source_macro_f1": source_f1,
        "target_accuracy": target["accuracy"],
        "target_macro_f1": target["macro_f1"],
        "maximum_recorded_loss": maximum_loss,
        "collapsed": source_accuracy < COLLAPSE_SOURCE_ACCURACY and target["accuracy"] < COLLAPSE_TARGET_ACCURACY,
        "extreme_loss": maximum_loss > EXTREME_LOSS_THRESHOLD,
    }


def original_rows() -> pd.DataFrame:
    task2 = pd.read_csv(TASK2 / "results" / "main_comparison_table.csv").set_index("method")
    task3 = pd.read_csv(TASK3 / "results" / "main_comparison_table.csv").set_index("method")
    mapping = {
        "dan": (task2.loc["dan"], "target_acc", "target_f1", "task2/results/training_curves/dan.csv"),
        "dann": (task2.loc["dann"], "target_acc", "target_f1", "task2/results/training_curves/dann.csv"),
        "dan_dg": (task3.loc["DAN-DG"], "sketch_acc", "sketch_f1", "task3/results/training_curves/dan_dg.csv"),
    }
    rows = []
    for method, (row, target_acc, target_f1, curve_path) in mapping.items():
        curve = pd.read_csv(ROOT / curve_path)
        checkpoint_dir = TASK2 if method in {"dan", "dann"} else TASK3
        checkpoint = torch.load(checkpoint_dir / "results" / "checkpoints" /
                                f"{method}_seed{ORIGINAL_SEED}.pt",
                                map_location="cpu", weights_only=False)
        maximum_loss = float(curve[["classification_loss", curve.columns[2]]].max().max())
        rows.append({
            "seed": ORIGINAL_SEED,
            "method": method,
            "selected_epoch": int(checkpoint["epoch"]),
            "mean_source_accuracy": float(row.mean_source_acc),
            "mean_source_macro_f1": float(row.mean_source_f1),
            "target_accuracy": float(row[target_acc]),
            "target_macro_f1": float(row[target_f1]),
            "maximum_recorded_loss": maximum_loss,
            "collapsed": float(row.mean_source_acc) < COLLAPSE_SOURCE_ACCURACY and float(row[target_acc]) < COLLAPSE_TARGET_ACCURACY,
            "extreme_loss": maximum_loss > EXTREME_LOSS_THRESHOLD,
        })
    return pd.DataFrame(rows)


def write_outputs(new_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_frame = pd.DataFrame(new_rows).sort_values(["method", "seed"])
    new_frame.to_csv(OUT / "additional_seed_results.csv", index=False)
    combined = pd.concat((original_rows(), new_frame), ignore_index=True).sort_values(["method", "seed"])
    combined.to_csv(OUT / "combined_seed_results.csv", index=False)
    summary = combined.groupby("method", as_index=False).agg(
        seeds=("seed", "count"),
        source_accuracy_mean=("mean_source_accuracy", "mean"),
        source_accuracy_std=("mean_source_accuracy", "std"),
        target_accuracy_mean=("target_accuracy", "mean"),
        target_accuracy_std=("target_accuracy", "std"),
        collapse_count=("collapsed", "sum"),
        extreme_loss_count=("extreme_loss", "sum"),
    )
    summary["collapse_rate"] = summary.collapse_count / summary.seeds
    summary.to_csv(OUT / "multiseed_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for method, group in combined.groupby("method"):
        axes[0].plot(group.seed.astype(str), group.mean_source_accuracy, marker="o", label=method)
        axes[1].plot(group.seed.astype(str), group.target_accuracy, marker="o", label=method)
    axes[0].axhline(COLLAPSE_SOURCE_ACCURACY, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(COLLAPSE_TARGET_ACCURACY, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Training seed", ylabel="Mean source accuracy", ylim=(0, 1))
    axes[1].set(xlabel="Training seed", ylabel="Sketch accuracy", ylim=(0, 1))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(OUT / "multiseed_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return combined, summary


def protocol_record() -> dict:
    return {
        "original_split_seed": ORIGINAL_SEED,
        "additional_training_seeds": list(VALIDATION_SEEDS),
        "methods": {"dan": {"lambda_mmd": 1.0},
                    "dann": {"grl_ceiling": 1.0},
                    "dan_dg": {"lambda_dg": 1.0}},
        "collapse_definition": {
            "mean_source_accuracy_below": COLLAPSE_SOURCE_ACCURACY,
            "target_accuracy_below": COLLAPSE_TARGET_ACCURACY,
        },
        "extreme_loss_definition": {"maximum_recorded_loss_above": EXTREME_LOSS_THRESHOLD},
        "selection": "best mean source validation macro-F1; patience 5; maximum 30 epochs",
        "target_labels_used_during_training_or_selection": False,
    }


def run_validation(*, force=False, allow_cpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("CUDA is required for the full validation. Pass allow_cpu=True only if the long CPU runtime is intentional.")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    (OUT / "protocol.json").write_text(json.dumps(protocol_record(), indent=2) + "\n")
    protocol = prepare_protocol()
    rows = []
    for seed in VALIDATION_SEEDS:
        for method in ("dan", "dann"):
            checkpoint, history = train_uda(method, seed, protocol, device, amp_dtype, force=force)
            rows.append(evaluate_checkpoint(checkpoint, history, protocol, device, amp_dtype))
            write_outputs(rows)
        checkpoint, history = train_dan_dg(seed, protocol, device, amp_dtype, force=force)
        rows.append(evaluate_checkpoint(checkpoint, history, protocol, device, amp_dtype))
        write_outputs(rows)
    return write_outputs(rows)


def check_protocol_only() -> dict:
    (OUT / "protocol.json").write_text(json.dumps(protocol_record(), indent=2) + "\n")
    protocol = prepare_protocol()
    result = {
        "source_train_sizes": {domain: len(protocol.source_train[domain]) for domain in SOURCE_DOMAINS},
        "source_validation_sizes": {domain: len(protocol.source_val[domain]) for domain in SOURCE_DOMAINS},
        "target_adaptation_size": len(protocol.target_adaptation),
        "target_evaluation_size": len(protocol.target_eval),
        "class_to_idx": protocol.class_to_idx,
    }
    (OUT / "protocol_check.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate data/splits without training")
    parser.add_argument("--force", action="store_true", help="Retrain even when validation checkpoints exist")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit the very slow full run without CUDA")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(check_protocol_only(), indent=2))
    else:
        combined, summary = run_validation(force=args.force, allow_cpu=args.allow_cpu)
        print(combined.to_string(index=False))
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
