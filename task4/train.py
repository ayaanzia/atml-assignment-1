"""CLI training entry point. It never imports or loads CIFAR-100."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from task4.data.cifar10 import build_cifar10_loaders
from task4.methods.gcsc import train_gcsc
from task4.methods.proser import PROSER, train_proser
from task4.methods.vanilla import train_vanilla

ROOT = Path(__file__).resolve().parent


def load_config(method: str):
    with (ROOT / "configs" / f"{method}.yaml").open() as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("vanilla", "gcsc", "proser"))
    parser.add_argument("--data-root", default=str(ROOT / "data" / "cifar"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    config = load_config(args.method)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_cifar10_loaders(
        args.data_root, config["batch_size"], config["seed"],
        randaugment=bool(config.get("randaugment", False)),
        download=not args.no_download, pin_memory=device.type == "cuda",
    )
    checkpoint = ROOT / "results" / "checkpoints" / f"{args.method}_seed6304.pt"
    if args.method == "vanilla":
        train_vanilla(loaders["train"], loaders["val"], config, checkpoint, device)
    elif args.method == "gcsc":
        train_gcsc(loaders["train"], loaders["val"], config, checkpoint, device)
    else:
        vanilla = ROOT / config["pretrained_checkpoint"]
        if not vanilla.exists():
            raise FileNotFoundError("Train Vanilla first: " + str(vanilla))
        model = PROSER.from_vanilla_checkpoint(vanilla, config["num_dummy_classes"])
        train_proser(model, loaders["train"], loaders["val"], config, checkpoint, device)


if __name__ == "__main__":
    main()
